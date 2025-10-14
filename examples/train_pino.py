# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for training a PINO model
"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from apebench.scenarios import scenario_dict
from pdequinox import cycling_dataloader

from pde_emulator.models import PINO
from pde_emulator.utils import (
    load_and_combine_data_for_unroll, 
    create_optimizer, 
    create_pino_schedule,
    unrolled_pino_loss_fn, 
    train_PINOmodel,
    perform_rollout,
    evaluate_and_visualize
)

# --- Configuration ---

# Define scenario configurations
SCENARIO_PARAM_NAMES = {
    'diff_burgers':  ['convection_delta', 'diffusion_gamma'],
    'diff_kdv': ['convection_sc_delta', 'dispersion_gamma', 'hyp_diffusion_gamma'],
    'diff_ks_cons': ['convection_delta', 'diffusion_gamma', 'hyp_diffusion_gamma'],
    'diff_fisher': ['quadratic_delta', 'drag_gamma', 'diffusion_gamma'],
    'diff_adv_diff': ['advection_gamma', 'diffusion_gamma'],
}

ENCODING_DIM = 7
EQUATION_COEFFICIENTS = {
    'diff_burgers': lambda b, nu: jnp.array([0., 0., 0., b, nu, 0., 0.]), # hold out test
    'diff_kdv': lambda b, eps, zet: jnp.array([0., 0., 0., b, 0., eps, zet]),
    'diff_ks_cons': lambda b, nu, zet: jnp.array([0., 0., 0., b, nu, 0., zet]),
    'diff_fisher': lambda rg, rq, nu: jnp.array([rg, rq, 0., 0., nu, 0., 0.]),
    'diff_adv_diff': lambda c, nu: jnp.array([0., 0., c, 0., nu, 0., 0.]),
}

# Define training scenarios
# Use fewer points for the parameters in trade-off for the unrolled training
TRAIN_SCENARIOS = {
    'diff_kdv': [
        (b, eps, zet)
        for b in [-2.0, -1.5, -1.0]
        for eps in [-20.0, -16.5, -13.0, -9.5, -7.0]
        for zet in [-9.0, -6.0, -3.0]
    ],
    'diff_ks_cons': [
        (b, nu, zet)
        for b in [-2.0, -1.5, -1.0]
        for nu in [-2.0, -1.25, -0.5]
        for zet in [-27.0, -22.0, -17.0, -12.0]
    ],
    'diff_fisher': [
        (r, -r, nu)
        for r in [-0.05, -0.03, -0.01]
        for nu in [0.2, 1.5, 3.0, 5.0]
    ],
    'diff_adv_diff': [
        (c, nu)
        for c in [-4.0, -2.0, 0.0, 2.0, 4.0]
        for nu in [2.0, 4.0, 6.0, 8.0]
    ],
}

# Define test scenarios
TEST_SCENARIOS = {
    'diff_burgers': [(-1.5, 1.5), (-1.5, 5.0)],
    'diff_kdv': [(-1.8, -10.0, -9.0), (-2.0, -5.0, -9.0)],
    'diff_ks_cons': [(-1.8, -1.5, -23.0), (-1.2, -1.0, -15.0)],
    'diff_fisher': [(-0.04, 0.04, 1.5), (-0.02, 0.02, 4.0)],
    'diff_adv_diff': [(-3.0, 3.0), (2.0, 6.0)],
}

# --- Training Setup ---
DOMAIN_SIZE = 1.0
SPATIAL_RESOLUTION = 160
DX = DOMAIN_SIZE / SPATIAL_RESOLUTION
DT = 1 # Time step size for APEBench data
PINO_MAX_WEIGHT = 3.0e-3 
PINO_WARMUP_STEPS = 60#60000

SEED = 42
UNROLL_STEPS = 5
# Training Hyperparameters
NUM_TRAIN_STEPS = 100000 
BATCH_SIZE = 64
WARMUP_STEPS = 5000
PEAK_LEARNING_RATE = 3e-4 
WEIGHT_DECAY = 1e-5

def main():
    # Initialize JAX random keys
    key = jax.random.PRNGKey(SEED)
    train_key, model_key, loader_key = jax.random.split(key, 3)
    
    print("Loading data...")
    u_initial_flat, u_true_future_flat, encodings_flat = load_and_combine_data_for_unroll(TRAIN_SCENARIOS, train_seed=SEED, scenario_param_names=SCENARIO_PARAM_NAMES, equation_coefficients=EQUATION_COEFFICIENTS, scenario_dict=scenario_dict, unroll_steps=UNROLL_STEPS)

    # Calculate global data normalization constants after data loading
    DATA_MEAN_U = jnp.mean(u_initial_flat)
    DATA_STD_U = jnp.std(u_initial_flat)
    if DATA_STD_U < 1e-6: # Prevent division by zero if data is nearly constant
        DATA_STD_U = 1.0 
    ENCODING_MIN_VALS = jnp.min(encodings_flat, axis=0)
    ENCODING_MAX_VALS = jnp.max(encodings_flat, axis=0)

    print("Creating PINO model...")
    num_spatial_dims = 1
    in_channels = 1
    model = PINO(
        num_spatial_dims=num_spatial_dims,
        in_channels_u=in_channels, # Pass the original 'u' channels
        encoding_dim=ENCODING_DIM,
        key=model_key,
        encoding_min_vals=ENCODING_MIN_VALS,
        encoding_max_vals=ENCODING_MAX_VALS,
        data_mean=DATA_MEAN_U,
        data_std=DATA_STD_U,
    )

    print("Setting up optimizer...")
    optimizer = create_optimizer(
        peak_lr=PEAK_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=NUM_TRAIN_STEPS,
        weight_decay=WEIGHT_DECAY
    )

    pino_weight_schedule_fn = create_pino_schedule(
        max_weight=PINO_MAX_WEIGHT,
        warmup_steps=PINO_WARMUP_STEPS
    )


    print("Creating data loader...")
    train_data_loader = cycling_dataloader(
        (u_initial_flat, u_true_future_flat, encodings_flat),
        batch_size=BATCH_SIZE,
        num_steps=NUM_TRAIN_STEPS,
        key=loader_key
    )

    print("Starting training...")
    trained_model, losses = train_PINOmodel(
        model=model,
        data_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=unrolled_pino_loss_fn,
        num_steps=NUM_TRAIN_STEPS,
        DX=DX,
        DT=DT,
        pino_weight_schedule_fn=pino_weight_schedule_fn,
        unroll_steps=UNROLL_STEPS,
        perform_rollout_fn=perform_rollout,
        verbose=True
    )

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    print("Training loss plot saved as 'training_loss.png'")

    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'pino_model.eqx')
    eqx.tree_serialise_leaves(model_path, trained_model)
    print(f"Model saved to {model_path}")

    # Evaluate on test set
    print("\nEvaluating on test scenarios...")
    test_results, test_curves = evaluate_and_visualize(
        model=trained_model,
        test_scenarios=TEST_SCENARIOS,
        test_seed=SEED+1,
        scenario_param_names=SCENARIO_PARAM_NAMES,
        equation_coefficients=EQUATION_COEFFICIENTS,
        scenario_dict=scenario_dict,
        plot_samples=1
    )

    # Print final test score
    total_score = sum(test_results.values())
    print(f"\nFinal Test Score (GMean of nRMSE): {total_score:.6f}")

if __name__ == "__main__":
    main()
