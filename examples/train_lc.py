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
Example script for training a Learned Correction (LC) model
"""

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from apebench.scenarios import scenario_dict
from pdequinox import cycling_dataloader

from pde_emulator.models import LC
from pde_emulator.utils import (
    load_and_combine_data_correction, 
    create_optimizer, 
    correction_loss_fn, 
    train_model,
    evaluate_and_visualize,
    perform_correction_rollout
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
TRAIN_SCENARIOS = {
    'diff_kdv': [
        (b, eps, zet)
        for b in [-2.0, -1.5, -1.0]
        for eps in [-20.0, -13.0, -7.0]
        for zet in [-9.0, -6.0, -3.0]
    ],
    'diff_ks_cons': [
        (b, nu, zet)
        for b in [-2.0, -1.5, -1.0]
        for nu in [-2.0, -1.25, -0.5]
        for zet in [-27.0, -20.0, -12.0]
    ],
    'diff_fisher': [
        (r, -r, nu)
        for r in [-0.05, -0.03, -0.01]
        for nu in [0.2, 2.5, 5.0]
    ],
    'diff_adv_diff': [
        (c, nu)
        for c in [-4.0, 0.0, 4.0]
        for nu in [2.0, 5.0, 8.0]
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
SEED = 42
NUM_TRAIN_STEPS = 100000 
BATCH_SIZE = 128
WARMUP_STEPS = 5000
PEAK_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

def main():
    # Initialize JAX random keys
    key = jax.random.PRNGKey(SEED)
    train_key, model_key, loader_key = jax.random.split(key, 3)
    
    print("Loading correction data...")
    u_coarse_in, corrections_true, encodings = load_and_combine_data_correction(
        TRAIN_SCENARIOS,
        train_seed=SEED,
        scenario_param_names=SCENARIO_PARAM_NAMES,
        equation_coefficients=EQUATION_COEFFICIENTS,
        scenario_dict=scenario_dict
    )

    print("Creating LC model...")
    model = LC(
        num_spatial_dims=1, 
        in_channels=1, 
        encoding_dim=ENCODING_DIM, 
        key=model_key
    )

    print("Setting up optimizer...")
    optimizer = create_optimizer(
        peak_lr=PEAK_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=NUM_TRAIN_STEPS,
        weight_decay=WEIGHT_DECAY
    )

    print("Creating data loader...")
    train_data_loader = cycling_dataloader(
        (u_coarse_in, corrections_true, encodings),
        batch_size=BATCH_SIZE,
        num_steps=NUM_TRAIN_STEPS,
        key=loader_key
    )

    print("Starting training...")
    trained_model, losses = train_model(
        model=model,
        data_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=correction_loss_fn,
        num_steps=NUM_TRAIN_STEPS,
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
    plt.savefig('lc_training_loss.png')
    print("Training loss plot saved as 'lc_training_loss.png'")

    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'lc_model.eqx')
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
        plot_samples=1,
        rollout_fn=perform_correction_rollout,
        CORRECTION_MODE=True  # This flag tells the evaluator to get the coarse stepper for each scenario
    )

    # Print final test score
    total_score = sum(test_results.values())
    print(f"\nFinal Test Score (GMean of nRMSE): {total_score:.6f}")

if __name__ == "__main__":
    main()
