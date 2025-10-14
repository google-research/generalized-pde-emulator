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
Example script for training a Latent Space Corrector with FNO architecture
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from apebench.scenarios import scenario_dict
from pdequinox import cycling_dataloader

from pde_emulator.models import LSC_FNO
from pde_emulator.utils import (
    load_and_combine_data, 
    create_optimizer, 
    loss_fn, 
    train_model,
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
TRAIN_SCENARIOS = {
    'diff_kdv': [
        (b, eps, zet)
        for b in np.linspace(-2.0, -1.0, 6)
        for eps in np.linspace(-20.0, -7.0, 8)
        for zet in np.linspace(-9.0, -3.0, 6)
    ],
    'diff_ks_cons': [
        (b, nu, zet)
        for b in np.linspace(-2.0, -1.0, 6)
        for nu in np.linspace(-2.0, -0.5, 6)
        for zet in np.linspace(-27.0, -12.0, 8)
    ],
    'diff_fisher': [
        (r, -r, nu)
        for r in np.linspace(0.01, 0.05, 6)
        for nu in np.linspace(0.2, 5.0, 8)
    ],
    'diff_adv_diff': [
        (c, nu)
        for c in np.linspace(-4.0, 4.0, 8)
        for nu in np.linspace(2.0, 8.0, 8)
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
PEAK_LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5

def main():
    # Initialize JAX random keys
    key = jax.random.PRNGKey(SEED)
    train_key, model_key, loader_key = jax.random.split(key, 3)
    
    print("Loading data...")
    u_t, u_t_plus_1, encodings, data_mean, data_std, encoding_mean, encoding_std, _, _ = load_and_combine_data(
        TRAIN_SCENARIOS, 
        train_seed=SEED,
        scenario_param_names=SCENARIO_PARAM_NAMES,
        equation_coefficients=EQUATION_COEFFICIENTS, 
        scenario_dict=scenario_dict
    )

    print("Creating LSC-FNO model...")
    num_spatial_dims = 1
    in_channels = 1
    spatial_dim_size = u_t.shape[-1]
    model = LSC_FNO(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        encoding_dim=ENCODING_DIM,
        spatial_dim_size=spatial_dim_size,
        data_mean=data_mean, # Pass normalization stats
        data_std=data_std,   # Pass normalization stats
        encoding_mean=encoding_mean, # Pass encoding normalization stats
        encoding_std=encoding_std,   # Pass encoding normalization stats
        key=model_key,
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
        (u_t, u_t_plus_1, encodings),
        batch_size=BATCH_SIZE,
        num_steps=NUM_TRAIN_STEPS,
        key=loader_key
    )

    print("Starting training...")
    trained_model, losses = train_model(
        model=model,
        data_loader=train_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
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
    plt.savefig('lsc_fno_training_loss.png')
    print("Training loss plot saved as 'lsc_fno_training_loss.png'")

    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'lsc_fno_model.eqx')
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
