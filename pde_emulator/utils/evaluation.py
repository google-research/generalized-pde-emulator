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
Utilities for evaluating PDE emulators
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from typing import Tuple, Callable, Dict, List, Optional, Union, Any
import exponax as ex
from exponax import metrics as ex_metrics

from pde_emulator.utils.data import get_equation_encoding

def perform_rollout(model: eqx.Module, ic: jax.Array, num_steps: int, encoding: jax.Array) -> jax.Array:
    """
    Perform an autoregressive rollout for a given model.
    
    Args:
        model: PDE emulator model
        ic: Initial condition
        num_steps: Number of steps to roll out
        encoding: Equation encoding vector
        
    Returns:
        Predicted trajectory
    """
    stepper_fn = lambda u: model(u, encoding)
    # The rollout function repeatedly applies the stepper
    trajectory = ex.rollout(stepper_fn, n=num_steps, include_init=False)(ic)
    return trajectory


def perform_correction_rollout(model: eqx.Module, ic: jax.Array, num_steps: int, encoding: jax.Array, coarse_stepper) -> jax.Array:
    """
    Perform an autoregressive rollout for a correction model.
    
    Args:
        model: PDE emulator correction model
        ic: Initial condition
        num_steps: Number of steps to roll out
        encoding: Equation encoding vector
        coarse_stepper: Coarse solver stepper object
        
    Returns:
        Predicted trajectory
    """
    # Define the combined one-step function: coarse step + learned correction
    def combined_stepper_fn(u_prev):
        # 1. Get coarse prediction from the previous state
        u_coarse_next = coarse_stepper.step(u_prev)
        # 2. Get model's correction based on the coarse prediction
        correction = model(u_coarse_next, encoding)
        # 3. Apply correction to get the final prediction
        u_next = u_coarse_next + correction
        return u_next

    # The rollout utility repeatedly applies this combined stepper
    trajectory = ex.rollout(combined_stepper_fn, n=num_steps, include_init=False)(ic)
    return trajectory


def evaluate_on_scenario(model: eqx.Module, scenario_name: str, param_setting,
                         test_seed: int, scenario_param_names: Dict,
                         equation_coefficients: Dict, scenario_dict: Dict,
                         rollout_fn: Callable = perform_rollout,
                         CORRECTION_MODE=False) -> Tuple[float, jax.Array]:
    """
    Evaluate a model's performance on a specific scenario.
    
    Args:
        model: PDE emulator model
        scenario_name: Name of the PDE scenario
        param_setting: Parameter values for the PDE
        test_seed: Random seed for test data generation
        scenario_param_names: Dictionary mapping from scenario names to parameter names
        equation_coefficients: Dictionary mapping from scenario names to encoding functions
        scenario_dict: Dictionary mapping from scenario names to scenario classes
        rollout_fn: Function to perform rollout (depends on model type)
        coarse_stepper: Coarse solver stepper (only used for correction models)
        
    Returns:
        Geometric mean of nRMSE and nRMSE at each step
    """
    param_keys = scenario_param_names[scenario_name]
    param_values = param_setting if isinstance(param_setting, (list, tuple)) else (param_setting,)
    
    if len(param_keys) != len(param_values):
        raise ValueError(f"Mismatch between param_keys {param_keys} and param_values {param_values} for {scenario_name}")

    # 1. Get the test data for this specific scenario
    scenario_kwargs = dict(zip(param_keys, param_values))
    scenario_kwargs['test_seed'] = test_seed
    test_scenario = scenario_dict[scenario_name](**scenario_kwargs)
    test_ics = test_scenario.get_test_ic_set()
    ref_trjs = test_scenario.get_test_data()[:, 1:] # Exclude initial condition
    num_rollout_steps = ref_trjs.shape[1]

    # 2. Get the corresponding equation encoding
    encoding = get_equation_encoding(scenario_name, param_values, equation_coefficients)

    # 3. Perform rollouts for all initial conditions
    if CORRECTION_MODE == True:
        coarse_stepper = test_scenario.get_coarse_stepper()
        # For correction models, we need to provide the coarse stepper
        pred_trjs = jax.vmap(rollout_fn, in_axes=(None, 0, None, None, None))(
            model, test_ics, num_rollout_steps, encoding, coarse_stepper
        )
    else:
        # For standard emulator models
        pred_trjs = jax.vmap(rollout_fn, in_axes=(None, 0, None, None))(
            model, test_ics, num_rollout_steps, encoding
        )

    # 4. Calculate mean nRMSE at each time step
    mean_nrmse_on_batch = lambda pred_batch, ref_batch: ex_metrics.mean_metric(
        ex_metrics.nRMSE, pred_batch, ref_batch
    )
    mean_nrmse_at_each_step = jax.vmap(mean_nrmse_on_batch, in_axes=(1, 1))(pred_trjs, ref_trjs)

    # 5. Aggregate with Geometric Mean over the first 100 steps
    gmean_nrmse = gmean(mean_nrmse_at_each_step[:100])

    return gmean_nrmse, mean_nrmse_at_each_step


def plot_rollout_comparison(truth, prediction, title="Model Prediction vs. Ground Truth", figsize=(18, 5)):
    """
    Generate comparison plots between ground truth and model prediction.
    
    Args:
        truth: Ground truth trajectory
        prediction: Model prediction trajectory
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    error = np.abs(truth - prediction)
    
    # Determine consistent color limits for truth and prediction plots
    vmin, vmax = truth.min(), truth.max()
    
    # Create the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Ground Truth
    im1 = axes[0].imshow(truth, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    
    # Model Prediction
    im2 = axes[1].imshow(prediction, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Model Prediction')
    
    # Absolute Error
    im3 = axes[2].imshow(error, aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Error')
    
    # Add colorbars and labels
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('Spatial Dimension (x)')
        ax.set_ylabel('Time Steps')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def evaluate_and_visualize(model: eqx.Module, test_scenarios: Dict, test_seed: int, scenario_param_names: Dict,
                          equation_coefficients: Dict, scenario_dict: Dict, plot_samples: int = 1,
                          rollout_fn: Callable = perform_rollout, CORRECTION_MODE=False) -> Dict:
    """
    Evaluate model on test scenarios and generate visualizations.
    
    Args:
        model: PDE emulator model
        test_scenarios: Dictionary of scenarios and parameter settings to test
        test_seed: Random seed for test data generation
        scenario_param_names: Dictionary mapping from scenario names to parameter names
        equation_coefficients: Dictionary mapping from scenario names to encoding functions
        scenario_dict: Dictionary mapping from scenario names to scenario classes
        plot_samples: Number of samples to plot for each scenario
        rollout_fn: Function to perform rollout (depends on model type)
        coarse_stepper: Coarse solver stepper (only used for correction models)
        
    Returns:
        Dictionary of test results
    """
    test_results = {}
    test_rollout_curves = {}
    
    for scenario_name, list_of_param_settings in test_scenarios.items():
        param_keys = scenario_param_names[scenario_name]
        for param_setting in list_of_param_settings:
            test_score, nrmse_curve = evaluate_on_scenario(
                    model, scenario_name, param_setting, test_seed,
                    scenario_param_names, equation_coefficients, scenario_dict,
                    rollout_fn=rollout_fn, CORRECTION_MODE=CORRECTION_MODE
                )
            # Format parameter string for display
            if isinstance(param_setting, (list, tuple)):
                param_str = ','.join([f"{k.split('_')[0]}={v}" for k, v in zip(param_keys, param_setting)])
            else:
                param_str = f"{param_keys[0].split('_')[0]}={param_setting}"
            result_key = f"{scenario_name}::{param_str}"
            
            test_results[result_key] = float(test_score)
            test_rollout_curves[result_key] = nrmse_curve
            
            print(f"  > Test Score (GMean of nRMSE) for {result_key}: {test_score:.4f}")
            
            # Generate visualizations for selected samples
            if plot_samples > 0:
                # Get test data
                scenario_kwargs = dict(zip(param_keys, param_setting if isinstance(param_setting, (list, tuple)) else [param_setting]))
                scenario_kwargs.update({'test_seed': test_seed, 'num_test_samples': plot_samples})
                scenario = scenario_dict[scenario_name](**scenario_kwargs)
                
                # Get ground truth trajectories and initial conditions
                ref_trajectories_full = scenario.get_test_data()
                initial_conditions = ref_trajectories_full[:, 0]
                num_rollout_steps = ref_trajectories_full.shape[1] - 1
                
                # Get the corresponding equation encoding for the model
                encoding = get_equation_encoding(scenario_name, param_setting, equation_coefficients)
                
                # Perform the rollout using the trained model
                if CORRECTION_MODE == True:
                    coarse_stepper = scenario.get_coarse_stepper()
                    # Perform the rollout using the trained model
                    pred_trajectories = jax.vmap(rollout_fn, in_axes=(None, 0, None, None, None))(
                        model, initial_conditions, num_rollout_steps, encoding, coarse_stepper
                    )
                else:
                    pred_trajectories = jax.vmap(rollout_fn, in_axes=(None, 0, None, None))(
                        model, initial_conditions, num_rollout_steps, encoding
                    )
                    
                pred_trajectories_full = jnp.concatenate([initial_conditions[:, None, ...], pred_trajectories], axis=1)
                
                # Plot each sample
                for i in range(plot_samples):
                    truth = np.array(ref_trajectories_full[i, :, 0, :])
                    prediction = np.array(pred_trajectories_full[i, :, 0, :])
                    
                    plot_rollout_comparison(truth, prediction, title=f'Sample Rollout for {scenario_name} ({param_str})')
                    plt.show()
    
    return test_results, test_rollout_curves
