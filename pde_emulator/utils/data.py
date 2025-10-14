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
Utilities for data processing, equation encodings, and handling PDE data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm.auto import tqdm
from typing import List, Dict, Union, Tuple, Callable, Optional


def get_equation_encoding(
    scenario_name: str,
    param_values: Union[float, List[float], Tuple[float, ...]],
    equation_coefficients: Dict
) -> jnp.ndarray:
    """
    Returns the encoding vector for a given PDE scenario and parameter values.

    Args:
        scenario_name: Name of the PDE scenario.
        param_values: Parameter values for the PDE.
        equation_coefficients: Dictionary mapping scenario names to encoding functions.

    Returns:
        Encoding vector for the PDE.
    """
    if scenario_name not in equation_coefficients:
        raise ValueError(f"Unknown scenario for encoding: {scenario_name}")
    if not isinstance(param_values, (list, tuple)):
        param_values = (param_values,)
    return equation_coefficients[scenario_name](*param_values)


def load_and_combine_data(
    scenarios_to_load: Dict[str, List],
    train_seed: int,
    scenario_param_names: Dict,
    equation_coefficients: Dict,
    scenario_dict: Dict
):
    """
    Generates and combines training data and equation encodings from multiple scenarios.

    Args:
        scenarios_to_load: Dictionary of scenarios and parameter settings to load.
        train_seed: Random seed for data generation.
        scenario_param_names: Dictionary mapping scenario names to parameter names.
        equation_coefficients: Dictionary mapping scenario names to encoding functions.
        scenario_dict: Dictionary mapping scenario names to scenario classes.

    Returns:
        u_t_flat: Input states, shape (steps, channels, space).
        u_t_plus_1_flat: Target states, shape (steps, channels, space).
        encodings_flat: Equation encodings, shape (steps, encoding_dim).
        data_mean: Mean of training data.
        data_std: Standard deviation of training data.
        encoding_mean: Mean of equation encodings.
        encoding_std: Standard deviation of equation encodings.
        encoding_min: Minimum values of equation encodings.
        encoding_max: Maximum values of equation encodings.
    """
    all_trajectories = []
    all_encodings = []
    raw_encoding_vectors = []

    print(f"Generating data for {len(scenarios_to_load)} equations...")
    for scenario_name, list_of_param_settings in tqdm(scenarios_to_load.items()):
        param_keys = scenario_param_names[scenario_name]
        for param_setting in list_of_param_settings:
            param_values = param_setting if isinstance(param_setting, (list, tuple)) else (param_setting,)
            if len(param_keys) != len(param_values):
                raise ValueError(f"Mismatch between param_keys {param_keys} and param_values {param_values} for {scenario_name}")
            scenario_kwargs = dict(zip(param_keys, param_values))
            scenario_kwargs['train_seed'] = train_seed
            scenario = scenario_dict[scenario_name](**scenario_kwargs)
            trajectories = scenario.get_train_data()
            num_samples = trajectories.shape[0]
            if np.isnan(trajectories).any():
                print(f"{scenario_name}, {param_values} has nans")
                continue
            encoding_vector = get_equation_encoding(scenario_name, param_values, equation_coefficients)
            all_trajectories.append(trajectories)
            encodings_for_samples = jnp.tile(encoding_vector, (num_samples, 1))
            all_encodings.append(encodings_for_samples)
            raw_encoding_vectors.append(encoding_vector)

    combined_trajectories = jnp.concatenate(all_trajectories, axis=0)
    combined_encodings = jnp.concatenate(all_encodings, axis=0)
    combined_raw_encodings = jnp.stack(raw_encoding_vectors)

    data_mean = jnp.mean(combined_trajectories)
    data_std = jnp.std(combined_trajectories)
    encoding_mean = jnp.mean(combined_raw_encodings, axis=0)
    encoding_std = jnp.std(combined_raw_encodings, axis=0)
    encoding_min = jnp.min(combined_raw_encodings, axis=0)
    encoding_max = jnp.max(combined_raw_encodings, axis=0)

    u_t = combined_trajectories[:, :-1]
    u_t_plus_1 = combined_trajectories[:, 1:]
    encodings_for_steps = jnp.tile(combined_encodings[:, None, :], (1, u_t.shape[1], 1))

    num_total_samples, num_time_steps, C, N = u_t.shape
    u_t_flat = u_t.reshape(num_total_samples * num_time_steps, C, N)
    u_t_plus_1_flat = u_t_plus_1.reshape(num_total_samples * num_time_steps, C, N)
    encodings_flat = encodings_for_steps.reshape(num_total_samples * num_time_steps, encoding_vector.shape[0])

    print(f"Total training steps created: {u_t_flat.shape[0]}")
    return (
        u_t_flat,
        u_t_plus_1_flat,
        encodings_flat,
        data_mean,
        data_std,
        encoding_mean,
        encoding_std,
        encoding_min,
        encoding_max,
    )


def load_and_combine_data_for_unroll(
    scenarios_to_load: Dict[str, List],
    train_seed: int,
    scenario_param_names: Dict,
    equation_coefficients: Dict,
    scenario_dict: Dict,
    unroll_steps: int
):
    """
    Generates and combines training data for unrolled training.

    Args:
        scenarios_to_load: Dictionary of scenarios and parameter settings to load.
        train_seed: Random seed for data generation.
        scenario_param_names: Dictionary mapping scenario names to parameter names.
        equation_coefficients: Dictionary mapping scenario names to encoding functions.
        scenario_dict: Dictionary mapping scenario names to scenario classes.
        unroll_steps: Number of steps to unroll during training.

    Returns:
        u_initial_flat: Initial states, shape (sequences, channels, space).
        u_true_future_flat: Future states for unrolling, shape (sequences, unroll_steps, channels, space).
        encodings_flat_unroll: Equation encodings, shape (sequences, encoding_dim).
    """
    combined_u_initial = []
    combined_u_true_future = []
    combined_encodings_unroll = []

    print(f"Generating data for {len(scenarios_to_load)} equations for unrolled training (unroll_steps={unroll_steps})...")
    for scenario_name, list_of_param_settings in tqdm(scenarios_to_load.items()):
        param_keys = scenario_param_names[scenario_name]
        for param_setting in list_of_param_settings:
            param_values = param_setting if isinstance(param_setting, (list, tuple)) else (param_setting,)
            if len(param_keys) != len(param_values):
                raise ValueError(f"Mismatch between param_keys {param_keys} and param_values {param_values} for {scenario_name}")
            scenario_kwargs = dict(zip(param_keys, param_values))
            scenario_kwargs['train_seed'] = train_seed
            scenario = scenario_dict[scenario_name](**scenario_kwargs)
            trajectories = scenario.get_train_data()
            num_samples = trajectories.shape[0]
            num_time_steps = trajectories.shape[1]
            if np.isnan(trajectories).any():
                print(f"{scenario_name}, {param_values} has nans")
                continue
            encoding_vector = get_equation_encoding(scenario_name, param_values, equation_coefficients)
            if num_time_steps < unroll_steps + 1:
                print(f"Warning: Scenario {scenario_name} param {param_values} has too few time steps ({num_time_steps}) for unroll_steps ({unroll_steps}). Skipping.")
                continue
            for i in range(num_samples):
                for t in range(num_time_steps - unroll_steps):
                    combined_u_initial.append(trajectories[i, t])
                    combined_u_true_future.append(trajectories[i, t+1 : t+1+unroll_steps])
                    combined_encodings_unroll.append(encoding_vector)

    u_initial_flat = jnp.stack(combined_u_initial, axis=0)
    u_true_future_flat = jnp.stack(combined_u_true_future, axis=0)
    encodings_flat_unroll = jnp.stack(combined_encodings_unroll, axis=0)

    print(f"Total training sequences created: {u_initial_flat.shape[0]}")
    return u_initial_flat, u_true_future_flat, encodings_flat_unroll


def load_and_combine_data_correction(
    scenarios_to_load: Dict[str, List],
    train_seed: int,
    scenario_param_names: Dict,
    equation_coefficients: Dict,
    scenario_dict: Dict
):
    """
    Generates and combines training data for the correction task.

    Args:
        scenarios_to_load: Dictionary of scenarios and parameter settings to load.
        train_seed: Random seed for data generation.
        scenario_param_names: Dictionary mapping scenario names to parameter names.
        equation_coefficients: Dictionary mapping scenario names to encoding functions.
        scenario_dict: Dictionary mapping scenario names to scenario classes.

    Returns:
        combined_coarse_preds: Coarse solver predictions, shape (steps, channels, space).
        combined_corrections: True correction values, shape (steps, channels, space).
        combined_encodings: Equation encodings, shape (steps, encoding_dim).
    """
    all_coarse_preds = []
    all_corrections = []
    all_encodings = []

    print(f"Generating correction data for {len(scenarios_to_load)} equations...")
    for scenario_name, list_of_param_settings in tqdm(scenarios_to_load.items()):
        param_keys = scenario_param_names[scenario_name]
        for param_setting in list_of_param_settings:
            param_values = param_setting if isinstance(param_setting, (list, tuple)) else (param_setting,)
            scenario_kwargs = dict(zip(param_keys, param_values))
            scenario_kwargs['train_seed'] = train_seed
            scenario = scenario_dict[scenario_name](**scenario_kwargs)
            coarse_stepper = scenario.get_coarse_stepper()
            ref_trajectories = scenario.get_train_data()
            num_samples = ref_trajectories.shape[0]
            if np.isnan(ref_trajectories).any():
                print(f"Warning: NaN found in reference data for {scenario_name}, {param_values}. Skipping.")
                continue
            u_t_true = ref_trajectories[:, :-1]
            u_t_plus_1_true = ref_trajectories[:, 1:]
            num_total_samples, num_time_steps, C, N = u_t_true.shape
            u_t_true_flat = u_t_true.reshape(num_total_samples * num_time_steps, C, N)
            u_t_plus_1_coarse_flat = jax.vmap(coarse_stepper.step)(u_t_true_flat)
            u_t_plus_1_true_flat = u_t_plus_1_true.reshape(num_total_samples * num_time_steps, C, N)
            correction_flat = u_t_plus_1_true_flat - u_t_plus_1_coarse_flat
            if np.isnan(u_t_plus_1_coarse_flat).any() or np.isnan(correction_flat).any():
                print(f"Warning: NaN found in coarse data or correction for {scenario_name}, {param_values}. Skipping.")
                continue
            encoding_vector = get_equation_encoding(scenario_name, param_values, equation_coefficients)
            encodings_for_samples = jnp.tile(encoding_vector, (num_samples, 1))
            encodings_for_steps = jnp.tile(encodings_for_samples[:, None, :], (1, u_t_true.shape[1], 1))
            encodings_flat = encodings_for_steps.reshape(num_total_samples * num_time_steps, encoding_vector.shape[0])
            all_coarse_preds.append(u_t_plus_1_coarse_flat)
            all_corrections.append(correction_flat)
            all_encodings.append(encodings_flat)

    combined_coarse_preds = jnp.concatenate(all_coarse_preds, axis=0)
    combined_corrections = jnp.concatenate(all_corrections, axis=0)
    combined_encodings = jnp.concatenate(all_encodings, axis=0)

    print(f"Total training steps created: {combined_coarse_preds.shape[0]}")
    return combined_coarse_preds, combined_corrections, combined_encodings


@eqx.filter_jit
def child_spectral_deriv(u: jax.Array, dx: float, order: int) -> jax.Array:
    """
    Computes the n-th order spatial derivative of u using Fourier transforms.

    Args:
        u: Array of shape (channels, spatial_points).
        dx: Spatial grid spacing.
        order: Order of the derivative (e.g., 1 for u_x, 2 for u_xx).

    Returns:
        Array of the same shape as u, containing the n-th order derivative.
    """
    N = u.shape[-1]
    k_physical = jnp.fft.rfftfreq(N, d=dx) * 2 * jnp.pi
    ik_pow_n = jnp.power(1j * k_physical, order)
    u_hat = jnp.fft.rfft(u, axis=-1)
    u_deriv_hat = u_hat * jnp.expand_dims(ik_pow_n, axis=0)
    u_deriv = jnp.fft.irfft(u_deriv_hat, n=N, axis=-1)
    return u_deriv


def get_pde_coefficients(encoding: jax.Array) -> Dict[str, float]:
    """
    Parses the 7-dimensional encoding vector into named PDE coefficients.

    Args:
        encoding: Array of shape (7,).

    Returns:
        Dictionary mapping coefficient names to their values.
    """
    return {
        'c_u': encoding[0],
        'c_u2': encoding[1],
        'c_ux': encoding[2],
        'c_uux': encoding[3],
        'c_uxx': encoding[4],
        'c_uxxx': encoding[5],
        'c_uxxxx': encoding[6],
    }


@eqx.filter_jit
def calculate_pde_residual(
    u_current: jax.Array,
    u_next_true: jax.Array,
    encoding: jax.Array,
    dx: float,
    dt: float
) -> jax.Array:
    """
    Calculates the PDE residual given the current true state, next true state,
    equation encoding, and grid parameters.

    Args:
        u_current: Array of shape (channels, spatial_points), u(t).
        u_next_true: Array of shape (channels, spatial_points), u(t+1).
        encoding: Array of shape (encoding_dim,).
        dx: Spatial grid spacing.
        dt: Time step.

    Returns:
        Array of residuals, same shape as u_current.
    """
    coeffs = get_pde_coefficients(encoding)
    u_t_approx = (u_next_true - u_current) / dt
    u_x = child_spectral_deriv(u_current, dx, 1)
    u_xx = child_spectral_deriv(u_current, dx, 2)
    u_xxx = child_spectral_deriv(u_current, dx, 3)
    u_xxxx = child_spectral_deriv(u_current, dx, 4)
    rhs = (
        coeffs['c_u'] * u_current +
        coeffs['c_u2'] * u_current**2 +
        coeffs['c_ux'] * u_x +
        coeffs['c_uux'] * u_current * u_x +
        coeffs['c_uxx'] * u_xx +
        coeffs['c_uxxx'] * u_xxx +
        coeffs['c_uxxxx'] * u_xxxx
    )
    residual = u_t_approx - rhs
    return residual
