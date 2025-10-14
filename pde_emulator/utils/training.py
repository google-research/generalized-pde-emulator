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
Utilities for training PDE emulators
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from tqdm.auto import tqdm
from typing import Tuple, Callable, Dict, List, Optional, Union, Any
import exponax as ex
from pde_emulator.utils.data import calculate_pde_residual


def create_optimizer(peak_lr: float, warmup_steps: int, total_steps: int, weight_decay: float = 1e-5):
    """
    Create optimizer with warmup and cosine decay schedule.
    
    Args:
        peak_lr: Peak learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        weight_decay: Weight decay factor for adamw
        
    Returns:
        Optimizer
    """
    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps),
            optax.cosine_decay_schedule(init_value=peak_lr, decay_steps=total_steps - warmup_steps)
        ],
        boundaries=[warmup_steps]
    )
    
    return optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay)


def create_pino_schedule(max_weight: float, warmup_steps: int,):
    """
    Create weight schedule for PINO (Physics-Informed Neural Operator) loss.
    
    Args:
        max_weight: Maximum weight for PINO loss term
        warmup_steps: Number of steps for linear warmup
        
    Returns:
        Schedule function for PINO weight
    """
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=max_weight, transition_steps=warmup_steps),
            optax.constant_schedule(max_weight) # Keep constant after warmup
        ],
        boundaries=[warmup_steps]
    )


def loss_fn(model: eqx.Module, u_t: jax.Array, u_t_plus_1: jax.Array, encoding: jax.Array) -> jax.Array:
    """
    Standard one-step prediction loss.
    
    Args:
        model: PDE emulator model
        u_t: Current state
        u_t_plus_1: Target next state
        encoding: Equation encoding vector
        
    Returns:
        L1 loss between prediction and target
    """
    pred_u_t_plus_1 = model(u_t, encoding)
    return jnp.mean(jnp.abs(pred_u_t_plus_1 - u_t_plus_1))


def pino_loss_fn(model: eqx.Module, u_t: jax.Array, u_t_plus_1: jax.Array, encoding: jax.Array, 
                dx: float, dt: float, pino_weight: float, spectral_deriv_fn: Callable) -> jax.Array:
    """
    Loss function incorporating PINO (Physics-Informed Neural Operator) residual term.
    
    Args:
        model: PDE emulator model
        u_t: Current state
        u_t_plus_1: Target next state
        encoding: Equation encoding vector
        dx: Spatial grid spacing
        dt: Time step size
        pino_weight: Weight for PINO loss term
        spectral_deriv_fn: Function to compute spectral derivatives
        
    Returns:
        Combined data and physics-informed loss
    """
    pred_u_t_plus_1 = model(u_t, encoding)
    
    # Data fitting loss
    data_loss = jnp.mean(jnp.abs(pred_u_t_plus_1 - u_t_plus_1))
    
    # Physics-informed loss: PDE residual on ground truth data
    residual = calculate_pde_residual(u_t, u_t_plus_1, encoding, dx, dt, spectral_deriv_fn)
    physics_loss = jnp.mean(jnp.abs(residual))
    
    # Combine losses
    total_loss = data_loss + pino_weight * physics_loss
    
    return total_loss


def unrolled_loss_fn(model: eqx.Module, u_initial: jax.Array, u_true_future_steps: jax.Array, 
                    encoding: jax.Array, unroll_steps: int, perform_rollout_fn: Callable) -> jax.Array:
    """
    Loss function for unrolled training.
    
    Args:
        model: PDE emulator model
        u_initial: Initial state
        u_true_future_steps: True future states for each step in the unroll
        encoding: Equation encoding vector
        unroll_steps: Number of steps to unroll
        perform_rollout_fn: Function to perform autoregressive rollout
        
    Returns:
        Mean L1 loss over the unrolled trajectory
    """
    # Perform unrolled prediction
    pred_trajectory = perform_rollout_fn(model, u_initial, unroll_steps, encoding)
    
    # Compute L1 loss across all predicted steps
    return jnp.mean(jnp.abs(pred_trajectory - u_true_future_steps))


def unrolled_pino_loss_fn(model: eqx.Module, u_initial: jax.Array, u_true_future_steps: jax.Array, 
                         encoding: jax.Array, dx: float, dt: float, pino_weight: float,
                         unroll_steps: int, perform_rollout_fn: Callable) -> jax.Array:
    """
    Loss function for unrolled training with PINO regularization.
    
    Args:
        model: PDE emulator model
        u_initial: Initial state
        u_true_future_steps: True future states for each step in the unroll
        encoding: Equation encoding vector
        dx: Spatial grid spacing
        dt: Time step size
        pino_weight: Weight for PINO loss term
        unroll_steps: Number of steps to unroll
        perform_rollout_fn: Function to perform autoregressive rollout
                
    Returns:
        Combined data and physics-informed loss
    """
    # Prediction loss: model now directly predicts u_next, not residual
    pred_trajectory = perform_rollout_fn(model, u_initial, unroll_steps, encoding)
    prediction_loss = jnp.mean(jnp.abs(pred_trajectory - u_true_future_steps))

    # PINO loss (calculated on ground truth data to regularize predictions)
    pino_losses = []
    
    # Calculate PINO loss for the first step: u_initial -> u_true_future_steps[0]
    residual_0 = calculate_pde_residual(u_initial, u_true_future_steps[0], encoding, dx, dt)
    pino_losses.append(jnp.mean(jnp.abs(residual_0)))

    # Calculate PINO loss for subsequent steps: u_true_future_steps[i-1] -> u_true_future_steps[i]
    for i in range(1, unroll_steps):
        residual_i = calculate_pde_residual(u_true_future_steps[i-1], u_true_future_steps[i], encoding, dx, dt)
        pino_losses.append(jnp.mean(jnp.abs(residual_i)))
    
    pino_loss = jnp.mean(jnp.array(pino_losses))

    return prediction_loss + pino_weight * pino_loss


def correction_loss_fn(model: eqx.Module, u_coarse: jax.Array, true_correction: jax.Array, 
                      encoding: jax.Array) -> jax.Array:
    """
    Loss function for training a correction model.
    
    Args:
        model: PDE emulator correction model
        u_coarse: Coarse solver prediction
        true_correction: True correction value
        encoding: Equation encoding vector
        
    Returns:
        L1 loss between predicted and true correction
    """
    pred_correction = model(u_coarse, encoding)
    return jnp.mean(jnp.abs(pred_correction - true_correction))


def train_model(model: eqx.Module, data_loader, optimizer, loss_fn: Callable, 
               num_steps: int, verbose: bool = True) -> Tuple[eqx.Module, List[float]]:
    """
    Generic training function for PDE emulator models.
    
    Args:
        model: PDE emulator model
        data_loader: Data loader providing training batches
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        num_steps: Number of training steps
        verbose: Whether to display progress bar
        
    Returns:
        Trained model and list of loss values
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    
    # Create grad function for batched loss computation
    loss_and_grad_fn = eqx.filter_value_and_grad(
        lambda m, *batch: jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(m, *batch))
    )
    
    # Define single training step
    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        loss_val, grads = loss_and_grad_fn(model, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val
    
    # Main training loop
    start_time = time.time()
    iterator = enumerate(data_loader)
    if verbose:
        iterator = tqdm(iterator, total=num_steps, desc="Training")
        
    for step, batch in iterator:
        if step >= num_steps:
            break
            
        model, opt_state, loss_val = train_step(model, opt_state, batch)
        losses.append(float(loss_val))
        
        if verbose and step % 100 == 0:
            elapsed = time.time() - start_time
            iterator.set_postfix({"loss": f"{loss_val:.4e}", "time": f"{elapsed:.1f}s"})
    
    return model, losses


def train_PINOmodel(model: eqx.Module, data_loader, optimizer, loss_fn: Callable, 
               num_steps: int, DX: float, DT: float, pino_weight_schedule_fn: Callable, unroll_steps: int, perform_rollout_fn: Callable, verbose: bool = True) -> Tuple[eqx.Module, List[float]]:
    """
    Generic training function for PINO models.
    
    Args:
        model: PDE emulator model
        data_loader: Data loader providing training batches
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        num_steps: Number of training steps
        DX: Spatial grid spacing
        DT: Time step size
        pino_weight_schedule_fn: Function to schedule PINO weight
        unroll_steps: Number of steps to unroll
        perform_rollout_fn: Function to perform autoregressive rollout
        verbose: Whether to display progress bar
        
    Returns:
        Trained model and list of loss values
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    
    # Create a grad function for the entire batch, vmapping over states and encodings
    # Batch dimensions for u_initial_batch: (B, C, N)
    # Batch dimensions for u_true_future_batch: (B, UNROLL_STEPS, C, N)
    # Batch dimensions for encodings_batch: (B, ENCODING_DIM)
    loss_and_grad_fn = eqx.filter_value_and_grad(
        lambda m, u_init_batch, u_true_fut_batch, enc_batch, dx_val, dt_val, pino_weight_val, unroll_steps, perform_rollout_fn: \
            jnp.mean(jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None, None, None))(m, u_init_batch, u_true_fut_batch, enc_batch, dx_val, dt_val, pino_weight_val, unroll_steps, perform_rollout_fn))
    )
    
    # Define single training step
    @eqx.filter_jit
    def train_step(model, opt_state, batch, dx_val, dt_val, pino_weight_val):
        """Performs a single training step with unrolled loss + PINO loss."""
        u_initial_batch, u_true_future_batch, encodings_batch = batch
        loss_val, grads = loss_and_grad_fn(model, u_initial_batch, u_true_future_batch, encodings_batch, dx_val, dt_val, pino_weight_val, unroll_steps, perform_rollout_fn)

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_val
    
    # Main training loop
    start_time = time.time()
    iterator = enumerate(data_loader)
    if verbose:
        iterator = tqdm(iterator, total=num_steps, desc="Training")
        
    for step, batch in iterator:
        if step >= num_steps:
            break
        
        current_pino_weight = pino_weight_schedule_fn(step) # Get the scheduled PINO weight   
        model, opt_state, loss_val = train_step(model, opt_state, batch, DX, DT, current_pino_weight)
        losses.append(float(loss_val))
        
        if verbose and step % 100 == 0:
            elapsed = time.time() - start_time
            iterator.set_postfix({"loss": f"{loss_val:.4e}", "time": f"{elapsed:.1f}s"})
    
    return model, losses
