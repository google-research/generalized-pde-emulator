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

"""Utilities for PDE emulation"""

from .data import (
    get_equation_encoding,
    load_and_combine_data,
    load_and_combine_data_for_unroll,
    load_and_combine_data_correction,
    child_spectral_deriv,
    get_pde_coefficients,
    calculate_pde_residual,
)
from .training import (
    create_optimizer,
    create_pino_schedule,
    loss_fn,
    pino_loss_fn,
    unrolled_loss_fn,
    unrolled_pino_loss_fn,
    correction_loss_fn,
    train_model,
    train_PINOmodel,    
)
from .evaluation import (
    perform_rollout,
    perform_correction_rollout,
    evaluate_on_scenario,
    plot_rollout_comparison,
    evaluate_and_visualize,
)

__all__ = [
    "get_equation_encoding",
    "load_and_combine_data",
    "load_and_combine_data_for_unroll",
    "load_and_combine_data_correction",
    "child_spectral_deriv",
    "get_pde_coefficients",
    "calculate_pde_residual",
    "create_optimizer",
    "create_pino_schedule",
    "loss_fn",
    "pino_loss_fn",
    "unrolled_loss_fn",
    "unrolled_pino_loss_fn",
    "correction_loss_fn",
    "train_model",
    "train_PINOmodel",
    "perform_rollout",
    "perform_correction_rollout",
    "evaluate_on_scenario",
    "plot_rollout_comparison",
    "evaluate_and_visualize",
]
