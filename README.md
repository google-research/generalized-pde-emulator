# Generalizing PDE Emulation with Equation-Aware Neural Operators

This repository contains the code for the paper "Generalizing PDE Emulation with Equation-Aware Neural Operators". It provides a modular framework for building neural network-based emulators for partial differential equations (PDEs), with a focus on generalization across different equations and parameters. The library is built using [Equinox](https://github.com/patrick-kidger/equinox) and [APEBench](https://github.com/tum-pbs/apebench) benchmark suite.

## Abstract

Solving partial differential equations (PDEs) can be prohibitively expensive using traditional numerical methods. Deep learning-based surrogate models typically specialize in a single PDE with fixed parameters. We present a framework for equation-aware emulation that generalizes to unseen PDEs, conditioning a neural model on a vector encoding representing the terms in a PDE  and their coefficients. We present a baseline of four distinct modeling technqiues, trained on a family of 1D PDEs from the APEBench suite. Our approach achieves strong performance on parameter sets held out from the training distribution, with strong stability for rollout beyond the training window, and generalization to an entirely unseen PDE. This work was developed as part of a broader effort exploring AI systems that automate the creation of expert-level empirical software for scorable scientific tasks, see [An AI system to help scientists write expert-level empirical software](https://arxiv.org/abs/2509.06503).
## Repository Structure

The repository is organized as follows:

-   `pde_emulator/`: Contains the source code for the models, utility functions, and layers.
-   `examples/`: Includes scripts to train the various models from scratch.
-   `data/`:
    -   `model_data/`: Contains the trained model weights and parameters from the paper (Please download these from Zenodo: https://doi.org/10.5281/zenodo.17593856).
    -   `figure_data/`: Contains the data required to reproduce the plots in the paper.
    -   `reproduce_the_figures.ipynb`: A notebook to reproduce the figures from the paper.
    -   `load_trained_models.ipynb`: A notebook demonstrating how to load and use the pre-trained models.

## Installation

To get started, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/google-research/generalized-pde-emulator.git
cd generalized-PDE-emulator
pip install -e .
```

## Quick Start: Using a Pre-trained Model

This example shows how to load a pre-trained `PI-FNO-UNET` model and use it to predict the solution for a PDE scenario from APEBench.

```python
import os
import jax
import jax.numpy as jnp
import equinox as eqx

# Import from apebench and pdequinox
import apebench
from apebench.scenarios import scenario_dict
from pde_emulator.utils import evaluate_and_visualize

# --- 1. Load Pre-trained Model ---
from pde_emulator.models import PI_FNO_UNET

IN_CHANNELS = 1
ENCODING_DIM = 7
MODEL_LOAD_PATH = "./data/model_data/PI-FNO-UNET.eqx"

# Define a template model instance. The weights will be overwritten from the saved file.
key = jax.random.PRNGKey(99)
dummy_mean = jnp.zeros(ENCODING_DIM)
dummy_std = jnp.ones(ENCODING_DIM)
model_template = PI_FNO_UNET(
    num_spatial_dims=1,
    in_channels=IN_CHANNELS,
    encoding_dim=ENCODING_DIM,
    key=key,
    encoding_mean=dummy_mean,
    encoding_std=dummy_std,
)

# Load the saved weights into the model structure
print(f"Loading model from {MODEL_LOAD_PATH}...")
trained_model = eqx.tree_deserialise_leaves(MODEL_LOAD_PATH, model_template)
print("Model 1 (PI-FNO-UNET) loaded successfully.")

# --- 2. Prepare Input Data from a Test Scenario ---
# Get a test scenario (e.g., Burgers' equation) and corresponding equation coefficients
TEST_SCENARIOS = {
    'diff_burgers': [(-1.5, 1.5), (-1.5, 5.0)],
}
SCENARIO_PARAM_NAMES = {
    'diff_burgers':  ['convection_delta', 'diffusion_gamma'],
}
EQUATION_COEFFICIENTS = {
    'diff_burgers': lambda b, nu: jnp.array([0., 0., 0., b, nu, 0., 0.]),
}

# --- 3. Run Prediction and Evaluate ---
test_results, test_curves = evaluate_and_visualize(
    model=trained_model,
    test_scenarios=TEST_SCENARIOS,
    test_seed=84,
    scenario_param_names=SCENARIO_PARAM_NAMES,
    equation_coefficients=EQUATION_COEFFICIENTS,
    scenario_dict=scenario_dict,
    plot_samples=1
)

```

## Training a New Model

The `examples/` directory contains scripts for training each of the four models discussed in the paper (`PI-FNO-UNET`, `LSC-FNO`, `PINO`, and `LC`). For instance, to train the `PI-FNO-UNET` model, you can run:

```bash
python examples/train_pi_fno_unet.py
```

You can customize the training scenarios, hyperparameters, and other settings within the script.

## Reproducing Paper Results

You can reproduce the figures and results from the paper using the provided notebooks and data:

-   **`data/load_trained_models.ipynb`**: This notebook provides a detailed walkthrough of loading the pre-trained models and using them for inference.
-   **`data/reproduce_the_figures.ipynb`**: This notebook uses the data in `data/figure_data/` to generate the plots shown in the paper.

## Acknowledgements

This work is built upon the following excellent libraries:

-   **[APEBench](https://github.com/tum-pbs/apebench)**: A benchmark suite for data-driven PDE solvers.
-   **[Equinox](https://github.com/patrick-kidger/equinox)**: A JAX library for building and training neural networks.

We thank the authors of these libraries for their valuable contributions to the community.

## Citation
This package was developed as part of the paper: Generalizing PDE Emulation with Equation-Aware Neural Operators, accepted by NeurIPS workshop ML4PS 2025. If you find it useful for your research, please consider citing it:

```bibtex
@inproceedings{
anonymous2025generalizing,
title={Generalizing {PDE} Emulation with Equation-Aware Neural Operators},
author={Anonymous},
booktitle={Machine Learning and the Physical Sciences Workshop @ NeurIPS 2025},
year={2025},
url={https://openreview.net/forum?id=e4QzheEePy}
}
```

For any questions, please email qianzezhu@g.harvard.edu.

## License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
