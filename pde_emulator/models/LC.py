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

import jax
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import jax.numpy as jnp
from typing import List, Callable, Tuple

class EquationEncoder(eqx.Module):
    """
    MLP encoder for equation parameters and state features.
    """
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, activation: Callable, *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=key2),
            eqx.nn.Linear(hidden_dim, out_dim, key=key3)
        ]
        self.activation = activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

class FiLMLayer(eqx.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    """
    to_gamma_beta: eqx.nn.Linear

    def __init__(self, embedding_dim: int, num_channels: int, *, key):
        self.to_gamma_beta = eqx.nn.Linear(embedding_dim, 2 * num_channels, key=key)

    def __call__(self, x, embedding):
        gamma_beta = self.to_gamma_beta(embedding)
        gamma, beta = jnp.split(gamma_beta, 2)
        gamma = jnp.expand_dims(gamma, axis=-1)
        beta = jnp.expand_dims(beta, axis=-1)
        return x * gamma + beta

class SpectralConv1d(eqx.Module):
    """
    1D spectral convolution layer using truncated Fourier modes.
    """
    weights: jax.Array
    num_modes: int
    in_channels: int
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int, num_modes: int, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        scale = jnp.sqrt(1 / (2 * self.in_channels))
        wkey1, _ = jr.split(key)
        self.weights = jr.normal(wkey1, (in_channels, out_channels, num_modes, 2)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        N = x.shape[-1]
        x_fft = jnp.fft.rfft(x, axis=-1)
        x_fft_truncated = x_fft[..., :self.num_modes]
        complex_weights = jax.lax.complex(self.weights[..., 0], self.weights[..., 1])
        x_fft_processed = jnp.einsum('im, iom -> om', x_fft_truncated, complex_weights)
        x_fft_full = jnp.pad(x_fft_processed, ((0, 0), (0, N // 2 + 1 - self.num_modes)), mode='constant')
        x_spatial = jnp.fft.irfft(x_fft_full, n=N, axis=-1)
        return x_spatial

class FiLMedResBlock(eqx.Module):
    """
    Residual block with two spectral convolutions and FiLM conditioning.
    """
    spectral_conv1: SpectralConv1d
    spectral_conv2: SpectralConv1d
    film1: FiLMLayer
    film2: FiLMLayer
    activation: Callable
    num_channels: int

    def __init__(self, num_channels: int, embedding_dim: int, num_spatial_points: int, activation: Callable, num_modes_val: int, *, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        self.activation = activation
        self.num_channels = num_channels
        num_modes = num_modes_val
        self.spectral_conv1 = SpectralConv1d(num_channels, num_channels, num_modes, key=key1)
        self.film1 = FiLMLayer(embedding_dim, num_channels, key=key2)
        self.spectral_conv2 = SpectralConv1d(num_channels, num_channels, num_modes, key=key3)
        self.film2 = FiLMLayer(embedding_dim, num_channels, key=key4)

    def __call__(self, x, embedding):
        residual = x
        x = self.spectral_conv1(x)
        x = self.film1(x, embedding)
        x = self.activation(x)
        x = self.spectral_conv2(x)
        x = self.film2(x, embedding)
        x = self.activation(x)
        return x + residual

class FiLMedConvNet(eqx.Module):
    """
    Stacked FiLMedResBlock network with input/output projections.
    """
    in_proj: eqx.nn.Conv1d
    out_proj: eqx.nn.Conv1d
    blocks: List[FiLMedResBlock]
    activation: Callable

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, embedding_dim: int, depth: int, num_spatial_points: int, activation: Callable, num_modes_val: int, *, key):
        keys = jr.split(key, depth + 2)
        self.activation = activation
        self.in_proj = eqx.nn.Conv1d(in_channels, hidden_channels, kernel_size=1, key=keys[0])
        self.blocks = [
            FiLMedResBlock(hidden_channels, embedding_dim, num_spatial_points, activation, num_modes_val, key=k) for k in keys[1:-1]
        ]
        self.out_proj = eqx.nn.Conv1d(hidden_channels, out_channels, kernel_size=1, key=keys[-1])

    def __call__(self, x, embedding):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, embedding)
        x = self.out_proj(x)
        return x

class EquationAwareModel(eqx.Module):
    """
    Model that combines equation encoding, state features, and FiLM-conditioned spectral network.
    """
    context_encoder: EquationEncoder
    network: FiLMedConvNet
    state_mlp: eqx.nn.MLP
    output_scaler_proj: eqx.nn.Linear 
    encoding_scale: jax.Array = eqx.field(static=True)

    def __init__(self, context_encoder: EquationEncoder, network: FiLMedConvNet, embedding_dim: int, activation_fn: Callable, *, key):
        mlp_key, scaler_key = jr.split(key)
        self.context_encoder = context_encoder
        self.network = network
        self.state_mlp = eqx.nn.MLP(in_size=3, out_size=16, width_size=64, depth=2, activation=activation_fn, key=mlp_key)
        self.output_scaler_proj = eqx.nn.Linear(embedding_dim, 1, key=scaler_key)
        self.encoding_scale = jnp.array([
            0.05,  # u (Fisher r)
            0.05,  # u^2 (Fisher -r)
            4.0,   # u_x (Advection c)
            2.0,   # u u_x (Burgers/KdV/KS b)
            8.0,   # u_xx (Diffusion nu)
            20.0,  # u_xxx (KdV eps)
            27.0   # u_xxxx (KdV/KS zet)
        ]) + 1e-6

    def __call__(self, u_coarse, encoding_vector):
        """
        Forward pass for the EquationAwareModel.

        Args:
            u_coarse: Input coarse state, shape (channels, N)
            encoding_vector: Equation encoding vector, shape (encoding_dim,)

        Returns:
            predicted_correction: Model output, shape (channels, N)
        """
        u_coarse_max_abs = jnp.maximum(jnp.max(jnp.abs(u_coarse), axis=(-2, -1), keepdims=False), 1e-4)
        u_coarse_mean = jnp.mean(u_coarse, axis=(-2, -1), keepdims=False)
        u_coarse_std = jnp.maximum(jnp.std(u_coarse, axis=(-2, -1), keepdims=False), 1e-4)
        state_features_raw = jnp.array([
            u_coarse_max_abs,
            u_coarse_mean,
            u_coarse_std
        ])
        processed_state_features = self.state_mlp(state_features_raw)
        normalized_encoding_vector = encoding_vector / self.encoding_scale
        context_vector = jnp.concatenate([normalized_encoding_vector, processed_state_features], axis=-1)
        embedding = self.context_encoder(context_vector)
        N = u_coarse.shape[-1]
        embedding_tiled = jnp.tile(jnp.expand_dims(embedding, axis=-1), (1, N))
        processed_state_features_tiled = jnp.tile(jnp.expand_dims(processed_state_features, axis=-1), (1, N))
        u_network_input = jnp.concatenate([u_coarse, processed_state_features_tiled, embedding_tiled], axis=-2)
        network_raw_output = self.network(u_network_input, embedding)
        learned_scale = jnp.exp(self.output_scaler_proj(embedding))
        predicted_correction = network_raw_output * learned_scale
        return predicted_correction

def LC(num_spatial_dims: int, in_channels: int, encoding_dim: int, key):
    """
    Factory function to instantiate the EquationAwareModel with default hyperparameters.

    Args:
        num_spatial_dims: Number of spatial dimensions (unused, for API compatibility)
        in_channels: Number of input channels
        encoding_dim: Dimension of the equation encoding vector
        key: JAX PRNGKey for initialization

    Returns:
        EquationAwareModel instance
    """
    key_model_init, encoder_key, network_key = jr.split(key, 3)
    embedding_dim = 32
    encoder_hidden = 64
    cnn_hidden = 160
    cnn_depth = 14
    activation_fn = jnn.gelu
    num_spatial_points = 160
    num_modes_fraction = 1/3 
    actual_num_modes = max(1, int(num_spatial_points * num_modes_fraction))
    augmented_in_channels = in_channels + 16 + embedding_dim

    print(f"\n>>> Instantiating GENERALIZED FiLM Correction model with parameters: "
          f"embedding_dim={embedding_dim}, cnn_hidden={cnn_hidden}, cnn_depth={cnn_depth}, "
          f"using SpectralConv1d with N={num_spatial_points}, num_modes_fraction={num_modes_fraction}, "
          f"and improved complex weight initialization. "
          f"Augmented input channels: {augmented_in_channels} <<<")
    
    context_encoder = EquationEncoder(
        in_dim=encoding_dim + 16,
        out_dim=embedding_dim,
        hidden_dim=encoder_hidden,
        activation=activation_fn,
        key=encoder_key
    )
    network = FiLMedConvNet(
        in_channels=augmented_in_channels,
        out_channels=in_channels,
        hidden_channels=cnn_hidden,
        embedding_dim=embedding_dim,
        depth=cnn_depth,
        num_spatial_points=num_spatial_points,
        activation=activation_fn,
        num_modes_val=actual_num_modes,
        key=network_key
    )
    return EquationAwareModel(context_encoder, network, embedding_dim, activation_fn=activation_fn, key=key_model_init)
