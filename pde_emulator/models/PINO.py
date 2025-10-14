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
from typing import List, Callable

@eqx.filter_jit
def child_normalize_encoding(encoding: jax.Array, ENCODING_MIN_VALS: jax.Array, ENCODING_MAX_VALS: jax.Array) -> jax.Array:
    """Normalize the equation encoding vector to [-1, 1] using global min/max."""
    range_vals = ENCODING_MAX_VALS - ENCODING_MIN_VALS
    normalized = jnp.where(
        range_vals == 0,
        0.0,
        2 * ((encoding - ENCODING_MIN_VALS) / range_vals) - 1
    )
    return normalized

class EquationEncoder(eqx.Module):
    """MLP to process the equation encoding vector."""
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

class child_FiLMLayer(eqx.Module):
    """Feature-wise Linear Modulation (FiLM) layer."""
    to_gamma_beta: eqx.nn.Linear

    def __init__(self, embedding_dim: int, num_channels: int, *, key):
        self.to_gamma_beta = eqx.nn.Linear(embedding_dim, 2 * num_channels, key=key)

    def __call__(self, x, embedding):
        gamma_beta = self.to_gamma_beta(embedding)
        gamma, beta = jnp.split(gamma_beta, 2)
        gamma = jnp.expand_dims(gamma, axis=-1)
        beta = jnp.expand_dims(beta, axis=-1)
        return x * gamma + beta

class child_SpectralConv1d(eqx.Module):
    """Spectral convolution layer for 1D FNO."""
    weights_real: jax.Array
    weights_imag: jax.Array
    num_modes: int

    def __init__(self, in_channels: int, out_channels: int, num_modes: int, *, key):
        key_real, key_imag = jr.split(key)
        self.num_modes = num_modes
        scale = jnp.sqrt(1.0 / (in_channels * num_modes))
        self.weights_real = jax.random.normal(key_real, (out_channels, in_channels, num_modes)) * scale
        self.weights_imag = jax.random.normal(key_imag, (out_channels, in_channels, num_modes)) * scale

    def __call__(self, x):
        num_spatial_points = x.shape[-1]
        x_fft = jnp.fft.rfft(x, axis=-1)
        effective_num_modes = min(self.num_modes, x_fft.shape[-1])
        weights = self.weights_real[..., :effective_num_modes] + 1j * self.weights_imag[..., :effective_num_modes]
        transformed_modes = jnp.einsum('oim,im->om', weights, x_fft[:, :effective_num_modes])
        out_fft = jnp.zeros(
            (weights.shape[0], x_fft.shape[-1]),
            dtype=jnp.complex64
        )
        out_fft = jax.lax.dynamic_update_slice(out_fft, transformed_modes, (0, 0))
        out_spatial = jnp.fft.irfft(out_fft, n=num_spatial_points, axis=-1)
        return out_spatial

class FiLMedFNOBlock(eqx.Module):
    """A single FiLMed FNO block."""
    spectral_conv: child_SpectralConv1d
    mlp: eqx.nn.Conv1d 
    film: child_FiLMLayer
    activation: Callable
    
    def __init__(self, channels: int, num_modes: int, embedding_dim: int, activation: Callable, *, key):
        key_sc, key_mlp, key_film = jr.split(key, 3) 
        self.activation = activation
        self.spectral_conv = child_SpectralConv1d(channels, channels, num_modes, key=key_sc)
        self.mlp = eqx.nn.Conv1d(channels, channels, kernel_size=1, key=key_mlp) 
        self.film = child_FiLMLayer(embedding_dim, channels, key=key_film)

    def __call__(self, x, embedding):
        x_spectral = self.spectral_conv(x)
        x_pointwise = self.mlp(x)
        x_combined = x_spectral + x_pointwise
        x_film_modulated = self.film(x_combined, embedding)
        x_activated = self.activation(x_film_modulated)
        x_out = x_activated + x
        return x_out

class FiLMedFNO(eqx.Module):
    """Stacked FiLMed FNO blocks with input/output projections."""
    in_proj: eqx.nn.Conv1d
    out_proj: eqx.nn.Conv1d
    blocks: List[FiLMedFNOBlock]
    activation: Callable

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_modes: int, embedding_dim: int, depth: int, activation: Callable, *, key):
        keys = jr.split(key, depth + 2)
        self.activation = activation
        self.in_proj = eqx.nn.Conv1d(in_channels, hidden_channels, kernel_size=1, key=keys[0])
        self.blocks = [
            FiLMedFNOBlock(hidden_channels, num_modes, embedding_dim, activation, key=k) 
            for k in keys[1:-1]
        ]
        self.out_proj = eqx.nn.Conv1d(hidden_channels, out_channels, kernel_size=1, key=keys[-1])

    def __call__(self, x, embedding):
        x = self.activation(self.in_proj(x))
        for block in self.blocks:
            x = block(x, embedding)
        x = self.out_proj(x)
        return x

class EquationAwareModel(eqx.Module):
    """
    Top-level model using FiLM + FNO architecture.
    Takes input state `u` and an equation encoding vector.
    """
    encoder: EquationEncoder
    input_proj_encoder: eqx.nn.Linear
    network: FiLMedFNO
    encoding_min_vals: jax.Array = eqx.static_field()
    encoding_max_vals: jax.Array = eqx.static_field()
    data_mean: float = eqx.static_field()
    data_std: float = eqx.static_field()

    def __init__(self, encoder: EquationEncoder, network: FiLMedFNO, encoding_dim: int, encoding_projection_channels: int, encoding_min_vals: jax.Array, encoding_max_vals: jax.Array, data_mean: float, data_std: float, *, key):
        self.encoder = encoder
        self.network = network
        self.input_proj_encoder = eqx.nn.Linear(encoding_dim, encoding_projection_channels, key=key)
        self.encoding_min_vals = encoding_min_vals
        self.encoding_max_vals = encoding_max_vals
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, u, encoding_vector):
        u_norm = (u - self.data_mean) / self.data_std
        encoding_norm = child_normalize_encoding(encoding_vector, self.encoding_min_vals, self.encoding_max_vals)
        film_embedding = self.encoder(encoding_norm)
        input_embedding_channel = self.input_proj_encoder(encoding_norm)
        u_with_embedding = jnp.concatenate([
            u_norm, 
            jnp.tile(input_embedding_channel[:, None], (1, u_norm.shape[-1]))
        ], axis=0)
        u_next_norm_pred = self.network(u_with_embedding, film_embedding)
        u_next_pred = u_next_norm_pred * self.data_std + self.data_mean
        return u_next_pred

def PINO(
    num_spatial_dims: int,
    in_channels_u: int,
    encoding_dim: int,
    key,
    encoding_min_vals: jax.Array,
    encoding_max_vals: jax.Array,
    data_mean: float,
    data_std: float,
):
    """
    Defines a generalized model that takes state `u` and an equation encoding vector.
    Returns an EquationAwareModel instance.
    """
    embedding_dim = 96
    encoder_hidden = 192
    fno_hidden = 256
    num_fno_modes = 32
    fno_depth = 6
    encoding_projection_channels = 2
    activation_fn = jnn.silu

    print(f"\n>>> Instantiating GENERALIZED FiLM + FNO model with parameters: "
          f"embedding_dim={embedding_dim}, encoder_hidden={encoder_hidden}, "
          f"fno_hidden={fno_hidden}, num_fno_modes={num_fno_modes}, fno_depth={fno_depth}, "
          f"encoding_projection_channels={encoding_projection_channels} <<<")

    encoder_key, input_proj_key, network_key = jr.split(key, 3)

    encoder = EquationEncoder(
        in_dim=encoding_dim,
        out_dim=embedding_dim,
        hidden_dim=encoder_hidden,
        activation=activation_fn,
        key=encoder_key,
    )

    fno_input_channels = in_channels_u + encoding_projection_channels

    network = FiLMedFNO(
        in_channels=fno_input_channels,
        out_channels=in_channels_u,
        hidden_channels=fno_hidden,
        num_modes=num_fno_modes,
        embedding_dim=embedding_dim,
        depth=fno_depth,
        activation=activation_fn,
        key=network_key,
    )

    generalized_model = EquationAwareModel(
        encoder, 
        network, 
        encoding_dim=encoding_dim, 
        encoding_projection_channels=encoding_projection_channels, 
        encoding_min_vals=encoding_min_vals,
        encoding_max_vals=encoding_max_vals,
        data_mean=data_mean,
        data_std=data_std,
        key=input_proj_key
    )
    return generalized_model
