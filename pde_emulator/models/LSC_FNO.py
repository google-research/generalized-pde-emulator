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

class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, activation: Callable, *, key):
        keys = jr.split(key, num_layers)
        self.layers = []
        if num_layers == 1:
            self.layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[0]))
        else:
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[0]))
            for i in range(1, num_layers - 1):
                self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))
        self.activation = activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

class FiLMLayer(eqx.Module):
    to_gamma_beta: eqx.nn.Linear

    def __init__(self, embedding_dim: int, num_channels: int, *, key):
        self.to_gamma_beta = eqx.nn.Linear(embedding_dim, 2 * num_channels, key=key)

    def __call__(self, x, embedding):
        gamma_beta = self.to_gamma_beta(embedding)
        gamma, beta = jnp.split(gamma_beta, 2)
        gamma = jnp.expand_dims(gamma, axis=-1)
        beta = jnp.expand_dims(beta, axis=-1)
        return x * gamma + beta

class GatedSpectralConv1d(eqx.Module):
    weights_real: jax.Array
    weights_imag: jax.Array
    gate_mlp: eqx.nn.Linear
    num_modes: int
    in_channels: int
    out_channels: int
    spatial_dim_size: int

    def __init__(self, in_channels: int, out_channels: int, num_modes: int, spatial_dim_size: int, embedding_dim: int, *, key):
        self.num_modes = num_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim_size = spatial_dim_size

        key_real, key_imag, key_gate = jr.split(key, 3)
        self.weights_real = jr.normal(key_real, (self.num_modes, in_channels, out_channels)) * 0.1
        self.weights_imag = jr.normal(key_imag, (self.num_modes, in_channels, out_channels)) * 0.1
        self.gate_mlp = eqx.nn.Linear(embedding_dim, num_modes, key=key_gate)

    def __call__(self, x: jax.Array, embedding: jax.Array):
        original_shape = x.shape
        is_batched = x.ndim == 3
        if not is_batched:
            x = jnp.expand_dims(x, axis=0)

        batch_size = x.shape[0]

        x_ft = jnp.fft.rfft(x, axis=-1)

        effective_num_modes = min(self.num_modes, x_ft.shape[-1])

        if batch_size > 1:
            raw_gate_weights = jax.vmap(self.gate_mlp)(embedding)
        else:
            raw_gate_weights = self.gate_mlp(embedding)
            raw_gate_weights = jnp.expand_dims(raw_gate_weights, axis=0)

        gate_weights = jnn.softplus(raw_gate_weights)

        out_ft = jnp.zeros((batch_size, self.out_channels, x_ft.shape[-1]), dtype=jnp.complex64)

        x_ft_truncated = x_ft[..., :effective_num_modes]
        gated_x_ft_truncated = x_ft_truncated * gate_weights[:, None, :effective_num_modes]

        weights_complex = self.weights_real[:effective_num_modes, :, :] + 1j * self.weights_imag[:effective_num_modes, :, :]
        out_ft_processed = jnp.einsum('bck, kco->bok', gated_x_ft_truncated, weights_complex)

        padded_out_ft = out_ft.at[:, :, :effective_num_modes].set(out_ft_processed)

        x_inv_ft = jnp.fft.irfft(padded_out_ft, n=self.spatial_dim_size, axis=-1)

        if not is_batched:
            x_inv_ft = jnp.squeeze(x_inv_ft, axis=0)
        return x_inv_ft

class EmbeddingToKVProj(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, activation: Callable, *, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        ]
        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.layers[0](x))
        x = self.layers[1](x)
        return x

class ConditionedFNOBlock(eqx.Module):
    spectral_conv: GatedSpectralConv1d
    local_mlp: eqx.nn.Conv1d
    film_layer: FiLMLayer
    attention: eqx.nn.MultiheadAttention
    embedding_to_kv_proj: MLP
    activation: Callable
    skip_conn_mlp: eqx.nn.Identity 

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, num_modes: int, spatial_dim_size: int, activation: Callable, *, key):
        key1, key2, key_attn_proj_mlp, key_attn, key3 = jr.split(key, 5) 
        self.activation = activation
        self.spectral_conv = GatedSpectralConv1d(in_channels, out_channels, num_modes, spatial_dim_size, embedding_dim, key=key1)
        self.local_mlp = eqx.nn.Conv1d(in_channels, out_channels, kernel_size=1, key=key2)
        
        num_heads = 4
        kv_proj_hidden = embedding_dim // 2 
        self.embedding_to_kv_proj = MLP(embedding_dim, out_channels, kv_proj_hidden, num_layers=2, activation=activation, key=key_attn_proj_mlp)
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads, 
            query_size=out_channels, 
            key_size=out_channels, 
            value_size=out_channels, 
            output_size=out_channels, 
            key=key_attn
        )
        
        self.film_layer = FiLMLayer(embedding_dim, out_channels, key=key3)
        self.skip_conn_mlp = eqx.nn.Identity()

    def __call__(self, x, embedding):
        x_spectral = self.spectral_conv(x, embedding)
        x_local = self.activation(self.local_mlp(x))
        x_combined_fno_output = x_spectral + x_local

        is_batched = x_combined_fno_output.ndim == 3
        
        if is_batched:
            query = jnp.transpose(x_combined_fno_output, (0, 2, 1)) # (B, N, C)
            kv_from_embedding = jax.vmap(self.embedding_to_kv_proj)(embedding) # (B, C)
            kv_from_embedding = jnp.expand_dims(kv_from_embedding, axis=1) # (B, 1, C)
            key = value = kv_from_embedding
            
            x_attended = self.attention(query, key, value) # (B, N, C)
            x_attended = jnp.transpose(x_attended, (0, 2, 1)) # (B, C, N)
        else:
            query = jnp.transpose(x_combined_fno_output, (1, 0)) # (N, C)
            kv_from_embedding = self.embedding_to_kv_proj(embedding) # (C,)
            kv_from_embedding = jnp.expand_dims(kv_from_embedding, axis=0) # (1, C)
            key = value = kv_from_embedding

            x_attended = self.attention(query, key, value) # (N, C)
            x_attended = jnp.transpose(x_attended, (1, 0)) # (C, N)

        x_after_attention = x_combined_fno_output + x_attended # Residual connection for attention

        x_conditioned = self.film_layer(x_after_attention, embedding)
        
        skip_x = self.skip_conn_mlp(x)

        return x_conditioned + skip_x


class FNO(eqx.Module):
    in_proj: eqx.nn.Conv1d
    out_proj: eqx.nn.Conv1d
    blocks: List[ConditionedFNOBlock]
    activation: Callable

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, embedding_dim: int, depth: int, num_modes: int, spatial_dim_size: int, activation: Callable, *, key):
        keys = jr.split(key, depth + 2)
        self.activation = activation

        self.in_proj = eqx.nn.Conv1d(in_channels, hidden_channels, kernel_size=1, key=keys[0])

        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                ConditionedFNOBlock(hidden_channels, hidden_channels, embedding_dim, num_modes, spatial_dim_size, activation, key=keys[i+1])
            )

        self.out_proj = eqx.nn.Conv1d(hidden_channels, in_channels, kernel_size=1, key=keys[-1])

    def __call__(self, x, embedding):
        x = self.activation(self.in_proj(x))
        for block in self.blocks:
            x = block(x, embedding)
        x = self.out_proj(x)
        return x

class EquationAwareModel(eqx.Module):
    equation_embedding_mlp: MLP
    encoder_conv: eqx.nn.Conv1d
    decoder_conv: eqx.nn.ConvTranspose1d
    network: FNO
    data_mean: jax.Array
    data_std: jax.Array
    encoding_mean: jax.Array
    encoding_std: jax.Array

    def __init__(self, equation_embedding_mlp: MLP, encoder_conv: eqx.nn.Conv1d, decoder_conv: eqx.nn.ConvTranspose1d, network: FNO, data_mean: jax.Array, data_std: jax.Array, encoding_mean: jax.Array, encoding_std: jax.Array):
        self.equation_embedding_mlp = equation_embedding_mlp
        self.encoder_conv = encoder_conv
        self.decoder_conv = decoder_conv
        self.network = network
        self.data_mean = data_mean
        self.data_std = data_std
        self.encoding_mean = encoding_mean
        self.encoding_std = encoding_std

    def normalize_u(self, x):
        # Handle zero standard deviation if all values are identical
        safe_std = jnp.where(self.data_std == 0, 1.0, self.data_std)
        return (x - self.data_mean) / safe_std

    def denormalize_u(self, x):
        safe_std = jnp.where(self.data_std == 0, 1.0, self.data_std)
        return x * safe_std + self.data_mean
    
    def normalize_encoding(self, encoding):
        safe_encoding_std = jnp.where(self.encoding_std == 0, 1e-6, self.encoding_std)
        return (encoding - self.encoding_mean) / safe_encoding_std

    def __call__(self, u, encoding_vector):
        u_normalized = self.normalize_u(u)
        encoding_normalized = self.normalize_encoding(encoding_vector)
        embedding = self.equation_embedding_mlp(encoding_normalized)

        latent_u = self.encoder_conv(u_normalized)
        latent_u = jnn.silu(latent_u) 

        normalized_latent_residual = self.network(latent_u, embedding)
        pred_latent_u_normalized = latent_u + normalized_latent_residual

        pred_u_normalized = self.decoder_conv(pred_latent_u_normalized)
        pred_u = self.denormalize_u(pred_u_normalized)
        return pred_u

def LSC_FNO(
    num_spatial_dims: int,
    in_channels: int,
    encoding_dim: int,
    spatial_dim_size: int, # Original spatial dim size (160)
    data_mean: jax.Array,
    data_std: jax.Array,
    encoding_mean: jax.Array,
    encoding_std: jax.Array,
    key,
):
    """Defines a generalized model that takes state `u` and an equation encoding vector."""

    N_LATENT = 80
    
    embedding_dim = 64
    encoder_hidden = 64
    fno_hidden = 128
    fno_depth = 12
    num_fourier_modes = N_LATENT // 2
    activation_fn = jnn.silu

    print(f"\n>>> Instantiating GENERALIZED FNO (LNO variant) model with parameters: "
          f"latent_dim={N_LATENT}, embedding_dim={embedding_dim}, fno_hidden={fno_hidden}, fno_depth={fno_depth}, "
          f"num_modes={num_fourier_modes}, activation={activation_fn.__name__}, original_spatial_dim_size={spatial_dim_size}, encoding_dim={encoding_dim}, encoder_hidden={encoder_hidden} <<<")

    encoder_key, network_key, enc_conv_key, dec_conv_key = jr.split(key, 4)

    equation_embedding_mlp = MLP(
        in_dim=encoding_dim,
        out_dim=embedding_dim,
        hidden_dim=encoder_hidden,
        num_layers=3,
        activation=activation_fn,
        key=encoder_key,
    )
    
    encoder_conv = eqx.nn.Conv1d(in_channels, fno_hidden, kernel_size=2, stride=2, key=enc_conv_key)
    decoder_conv = eqx.nn.ConvTranspose1d(fno_hidden, in_channels, kernel_size=2, stride=2, key=dec_conv_key)


    network = FNO(
        in_channels=fno_hidden,
        out_channels=fno_hidden,
        hidden_channels=fno_hidden,
        embedding_dim=embedding_dim,
        depth=fno_depth,
        num_modes=num_fourier_modes,
        spatial_dim_size=N_LATENT,
        activation=activation_fn,
        key=network_key,
    )

    generalized_model = EquationAwareModel(equation_embedding_mlp, encoder_conv, decoder_conv, network, data_mean, data_std, encoding_mean, encoding_std)
    return generalized_model
