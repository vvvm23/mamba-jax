from functools import partial
from typing import List, Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

from .blocks import MambaBlock, ResidualBlock


def create_block(
    dim: int,
    state_dim: int = 16,
    kernel_size: int = 4,
    expand: int = 2,
    dt_rank: Literal["auto"] | int = "auto",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: Literal["constant", "random"] = "random",
    dt_scale: float = 1.0,
    dt_init_floor: float = 1e-4,
    conv_bias: bool = True,
    bias: bool = False,
    use_kernel: bool = False,
    layer_idx: int = None,
    norm_eps: float = 1e-5,
    # TODO: add norm type
    res_dtype: jnp.dtype = jnp.float32,
    dtype: jnp.dtype = jnp.float32,
    key: jax.random.PRNGKey = None,
) -> ResidualBlock:

    mixer_factory = partial(
        MambaBlock,
        state_dim=state_dim,
        kernel_size=kernel_size,
        expand=expand,
        dt_rank=dt_rank,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_init=dt_init,
        dt_scale=dt_scale,
        dt_init_floor=dt_init_floor,
        conv_bias=conv_bias,
        bias=bias,
        use_kernel=use_kernel,
        layer_idx=layer_idx,
        dtype=dtype,
    )

    norm_factory = partial(nn.RMSNorm, eps=norm_eps)

    return ResidualBlock(
        dim, mixer_factory=mixer_factory, norm_factory=norm_factory, res_dtype=res_dtype, layer_idx=layer_idx, key=key
    )


# Model with Mamba sequence mixer, with no specific task head
# corresponds to the `MixerModel`
class MambaModel(eqx.Module):

    embedding: nn.Embedding
    layers: List[ResidualBlock]

    norm_f: nn.RMSNorm

    def __init__(
        self,
        dim: int,
        num_layers: int,
        vocab_size: int,
        state_dim: int = 16,
        kernel_size: int = 4,
        expand: int = 2,
        dt_rank: Literal["auto"] | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: Literal["constant", "random"] = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_kernel: bool = False,
        norm_eps: float = 1e-5,
        # TODO: add norm type (rms or layer)
        res_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()

        key, subkey = jax.random.split(key)
        self.embedding = nn.Embedding(vocab_size, dim, key=subkey)

        key, *layer_keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            create_block(
                dim=dim,
                state_dim=state_dim,
                kernel_size=kernel_size,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=bias,
                use_kernel=use_kernel,
                layer_idx=i,
                norm_eps=norm_eps,
                res_dtype=res_dtype,
                dtype=dtype,
                key=subkey,
            )
            for i, subkey in enumerate(layer_keys)
        ]

        self.norm_f = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = jax.vmap(self.embedding)(input_ids)

        res = None

        for layer in self.layers:
            x, res = layer(x, res)

        res = x + res if res is not None else x
        x = jax.vmap(self.norm_f)(res.astype(self.norm_f.weight.dtype))

        return x


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
# essentially, the `MambaLMHeadModel` class
class MambaLLM(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        pass


if __name__ == "__main__":
    L, N = 4096, 128
    num_layers = 12
    vocab_size = 10_000
    key = jax.random.PRNGKey(0)

    x = jax.random.randint(key, (L,), minval=0, maxval=vocab_size)
    model = MambaModel(N, num_layers, vocab_size, key=key)

    y = model(x)
    print(y)
    print(y.shape)
