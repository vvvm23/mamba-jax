from functools import partial
from typing import Literal

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
    def __init__(self):
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        pass


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
# essentially, the `MambaLMHeadModel` class
class MambaLLM(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        pass


if __name__ == "__main__":
    L, N = 10, 8
    key = jax.random.PRNGKey(0)

    x = jax.random.uniform(key, (L, N))
    block = create_block(N, key=key)

    y, _ = block(x)
    print(y)
    print(y.shape)
