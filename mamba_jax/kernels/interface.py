from enum import Enum
from typing import Optional

import jax
import jax.numpy as jnp

from .reference import mamba_ssm as mamba_ssm_xla


class KernelType(Enum):
    PALLAS = 0
    XLA = 1


# TODO: populate function that arranges data as expected by kernel and calls it
# TODO: add `jax.custom_jvp` for calling correct kernel for fwd / bwd pass
def mamba_ssm(
    u: jax.Array,
    delta: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: Optional[jax.Array] = None,
    delta_bias: Optional[jax.Array] = None,
    delta_softplus: bool = False,
    mode: KernelType = KernelType.XLA,
) -> jax.Array:
    if mode == KernelType.PALLAS:
        raise NotImplementedError
    elif mode == KernelType.XLA:
        # let JAX handle backwards pass in reference kernel
        return mamba_ssm_xla(u, delta, A, B, C, D=D, delta_bias=delta_bias, delta_softplus=delta_softplus)


# TODO: add fused residual+norm layer
# TODO: add `jax.custom_jvp` for calling correct kernel for fwd / bwd pass
def add_norm():
    pass
