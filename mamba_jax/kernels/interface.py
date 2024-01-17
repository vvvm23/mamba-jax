from enum import Enum
from typing import Optional

import jax
import jax.numpy as jnp

from .reference import mamba_ssm as mamba_ssm_xla


class KernelType(Enum):
    PALLAS = 0
    XLA = 1
    XLA_ASSOCIATIVE = 2


KernelTypeMapping = {"pallas": KernelType.PALLAS, "xla": KernelType.XLA, "xla_associative": KernelType.XLA_ASSOCIATIVE}


# TODO: populate function that arranges data as expected by kernel and calls it
# TODO: add `jax.custom_jvp` for calling correct kernel for fwd / bwd pass, when
# use Pallas mode.
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
        return jax.checkpoint(mamba_ssm_xla, static_argnums=(-1, -2))(
            u, delta, A, B, C, D, delta_bias, delta_softplus, False
        )
    elif mode == KernelType.XLA_ASSOCIATIVE:
        # reference kernel with associative scan
        return jax.checkpoint(mamba_ssm_xla, static_argnums=(-1, -2))(
            u, delta, A, B, C, D, delta_bias, delta_softplus, True
        )


# TODO: add fused residual+norm layer
# TODO: add `jax.custom_jvp` for calling correct kernel for fwd / bwd pass
def add_norm():
    pass
