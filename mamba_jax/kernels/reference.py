from typing import Optional

import jax
import jax.numpy as jnp


# TODO: complete reference implementation
def mamba_ssm(
    u: jax.Array,
    delta: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: Optional[jax.Array] = None,
    delta_bias: Optional[jax.Array] = None,
    delta_softplus: bool = False,
) -> jax.Array:
    pass


# TODO: add reference residual+norm layer
# this could be better served as a regular layer
def add_norm():
    pass
