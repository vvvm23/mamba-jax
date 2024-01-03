from typing import Optional

import jax
import jax.numpy as jnp
from einops import einsum


def mamba_ssm(
    u: jax.Array,
    delta: jax.Array,  # time-variant
    A: jax.Array,  # time-invariant
    B: jax.Array,  # time-variant
    C: jax.Array,  # time variant
    D: Optional[jax.Array] = None,  # time-invariant
    delta_bias: Optional[jax.Array] = None,
    delta_softplus: bool = False,
) -> jax.Array:
    # Adapted from Mamba Minimal by johnma2006
    # https://github.com/johnma2006/mamba-minimal/blob/c91c81d99480cf89fe39d3b04a2115ce984612e2/model.py#L275

    if delta_bias is not None:
        raise NotImplementedError("delta_bias not implemented yet.")

    l, d_in = u.shape
    n = A.shape[1]

    delta = jnp.asarray(delta, dtype=jnp.float32)

    if delta_softplus:
        delta = jax.nn.softplus(delta)

    delta_A = jnp.exp(einsum(delta, A, "l d_in, d_in n -> l d_in n"))
    delta_B_u = einsum(delta, B, u, "l d_in, l n, l d_in -> l d_in n")

    x = jnp.zeros((d_in, n))

    def _scan_fn(x, params):
        d_A, d_Bu, C = params

        x = d_A * x + d_Bu
        return x, einsum(x, C, "d_in n, n -> d_in")

    # don't do jax.lax.associative_scan as this will materialise full state and likely OOM
    _, y = jax.lax.scan(_scan_fn, init=x, xs=[delta_A, delta_B_u, C])

    y = y + u * D
    return y


# TODO: add reference residual+norm layer
# this could be better served as a regular layer
def add_norm():
    pass


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    L = 128
    D = 16
    N = 8
    u_key, delta_key, A_key, B_key, C_key, D_key = jax.random.split(key, 6)

    u = jax.random.normal(u_key, (L, D))
    delta = jax.random.normal(delta_key, (L, D))
    A = jax.random.normal(A_key, (D, N))
    B = jax.random.normal(B_key, (L, N))
    C = jax.random.normal(C_key, (L, N))
    D = jax.random.normal(D_key, (D,))

    y = mamba_ssm(u, delta, A, B, C, D)
    print()
