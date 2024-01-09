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
    associative_scan: bool = True,
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

    def _associative_scan_fn(s, c):
        return tuple((c[0] * s[0], c[0] * s[1] + c[1]))

    if associative_scan:
        _, y = jax.lax.associative_scan(_associative_scan_fn, (delta_A, delta_B_u))
        y = einsum(y, C, "L d_in n, L n -> L d_in")
    else:
        _, y = jax.lax.scan(_scan_fn, init=x, xs=[delta_A, delta_B_u, C])

    y = y + u * D
    return y


# TODO: add reference residual+norm layer
# this could be better served as a regular layer
def add_norm():
    pass


if __name__ == "__main__":
    import time
    from functools import partial

    key = jax.random.PRNGKey(0)

    L, D, N = 2048, 1024, 16

    u = jax.random.normal(key, (L, D), dtype=jnp.bfloat16) / 10
    delta = jax.random.normal(key, (L, D), dtype=jnp.bfloat16) / 10
    A = jax.random.normal(key, (D, N), dtype=jnp.bfloat16) / 10
    B = jax.random.normal(key, (L, N), dtype=jnp.bfloat16) / 10
    C = jax.random.normal(key, (L, N), dtype=jnp.bfloat16) / 10
    D = 0

    delta_softplus = True

    scan_fn = jax.jit(partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=False))
    associative_scan_fn = jax.jit(partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=True))

    scan_result = scan_fn(u, delta, A, B, C, D)
    associative_scan_result = associative_scan_fn(u, delta, A, B, C, D)

    start_time = time.time()
    for _ in range(100):
        _ = scan_fn(u, delta, A, B, C, D).block_until_ready()
    scan_time = (time.time() - start_time) / 100

    start_time = time.time()
    for _ in range(100):
        _ = associative_scan_fn(u, delta, A, B, C, D).block_until_ready()
    associative_scan_time = (time.time() - start_time) / 100

    print(scan_time, associative_scan_time)
    assert scan_result.shape == associative_scan_result.shape
