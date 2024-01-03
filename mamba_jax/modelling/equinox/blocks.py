import math
from typing import Literal, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from einops import repeat

from ...kernels.interface import KernelType, mamba_ssm


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
class MambaBlock(eqx.Module):
    in_proj: nn.Linear
    conv1d: nn.Conv1d
    x_proj: nn.Linear
    dt_proj: nn.Linear

    A_log: jax.Array
    D: Optional[jax.Array]

    out_proj: nn.Linear

    use_kernel: bool = False
    layer_idx: int = None
    dt_rank: int
    state_dim: int

    def __init__(
        self,
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
        layer_idx: int = None,  # used in fused add-norm
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        self.use_kernel = use_kernel
        self.layer_idx = layer_idx

        inner_dim = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank
        self.state_dim = state_dim

        key, subkey = jax.random.split(key)
        self.in_proj = nn.Linear(dim, 2 * inner_dim, use_bias=bias, key=subkey)

        key, subkey = jax.random.split(key)
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            use_bias=conv_bias,
            kernel_size=kernel_size,
            groups=inner_dim,
            padding=kernel_size - 1,
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.x_proj = nn.Linear(inner_dim, self.dt_rank + state_dim * 2, use_bias=False, key=subkey)

        key, subkey = jax.random.split(key)
        self.dt_proj = nn.Linear(self.dt_rank, inner_dim, use_bias=True, key=subkey)

        dt_init_std = self.dt_rank**-0.5 * dt_scale

        key, subkey = jax.random.split(key)
        dt = jnp.exp(
            jax.random.uniform(subkey, (inner_dim,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = jnp.clip(dt, a_min=dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))

        self.dt_proj = eqx.tree_at(lambda l: l.bias, self.dt_proj, inv_dt)

        if dt_init == "constant":
            new_weight = jnp.zeros_like(self.dt_proj.weight) + dt_init_std
        elif dt_init == "random":
            key, subkey = jax.random.split(key)
            new_weight = jax.random.uniform(subkey, self.dt_proj.weight.shape, minval=-dt_init_std, maxval=dt_init_std)
        else:
            raise NotImplementedError

        self.dt_proj = eqx.tree_at(lambda l: l.weight, self.dt_proj, new_weight)

        # S4D (diagonal) real initialisation
        A = repeat(jnp.arange(1, state_dim + 1), "n -> d n", d=inner_dim)

        self.A_log = jnp.log(A)

        self.D = jnp.ones((inner_dim,))

        key, subkey = jax.random.split(key)
        self.out_proj = nn.Linear(inner_dim, dim, use_bias=bias, key=subkey)

    def __call__(self, x: jax.Array) -> jax.Array:
        L, _ = x.shape
        x, z = jnp.split(jax.vmap(self.in_proj)(x), 2, axis=-1)

        A = -jnp.exp(jnp.asarray(self.A_log, dtype=jnp.float32))

        x = jax.nn.silu(self.conv1d(x)[..., :L])

        dt, B, C = jnp.split(jax.vmap(self.x_proj)(x), [self.dt_rank, self.state_dim + 1], axis=-1)

        assert B.shape[-1] == self.state_dim
        assert C.shape[-1] == self.state_dim

        dt = jax.vmap(self.dt_proj)(dt)

        y = mamba_ssm(
            x,
            dt,
            A,
            B,
            C,
            D=self.D,
            delta_bias=None,
            delta_softplus=True,
            mode=KernelType.PALLAS if self.use_kernel else KernelType.XLA,
        )

        y = y * jax.nn.silu(z)

        return jax.vmap(self.out_proj)(y)


class ResidualBlock(eqx.Module):
    mixer: MambaBlock
    norm: eqx.Module

    def __init__(
        self,
        dim: int,
        mixer_factory,
        norm_factory=nn.RMSNorm,
        fused_add_norm: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        self.mixer = mixer_factory(dim, key=key)
        self.norm = norm_factory(dim)

    def __call__(self, x: jax.Array, res: Optional[jax.Array] = None) -> jax.Array:
        # TODO: add fused residual add +norm followed by mamba mixer
        # correspnds to `Block` in reference
        res = x if res is None else x + res

        x = jax.vmap(self.norm)(res)  # TODO: cast dtype here?
        x = self.mixer(x)

        return x, res


if __name__ == "__main__":
    from functools import partial

    key = jax.random.PRNGKey(0)
    model_key, input_key = jax.random.split(key)

    D = 8
    L = 16

    mixer_cls = partial(MambaBlock)
    norm_cls = partial(nn.RMSNorm, eps=1e-5)

    model = ResidualBlock(D, mixer_cls, norm_cls, key=model_key)

    x = jax.random.normal(input_key, (L, D))

    y, _ = model(x)

    assert x.shape == y.shape
    print(y)
    print()
