import math
from functools import partial
from typing import Literal, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat

from ...kernels.interface import KernelType, mamba_ssm
from .utils import cast_eqx_layer


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
class MambaBlock(eqx.Module):
    in_proj: nn.Linear
    conv1d: nn.Conv1d
    x_proj: nn.Linear
    dt_proj: nn.Linear

    A_log: jax.Array
    D: jax.Array

    out_proj: nn.Linear

    dtype: jnp.dtype = jnp.float32
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
        layer_idx: int = None,  # used to access cache
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        self.use_kernel = use_kernel
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.state_dim = state_dim

        inner_dim = int(expand * dim)

        # default dt low rank is dim / 16
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank

        keys = jax.random.split(key, 7)

        # first linear projection in mamba block (fused both branches into one layer)
        self.in_proj = cast_eqx_layer(nn.Linear(dim, 2 * inner_dim, use_bias=bias, key=keys[0]), dtype=self.dtype)

        # conv1d layer in SSM path in Mamba block
        self.conv1d = cast_eqx_layer(
            nn.Conv1d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                use_bias=conv_bias,
                kernel_size=kernel_size,
                groups=inner_dim,
                padding=kernel_size - 1,
                key=keys[1],
            ),
            dtype=self.dtype,
        )

        # hypernetwork that predicts B, C and dt
        self.x_proj = cast_eqx_layer(
            nn.Linear(inner_dim, self.dt_rank + state_dim * 2, use_bias=False, key=keys[2]), dtype=self.dtype
        )

        # dt proj has a special initialisation
        self.dt_proj = nn.Linear(self.dt_rank, inner_dim, use_bias=True, key=keys[3])
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = jnp.exp(
            jax.random.uniform(keys[4], (inner_dim,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = jnp.clip(dt, a_min=dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))

        self.dt_proj = eqx.tree_at(lambda l: l.bias, self.dt_proj, inv_dt)

        if dt_init == "constant":
            new_weight = jnp.zeros_like(self.dt_proj.weight) + dt_init_std
        elif dt_init == "random":
            new_weight = jax.random.uniform(keys[5], self.dt_proj.weight.shape, minval=-dt_init_std, maxval=dt_init_std)
        else:
            raise NotImplementedError

        self.dt_proj = eqx.tree_at(lambda l: l.weight, self.dt_proj, new_weight)
        self.dt_proj = cast_eqx_layer(self.dt_proj, dtype=self.dtype)

        # S4D (diagonal) real initialisation of A matrix
        A = repeat(jnp.arange(1, state_dim + 1), "n -> d n", d=inner_dim)
        self.A_log = jnp.log(A)

        # init D to be all ones
        self.D = jnp.ones((inner_dim,))

        # output projection following multiplicative gating
        self.out_proj = cast_eqx_layer(nn.Linear(inner_dim, dim, use_bias=bias, key=keys[6]), dtype=self.dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        L, _ = x.shape
        x = x.astype(dtype=self.dtype)
        x, z = jnp.split(jax.vmap(self.in_proj)(x), 2, axis=-1)

        A = -jnp.exp(self.A_log.astype(dtype=jnp.float32))

        x = x.astype(dtype=self.dtype)
        x = jnp.transpose(x)
        x = jax.nn.silu(self.conv1d(x))
        x = jnp.transpose(x)[:L]

        x = x.astype(dtype=self.dtype)
        dt, B, C = jnp.split(jax.vmap(self.x_proj)(x), [self.dt_rank, self.state_dim + self.dt_rank], axis=-1)

        assert B.shape[-1] == self.state_dim
        assert C.shape[-1] == self.state_dim

        dt = jax.vmap(self.dt_proj)(dt)

        x = x.astype(dtype=self.dtype)
        y = mamba_ssm(
            x,
            dt,
            A,
            B,
            C,
            D=self.D.astype(dtype=jnp.float32),
            delta_bias=None,
            delta_softplus=True,
            mode=KernelType.PALLAS if self.use_kernel else KernelType.XLA,
        )

        y = y * jax.nn.silu(z)

        y = y.astype(dtype=self.dtype)
        return jax.vmap(self.out_proj)(y)

    # in this instance, x is a 1d tensor, representing a single token
    def generate_step(self, x: jax.Array, cache) -> jax.Array:
        conv_state, ssm_state = cache[self.layer_idx]

        x = x.astype(dtype=self.dtype)
        x, z = jnp.split(self.in_proj(x), 2, axis=-1)

        A = -jnp.exp(self.A_log.astype(dtype=jnp.float32))

        conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
        conv_state = conv_state.at[:, -1].set(x)

        x = jnp.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), axis=-1)

        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias[:, 0]

        x = jax.nn.silu(x).astype(dtype=self.dtype)
        dt, B, C = jnp.split(self.x_proj(x), [self.dt_rank, self.state_dim + self.dt_rank], axis=-1)

        assert B.shape[-1] == self.state_dim
        assert C.shape[-1] == self.state_dim

        dt = jax.nn.softplus(self.dt_proj(dt))
        x = x.astype(dtype=self.dtype)

        delta_A = jnp.exp(einsum(dt, A, "d, d n -> d n"))
        delta_B = einsum(dt, B, "d, n -> d n")

        ssm_state = ssm_state * delta_A + rearrange(x, "d -> d 1") * delta_B
        ssm_state = ssm_state.astype(dtype=self.dtype)

        y = einsum(ssm_state, C, "d n, n -> d")
        y = y + x * self.D

        y = y * jax.nn.silu(z)
        y = y.astype(dtype=self.dtype)

        cache[self.layer_idx] = (conv_state, ssm_state)
        return self.out_proj(y)


class ResidualBlock(eqx.Module):
    mixer: MambaBlock
    norm: eqx.Module

    res_dtype: jnp.dtype = jnp.float32
    layer_idx: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        dim: int,
        mixer_factory,
        norm_factory=nn.RMSNorm,
        fused_add_norm: bool = False,
        res_dtype: jnp.dtype = jnp.float32,
        layer_idx: Optional[int] = None,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.res_dtype = res_dtype
        self.dtype = dtype
        self.mixer = mixer_factory(dim, dtype=dtype, key=key)
        self.norm = cast_eqx_layer(norm_factory(dim), dtype=dtype)

    def __call__(self, x: jax.Array, res: Optional[jax.Array] = None) -> jax.Array:
        # TODO: add fused residual add +norm followed by mamba mixer
        # correspnds to `Block` in reference
        res = x if res is None else x + res

        x = jax.vmap(self.norm)(res.astype(dtype=self.norm.weight.dtype))
        x = x.astype(self.dtype)
        x = self.mixer(x)

        return x, res.astype(dtype=self.res_dtype)

    def generate_step(self, x: jax.Array, res: Optional[jax.Array] = None, cache=None) -> jax.Array:
        res = x if res is None else x + res
        x = self.norm(res.astype(dtype=self.norm.weight.dtype))

        x = x.astype(self.dtype)
        x = self.mixer.generate_step(x, cache)

        return x, res.astype(dtype=self.res_dtype)


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
    )

    norm_factory = partial(nn.RMSNorm, eps=norm_eps)

    return ResidualBlock(
        dim,
        mixer_factory=mixer_factory,
        norm_factory=norm_factory,
        res_dtype=res_dtype,
        layer_idx=layer_idx,
        dtype=dtype,
        key=key,
    )
