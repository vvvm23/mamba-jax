import math
from typing import Literal, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from einops import repeat


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
        dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank

        key, subkey = jax.random.split(key)
        self.in_proj = nn.Linear(dim, 2 * inner_dim, bias=bias, key=subkey)

        key, subkey = jax.random.split(key)
        self.conv1d = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=inner_dim,
            padding=kernel_size - 1,
            key=subkey,
        )

        key, subkey = jax.random.split(key)
        self.x_proj = nn.Linear(inner_dim, dt_rank + state_dim * 2, bias=False, key=subkey)

        key, subkey = jax.random.split(key)
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True, key=subkey)

        dt_init_std = dt_rank**-0.5 * dt_scale

        key, subkey = jax.random.split(key)
        dt = jnp.exp(
            jax.random.uniform(subkey, (inner_dim,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = jnp.clip(dt, min=dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))

        self.dt_proj.bias = inv_dt

        if dt_init == "constant":
            self.dt_proj.weight = jnp.zeros_like(self.dt_proj.weight) + dt_init_std
        elif dt_init == "random":
            key, subkey = jax.random.split(key)
            self.dt_proj.weight = jax.random.uniform(
                subkey, self.dt_proj.weight.shape, minval=-dt_init_std, maxval=dt_init_std
            )
        else:
            raise NotImplementedError

        # S4D (diagonal) real initialisation
        A = repeat(jnp.arange(state_dim + 1), "n -> d n", d=inner_dim) + 1

        self.A_log = jnp.log(A)

        self.D = jnp.ones((inner_dim,))

        key, subkey = jax.random.split(key)
        self.out_proj = nn.Linear(inner_dim, dim, bias=bias, key=subkey)

    def forward(self, x: jax.Array) -> jax.Array:
        # TODO: forward pass that calls into kernel interface
        pass


class ResidualBlock(eqx.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.block = MambaBlock(dim)

    def forward(self, x: jax.Array) -> jax.Array:
        # TODO: add fused residual add +norm followed by mamba mixer
        # correspnds to `Block` in reference
        pass
