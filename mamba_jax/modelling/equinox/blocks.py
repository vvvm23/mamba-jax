import equinox as eqx
import equinox.nn as nn
import jax


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
class MambaBlock(eqx.Module):
    def __init__(self, dim: int):
        # TODO: add mamba specific layers (`Mamba` class in official)
        pass

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
