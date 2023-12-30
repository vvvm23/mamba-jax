import equinox as eqx
import jax


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
class MambaBlock(eqx.Module):
    def __init__(self):
        pass

    def forward(self, x: jax.Array) -> jax.Array:
        pass
