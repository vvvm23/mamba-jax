import equinox as eqx
import jax


# Model with Mamba sequence mixer, with no specific task head
# corresponds to the `MixerModel`
class MambaModel(eqx.Module):
    def __init__(self):
        pass

    def forward(self, x: jax.Array) -> jax.Array:
        pass


# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
# essentially, the `MambaLMHeadModel` class
class MambaLLM(eqx.Module):
    def __init__(self):
        pass

    def forward(self, x: jax.Array) -> jax.Array:
        pass
