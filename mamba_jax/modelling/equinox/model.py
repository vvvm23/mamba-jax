from typing import Dict, List, Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

from ...kernels.interface import KernelType, KernelTypeMapping
from .blocks import ResidualBlock, create_block
from .utils import cast_eqx_layer

# corresponds to implementation in:
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py

# Model with Mamba sequence mixer, with no specific task head
# corresponds to the `MixerModel` in original implementation
class MambaModel(eqx.Module):

    embedding: nn.Embedding
    layers: List[ResidualBlock]

    norm_f: nn.RMSNorm

    dim: int
    num_layers: int
    state_dim: int = 16
    kernel_size: int = 4
    expand: int = 2

    dtype: jnp.dtype = jnp.float32

    def __init__(
        self,
        dim: int,
        num_layers: int,
        vocab_size: int,
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
        kernel_mode: KernelType = KernelType.XLA,
        pad_vocab_mult: int = 0,
        norm_eps: float = 1e-5,
        # TODO: add norm type (rms or layer)
        res_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.kernel_size = kernel_size
        self.expand = expand
        self.num_layers = num_layers
        self.dtype = dtype

        if pad_vocab_mult != 0 and vocab_size % pad_vocab_mult != 0:
            vocab_size += pad_vocab_mult - (vocab_size % pad_vocab_mult)

        key, subkey = jax.random.split(key)
        self.embedding = cast_eqx_layer(nn.Embedding(vocab_size, dim, key=subkey), dtype=dtype)

        key, *layer_keys = jax.random.split(key, num_layers + 1)
        self.layers = [
            create_block(
                dim=dim,
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
                kernel_mode=kernel_mode,
                layer_idx=i,
                norm_eps=norm_eps,
                res_dtype=res_dtype,
                dtype=dtype,
                key=subkey,
            )
            for i, subkey in enumerate(layer_keys)
        ]

        self.norm_f = cast_eqx_layer(nn.RMSNorm(dim, eps=norm_eps), dtype=dtype)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = jax.vmap(self.embedding)(input_ids)

        res = None
        for layer in self.layers:
            x = x.astype(self.dtype)
            x, res = layer(x, res)

        res = x + res if res is not None else x
        x = jax.vmap(self.norm_f)(res.astype(self.norm_f.weight.dtype))

        return x

    def generate_step(self, input_ids: jax.Array, cache=None) -> jax.Array:
        x = self.embedding(input_ids)

        res = None
        for layer in self.layers:
            x = x.astype(self.dtype)
            x, res = layer.generate_step(x, res, cache=cache)

        res = x + res if res is not None else x
        x = self.norm_f(res.astype(self.norm_f.weight.dtype))

        return x, cache

    def init_cache(self):
        cache = []
        for _ in range(self.num_layers):
            conv_state = jnp.zeros((self.dim * self.expand, self.kernel_size), dtype=self.dtype)
            ssm_state = jnp.zeros((self.dim * self.expand, self.state_dim), dtype=self.dtype)

            cache.append((conv_state, ssm_state))

        return cache


# corresponds to `MambaLMHeadModel` class in original implementation
class MambaLLM(eqx.Module):
    model: MambaModel
    lm_head: nn.Linear

    def __init__(
        self,
        dim: int,
        num_layers: int,
        vocab_size: int,
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
        kernel_mode: KernelType = KernelType.XLA,
        pad_vocab_mult: int = 8,
        norm_eps: float = 1e-5,
        # TODO: add norm type (rms or layer)
        res_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        key: jax.random.PRNGKey = None,
    ):
        super().__init__()
        model_key, head_key = jax.random.split(key)

        self.model = MambaModel(
            dim=dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
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
            kernel_mode=kernel_mode,
            pad_vocab_mult=pad_vocab_mult,
            norm_eps=norm_eps,
            # TODO: add norm type (rms or layer)
            res_dtype=res_dtype,
            dtype=dtype,
            key=model_key,
        )

        # keep in full precision
        self.lm_head = nn.Linear(dim, vocab_size, use_bias=False, key=head_key)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.model(input_ids)
        x = x.astype(self.lm_head.weight.dtype)
        return jax.vmap(self.lm_head)(x)

    # init cache for efficient sampling
    def init_cache(self):
        return self.model.init_cache()

    # performs a single generate step
    def generate_step(self, input_ids: jax.Array, cache) -> jax.Array:
        x, cache = self.model.generate_step(input_ids, cache)
        return self.lm_head(x), cache

    # this generate call will sample from the model without interacting with the host.
    # it won't stop when eos_token_id is generated, so it needs to be
    # post-processed to remove trailing tokens.
    def generate(
        self, input_ids: jax.Array, gen_len: int, temperature: float = 1.0, key: jax.random.PRNGKey = None
    ) -> jax.Array:
        cache = self.init_cache()

        def prefill_scan(cache, x):
            _, cache = self.generate_step(x, cache=cache)
            return cache, None

        cache, _ = jax.lax.scan(prefill_scan, init=cache, xs=input_ids[:-1])

        def generate_scan(carry, _):
            cache, input_id, key = carry

            key, subkey = jax.random.split(key)
            logits, cache = self.generate_step(input_id, cache=cache)
            logits = logits / temperature
            output_id = jax.random.categorical(subkey, logits)

            return (cache, output_id, key), output_id

        _, output_ids = jax.lax.scan(generate_scan, init=(cache, input_ids[-1], key), xs=jnp.arange(gen_len)[:, None])

        return jnp.concatenate([input_ids, output_ids])

    # TODO: does this belong here? weakly couples MambaLLM implementation with args in train.py
    @staticmethod
    def args_namespace_to_model_kwargs(args):
        model_kwargs = {
            "dim": args.dim,
            "num_layers": args.num_layers,
            "vocab_size": args.vocab_size,
            "state_dim": args.state_dim,
            "kernel_size": args.kernel_size,
            "expand": args.expand,
            "dt_rank": args.dt_rank,
            "dt_min": args.dt_min,
            "dt_max": args.dt_max,
            "dt_init": args.dt_init,
            "dt_scale": args.dt_scale,
            "dt_init_floor": args.dt_init_floor,
            "conv_bias": args.no_conv_bias,
            "bias": args.bias,
            "kernel_mode": KernelTypeMapping[args.kernel_mode],
            "pad_vocab_mult": args.pad_vocab_mult,
            "norm_eps": args.norm_eps,
            "res_dtype": jnp.bfloat16 if args.res_in_bf16 else jnp.float32,
            "dtype": jnp.bfloat16 if args.bf16 else jnp.float32,
        }

        return model_kwargs
