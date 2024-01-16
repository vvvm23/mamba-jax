import argparse
import json
import string
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from mamba_jax.kernels import KernelTypeMapping
from mamba_jax.modelling.equinox import MambaLLM
from mamba_jax.modelling.equinox.loader import load_pretrained


class Text8Tokenizer:
    def __init__(self):
        lower, upper = string.ascii_lowercase[0], string.ascii_lowercase[-1]
        self.lower, self.upper = ord(lower), ord(upper)
        self.space = self.upper + 1

        self.eos_token_id = -1

    def __call__(self, prompt, *args, **kwargs):
        prompt = prompt.lower()
        assert all(c in string.ascii_lowercase + " " for c in prompt)

        bytes = np.array([ord(c) for c in prompt], dtype=int)
        bytes[bytes == ord(" ")] = self.space
        input_ids = bytes - self.lower

        return SimpleNamespace(input_ids=[input_ids])

    def decode(self, input_ids, *args, **kwargs):
        if isinstance(input_ids, list):
            bytes = [input_ids[0] + self.lower]
        else:
            bytes = input_ids + self.lower
        decoded = "".join([chr(b) if b != self.space else " " for b in bytes])

        return decoded


def load_local_model(args):
    with open(args.config, mode="r") as f:
        config = json.load(f)

    config = SimpleNamespace(**config)

    # TODO: move this under a 'model_config' sub-dictionary so we can just do **config.model_config
    model_kwargs = {
        "dim": config.dim,
        "num_layers": config.num_layers,
        "vocab_size": config.vocab_size,
        "state_dim": config.state_dim,
        "kernel_size": config.kernel_size,
        "expand": config.expand,
        "dt_rank": config.dt_rank,
        "dt_min": config.dt_min,
        "dt_max": config.dt_max,
        "dt_init": config.dt_init,
        "dt_scale": config.dt_scale,
        "dt_init_floor": config.dt_init_floor,
        "conv_bias": config.no_conv_bias,
        "bias": config.bias,
        "kernel_mode": KernelTypeMapping[config.kernel_mode],
        "pad_vocab_mult": config.pad_vocab_mult,
        "norm_eps": config.norm_eps,
        "res_dtype": jnp.bfloat16 if config.res_in_bf16 else jnp.float32,
        "dtype": jnp.bfloat16 if config.bf16 else jnp.float32,
        "key": jax.random.PRNGKey(0),  # dummy key as we will overwrite weights later
    }
    model = MambaLLM(**model_kwargs)
    model = eqx.tree_deserialise_leaves(args.model, like=model)

    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # TODO: for now, the training script only trains on text8, so we just use
    # tokeniser for that, which can be a simple function
    tokenizer = Text8Tokenizer()

    return model, tokenizer


def main(args):
    # TODO: make this more robust as future models may not be under state-spaces namespace
    if args.model.startswith("state-spaces"):
        model, tokenizer = load_pretrained(
            args.model,
            dtype=jnp.bfloat16 if args.bf16 else jnp.float32,
            kernel_mode=KernelTypeMapping[args.kernel_mode],
        )
    else:  # is probably local model
        model, tokenizer = load_local_model(args)

    prompt = args.prompt

    gen_len = args.gen_len
    temperature = args.temperature

    generate_step = eqx.filter_jit(model.generate_step)

    generate_fn = None
    if args.scan:
        generate_fn = eqx.filter_jit(model.generate)

    for p in range(args.seed_iters):
        output = []
        cache = model.init_cache()

        input_ids = tokenizer(prompt, return_tensors="np").input_ids[0]

        key = jax.random.PRNGKey(args.seed + p)

        if generate_fn is not None:
            output_ids = generate_fn(input_ids, gen_len, temperature=temperature, key=key)
            print(tokenizer.decode(output_ids, skip_special_tokens=True))
            print("\n\n")
            continue

        print()
        print(tokenizer.decode(input_ids, skip_special_tokens=True), flush=True, end="")

        # prefill
        for input_id in input_ids[:-1]:
            _, cache = generate_step(input_id, cache=cache)

        input_id = input_ids[-1]
        output = input_ids.tolist()

        for _ in range(gen_len):
            logits, cache = generate_step(input_id, cache=cache)
            logits = logits / temperature
            key, subkey = jax.random.split(key)
            input_id = jax.random.categorical(subkey, logits)
            if input_id == tokenizer.eos_token_id:
                print("\n\n\n")
                break

            output.append(input_id)
            print(tokenizer.decode([output[-1]]), flush=True, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompt", type=str, default="Aloha, World! ", help="Starting prompt for generation.")
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-2.8b",
        help="Local JAX model checkpoint or PyTorch model repo id as on Huggingface Hub.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to model config file if using a local JAX model checkpoint."
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for inference. If using `--config` the value there will overwrite this flag.",
    )
    parser.add_argument("--gen_len", type=int, default=1024, help="Length of generated sequence.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for PRNG initialisation.")
    parser.add_argument("--seed_iters", type=int, default=1, help="Number of seeds to generate, starting from --seed.")
    parser.add_argument("--scan", action="store_true", help="Use jax.lax.scan version of generate loop.")
    args = parser.parse_args()

    main(args)
