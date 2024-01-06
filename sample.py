import argparse

import equinox as eqx
import jax
import jax.numpy as jnp

from mamba_jax.modelling.equinox.loader import load_pretrained


def main(args):
    model, tokenizer = load_pretrained(args.model, dtype=jnp.bfloat16 if args.bf16 else jnp.float32)

    # prompt = "Instructions for building a time machine, by Albert Einstein:\n\n"
    prompt = args.prompt

    gen_len = args.gen_len
    temperature = args.temperature

    generate_step = eqx.filter_jit(model.generate_step)

    for p in range(args.seed_iters):
        output = []
        cache = model.init_cache()

        input_ids = tokenizer(prompt, return_tensors="np").input_ids[0]

        print()
        print(tokenizer.decode(input_ids, skip_special_tokens=True), flush=True, end="")
        key = jax.random.PRNGKey(args.seed + p)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="bar")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gen-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_iters", type=int, default=1)
    args = parser.parse_args()

    main(args)
