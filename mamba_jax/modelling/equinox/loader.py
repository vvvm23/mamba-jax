import json
from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import torch  # TODO: can remove dependency once mamba converted to safetensors
from huggingface_hub import hf_hub_download

from .model import MambaLLM


def get_pt_checkpoint(repo_id: str, config_path: str = "config.json", checkpoint_path: str = "pytorch_model.bin"):
    config_path = hf_hub_download(repo_id=repo_id, filename=config_path)
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_path)

    with open(config_path, mode="r") as f:
        config = json.load(f)

    sd = torch.load(checkpoint_path, weights_only=True)

    return sd, config


def pt_to_raw_pytree(sd):
    def _key_rename(k: str):
        # TODO: consider just renaming this in modelling code 🤷
        k = k.replace("backbone", "model")
        return k

    return {_key_rename(k): jnp.asarray(v) for k, v in sd.items()}


# TODO: add test for this, comparing all weights as a sanity check
def init_mamba_from_raw_pytree(tree, config):
    # TODO: use other options from config
    N = config["d_model"]
    num_layers = config["n_layer"]
    vocab_size = config["vocab_size"]
    model = MambaLLM(N, num_layers, vocab_size, key=jax.random.PRNGKey(0))

    def where_fn(model):
        where = []
        for k in tree.keys():
            path = k.split(".")

            node = model
            for p in path:
                if p.isnumeric():
                    node = node[int(p)]
                else:
                    node = getattr(node, p)

            where.append(node)

        return where

    def generate_replace(tree):
        replace = []
        for k, v in tree.items():
            # paranoid about ordering..
            if "conv1d.bias" in k:
                replace.append(jnp.expand_dims(v, axis=-1))
                continue
            replace.append(v)

        return replace

    import ipdb

    ipdb.set_trace()
    replace = generate_replace(tree)
    model = eqx.tree_at(where_fn, model, replace=replace)

    return model


if __name__ == "__main__":
    from transformers import AutoTokenizer

    sd, config = get_pt_checkpoint("state-spaces/mamba-130m")
    tree = pt_to_raw_pytree(sd)
    model = init_mamba_from_raw_pytree(tree, config)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompt = "Aloha, World! "
    input_ids = tokenizer(prompt, return_tensors="np").input_ids[0]

    import ipdb

    ipdb.set_trace()
    logits = model(input_ids)
    output_ids = jnp.argmax(logits, axis=-1)
    print(tokenizer.decode(output_ids))
