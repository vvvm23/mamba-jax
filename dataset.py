import string
from typing import Optional

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def chunk_dataset(dataset, args, target_field: Optional[str] = None):
    if target_field is None:
        if hasattr(args, "dataset_text_field"):
            target_field = args.dataset_text_field
        else:
            target_field = "text"

    def map_fn(batch):
        chunks = []
        for example in batch[target_field]:
            chunks += [example[i : i + args.sequence_length] for i in range(0, len(example), args.sequence_length)]

        return {target_field: chunks}

    dataset = dataset.map(map_fn, batched=True)

    return dataset


def pretokenize_dataset(dataset, tokenizer: AutoTokenizer, args, target_field: Optional[str] = None):
    if target_field is None:
        if hasattr(args, "dataset_text_field"):
            target_field = args.dataset_text_field
        else:
            target_field = "text"

    def map_fn(batch):
        batch = batch[target_field]
        # TODO: check if this is correct: we pad to +1 so when we slice for
        # labels and input_ids, the size is equal to sequence_length
        # should be fine so long as first element is SOS and last is EOS..
        return tokenizer(batch, padding="max_length", max_length=args.sequence_length + 1)

    dataset = dataset.map(map_fn, batched=True)

    return dataset


# get dataset from huggingface based on args.dataset
# TODO: support datasets with advanced tokenizers, and pre-tokenisation
def setup_text8_dataset(args, dataset_text_field: Optional[str] = None):
    if dataset_text_field is None:
        if hasattr(args, "dataset_text_field"):
            dataset_text_field = args.dataset_text_field
        else:
            dataset_text_field = "text"

    lower, upper = string.ascii_lowercase[0], string.ascii_lowercase[-1]
    lower, upper = ord(lower), ord(upper)
    space_or_pad = upper + 1

    def transform_fn(batch):
        batch = batch[dataset_text_field]
        batch = [example.ljust(args.sequence_length + 1) for example in batch]
        bytes = torch.tensor([[ord(c) for c in example] for example in batch], dtype=int)
        bytes[bytes == ord(" ")] = space_or_pad
        input_ids = bytes - lower

        return {"input_ids": input_ids}

    dataset = datasets.load_dataset(args.dataset)
    dataset = chunk_dataset(dataset, args, target_field="text")

    for split in ["train", "validation"]:
        dataset[split].set_transform(transform_fn)

    return dataset["train"], dataset["validation"]


def setup_hf_dataset(args, dataset_text_field: Optional[str] = None):
    dataset = datasets.load_dataset(args.dataset)

    # TODO: check if splits already exist
    if "validation" not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=0.1)  # TODO: make this configurable
        dataset["validation"] = dataset["test"]
        del dataset["test"]

    tokenizer = get_tokenizer(args)

    dataset = chunk_dataset(dataset, args, target_field=dataset_text_field)
    dataset = pretokenize_dataset(dataset, tokenizer, args, target_field=dataset_text_field)
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col != "input_ids"])
    dataset.set_format("np")

    return dataset


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_dataloaders(args, train_dataset, validation_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    return train_loader, validation_loader


def torch_to_np_batch(batch):
    return {k: v.numpy() for k, v in batch.items()}


def setup_dataset(args, dataset_text_field: Optional[str] = None):
    if "text8" in args.dataset:
        return setup_text8_dataset(args, dataset_text_field=dataset_text_field)

    return setup_hf_dataset(args, dataset_text_field=dataset_text_field)
