import string

import datasets
import torch
from torch.utils.data import DataLoader


# get dataset from huggingface based on args.dataset
# TODO: support datasets with advanced tokenizers, and pre-tokenisation
def setup_dataset(args):
    lower, upper = string.ascii_lowercase[0], string.ascii_lowercase[-1]
    lower, upper = ord(lower), ord(upper)
    space_or_pad = upper + 1

    def transform_fn(batch):
        batch = batch["text"]
        batch = [example.ljust(args.sequence_length + 1) for example in batch]
        bytes = torch.tensor([[ord(c) for c in example] for example in batch], dtype=int)
        bytes[bytes == ord(" ")] = space_or_pad
        input_ids = bytes - lower

        return {"input_ids": input_ids}

    dataset = datasets.load_dataset(args.dataset)

    # TODO: chunk dataset to args.sequence_length (potentially adding extra rows)

    for split in ["train", "validation"]:
        dataset[split].set_transform(transform_fn)

    return dataset["train"], dataset["validation"]


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
