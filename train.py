import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from loguru import logger

from mamba_jax.modelling.equinox import MambaLLM
from train_utils import (
    consolidate_metrics,
    make_experiment_directory,
    save_checkpoint,
    seed_others,
    update_metrics,
    wandb_init,
)


def setup_dataset(args):
    pass


def setup_dataloaders(args, dataset):
    pass


def setup_optimiser(args):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    warmup_proportion = args.warmup_proportion

    warmup_start_lr = args.warmup_start_lr
    end_learning_rate = args.end_learning_rate
    max_steps = args.max_steps
    max_grad_norm = args.max_grad_norm

    # TODO: update for sharding
    gradient_accumulation = args.batch_size // args.micro_batch_size


def create_step_fn(args, model, optimiser):
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(model, batch):
        pass

    def prepare_batch(batch):
        pass

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        metrics = None
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, batch):
        metrics = None
        return metrics

    return train_step, eval_step, opt_state


def main(args):
    logger.info("Starting training script..")

    logger.info(f"Initialising PRNG state from seed {args.seed}")
    key = jax.random.PRNGKey(args.seed)
    seed_others(args.seed)

    if args.micro_batch_size is None:
        args.micro_batch_size = args.micro_batch

    # TODO: change micro batch size based on number of data parallel shards

    assert args.batch_size % args.micro_batch_size == 0, "Micro batch size must perfectly divide batch size"

    key, model_key = jax.random.split(key)

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
        "conv_bias": args.conv_bias,
        "bias": args.bias,
        "use_kernel": args.use_kernel,  # TODO: update to use kernel mode
        "pad_vocab_mult": args.pad_vocab_mult,
        "norm_eps": args.norm_eps,
        "res_dtype": jnp.bfloat16 if args.res_in_bf16 else jnp.float32,
        "dtype": jnp.bfloat16 if args.bf16 else jnp.float32,
        "key": model_key,
    }
    logger.info("Initialising model with arguments:")
    for k, v in model_kwargs.items():
        logger.info(f"\t{k}: {v}")
    model = MambaLLM(**model_kwargs)

    num_parameters = jax.tree_util.tree_reduce(lambda s, p: s + (p.size if eqx.is_array(p) else 0), model, 0)
    logger.info(f"Model has {num_parameters:,} parameters.")

    logger.info(f"Initialising '{args.dataset}' dataset")

    train_dataset, eval_dataset = setup_dataset(args)
    train_loader, eval_loader = setup_dataloaders(args, train_dataset), setup_dataloaders(args, eval_dataset)

    optimiser = setup_optimiser(args)

    train_step, eval_step, opt_state = create_step_fn(args, model, optimiser)

    exp_dir = make_experiment_directory(args)
    logger.info(f"Experiment directory: {exp_dir}")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.wandb:
        logger.info("Initialising W&B")
    wandb_logger = wandb_init(args)

    # TODO: update for sharding
    logger.info("Starting training loop..")
    try:
        train_metrics = None
        for step_idx in range(args.max_steps):
            # train phase
            batch = next(train_loader)
            model, opt_state, metrics = train_step(model, opt_state, batch)

            train_metrics = update_metrics(metrics, train_metrics)

            if step_idx > 0 and step_idx % args.log_freq == 0:
                metrics, train_metrics = consolidate_metrics(train_metrics, args.log_freq, "train")
                if args.wandb:
                    wandb_logger.log(metrics, step=step_idx)

                logger.info(f"[Train] Step {step_idx}: {metrics}")

            if step_idx > 0 and step_idx % args.eval_freq == 0:
                # eval phase
                eval_metrics = None
                for _ in range(args.eval_iters):
                    eval_batch = next(eval_loader)
                    metrics = eval_step(model, eval_batch)
                    eval_metrics = update_metrics(metrics, eval_metrics)

                metrics, eval_metrics = consolidate_metrics(eval_metrics, args.eval_iters, "eval")
                if args.wandb:
                    wandb_logger.log(metrics, step=step_idx)

                logger.info(f"[Eval] Step {step_idx}: {metrics}")

            if step_idx > 0 and step_idx % args.save_freq == 0:
                # save checkpoint
                save_checkpoint(args, exp_dir, model, opt_state)

    except BaseException as e:
        logger.warning("Caught exception.. Saving checkpoint before closing..")
        save_checkpoint(args, exp_dir, model, opt_state)
        raise e

    logger.info("Finished training.. Saving final checkpoint..")
    save_checkpoint(args, exp_dir, model, opt_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for PRNG initialisation.")

    # logging args
    parser.add_argument("--max_steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--log_freq", type=int, default=100, help="Frequency of logging train metrics.")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency of evaluation phase.")
    parser.add_argument("--eval_iters", type=int, default=10, help="Number of iterations during evaluation phase.")
    parser.add_argument("--save_freq", type=int, default=10000, help="Frequency of saving checkpoint.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")

    # data args
    parser.add_argument("--dataset", type=str, default="openwebtext", help="Dataset to use as on Huggingface hub.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Micro batch size, used to calculate gradient accumulation steps.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")

    # optimiser args
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate after warmup phase.")
    parser.add_argument("--end_learning_rate", type=float, default=1e-6, help="End learning rate.")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-5, help="Warmup start learning rate.")
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.1, help="Proportion of warmup steps out of total steps."
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")

    # MambaLM args
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension.")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers.")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocab size of the model.")
    parser.add_argument("--state_dim", type=int, default=16, help="State size of SSM model.")
    parser.add_argument("--kernel_size", type=int, default=4, help="Kernel size of Conv layer in Mamba block.")
    parser.add_argument("--expand", type=int, default=2, help="Expansion factor in Mamba block.")
    parser.add_argument("--dt_rank", type=str, default="auto", help="Rank of the delta projection layer.")
    parser.add_argument("--dt_min", type=float, default=0.001, help="Minimum value of delta.")
    parser.add_argument("--dt_max", type=float, default=0.1, help="Maximum value of delta.")
    parser.add_argument("--dt_init", type=str, default="random", help="Initialisation method of delta projection")
    parser.add_argument("--dt_scale", type=float, default=1.0, help="Scale of initialisation of delta projection")
    parser.add_argument("--dt_init_floor", type=float, default=1e-4, help="TODO")
    parser.add_argument("--conv_bias", action="store_true", help="Use bias in Conv layer in Mamba block.")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layers.")
    parser.add_argument("--use_kernel", action="store_true", help="TODO: replace with kernel mode Literal")
    parser.add_argument("--pad_vocab_mult", type=int, default=8, help="Pad vocab multiplier.")
    parser.add_argument("--norm_eps", type=float, default=1e-5, help="RMSNorm epsilon")
    parser.add_argument(
        "--res_in_bf16", action="store_true", help="Use bfloat16 for residual connections. Otherwise use float32."
    )

    args = parser.parse_args()

    main(args)
