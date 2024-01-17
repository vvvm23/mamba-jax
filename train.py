import argparse
import itertools
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from loguru import logger

from dataset import setup_dataloaders, setup_dataset, torch_to_np_batch
from mamba_jax.modelling.equinox import MambaLLM
from train_utils import (
    consolidate_metrics,
    make_experiment_directory,
    save_checkpoint,
    seed_others,
    update_metrics,
    wandb_init,
)


# setting up Optax optimiser with optional lr scheduler, weight decay, and
# gradient accumulation
def setup_optimiser(args, model):
    lr = args.learning_rate

    if args.use_lr_scheduler:
        logger.info("Using learning rate scheduler")
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        logger.info(f"{args.warmup_start_lr} -> {lr} (for {warmup_steps:,} steps)")
        logger.info(f"{lr} -> {args.end_learning_rate} (for {args.max_steps - warmup_steps:,} steps)")
        lr = optax.join_schedules(
            [
                optax.linear_schedule(
                    args.warmup_start_lr,
                    lr,
                    warmup_steps,
                ),
                optax.linear_schedule(
                    lr,
                    args.end_learning_rate,
                    args.max_steps - warmup_steps,
                ),
            ],
            [warmup_steps],
        )

    decay_spec = jax.tree_map(lambda _: "no_decay", eqx.filter(model, eqx.is_inexact_array))
    is_decay_weight = lambda p: hasattr(p, "weight") and not hasattr(p, "num_embeddings")
    where_decay_weight = lambda m: tuple(
        p.weight for p in jax.tree_util.tree_leaves(m, is_leaf=is_decay_weight) if is_decay_weight(p)
    )
    decay_spec = eqx.tree_at(where_decay_weight, decay_spec, replace_fn=lambda _: "decay")

    optimiser = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.multi_transform(
            {
                "decay": optax.adamw(learning_rate=lr, weight_decay=args.weight_decay, b1=args.beta1, b2=args.beta2),
                "no_decay": optax.adamw(learning_rate=lr, weight_decay=0.0, b1=args.beta1, b2=args.beta2),
            },
            decay_spec,
        ),
    )

    # TODO: update steps for sharding (essentially multiply micro_batch_size by num_devices)
    optimiser = optax.MultiSteps(optimiser, args.batch_size // args.micro_batch_size)

    return optimiser


# create jit-compiled train & eval steps, and also initialise optimiser
def create_step_fn(args, model, optimiser):
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(model, batch):
        # TODO: fix this as this actually results in -1 the sequence length
        input_ids, labels = jnp.copy(batch[:, :-1]), jnp.copy(batch[:, 1:])
        logits = jax.vmap(model[0])(input_ids)
        num_tokens = (labels != -100).sum()
        accuracy = jnp.argmax(logits, axis=-1) == labels
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

        accuracy = jnp.where(labels == -100, 0, accuracy).sum() / num_tokens
        loss = jnp.where(labels == -100, 0, loss).sum() / num_tokens

        return loss, accuracy

    def prepare_batch(batch):
        return batch["input_ids"]

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        batch = prepare_batch(batch)
        (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch)
        updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        metrics = {"loss": loss, "accuracy": accuracy, "bpt": loss / jnp.log(2)}
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, batch):
        batch = prepare_batch(batch)
        loss, accuracy = loss_fn(model, batch)
        metrics = {"loss": loss, "accuracy": accuracy, "bpt": loss / jnp.log(2)}
        return metrics

    return train_step, eval_step, opt_state


def main(args):
    logger.info("Starting training script..")

    # seed prng
    logger.info(f"Initialising PRNG state from seed {args.seed}")
    key = jax.random.PRNGKey(args.seed)
    seed_others(args.seed)

    # calculating micro batch size and accumulation steps
    # TODO: change micro batch size based on number of data parallel shards
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    assert args.batch_size % args.micro_batch_size == 0, "Micro batch size must perfectly divide batch size"
    grad_accumulation_steps = args.batch_size // args.micro_batch_size

    # initialising random model
    key, model_key = jax.random.split(key)
    model_kwargs = MambaLLM.args_namespace_to_model_kwargs(args)

    logger.info("Initialising model with arguments:")
    for k, v in model_kwargs.items():
        logger.info(f"\t{k}: {v}")
    model = MambaLLM(**model_kwargs, key=model_key)

    num_parameters = jax.tree_util.tree_reduce(lambda s, p: s + (p.size if eqx.is_array(p) else 0), model, 0)
    logger.info(f"Model has {num_parameters:,} parameters.")

    # initialising dataset
    logger.info(f"Initialising '{args.dataset}' dataset")
    train_dataset, eval_dataset = setup_dataset(args)
    train_loader, eval_loader = setup_dataloaders(args, train_dataset, eval_dataset)
    train_iter, eval_iter = iter(train_loader), iter(eval_loader)

    model = [model]  # annoying hack with Equinox + Optax
    optimiser = setup_optimiser(args, model)

    # create the jit-compiled train & eval steps, as well as init optimiser
    train_step, eval_step, opt_state = create_step_fn(args, model, optimiser)

    # create training directory and dump config there
    exp_dir = make_experiment_directory(args)
    logger.info(f"Experiment directory: {exp_dir}")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # init wandb if args.wandb present
    if args.wandb:
        logger.info("Initialising W&B")
    wandb_logger = wandb_init(args)

    # TODO: update for sharding
    logger.info("Starting training loop..")
    try:
        train_metrics = None
        start_time = time.time()
        for step_idx in range(args.max_steps):
            for _ in range(grad_accumulation_steps):
                # train phase
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = torch_to_np_batch(batch)
                model, opt_state, metrics = train_step(model, opt_state, batch)

                train_metrics = update_metrics(metrics, train_metrics)

            # log train metrics and reset them
            if step_idx > 0 and step_idx % args.log_freq == 0:
                metrics, train_metrics = consolidate_metrics(
                    train_metrics, args.log_freq * grad_accumulation_steps, "train"
                )
                if args.wandb:
                    wandb_logger.log(metrics, step=step_idx)

                end_time = time.time()
                batches_per_second = grad_accumulation_steps * args.log_freq / (end_time - start_time)
                tokens_per_second = batches_per_second * batch["input_ids"].size
                logger.info(f"[Train] Step {step_idx}/{args.max_steps}: {metrics} | tokens/s: {tokens_per_second}")

            if step_idx > 0 and step_idx % args.eval_freq == 0:
                # eval phase
                eval_metrics = None
                num_eval_micro_batches = 0

                for _ in range(args.eval_iters) if args.eval_iters > 0 else itertools.count():
                    # ensures consistent for different micro batch sizes, given same global batch size
                    end = True
                    for _ in range(grad_accumulation_steps):
                        try:
                            eval_batch = next(eval_iter)
                        except StopIteration:
                            eval_iter = iter(eval_loader)
                            if args.eval_iters == 0:
                                break
                            eval_batch = next(eval_iter)
                        eval_batch = torch_to_np_batch(eval_batch)
                        metrics = eval_step(model, eval_batch)
                        eval_metrics = update_metrics(metrics, eval_metrics)
                        num_eval_micro_batches += 1
                    else:
                        # if we didn't break then continue the outer loop
                        # we would only break if args.eval_iters == 0 (eval on
                        # whole dataset) and dataset was exhausted
                        # this isn't possible otherwise
                        end = False

                    if end:
                        break

                metrics, eval_metrics = consolidate_metrics(eval_metrics, num_eval_micro_batches, "eval")
                if args.wandb:
                    wandb_logger.log(metrics, step=step_idx)

                logger.info(f"[Eval] Step {step_idx}/{args.max_steps}: {metrics}")

            if step_idx > 0 and step_idx % args.save_freq == 0:
                # save checkpoint
                save_checkpoint(args, exp_dir, step_idx, model, opt_state)

            if step_idx > 0 and step_idx % args.log_freq == 0:
                # reset train throughput timer
                # we delay this to the end of the loop to ensure no false
                # readings involving eval or checkpoint save phase.
                start_time = time.time()

    except BaseException as e:
        # if exception, save the model before closing
        logger.warning("Caught exception.. Saving checkpoint before closing..")
        save_checkpoint(args, exp_dir, "final", model, opt_state)
        raise e

    logger.info("Finished training.. Saving final checkpoint..")
    save_checkpoint(args, exp_dir, "final", model, opt_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for PRNG initialisation.")

    # logging args
    parser.add_argument("--max_steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument("--log_freq", type=int, default=10, help="Frequency of logging train metrics.")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Frequency of evaluation phase.")
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=0,
        help="Number of iterations during evaluation phase. Defaults to 0, which uses the entire evalulation dataset.",
    )
    parser.add_argument("--save_freq", type=int, default=1000, help="Frequency of saving checkpoint.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")

    # data args
    parser.add_argument(
        "--dataset", type=str, default="afmck/text8-chunked1024", help="Dataset to use as on Huggingface hub."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Micro batch size, used to calculate gradient accumulation steps. If None, becomes equal to `batch_size`",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length for training.")

    # optimiser args
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Initial learning rate after warmup phase.")
    parser.add_argument("--end_learning_rate", type=float, default=1e-6, help="End learning rate.")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-5, help="Warmup start learning rate.")
    parser.add_argument(
        "--warmup_proportion", type=float, default=0.1, help="Proportion of warmup steps out of total steps."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for the optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use learning rate scheduler.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2.")

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
    parser.add_argument("--no_conv_bias", action="store_false", help="Do not use bias in Conv layer in Mamba block.")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layers.")
    parser.add_argument("--kernel_mode", type=str, default="xla_associative", help="Selects which Mamba Kernel to use.")
    parser.add_argument("--pad_vocab_mult", type=int, default=8, help="Pad vocab multiplier.")
    parser.add_argument("--norm_eps", type=float, default=1e-5, help="RMSNorm epsilon")
    parser.add_argument(
        "--res_in_bf16", action="store_true", help="Use bfloat16 for residual connections. Otherwise use float32."
    )

    args = parser.parse_args()

    main(args)
