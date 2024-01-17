# Mamba x JAX
Unofficial Implementation of "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces" in JAX.

> ⚠️ **This is very much a work-in-progress implementation. Expect numerical**
> **mismatches, slower speeds, bad code, and general wrongness herein.** ⚠️

## Installation

As the plan is to eventually write custom Pallas kernels for the Mamba
recurrence scan, we need to install requirements that work with Pallas.

Unfortunately, Pallas is currently quite hard to install (see [this
issue](https://github.com/google/jax/issues/18603)) and the required options
can't be fully specified in a `requirements.txt` file. So, to setup the
environment for this repository, take the following steps:
1. Create a Python 3.9 or 3.10 virtual environment.
2. Run `install-requirements.txt` and ensure none of the commands fail.

Such a kernel does not exist yet, and it is not clear how it would be
implemented. However, I optimistically pin the versions for now.

## Usage

### Sampling
The script `sample.py` is the main entry point to sample from a pretrained
Mamba model:
```
usage: sample.py [-h] [--prompt PROMPT] [--model MODEL] [--bf16] [--gen_len GEN_LEN]
                 [--temperature TEMPERATURE] [--seed SEED] [--seed_iters SEED_ITERS]
                 [--scan]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Starting prompt for generation. (default: Aloha, World! )
  --model MODEL         Model repo id as on Huggingface Hub. (default: state-
                        spaces/mamba-2.8b)
  --bf16                Use bfloat16 for inference (default: False)
  --gen_len GEN_LEN     Length of generated sequence. (default: 1024)
  --temperature TEMPERATURE
                        Sampling temperature. (default: 1.0)
  --seed SEED           Random seed for PRNG initialisation. (default: 0)
  --seed_iters SEED_ITERS
                        Number of seeds to generate, starting from --seed. (default: 1)
  --scan                Use jax.lax.scan version of generate loop. (default: False)
```

### Training
The script `train.py` can be used the `MambaLLM` for next token prediction.
Currently, it only supports Text8 style datasets (search for `afmck/text8` on
the Huggingface hub) for next-character prediction. Further dataset support
will come later, though it should be trivial to add this.

Usage is as follows:
```
usage: train.py [-h] [--bf16] [--seed SEED] [--max_steps MAX_STEPS]
                [--log_freq LOG_FREQ] [--eval_freq EVAL_FREQ]
                [--eval_iters EVAL_ITERS] [--save_freq SAVE_FREQ] [--wandb]
                [--dataset DATASET] [--batch_size BATCH_SIZE]
                [--micro_batch_size MICRO_BATCH_SIZE]
                [--num_workers NUM_WORKERS]
                [--sequence_length SEQUENCE_LENGTH]
                [--learning_rate LEARNING_RATE]
                [--end_learning_rate END_LEARNING_RATE]
                [--warmup_start_lr WARMUP_START_LR]
                [--warmup_proportion WARMUP_PROPORTION]
                [--weight_decay WEIGHT_DECAY] [--max_grad_norm MAX_GRAD_NORM]
                [--use_lr_scheduler] [--beta1 BETA1] [--beta2 BETA2]
                [--dim DIM] [--num_layers NUM_LAYERS]
                [--vocab_size VOCAB_SIZE] [--state_dim STATE_DIM]
                [--kernel_size KERNEL_SIZE] [--expand EXPAND]
                [--dt_rank DT_RANK] [--dt_min DT_MIN] [--dt_max DT_MAX]
                [--dt_init DT_INIT] [--dt_scale DT_SCALE]
                [--dt_init_floor DT_INIT_FLOOR] [--no_conv_bias] [--bias]
                [--kernel_mode KERNEL_MODE] [--pad_vocab_mult PAD_VOCAB_MULT]
                [--norm_eps NORM_EPS] [--res_in_bf16]

options:
  -h, --help            show this help message and exit
  --bf16                Use bfloat16 for training (default: False)
  --seed SEED           Random seed for PRNG initialisation. (default: 0)
  --max_steps MAX_STEPS
                        Number of training steps. (default: 10000)
  --log_freq LOG_FREQ   Frequency of logging train metrics. (default: 10)
  --eval_freq EVAL_FREQ
                        Frequency of evaluation phase. (default: 1000)
  --eval_iters EVAL_ITERS
                        Number of iterations during evaluation phase. Defaults
                        to 0, which uses the entire evalulation dataset.
                        (default: 0)
  --save_freq SAVE_FREQ
                        Frequency of saving checkpoint. (default: 1000)
  --wandb               Log metrics to Weights & Biases. (default: False)
  --dataset DATASET     Dataset to use as on Huggingface hub. (default:
                        afmck/text8-chunked1024)
  --batch_size BATCH_SIZE
                        Batch size for training. (default: 8)
  --micro_batch_size MICRO_BATCH_SIZE
                        Micro batch size, used to calculate gradient
                        accumulation steps. If None, becomes equal to
                        `batch_size` (default: None)
  --num_workers NUM_WORKERS
                        Number of worker processes for data loading. (default:
                        4)
  --sequence_length SEQUENCE_LENGTH
                        Sequence length for training. (default: 1024)
  --learning_rate LEARNING_RATE
                        Initial learning rate after warmup phase. (default:
                        0.0006)
  --end_learning_rate END_LEARNING_RATE
                        End learning rate. (default: 1e-06)
  --warmup_start_lr WARMUP_START_LR
                        Warmup start learning rate. (default: 1e-05)
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of warmup steps out of total steps.
                        (default: 0.1)
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer. (default: 0.1)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum gradient norm for gradient clipping. (default:
                        1.0)
  --use_lr_scheduler    Use learning rate scheduler. (default: False)
  --beta1 BETA1         Adam beta1. (default: 0.9)
  --beta2 BETA2         Adam beta2. (default: 0.95)
  --dim DIM             Model dimension. (default: 1024)
  --num_layers NUM_LAYERS
                        Number of layers. (default: 32)
  --vocab_size VOCAB_SIZE
                        Vocab size of the model. (default: 50257)
  --state_dim STATE_DIM
                        State size of SSM model. (default: 16)
  --kernel_size KERNEL_SIZE
                        Kernel size of Conv layer in Mamba block. (default: 4)
  --expand EXPAND       Expansion factor in Mamba block. (default: 2)
  --dt_rank DT_RANK     Rank of the delta projection layer. (default: auto)
  --dt_min DT_MIN       Minimum value of delta. (default: 0.001)
  --dt_max DT_MAX       Maximum value of delta. (default: 0.1)
  --dt_init DT_INIT     Initialisation method of delta projection (default:
                        random)
  --dt_scale DT_SCALE   Scale of initialisation of delta projection (default:
                        1.0)
  --dt_init_floor DT_INIT_FLOOR
                        TODO (default: 0.0001)
  --no_conv_bias        Do not use bias in Conv layer in Mamba block.
                        (default: True)
  --bias                Use bias in linear layers. (default: False)
  --kernel_mode KERNEL_MODE
                        Selects which Mamba Kernel to use. (default:
                        xla_associative)
  --pad_vocab_mult PAD_VOCAB_MULT
                        Pad vocab multiplier. (default: 8)
  --norm_eps NORM_EPS   RMSNorm epsilon (default: 1e-05)
  --res_in_bf16         Use bfloat16 for residual connections. Otherwise use
                        float32. (default: False)
```

### Mamba Components

The components of the full Mamba architecture can be imported as follows:
- An interface with the S6 (S4 with selective scan) model can be imported at the
path `mamba_jax.kernels.mamba_ssm`. This is a purely functional implementation
of Algorithm 2 in the paper which is agnostic of the neural network API you
use.  Currently, this just dispatches to a pure JAX implementation, though the
idea is you will be able to dispatch to an optimised Pallas kernel via the
`mode` argument in the future.
- [Equinox](https://github.com/patrick-kidger/equinox) Mamba language model and
sub-components of it can be found in `mamba_jax.modelling.equinox` as
`MambaBlock`, `ResidualBlock`, `MambaModel`, and `MambaLLM`.
- PRs for other neural network APIs (Flax, NNX) welcome.

## Roadmap
- [ ] Make this all pip installable.
- [ ] Testing to 100% verify parity with CUDA reference.
- [ ] Add efficient training code in pure JAX.
- [ ] Add efficient custom kernels for work-efficient associative scan, implemented in Pallas.
- [ ] Try to reproduce some training results from scratch.
- [ ] Complex number mode

---

### Acknowledgements

This implementation was based off a mix of:
- [Original implementation in PyTorch and CUDA](https://github.com/state-spaces/mamba)
- [Minimal Mamba implementation in PyTorch](https://github.com/johnma2006/mamba-minimal)

A lot of understanding of how S4 models work was derived from:
- [Albert Gu's amazing thesis on the topic](https://stacks.stanford.edu/file/druid:mb976vf9362/gu_dissertation-augmented.pdf)
- [The Annotated S4 by Sasha Rush](https://srush.github.io/annotated-s4/)

And a lot of understanding on the associative scan recurrent form was derived from:
- [Appendix H of the S5 paper](https://arxiv.org/abs/2208.04933)

### References

[**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**](https://arxiv.org/abs/2312.00752)

*Albert Gu, Tri Dao*
```
@misc{gu2023mamba,
      title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
      author={Albert Gu and Tri Dao},
      year={2023},
      eprint={2312.00752},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
