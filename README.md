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
