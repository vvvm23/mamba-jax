## Pallas Kernel for Mamba Selective SSM

This directory contains the implementation for the Mamba Selective SSM
operation written in Pallas, an extension of JAX that allows for custom kernels
that compile to both GPU and TPU.

The kernel will perform the following operations:
- Load parameters Delta, A, B, and C from HBM into SRAM (`A = A_ref[...]` performs the load from HBM).
- Discretize A and B within SRAM, yielding arrays of size (B, L, D, N)
- Perform a parallel associative scan to compute the hidden states, possible
  due to the structure of the update rule. This yields an array of size (B, L,
  D, N) in SRAM.
- Multiply and sum with C, producing output of size (B, L, D) which we write
  back to SRAM (`out_ref[...] = out`).
- We checkpoint the regions with the full sized activations, so we don't need
  to store the large (B, L, D, N) after they have been used.

For efficient training, we also define custom kernels for the backwards pass
using `jax.custom_jvp`.

*We also include kernels for a fused residual+norm layer