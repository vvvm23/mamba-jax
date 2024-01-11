import sys
import time
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from einops import einsum

from mamba_jax.kernels.interface import KernelType
from mamba_jax.kernels.reference import mamba_ssm
from mamba_jax.modelling.equinox import load_pretrained


def grad_wrapper(f):
    f_wrapper = lambda *args, **kwargs: f(*args, **kwargs).mean()

    return jax.grad(f_wrapper)


def scan_vs_associative_graph():
    delta_softplus = True
    scan_fn = partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=False)
    associative_scan_fn = partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=True)

    key = jax.random.PRNGKey(0)
    D = 2048
    iters = 5

    Ls = [2**4, 2**6, 2**8, 2**10, 2**12, 2**14, 2**16]
    Ns = [16]

    max_L = max(Ls)
    max_N = max(Ns)
    uF = jax.random.normal(key, (max_L, D), dtype=jnp.bfloat16)
    deltaF = jax.random.normal(key, (max_L, D), dtype=jnp.bfloat16)
    AF = jax.random.normal(key, (D, max_N), dtype=jnp.bfloat16)
    BF = jax.random.normal(key, (max_L, max_N), dtype=jnp.bfloat16)
    CF = jax.random.normal(key, (max_L, max_N), dtype=jnp.bfloat16)
    for N in Ns:
        jax.clear_caches()
        scan_times = []
        associative_scan_times = []
        for L in Ls:
            print(N, L)
            # D = 0

            u = uF[:L, :]
            delta = deltaF[:L, :]
            A = AF[:, :N]
            B = BF[:L, :N]
            C = CF[:L, :N]

            scan_fn = jax.jit(scan_fn)
            associative_scan_fn = jax.jit(associative_scan_fn)

            scan_fn(u, delta, A, B, C, 0).block_until_ready()
            associative_scan_fn(u, delta, A, B, C, 0).block_until_ready()

            start_time = time.time()
            for _ in range(iters):
                scan_fn(u, delta, A, B, C, 0).block_until_ready()
            scan_time = (time.time() - start_time) / iters
            scan_times.append(scan_time)

            start_time = time.time()
            for _ in range(iters):
                associative_scan_fn(u, delta, A, B, C, 0).block_until_ready()
            associative_scan_time = (time.time() - start_time) / iters
            associative_scan_times.append(associative_scan_time)

            print(scan_time)
            print(associative_scan_time)
            print()

        plt.plot(Ls, scan_times, label=f"scan (state_dim={N})")
        plt.plot(Ls, associative_scan_times, label=f"associative scan (state_dim={N})")
    plt.xlabel("Sequence length")
    plt.ylabel("Time")
    plt.yscale("log")
    plt.legend()
    plt.savefig("benchmark.png")


def mamba_llm_sample_graph():
    model_list = [
        "state-spaces/mamba-130m",
        "state-spaces/mamba-370m",
        "state-spaces/mamba-790m",
        "state-spaces/mamba-1.4b",
        "state-spaces/mamba-2.8b",
    ]

    iters = 10_000

    times = []
    for model_name in model_list:
        eqx.clear_caches()
        jax.clear_caches()
        print(model_name)
        model, _ = load_pretrained(model_name, dtype=jnp.bfloat16)

        @eqx.filter_jit
        def sample_step(model, input_id, cache):
            logits, cache = model.generate_step(input_id, cache=cache)
            return jnp.argmax(logits), cache

        cache = model.init_cache()
        idx = jnp.array(111, dtype=int)
        sample_step(model, idx, cache)

        start_time = time.time()
        for _ in tqdm.trange(iters):
            idx, cache = sample_step(model, idx, cache)
        time_per_token = (time.time() - start_time) / iters

        times.append(1 / time_per_token)

    plt.xlabel("Model")
    plt.xticks(rotation=90)
    plt.ylabel("Tokens / s")
    plt.bar([m.split("/")[-1] for m in model_list], times)
    plt.savefig("sample-benchmark.png")


def mamba_llm_throughput_graph():
    model_list = [
        "state-spaces/mamba-130m",
        "state-spaces/mamba-370m",
        "state-spaces/mamba-790m",
        "state-spaces/mamba-1.4b",
        "state-spaces/mamba-2.8b",
    ]

    iters = 5
    L = 2**12
    input_ids = jnp.array([111] * L, dtype=int)

    labels = []

    times = []
    mode = KernelType.XLA_ASSOCIATIVE
    for model_name in model_list:
        eqx.clear_caches()
        jax.clear_caches()
        print(model_name)
        model, _ = load_pretrained(model_name, dtype=jnp.bfloat16, kernel_mode=mode)
        model = eqx.filter_jit(model)

        model(input_ids)

        start_time = time.time()
        for _ in tqdm.trange(iters):
            model(input_ids)
        time_per_token = (time.time() - start_time) / iters

        labels.append(model_name + "+associative")
        times.append(time_per_token)

    mode = KernelType.XLA
    for model_name in model_list:
        eqx.clear_caches()
        jax.clear_caches()
        print(model_name)
        model, _ = load_pretrained(model_name, dtype=jnp.bfloat16, kernel_mode=mode)
        model = eqx.filter_jit(model)

        model(input_ids)

        start_time = time.time()
        for _ in tqdm.trange(iters):
            model(input_ids)
        time_per_token = (time.time() - start_time) / iters

        labels.append(model_name + "+scan")
        times.append(time_per_token)

    plt.xlabel("Model")
    plt.xticks(rotation=90)
    plt.ylabel("Time s")
    plt.bar(labels, times)
    plt.savefig("sample-benchmark.png", bbox_inches="tight")


if __name__ == "__main__":
    if sys.argv[1] == "graph":
        scan_vs_associative_graph()
        exit()

    if sys.argv[1] == "sample":
        mamba_llm_sample_graph()
        exit()

    if sys.argv[1] == "throughput":
        mamba_llm_throughput_graph()
        exit()

    with jax.profiler.trace("/tmp/tensorboard"):
        key = jax.random.PRNGKey(0)

        L, D, N = 8192 * 8, 16, 32

        u = jax.random.normal(key, (L, D))
        delta = jax.random.normal(key, (L, D))
        A = jax.random.normal(key, (D, N))
        B = jax.random.normal(key, (L, N))
        C = jax.random.normal(key, (L, N))
        D = 0

        delta_softplus = True

        scan_fn = partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=False)
        associative_scan_fn = partial(mamba_ssm, delta_softplus=delta_softplus, associative_scan=True)

        if len(sys.argv) > 1 and sys.argv[1] == "scan":
            scan_result = jax.jit(grad_wrapper(scan_fn))(u, delta, A, B, C, D).block_until_ready()
        elif len(sys.argv) > 1 and sys.argv[1] == "associative":
            associative_scan_result = jax.jit(grad_wrapper(associative_scan_fn))(
                u, delta, A, B, C, D
            ).block_until_ready()
        else:
            scan_fn = jax.jit(scan_fn)
            associative_scan_fn = jax.jit(associative_scan_fn)

            scan_fn(u, delta, A, B, C, D).block_until_ready()
            associative_scan_fn(u, delta, A, B, C, D).block_until_ready()

            start_time = time.time()
            for _ in range(100):
                scan_result = scan_fn(u, delta, A, B, C, D).block_until_ready()
            scan_time = (time.time() - start_time) / 100

            start_time = time.time()
            for _ in range(100):
                associative_scan_result = associative_scan_fn(u, delta, A, B, C, D).block_until_ready()
            associative_scan_time = (time.time() - start_time) / 100

            print(scan_time, associative_scan_time)

            assert scan_result.shape == associative_scan_result.shape
            assert np.allclose(np.asarray(scan_result), np.asarray(associative_scan_result))
