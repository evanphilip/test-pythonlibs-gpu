# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax[cuda]>=0.5.0",
# ]
# ///
"""
Checks if JAX is able to leverage the GPU by running a minimal benchmark.
"""

import timeit

import jax
import numpy as np


def job(gpu):
    device = jax.devices("gpu")[0] if gpu else jax.devices("cpu")[0]
    x = jax.device_put(np.random.randn(1000, 1000).astype(np.float32), device=device)
    f = jax.jit(lambda x: x.T @ (x - x.mean(axis=0)), device=device)
    f(x).block_until_ready()
    iterations, timing = timeit.Timer(
        "f(x).block_until_ready()", globals=locals()
    ).autorange()
    return timing / iterations


print(__doc__)
print(f"Default backend available to JAX : {jax.default_backend()}\n")
print(f"Time per job on CPU\t: {job(gpu=False):#.2e}s")
print(f"Time per job on GPU\t: {job(gpu=True):#.2e}s")
