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

gpu_device = jax.devices("gpu")[0]
cpu_device = jax.devices("cpu")[0]


def f(x):
    return x.T @ (x - x.mean(axis=0))


print(__doc__)
print(f"Default backend available to JAX : {jax.default_backend()}\n")

x = np.random.randn(1000, 1000).astype(np.float32)
x_cpu = jax.device_put(x, device=cpu_device)
x_gpu = jax.device_put(x, device=gpu_device)

f_cpu = jax.jit(f, device=cpu_device)
f_gpu = jax.jit(f, device=gpu_device)

f_cpu(x_cpu).block_until_ready()
f_gpu(x_gpu).block_until_ready()

iterations, timing = timeit.Timer(
    "f_cpu(x_cpu).block_until_ready()", globals=globals()
).autorange()
t_cpu = timing / iterations
iterations, timing = timeit.Timer(
    "f_gpu(x_gpu).block_until_ready()", globals=globals()
).autorange()
t_gpu = timing / iterations

print(f"Time per job on CPU\t: {t_cpu:#.2e}s")
print(f"Time per job on GPU\t: {t_gpu:#.2e}s")
