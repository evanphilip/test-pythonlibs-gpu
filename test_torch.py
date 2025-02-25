# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.6.0",
# ]
# ///
"""
Checks if PyTorch is able to leverage the GPU by running a minimal benchmark.
"""

import torch
import torch.utils.benchmark as benchmark


def job(gpu):
    device = "cuda" if gpu else "cpu"
    x = torch.randn(10000, 1024, device=device)
    return benchmark.Timer(
        stmt="x.mul(x).sum(-1)",
        globals={"x": x},
        label="Benchmark",
        description=device,
    ).timeit(100)


print(__doc__)
print(f"GPU Device Name : {torch.cuda.get_device_name()}\n")
results = [job(gpu=False), job(gpu=True)]
benchmark.Compare(results).print()
