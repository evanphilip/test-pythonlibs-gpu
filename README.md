Checks if Python libraries like PyTorch and JAX are able to leverage an Nvidia GPU by running a minimal benchmark.

## Installation

1. Clone this repository. For example, using
    ```bash
    git clone https://github.com/evanphilip/test-pythonlibs-gpu.git
    ```
    
2. Go to `test-pythonlibs-gpu` folder and use a tool of your choice to run the Python scripts. For example,
    
    1. run it using [uv](https://docs.astral.sh/uv/)
        ```bash
        uv run test_torch.py
        ```

    2. run it using [hatch](https://hatch.pypa.io/latest/)
        ```bash
        hatch run test_torch.py
        ```
