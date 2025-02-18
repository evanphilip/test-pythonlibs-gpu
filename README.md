## Installation

1. Clone this repository. For example, using
    ```bash
    git clone https://github.com/evanphilip/test-pythonlibs-gpu.git
    ```
    
2. Go to `test-pythonlibs-gpu` folder and use a tool of your choice to run the Python scripts. For example,
    
    1. run it using [uv](https://docs.astral.sh/uv/)
        ```bash
        uv run gpu_torch.py
        ```

    2. run it using [hatch](https://hatch.pypa.io/latest/)
        ```bash
        hatch run python gpu_torch.py
        ```
        
    > _Note_: `pyproject.toml` does not use any `uv` or `hatch` specific configuration, so you may also use other tools. For example, you could use a combination of `pyenv`, `venv`  and `pip`.
