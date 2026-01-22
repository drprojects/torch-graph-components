# Centralized optional dependencies (PyG, torch-scatter, etc.)

_DEPS_IMPORT_ERROR = """
torch-graph-components requires the following external dependencies,
which are NOT installed automatically:

- PyTorch (torch)
- torch-geometric
- torch-scatter

These packages depend on your Python, PyTorch, and CUDA versions
and must be installed manually by the user.

Please install them following the official instructions:
- PyTorch: https://pytorch.org
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- PyTorch Scatter: https://pytorch-geometric.readthedocs.io 
"""

# Initialize module attributes
torch = None
torch_scatter = None
torch_geometric = None

try:
    import torch
    import torch_scatter
    import torch_geometric
except ImportError as e:
    _IMPORT_EXCEPTION = e

    # Lazy attribute access
    class _LazyModule:
        def __init__(self, name):
            self._name = name

        def __getattr__(self, item):
            raise ImportError(_DEPS_IMPORT_ERROR) from _IMPORT_EXCEPTION

    torch = torch or _LazyModule("torch")
    torch_scatter = torch_scatter or _LazyModule("torch_scatter")
    torch_geometric = torch_geometric or _LazyModule("torch_geometric")

# Expose all modules
__all__ = ["torch", "torch_scatter", "torch_geometric"]
