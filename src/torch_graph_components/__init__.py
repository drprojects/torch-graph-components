# src/torch_graph_components/__init__.py

from .wcc import wcc_by_max_propagation
from .merge import merge_components_by_contour_prior

__all__ = [
    "wcc_by_max_propagation",
    "merge_components_by_contour_prior",
]
