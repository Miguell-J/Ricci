"""Ricci: typed contraction diagrams for scientific computing.

Importing the core does not initialize PyTorch, CUDA or any network connection.
Numerical entry points are available from ``ricci.backends.torch``.
"""

from ricci.core import (
    BatchGroup,
    Edge,
    Graph,
    Index,
    IndexSpace,
    NodeSpec,
    Port,
    RicciError,
    SpaceKind,
    Variance,
)

__version__ = "0.1.0a1"

__all__ = [
    "BatchGroup",
    "Edge",
    "Graph",
    "Index",
    "IndexSpace",
    "NodeSpec",
    "Port",
    "RicciError",
    "SpaceKind",
    "Variance",
    "__version__",
]
