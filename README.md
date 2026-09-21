# Ricci

**Typed tensor contraction diagrams for differentiable scientific computing.**

Ricci connects explicit tensor legs, checks spaces and duality, and executes the
result with PyTorch. Inspired by string diagrams, it separates the mathematical
expression from contraction planning, numerical data and future MCP transport.

**Status: 0.1.0a1, experimental foundation.** Package distribution name:
`ricci-tensor`; Python import: `ricci`. No PyPI publication is implied.

## What works now

- Immutable, backend-independent spaces, indices, node specifications and graphs.
- Explicit edges, traces, multiple edges, outer products and ordered outputs.
- Strict geometric variance, space/basis/fiber identity and preserved batch groups.
- Dense real/complex PyTorch execution with ordinary autograd and fresh bindings.
- Optional `opt_einsum` planning, bounded path cache and intermediate-size estimates.
- Versioned JSON graph exchange with a generated schema, without tensor serialization.
- Executable examples: Minkowski metric, continuum strain energy, supplied curvature,
  Bell state/MPS contraction and trainable attention.

## Install from this repository

Python 3.12 or newer. Select a PyTorch build suitable for your hardware first if needed.

```bash
git clone https://github.com/Miguell-J/Ricci.git
cd Ricci
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest
```

Maintainers use the committed `uv.lock`: `uv sync --locked --extra dev`.
The lock is a development snapshot; library dependencies have compatible ranges.

## A contraction with geometric meaning

```python
import torch
from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, Variance
from ricci.backends.torch import contract

V = IndexSpace("V", 3, basis="cartesian")
up = Index("i", V, Variance.UP)
down = Index("j", V, Variance.DOWN)

graph = Graph(
    nodes=(NodeSpec("A", (up, down)), NodeSpec("v", (up,))),
    edges=(Edge(Port("A", 1), Port("v", 0)),),
    outputs=(Port("A", 0),),
)
A = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
v = torch.randn(3, dtype=torch.float64, requires_grad=True)
result = contract(graph, {"A": A, "v": v})
torch.testing.assert_close(result.data, A @ v)
result.data.square().sum().backward()
```

The labels `i` and `j` are presentation only. Connectivity uses ports. Equal-size
spaces are not identified automatically; variance is stored on each leg. Metric
application and complex conjugation are explicit operations.

```bash
python examples/physics.py
python examples/quantum.py
python examples/curvature.py
python examples/attention.py
make check
make benchmark
```

## Boundaries

A contraction network represents multilinear tensor operations. Neural models
also contain nonlinearities, normalization, masks, state and optimization; these
compose with Ricci using PyTorch. A string diagram here denotes a contraction,
not the autograd dependency graph or a general-purpose categorical proof system.

The initial executor supports up to **52 distinct wires**, fixed positive sizes,
dense floating/complex tensors, exact batch matching and one common dtype/device.
No implicit broadcasting, coordinate changes, parallel transport, metric insertion
or conjugation. CPU is covered in CI; CUDA tests run only where hardware exists.
The direct path delegates to `torch.einsum` and does not promise an optimal path.
An intermediate-size budget is **not** a GPU-memory quota.

QR/SVD truncation, canonical MPS algorithms, Tensor Train layers, Tucker fitting,
symmetry sectors, fermionic signs and metric-derived curvature are roadmap items.
Christoffel symbols will need their own transformation semantics. Supplied tensor
components may already be contracted in any dimension supported by available memory.

## MCP-One / Stack

Ricci is a standalone library. A future **independent `ricci-mcp` server** will
consume it, expose native MCP tools and use the existing Scientific MCP Contract v1.
No gateway code or MCP dependency is imported by the core. This setup includes
an integration design and disabled manifest example; it does not activate a server.

[Architecture](docs/architecture.md) · [Mathematical contract](docs/mathematics.md) ·
[API](docs/api.md) · [MCP integration](docs/mcp-integration.md) ·
[Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md) ·
[Validation](docs/validation.md)

Licensing has not yet been selected by the repository owner. No distribution
license is granted by this setup; choose one before publishing a package.
