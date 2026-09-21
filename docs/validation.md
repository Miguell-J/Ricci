# Foundation validation

Validation date: 2026-09-21. Environment: Linux x86_64, CPython 3.12.14,
PyTorch 2.14.0, opt_einsum 3.4.0; CPU execution. Exact development resolution
is recorded in uv.lock. Hardware-specific timing is not a portable guarantee.

## Executed checks

| Check | Result |
| --- | --- |
| Ruff lint and format | Passed |
| Strict mypy on 17 source modules | Passed; vendor-specific opt_einsum untyped-import exception |
| pytest | 56 passed, 1 skipped (CUDA hardware unavailable) |
| Coverage including branches | 99.19%; required gate 85% |
| Generated graph JSON Schema drift | Passed |
| Physics example | g(u,u)=-3, gradient=(-4,2,0,0); strain energy=0.00015 |
| Quantum example | Bell-state norm=1 within rounding; expectation of Z on first qubit=0 |
| Curvature example | Constant-curvature n=3, K=2 gives scalar curvature 12 |
| Attention example | Three optimizer steps; output shape (2,4,8), all parameter gradients present |
| Build | Wheel and source distribution built successfully |
| Isolated wheel installation | Import and direct contraction/backward outside checkout; no planning/interop extras required |
| Benchmarks | Forward CPU baseline for size 4, 32 and 128 matrix chains |

Tests exercise explicit index relabeling, dimension/space/basis/fiber conflicts,
variance, port reuse, traces, outer products, preserved batch hyperedges, output
ordering, scalar nodes, noncontiguous storage, dtype/device policies and graph JSON
roundtrips. Numerical tests include first and second derivatives in float64 and
complex128, antilinearity of quantum inner products, scalar invariance under basis
changes, attention gradients and MPS chains with tied parameters/live views.

The missing coverage is in optional dependency failure handling and the planner's
post-search budget rejection. Coverage does not establish mathematical completeness.
The property test's wall-time deadline is disabled because first-use PyTorch
initialization is not a correctness property; numerical assertions remain enabled.

## Performance evidence

[Raw CPU baseline](benchmarks/cpu-baseline.json) records all samples and environment.
Run `make benchmark` to collect a new local result. Seven batches of 100 forward
evaluations follow ten warmup calls per method; no gradients are recorded in this
benchmark. Plans are built before timing. This measures executor overhead separately
from initial graph construction/path search.

| Matrix size | torch matmul median, µs | torch einsum | Ricci direct | Ricci greedy, preplanned |
| --- | ---: | ---: | ---: | ---: |
| 4 | 2.16 | 71.86 | 85.88 | 47.78 |
| 32 | 9.26 | 82.94 | 90.36 | 55.54 |
| 128 | 226.11 | 461.14 | 517.80 | 448.88 |

The typed wrapper has measurable overhead, especially for tiny operations. Explicit
path reuse reduced cost against direct einsum in these particular cases; handwritten
matmul was faster. These are illustrative measurements on shared hardware, not claims
of speedup across networks. Benchmark timings are intentionally not a CI gate.

## Scope of evidence

CUDA execution was not verified. Windows and Python 3.13 are configured in GitHub
Actions and require their remote runs to establish evidence. The declared PyTorch
compatibility range has not been exhaustively tested at every version.

There is no running Ricci MCP server, no activated stack manifest, no GPU-memory
enforcement, no metric-derived curvature engine and no SVD/TT compression algorithm
in this release. Integration alignment was checked against actual upstream contract
documents at the commit linked in [MCP integration](mcp-integration.md); native
server/gateway execution belongs to the later adapter milestone.
