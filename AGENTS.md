# Working on Ricci

## Goal and scope

Build a reusable, mathematically typed tensor-contraction library for quantum
physics, geometric mechanics, relativity and machine learning. Read README.md,
docs/mathematics.md and docs/architecture.md before changing public contracts.
The owner determines scientific models; implementations must state conventions
and test those models against independent mathematical references.

## Architectural invariants

- `ricci.core` uses the standard library only. Do not import torch, Pydantic, MCP,
  a gateway, network clients or environment-dependent configuration there.
- Metadata is immutable. Node data remains the caller's live torch.Tensor.
- Edges connect port occurrences; labels never infer connectivity.
- Do not silently identify spaces, bases, fibers, duals or complex conjugates.
- Preserve declared output order and batch axes. Never sum a batch implicitly.
- Do not make GR connection coefficients ordinary tensors under chart changes.
- Planning caches contain shapes/paths only, never tensor values or autograd state.
- Do not detach data, roundtrip through NumPy, or reconstruct tensors in execution.
- All trainable weights in future neural modules must be registered parameters.
- MCP belongs to a separately installable adapter/server. Reuse the upstream
  scientific contract; do not duplicate its result envelope in this library.
- Errors have stable codes. Update docs and compatibility tests when changing them.
- No global dtype, seed, device or autograd-mode changes at import time.

## Workflow

Use Python 3.12+, the src layout and the existing pyproject/lock. Keep public API
changes small and runnable. Document an ADR for changes to mathematical semantics,
IR, backend policy or adapter boundaries. Avoid speculative inheritance hierarchies,
empty placeholder classes and performance claims without benchmark evidence.

Run `make check` (format, lint, strict types, tests/coverage, schema drift),
`make examples` and `make build` for relevant changes. Mathematical tests must
include independent formulas, invalid cases and actual gradient values. Use
float64/complex128 for numerical checks and dtype-appropriate tolerances.
Optional CUDA tests should skip cleanly on CPU; do not claim GPU verification
when unavailable. Mark unrun gates explicitly in validation reports.

Update uv.lock intentionally with `uv lock` when dependency declarations change.
Do not hand-edit the generated lock or JSON schema. Regenerate the latter with
`python scripts/export_schema.py`. Keep development pins separate from library
dependency ranges. Review dependency updates and never add secrets, credentials,
trained weights, large datasets, virtual environments or generated build outputs.

## Completion evidence

Report what is implemented, what was tested, and what remains planned. Do not
describe a JSON schema as an operational MCP server or a contraction example as
a GR field solver. Preserve concurrent user edits and use ordinary non-force Git
updates. Do not choose a license or publish a package without owner direction.
