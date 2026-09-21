# ADR 0003: Immutable metadata, fresh data bindings

Status: accepted for the alpha foundation.

Graph construction, lowering, planning and execution are separate stages. Core
metadata uses only the standard library. PyTorch operands are bound at execution
time and are never copied/detached by the wrapper. Plans cache only shapes and paths.

Consequences: metadata-only validation and future remote planning are possible;
training steps do not reuse stale autograd graphs. Plan budgets describe the largest
intermediate, not total memory. The first lowering explicitly rejects more than
52 wire labels; a later stepwise executor can lift this without changing identity.
