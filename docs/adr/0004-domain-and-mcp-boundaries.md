# ADR 0004: Domain composition and an independent MCP adapter

Status: accepted for the alpha foundation.

Physics, quantum and neural helpers compose a common contraction core. They do not
inherit Graph to weaken invariants. Complex conjugation, geometric duality, field
derivatives and nonlinear neural operations retain distinct meanings.

A future ricci-mcp server depends on Ricci and the upstream scientific contracts.
The library imports neither MCP One nor the stack. GraphDocument is versioned
input metadata; it is not a duplicate ScientificResult or a running protocol server.

Consequences: native gateway discovery/routing is reusable without transport coupling.
Deployment, artifacts and session lifetimes belong outside the library. There is no
implied serialization of autograd across tool calls. End-to-end integration remains
a separate milestone with real-client tests before manifest activation.
