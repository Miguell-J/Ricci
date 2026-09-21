# Native MCP integration design

This document was aligned against `Miguell-J/mcp-stack` commit
`d0dff3a5eaec93bc4256daf6e93d0709d1a9e351` on 2026-09-21. Read upstream again
before implementing/deploying an adapter; the library does not pin gateway behavior.

Upstream inspected sources:

- [Scientific result v1](https://github.com/Miguell-J/mcp-stack/blob/d0dff3a5eaec93bc4256daf6e93d0709d1a9e351/docs/contracts/scientific-result-v1.md)
- [Errors v1](https://github.com/Miguell-J/mcp-stack/blob/d0dff3a5eaec93bc4256daf6e93d0709d1a9e351/docs/contracts/errors-v1.md)
- [Adding a server](https://github.com/Miguell-J/mcp-stack/blob/d0dff3a5eaec93bc4256daf6e93d0709d1a9e351/docs/adding-a-server.md)
- [Tool naming](https://github.com/Miguell-J/mcp-stack/blob/d0dff3a5eaec93bc4256daf6e93d0709d1a9e351/docs/contracts/tool-naming-v1.md)

## Ownership

Ricci owns tensor algebra. A future separately installable `ricci-mcp` service
installs Ricci and the existing `scientific_mcp_contracts` package as dependencies.
MCP One discovers/routes native tools, preserves results and handles failure policy.
mcp-stack owns manifests, container orchestration and integration fixtures.

Do not embed the library in the gateway, copy scientific contract models, recreate
legacy REST `/call` endpoints or introduce a custom success/error envelope. No SDK
version is pinned here because no MCP server is shipped in this release. The
inspected stack documents MCP 2026-07-28 and official Python SDK 2.2.0; compatibility
must be exercised with its pinned dependency when the adapter is implemented.

## Proposed public surface (not yet deployed)

| Local name | Gateway name | Concrete future output data |
| --- | --- | --- |
| validate | ricci.validate | Validated shape/topology summary and checks |
| plan | ricci.plan | Equation, strategy, estimates, graph/version identifier |
| contract | ricci.contract | ArtifactReference plus output shape/dtype/basis metadata |

Use namespace `ricci` exactly once. Discover schemas/annotations through native MCP;
do not duplicate tool definitions in the stack manifest. Include truthful read-only
annotations: planning is read-only; artifact-producing operations have side effects.

Successful calls use `CallToolResult.structuredContent = ScientificResult[T]` with
concrete T. Include concise human content, scientific context and actual provenance
(Ricci/backend versions, strategy, precision, conventions). Gateway request ids,
attempt counts, trace/span ids and routing data belong to gateway `_meta`, not
scientific provenance. A graph document is tool input, not a replacement envelope.

## Errors

Map syntax/shape/topology errors to upstream INVALID_ARGUMENT; incompatible
geometric contracts to INVALID_DOMAIN; absent backend features to
UNSUPPORTED_OPERATION. Preserve the specific Ricci code as structured details.
Use upstream ScientificError descriptors under
`_meta["io.github.miguell-j.scientific/error"]`, category=domain, retryable=false
where appropriate. Domain failures set `isError=true` and do not emit a fake T.
Do not change the upstream enum to insert library-specific error codes.

OOM, unavailable artifact stores and timeouts require an explicit operational
policy and may not map to the same category. Infrastructure errors must not be
fabricated by pretending they are scientific validity checks.

## Data and autograd lifetime

Pass small bounded graph metadata in requests. Resolve tensor operands from an
allowlisted artifact store/session controlled by the adapter; do not accept arbitrary
pickle, Python evaluation or unrestricted user URLs. Large results use the upstream
ArtifactReference and an implemented retrieval policy. The current gateway is
tools-only and does not transparently federate resource downloads.

Serialization preserves values/metadata, not a live autograd graph. A trainable
experiment must execute forward, chosen loss and backward in one job, or in an
explicitly managed session with bounded lifetime. Do not promise gradients across
independent serialized tool calls. Session/task APIs are deferred until resource,
cancellation, cleanup and idempotency semantics have been designed and tested.

Enforce input rank/size, estimated cost and wall-time budgets before numerical
execution. `max_intermediate` counts planned elements; it does not enforce peak
memory or preempt GPU work. Hard cancellation/memory isolation belongs in a worker
process or execution service. Never claim determinism solely from a supplied seed.

## Activation gate

`integration/mcp-stack/ricci.example.yaml` is a disabled example only. After a real
server exists, build a versioned image, test native discovery/output schemas/domain
errors and successful contraction through the pinned MCP One stack. Then enable a
single manifest, run stack validate/render and add a real-client integration test.
No stack repository or running service was modified by this setup.
