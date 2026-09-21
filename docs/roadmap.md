# Roadmap and acceptance criteria

Versions are milestones, not promises of a calendar date. Complete one runnable
capability at a time. Keep physical-model choices explicit and owner-led.

| Milestone | Scope | Acceptance evidence |
| --- | --- | --- |
| 0.1 foundation (implemented) | Explicit typed contractions, batch preservation, PyTorch/autograd, opt_einsum, metadata exchange | Formula/gradient tests, invariant scalar checks, four examples, build and CI |
| 0.2 network algebra | Group/split legs, QR/SVD with named partitions, truncation budgets and new bond spaces | Reconstruction residuals, rank/error curves, conditioning and degeneracy behavior |
| 0.3 quantum networks | MPS/MPO containers, canonicalization, expectation values, controlled sweeps | Dense small-system comparisons; norm and observable convergence; complex gradients |
| 0.4 neural networks | Tensor Train linear layers, Tucker parameterizations, higher-order attention experiments | Registered parameters/state_dict roundtrips; training baselines; parameter/FLOP/memory comparisons |
| 0.5 geometric mechanics | Explicit raising/lowering, frame transforms, structured constitutive contractions | Basis covariance; stress/energy references; no hidden Euclidean assumptions |
| 0.6 differential geometry | Tensor fields, coordinate dependence, connections and curvature conventions | Analytic flat/curved references; chart transformations; symmetry and Bianchi checks where applicable |
| MCP adapter milestone | Independent native service using ScientificResult v1 and artifacts | Real-client discovery, schema/domain-error tests and gateway integration with pinned stack |
| 1.0 | Stable documented API/interchange, supported platform matrix and migration policy | Release checklist, reproducible benchmarks, adapter compatibility, selected license |

The MCP adapter can be implemented after the core stabilizes; it does not need to
wait for all domain modules. Conversely, the core must stay usable without MCP.

## Specific research directions

- Quantum: Bell/GHZ networks, spin-chain observables and MPS ground-state methods.
- Continuum mechanics: constitutive tensors, anisotropy and differentiable strain energy.
- Relativity: supplied Riemann/Ricci contractions now; metric-derived fields later.
- AI: factorized weights and custom multilinear attention, including triadic scores.
  Explicitly measure cubic token-cost and intermediate growth before scaling a
  triadic construction. Softmax/nonlinearities stay explicit numerical operations.

For SVD-based training, rank truncation is a modeling decision and gradients may be
ill-conditioned at repeated singular values. For fermionic networks, explicit
graded swaps/parity are necessary. Do not imply those features are already covered
by dense real/complex component contractions.

## Performance development

First measure overhead with small 3D/4D tensors and throughput with larger bonds.
Only then consider fusion, compiled execution, slicing, symmetry blocks, distributed
contractions or extra backends. A planner optimizes a cost model; it does not guarantee
fastest execution on a particular device. Never specialize the core to dimensions 3/4.
