# Mathematical contract

## Spaces, legs and connectivity

A geometric UP leg belongs to V; DOWN belongs to V*. Binary contraction uses the
canonical pairing V* ⊗ V → scalar. The space id, dimension, basis and fiber must
all agree. Distinct identifiers of the same dimension are not interchangeable.
Two declarations with the same id and inconsistent metadata are rejected.

Names on indices are display labels; ports `(node_name, axis)` identify occurrences.
Contracting A's second leg with v's first leg expresses A^i_j v^j regardless of
the display labels. A trace joins two distinct legs on the same node. Several
edges may join the same pair of nodes. A port participates in at most one edge
or batch alignment. Disconnected components express outer products.

The graph is an ordered tensor expression, not a visual embedding. Moving boxes
on a screen does not change it. Ordinary vector spaces use ordinary symmetric
swaps; fermionic/graded signs and braided categories are not represented.

## Geometric versus component versus batch spaces

| Kind | Allowed variance | Binary contraction | Preserved alignment |
| --- | --- | --- | --- |
| geometric | UP or DOWN | Opposite variance, same space | Not implicit |
| component | NEUTRAL | Explicit component summation | Not implicit |
| batch | NEUTRAL | Rejected | Explicit BatchGroup |

Component contraction uses declared component bases. It does not promise covariance
under arbitrary basis transformations. This mode is useful for neural features,
tensor-network bonds and quantum coefficients in a fixed basis.

Every free port is listed exactly once in outputs. Each batch group contributes
exactly one representative. The tuple order determines numerical axis order and
metadata order. In attention `bhid,bhjd->bhij`, b and h are retained groups and d
is an edge. Repeated names alone neither sum nor align anything. Shapes match
exactly: no implicit NumPy-style broadcasting or silent size-one expansion.

## Metrics and coordinates

g_ij u^i v^j uses an explicit covariant bilinear form and two contravariant vectors.
The operation is bilinear, including with complex-valued components. Changing a
variance flag does not raise/lower a numerical tensor. `Index.dual()` changes only
slot metadata and is not a numerical raising/lowering operation.

`metric_pairing` evaluates the supplied form; it does not certify symmetry,
nondegeneracy or Lorentzian signature. Tests use signature (-,+,+,+). The API makes
no assumption that all three- or four-dimensional spaces have a particular metric.
Under component change u'=S u, the matrix transforms as g'=S^{-T}gS^{-1}.
Tests verify the paired scalar is unchanged, using several dimensions/transforms.

T_p M and T_q M are different fibers. A user may label the fiber in metadata;
the library neither computes transport nor knows coordinates of p and q. Likewise,
arrays sampled at points are not differentiable fields unless an explicit numerical
function retains their coordinate dependence in PyTorch.

The curvature example adopts R^a_bcd = K(delta^a_c g_bd - delta^a_d g_bc),
Ric_bd = R^a_bad and R = g^bd Ric_bd. It contracts supplied components. Computing
these from g(x) requires a future field/connection layer. Christoffel coefficients
have an inhomogeneous coordinate transformation and must not be treated as tensors.

## Complex numbers and differentiation

Numerical execution is multilinear without implicit conjugation. `Node.conjugate()`
conjugates data while preserving slot metadata. `quantum.inner(psi, phi)` explicitly
conjugates psi and sums corresponding component legs in a declared orthonormal
product basis. Nonorthonormal Hilbert bases need an explicit Gram matrix.

Quantum inner products are antilinear in the first argument. Geometric bilinear
forms, complex conjugation, dual-space typing and Hermitian adjoints are different
operations. The library never infers one from another.

Autograd differentiates the executed tensor operations; it does not provide a
connection, covariant derivative or physical field equation. Numerical tests use
float64/complex128, first/second derivative checks and independent gradient formulas.
Backpropagation of a complex-valued result needs an explicitly chosen real loss
or upstream gradient, following PyTorch conventions.
