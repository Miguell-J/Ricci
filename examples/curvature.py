"""Contract supplied constant-sectional-curvature components; no field derivatives."""

import torch

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, Variance
from ricci.backends.torch import contract


def main() -> None:
    # Convention R^a_bcd = K(delta^a_c g_bd - delta^a_d g_bc).
    n, k = 3, 2.0
    space = IndexSpace("tangent", n, fiber="p", basis="orthonormal")
    up, down = Index("a", space, Variance.UP), Index("b", space, Variance.DOWN)
    g = torch.eye(n, dtype=torch.float64)
    riemann = k * (torch.einsum("ac,bd->abcd", g, g) - torch.einsum("ad,bc->abcd", g, g))
    graph = Graph(
        (NodeSpec("R", (up, down, down, down)),),
        edges=(Edge(Port("R", 0), Port("R", 2)),),
        outputs=(Port("R", 1), Port("R", 3)),
    )
    ricci = contract(graph, {"R": riemann})
    torch.testing.assert_close(ricci.data, (n - 1) * k * g)
    scalar_graph = Graph(
        (ricci.spec, NodeSpec("inverse_metric", (up, up))),
        edges=tuple(Edge(Port("result", i), Port("inverse_metric", i)) for i in range(2)),
    )
    scalar = contract(scalar_graph, {"result": ricci.data, "inverse_metric": g})
    torch.testing.assert_close(scalar.data, torch.tensor(n * (n - 1) * k, dtype=torch.float64))
    print("Scalar curvature:", scalar.data.item())


if __name__ == "__main__":
    main()
