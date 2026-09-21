from dataclasses import replace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.autograd import gradcheck, gradgradcheck

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, RicciError, SpaceKind, Variance
from ricci.backends.torch import Node, contract, execute
from ricci.nn import attention_scores
from ricci.physics import metric_pairing
from ricci.planning import plan
from ricci.quantum import inner


def dot_graph(n):
    index = Index("i", IndexSpace("components", n, SpaceKind.COMPONENT))
    return Graph(
        (NodeSpec("a", (index,)), NodeSpec("b", (index,))),
        edges=(Edge(Port("a", 0), Port("b", 0)),),
    )


@pytest.mark.parametrize("strategy", ["direct", "greedy", "auto"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_first_second_and_complex_gradients(strategy, dtype):
    graph = dot_graph(3)
    a = torch.randn(3, dtype=dtype, requires_grad=True)
    b = torch.randn(3, dtype=dtype, requires_grad=True)

    def operation(x, y):
        return contract(graph, {"a": x, "b": y}, strategy=strategy).data

    torch.testing.assert_close(operation(a, b), (a * b).sum())
    assert gradcheck(operation, (a, b))
    assert gradgradcheck(operation, (a, b))


def test_reusing_plan_with_fresh_autograd_graphs():
    graph = dot_graph(3)
    compiled = plan(graph, "greedy")
    assert plan(graph, "greedy") is compiled
    assert compiled.estimated_flops > 0
    for value in (1.0, 2.0, 3.0):
        a = torch.full((3,), value, dtype=torch.float64, requires_grad=True)
        b = torch.arange(3, dtype=torch.float64)
        execute(compiled, {"a": a, "b": b}).data.backward()
        torch.testing.assert_close(a.grad, b)


@pytest.mark.parametrize(
    "bindings,code",
    [
        ({"a": torch.ones(3)}, "BINDING_MISMATCH"),
        ({"a": torch.ones(3), "b": torch.ones(4)}, "SHAPE_MISMATCH"),
        ({"a": torch.ones(3), "b": torch.ones(3, dtype=torch.float64)}, "BACKEND_MISMATCH"),
        (
            {"a": torch.ones(3, dtype=torch.int64), "b": torch.ones(3, dtype=torch.int64)},
            "UNSUPPORTED_DTYPE",
        ),
        (
            {"a": torch.ones(3, device="meta"), "b": torch.ones(3, device="meta")},
            "UNSUPPORTED_OPERATION",
        ),
        ({"a": torch.ones(3).to_sparse(), "b": torch.ones(3)}, "UNSUPPORTED_OPERATION"),
    ],
)
def test_execution_rejections(bindings, code):
    with pytest.raises(RicciError) as exc:
        contract(dot_graph(3), bindings)
    assert exc.value.descriptor()["code"] == code


def test_noncontiguous_operands():
    a = torch.arange(12, dtype=torch.float64).reshape(3, 4).T[:, 1]
    b = torch.arange(4, dtype=torch.float64)
    torch.testing.assert_close(contract(dot_graph(4), {"a": a, "b": b}).data, (a * b).sum())


@given(seed=st.integers(0, 1000), n=st.integers(2, 5))
@settings(max_examples=15, deadline=None)
def test_basis_invariance_of_metric_pairing(seed, n):
    generator = torch.Generator().manual_seed(seed)
    transform = torch.randn(n, n, dtype=torch.float64, generator=generator) + n * torch.eye(n)
    inverse = torch.linalg.inv(transform)
    g = torch.diag(torch.tensor([-1.0] + [1.0] * (n - 1), dtype=torch.float64))
    u = torch.randn(n, dtype=torch.float64, generator=generator)
    v = torch.randn(n, dtype=torch.float64, generator=generator)

    def pairing(matrix, x, y, basis):
        space = IndexSpace("V", n, basis=basis, fiber="p")
        lo, hi = Index("i", space, Variance.DOWN), Index("i", space, Variance.UP)
        return metric_pairing(
            Node("g", matrix, (lo, lo)), Node("u", x, (hi,)), Node("v", y, (hi,))
        ).data

    # u'=S u, g'=S^{-T} g S^{-1}; physical scalar must not change.
    original = pairing(g, u, v, "original")
    changed = pairing(inverse.T @ g @ inverse, transform @ u, transform @ v, "changed")
    torch.testing.assert_close(original, u @ g @ v)
    torch.testing.assert_close(changed, original, atol=1e-8, rtol=1e-8)


def test_metric_gradient_and_invalid_arguments():
    space = IndexSpace("Minkowski", 4)
    lo, hi = Index("i", space, Variance.DOWN), Index("i", space, Variance.UP)
    matrix = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    g = Node("g", matrix, (lo, lo))
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    u = Node("u", x, (hi,))
    metric_pairing(g, u, u).data.backward()
    torch.testing.assert_close(x.grad, 2 * matrix @ x)
    for metric, left in ((u, u), (Node("g", matrix, (hi, hi)), u), (g, Node("u", x, (lo,)))):
        with pytest.raises(RicciError):
            metric_pairing(metric, left, u)


def test_quantum_sesquilinearity_and_conjugation():
    i = Index("i", IndexSpace("H", 2, SpaceKind.COMPONENT))
    x = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128, requires_grad=True)
    y = torch.tensor([2 - 1j, 1 + 4j], dtype=torch.complex128)
    a, b = Node("psi", x, (i,)), Node("phi", y, (i,))
    torch.testing.assert_close(inner(a, b).data, torch.vdot(x, y))
    alpha = 2 + 3j
    torch.testing.assert_close(
        inner(Node("a", alpha * x, (i,)), b).data, alpha.conjugate() * torch.vdot(x, y)
    )
    inner(a, a).data.real.backward()
    torch.testing.assert_close(x.grad, 2 * x)
    conjugate = a.conjugate(name="conjugate")
    assert conjugate.indices == a.indices
    torch.testing.assert_close(conjugate.data, x.conj())
    with pytest.raises(RicciError):
        inner(Node("scalar", torch.tensor(1.0), ()), a)
    geometric = Index("i", IndexSpace("G", 2), Variance.UP)
    with pytest.raises(RicciError):
        inner(Node("v", x, (geometric,)), Node("w", y, (geometric,)))


def attention_nodes():
    b = Index("b", IndexSpace("batch", 2, SpaceKind.BATCH))
    h = Index("h", IndexSpace("head", 3, SpaceKind.BATCH))
    i = Index("i", IndexSpace("query_token", 4, SpaceKind.COMPONENT))
    j = Index("j", IndexSpace("key_token", 5, SpaceKind.COMPONENT))
    d = Index("d", IndexSpace("feature", 6, SpaceKind.COMPONENT))
    q = Node("q", torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True), (b, h, i, d))
    k = Node("k", torch.randn(2, 3, 5, 6, dtype=torch.float64, requires_grad=True), (b, h, j, d))
    return q, k


@pytest.mark.parametrize("scaled", [False, True])
def test_attention_values_and_gradients(scaled):
    q, k = attention_nodes()
    scores = attention_scores(q, k, scaled=scaled)
    reference = q.data @ k.data.transpose(-2, -1)
    if scaled:
        reference = reference / 6**0.5
    torch.testing.assert_close(scores.data, reference)
    actual_grads = torch.autograd.grad(scores.data.square().sum(), (q.data, k.data))
    expected_grads = torch.autograd.grad(reference.square().sum(), (q.data, k.data))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
    assert scores.data.shape == (2, 3, 4, 5)


def test_attention_rejects_undefined_semantics():
    q, k = attention_nodes()
    with pytest.raises(RicciError):
        attention_scores(Node("x", torch.ones(3), (q.indices[1],)), k)
    with pytest.raises(RicciError):
        attention_scores(Node("q", q.data.to(torch.complex128), q.indices), k)
    wrong = replace(q.indices[2], space=IndexSpace("wrong", 4), variance=Variance.UP)
    with pytest.raises(RicciError):
        attention_scores(Node("q", q.data, (*q.indices[:2], wrong, q.indices[3])), k)


def test_planning_policy_and_budgets():
    with pytest.raises(RicciError):
        plan(dot_graph(3), "optimal")
    for limit in (0, -1, True):
        with pytest.raises(RicciError):
            plan(dot_graph(3), "greedy", limit)
    with pytest.raises(RicciError):
        plan(dot_graph(3), "direct", 4)
    i = dot_graph(3).nodes[0].indices[0]
    graph = Graph((NodeSpec("a", (i,)),), outputs=(Port("a", 0),))
    with pytest.raises(RicciError, match="Output"):
        plan(graph, "greedy", 2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware not available")
def test_cuda_values_and_gradients():
    a = torch.randn(3, dtype=torch.float64, device="cuda", requires_grad=True)
    b = torch.randn_like(a)
    output = contract(dot_graph(3), {"a": a, "b": b}, strategy="greedy")
    torch.testing.assert_close(output.data, (a * b).sum())
    output.data.backward()
    torch.testing.assert_close(a.grad, b)
