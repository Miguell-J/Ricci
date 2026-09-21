"""PyTorch bindings preserve tensor identity, device, dtype and gradient history."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from ricci.core import Graph, Index, NodeSpec, RicciError
from ricci.planning import Plan, Strategy, plan


@dataclass(frozen=True, slots=True, eq=False)
class Node:
    name: str
    data: Tensor
    indices: tuple[Index, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if not isinstance(self.data, Tensor):
            raise RicciError("INVALID_ARGUMENT", "Node data must be a torch.Tensor")
        if tuple(self.data.shape) != self.spec.shape:
            raise RicciError("SHAPE_MISMATCH", f"{self.name}: expected {self.spec.shape}")

    @property
    def spec(self) -> NodeSpec:
        return NodeSpec(self.name, self.indices)

    def conjugate(self, *, name: str | None = None) -> "Node":
        """Complex conjugation only; slot types are unchanged."""
        return Node(name or self.name, self.data.conj(), self.indices)


def execute(compiled: Plan, bindings: Mapping[str, Tensor], *, name: str = "result") -> Node:
    """Execute using live operands; no detach, NumPy conversion or tensor caching."""
    if set(bindings) != set(compiled.ir.input_names):
        raise RicciError("BINDING_MISMATCH", "Bindings must exactly match the graph's node names")
    tensors: list[Tensor] = []
    for node, shape in zip(compiled.ir.input_names, compiled.ir.shapes, strict=True):
        tensor = bindings[node]
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
            raise RicciError("SHAPE_MISMATCH", f"{node}: expected shape {shape}", path=node)
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise RicciError(
                "UNSUPPORTED_OPERATION", "Execution requires materialized dense tensors"
            )
        if not (tensor.is_floating_point() or tensor.is_complex()):
            raise RicciError("UNSUPPORTED_DTYPE", "Use floating-point or complex tensors")
        if tensors and (tensor.dtype != tensors[0].dtype or tensor.device != tensors[0].device):
            raise RicciError("BACKEND_MISMATCH", "All operands must share dtype and device")
        tensors.append(tensor)
    if compiled.path is None:
        result = torch.einsum(compiled.ir.equation, *tensors)
    else:
        import opt_einsum as oe

        result = cast(
            Tensor,
            oe.contract(compiled.ir.equation, *tensors, optimize=compiled.path, backend="torch"),
        )
    return Node(name, result, compiled.graph.output_indices)


def contract(
    graph: Graph,
    bindings: Mapping[str, Tensor],
    *,
    strategy: Strategy = "direct",
    max_intermediate: int | None = None,
    name: str = "result",
) -> Node:
    return execute(plan(graph, strategy, max_intermediate), bindings, name=name)


__all__ = ["Node", "contract", "execute"]
