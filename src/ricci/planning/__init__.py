"""Plans cache metadata and paths only; never tensors or autograd graphs."""

from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import Literal

from ricci.core import ContractionIR, Graph, RicciError, lower

Strategy = Literal["direct", "greedy", "auto"]


@dataclass(frozen=True, slots=True)
class Plan:
    graph: Graph
    ir: ContractionIR
    strategy: Strategy
    path: tuple[tuple[int, ...], ...] | None
    estimated_flops: int | None
    largest_intermediate_elements: int | None


@lru_cache(maxsize=128, typed=True)
def plan(graph: Graph, strategy: Strategy = "direct", max_intermediate: int | None = None) -> Plan:
    """Build a reusable plan for fixed shapes.

    max_intermediate limits the largest planned intermediate (elements), not
    peak process/GPU memory. Autograd, inputs, workspace and allocator overhead
    need additional memory. The direct executor has no such estimate.
    """
    ir = lower(graph)
    if strategy not in ("direct", "greedy", "auto"):
        raise RicciError("INVALID_ARGUMENT", f"Unknown planning strategy: {strategy}")
    if max_intermediate is not None and (type(max_intermediate) is not int or max_intermediate < 1):
        raise RicciError("INVALID_ARGUMENT", "max_intermediate must be a positive integer")
    if strategy == "direct":
        if max_intermediate is not None:
            raise RicciError("UNSUPPORTED_OPERATION", "Use greedy/auto for intermediate budgeting")
        return Plan(graph, ir, strategy, None, None, None)
    try:
        import opt_einsum as oe
    except ImportError as exc:
        raise RicciError("MISSING_DEPENDENCY", "Install ricci-tensor[planning]") from exc
    if max_intermediate is not None and prod(ir.output_shape) > max_intermediate:
        raise RicciError("RESOURCE_LIMIT", "Output alone exceeds the intermediate budget")
    path, info = oe.contract_path(
        ir.equation, *ir.shapes, shapes=True, optimize=strategy, memory_limit=max_intermediate
    )
    largest = int(info.largest_intermediate)
    if max_intermediate is not None and largest > max_intermediate:
        raise RicciError("RESOURCE_LIMIT", "No acceptable path was found within the budget")
    return Plan(
        graph, ir, strategy, tuple(tuple(step) for step in path), int(info.opt_cost), largest
    )


__all__ = ["Plan", "Strategy", "plan"]
