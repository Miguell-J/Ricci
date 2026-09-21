"""Reproducible forward-only CPU comparisons; no timing gate in CI."""

import argparse
import json
import platform
import statistics
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch

from ricci import Edge, Graph, Index, IndexSpace, NodeSpec, Port, SpaceKind
from ricci.backends.torch import execute
from ricci.planning import plan


def measure(function, repeats=7, iterations=100):
    for _ in range(10):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/latest.json"))
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(1)
    results = []
    for size in (4, 32, 128):
        index = Index("i", IndexSpace("V", size, SpaceKind.COMPONENT))
        graph = Graph(
            tuple(NodeSpec(name, (index, index)) for name in ("A", "B", "C")),
            edges=(Edge(Port("A", 1), Port("B", 0)), Edge(Port("B", 1), Port("C", 0))),
            outputs=(Port("A", 0), Port("C", 1)),
        )
        a, b, c = (torch.randn(size, size, dtype=torch.float64) for _ in range(3))
        bindings = {"A": a, "B": b, "C": c}
        direct, greedy = plan(graph), plan(graph, "greedy")
        reference = (a @ b) @ c
        torch.testing.assert_close(execute(direct, bindings).data, reference)
        torch.testing.assert_close(execute(greedy, bindings).data, reference)
        results.append(
            {
                "size": size,
                "torch_matmul": measure(lambda a=a, b=b, c=c: (a @ b) @ c),
                "torch_einsum": measure(partial(torch.einsum, "ij,jk,kl->il", a, b, c)),
                "ricci_direct_cached_plan": measure(partial(execute, direct, bindings)),
                "ricci_greedy_cached_plan": measure(partial(execute, greedy, bindings)),
                "estimated_flops": greedy.estimated_flops,
                "largest_intermediate_elements": greedy.largest_intermediate_elements,
            }
        )
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "opt_einsum": version("opt-einsum"),
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "seed": 17,
        "mode": "forward without requires_grad; prebuilt plans; no GPU claims",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sizes": [r["size"] for r in results]}))


if __name__ == "__main__":
    main()
