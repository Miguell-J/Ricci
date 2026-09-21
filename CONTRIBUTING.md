# Contributing

Use a focused branch and explain the scientific or engineering problem being
solved. Include a runnable example and an independent reference for new algebra.
Conventions (basis, metric signature, index order, complex conjugation) are part
of the API, not incidental implementation details.

```bash
uv sync --locked --extra dev
uv run --locked --extra dev make check
uv run --locked --extra dev make examples
uv run --locked --extra dev make build
```

After activating `.venv`, the same Make targets work directly. Individual commands
in Makefile also work on platforms without Make. Public metadata stays immutable;
the library does not take ownership of caller tensor storage.

Tests cover numerical values, rejected inputs, output metadata and gradients.
Prefer property tests for basis covariance and index relabeling over snapshots
of implementation internals. Keep benchmark fixtures seeded, compare equivalent
operations and report environment/dtype/device. Do not put noisy timing thresholds
in pull-request CI.

The package is alpha: public changes may break compatibility, but must still be
documented in CHANGELOG.md. Graph exchange has its own version (`ricci.graph.v1`);
breaking interchange changes require a new version and migration tests.

Package publication and license selection remain owner decisions. CI builds and
checks distributions without uploading them to PyPI or publishing documentation.
