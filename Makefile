PYTHON ?= python

.PHONY: check lint format typecheck test schema schema-check examples benchmark build
check: lint typecheck test schema-check
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
typecheck:
	$(PYTHON) -m mypy
test:
	$(PYTHON) -m pytest --cov=ricci --cov-report=term-missing --cov-report=xml
schema:
	$(PYTHON) scripts/export_schema.py
schema-check:
	$(PYTHON) scripts/export_schema.py --check
examples:
	$(PYTHON) examples/physics.py
	$(PYTHON) examples/quantum.py
	$(PYTHON) examples/curvature.py
	$(PYTHON) examples/attention.py
benchmark:
	$(PYTHON) benchmarks/contraction.py
build:
	$(PYTHON) -m build --no-isolation
