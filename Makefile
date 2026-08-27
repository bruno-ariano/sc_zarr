.PHONY: install test test-unit test-integration lint clean

# Default command when running just 'make'
all:
	install test

# Install dependencies and sync local package in editable mode
install:
	uv sync

# Run all tests
test:
	uv run --group test pytest --cov=src --cov-report=term-missing

# Run unit tests only (fast)
test-unit:
	uv run --group test pytest tests/unit

# Run code formatters and static type checks
lint:
	uv run --group lint ruff check .
	uv run --group lint mypy src/
	uv run --group lint deptry .

# Clean temporary cache folders
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
