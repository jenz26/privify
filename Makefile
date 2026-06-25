.PHONY: install test lint format demo tad clean

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

demo:
	@echo "Run notebooks/demo_colab.ipynb in Google Colab"

tad:
	quarto render docs/technical_analysis.qmd --to typst

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
