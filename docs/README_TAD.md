# Technical Analysis Document — Build Instructions

The TAD source lives in `technical_analysis.qmd` (Quarto Markdown).
The compiled PDF is `technical_analysis.pdf`, also committed for
reviewers without a Quarto installation.

## Prerequisites

- Quarto CLI >= 1.5: https://quarto.org
- TinyTeX (via Quarto): `quarto install tinytex`

## Build

From the project root:

```
make tad
```

Or directly:

```
quarto render docs/technical_analysis.qmd --to pdf
```

Output: `docs/technical_analysis.pdf`

## Running the notebooks

The analysis notebooks under `notebooks/` run on the runtime stack in
`requirements.txt` (no pandas required). In particular,
`notebooks/ood_evaluation.ipynb` produces the recall table and comparison
figures used in sections 3 (*Experimental Results*) and 4 (*Failure
Analysis*).

To execute them in **VS Code**, install the development tooling so the
`.venv` kernel becomes selectable:

```
pip install ipykernel        # or: pip install -e ".[dev]"
```

Then open the notebook and pick the `.venv` interpreter as the kernel
(listed as `Python (.venv)` / `.venv (Python 3.x)`). The notebooks also run
on Google Colab via their self-contained setup cell.
