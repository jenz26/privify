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
