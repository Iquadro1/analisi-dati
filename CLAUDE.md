# CLAUDE.md

## Project
University "Analisi dei Dati" coursework repo.
- `FinalTask.pdf`- assignment for the project, read it carefully
- `Foglio1.ipynb`, `Foglio2.ipynb`, ... — worksheet exercises, cells labeled by exercise number (`## 1.12`, `## 2.3`, ...)
- `progetto/` — final project: gene selection & classification on the Golub et al. (1999) leukemia dataset (`golub.csv`, 7129 genes × 72 patients, ALL vs AML)

## Environment
- Package manager: `uv` (`pyproject.toml` + `uv.lock`), Python 3.14
- Sync deps: `uv sync`
- Run/execute notebooks: `uv run jupyter lab` / `uv run jupyter nbconvert --execute <notebook>`

## progetto/ pipeline (see `project.ipynb`, the canonical version)
1. Data prep & EDA
2. Statistical filtering: t-test + Mann-Whitney U, FDR correction (statsmodels)
3. Classification & variable selection: SVM / SVM-RFE, Logistic Regression (Lasso/ElasticNet), mRMR
4. Stability selection: bootstrap resampling, inclusion-probability threshold (≥80%)
5. Synthesis: concordance between ML-selected and statistically significant genes

Key libs beyond sklearn: `statsmodels` (FDR correction), `mrmr-selection` (+ `fakemp` for its parallelism), `matplotlib-venn` (gene-set overlap diagrams).

`big.ipynb` and `svm-rfe.ipynb` are exploratory variants of the `project.ipynb` pipeline, not the source of truth.

## Development philosophy
- Read the assignment carefully, understand the requirements and constraints.
- Do not overengineer: use the simplest solution that works, avoid unnecessary complexity, this is an assignment for a university course, not a biological research project.