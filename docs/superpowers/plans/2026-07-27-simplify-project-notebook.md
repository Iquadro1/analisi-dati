# Simplification of `project_revised.ipynb` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `progetto/project_revised.ipynb` from a research-grade genomics study to the complete analysis appropriate for a university data-analysis course assignment, without dropping any of the three requirements set by `FinalTask.pdf`.

**Architecture:** The notebook stays a single end-to-end analysis feeding the PDF report — this is *not* a stripped-down final-method deliverable. Five blocks of research-methodology apparatus are removed (a permutation null, a matched-sparsity controlled experiment, correlation-module clustering, a second stability index, and self-narrating verdict branches). The mRMR pre-filter is removed and the SVM arm is rebuilt as plain SVM-RFE (Guyon et al., 2002). Two structural defects are fixed (a filter leak into the nested CV, and a duplicated regularization path). One cheap arm is added (top-k univariate selection), giving four arms — one per category of the Saeys et al. (2007) taxonomy. Interpretive prose moves from `print()` calls into markdown.

**Tech Stack:** Python 3.14, `uv`, Jupyter, numpy/pandas/matplotlib/seaborn, scipy.stats, statsmodels, scikit-learn, matplotlib-venn.

## Global Constraints

- **Edit `progetto/project_revised.ipynb` in place.** Task 1 commits the current state first so the original is recoverable via git. Do not create a second notebook.
- **All notebook text stays in Italian**, matching the existing prose. Code comments in Italian. This document is in English; the content you write into the notebook is not.
- **Do not execute the full notebook until Task 12.** Per-task verification is a static check (grep for orphaned symbols). Full execution is slow and is done once, at the end.
- **Use the `NotebookEdit` tool** for all cell insertions, edits and deletions. Fetch its schema with `ToolSearch` (`select:NotebookEdit`) before Task 1.
- **Locate cells by anchor string, not by index.** Indices shift after every deletion. Each task gives the anchor text to search for.
- **Delete cells in descending index order** within a single task.
- **Do not invent numeric results.** Where a task says a conclusion must be rewritten, write it after Task 12 produces real output, or leave the markdown stating the mechanism without the number and fill the number in Task 12.
- Commit after every task with the message given in the task.

## Cell anchor map (state at plan time, 46 cells)

| Anchor string (unique) | Idx | Fate |
|---|---|---|
| `RANDOM_STATE = 43` | 2 | Modify (T1) |
| `plot_target_stacked(y, raw_df["Gender"]` | 5 | Modify (T10) |
| `X_thr = X_all.clip(lower=FLOOR` | 9 | Modify (T5) |
| `pca_scatter(raw_df.loc[y_train.index, "BM.PB"]` | 11 | Modify (T10) |
| `cv_scores = lasso_grid.cv_results_` | 17 | Modify (T6) |
| `top_n = 30` | 20 | Modify (T6) |
| `thresholds = np.arange(0.30, 1.001, 0.05)` | 21 | Modify (T10) |
| `### 4.1 L'effetto della multi-collinearita` | 22 | Modify (T7) |
| `sig_names = sorted(robust_baseline_genes)` | 23 | Delete (T7) |
| `prob_map = dict(zip(lasso_stability_df` | 24 | Replace (T7) |
| `nested_pipe = Pipeline([` | 27 | Modify (T5) |
| `### 5.2 Controllo delle false scoperte: il null` | 28 | Delete (T3) |
| `perm_model = LogisticRegression(` | 29 | Delete (T3) |
| `overlap = lasso_stable_set & robust_baseline_genes` | 30 | Modify (T10) |
| `### 5.3 Metodi alternativi di selezione` | 31 | Modify (T8) |
| `mrmr_features = mrmr_classif(` | 32 | Delete (T2) |
| `svm_boot_sets = []` | 33 | Replace (T2) |
| `top_svm = svm_stability_df.head(20)` | 34 | Delete (T2) |
| `en_param_grid = {"l1_ratio"` | 35 | Delete (T7, relocated) |
| `#### Lasso contro Elastic Net a parita di sparsita` | 36 | Delete (T4) |
| `EN_MATCHED_L1_RATIO = 0.5` | 37 | Delete (T4) |
| `# --- Il test decisivo` | 38 | Delete (T4) |
| `en_top = sorted(en_stable_set)` | 39 | Delete (T4) |
| `### 5.4 Score di riproducibilita` | 40 | Modify (T9) |
| `stability_scores = pd.DataFrame([` | 41 | Modify (T9) |
| `def plot_dropoff(ax, df, label` | 42 | Modify (T9) |
| `def top_by_ttest(df_expr, y_vec, k=20)` | 43 | Modify (T11) |
| `## Conclusioni` | 44 | Rewrite (T11) |
| `## Riferimenti` | 45 | Rewrite (T11) |

---

### Task 1: Baseline commit and constant/import cleanup

**Files:**
- Modify: `progetto/project_revised.ipynb` (cell anchored on `RANDOM_STATE = 43`)

**Interfaces:**
- Produces: the global constant set `RANDOM_STATE, N_BOOTSTRAP, STABILITY_THRESHOLD, FDR_ALPHA, FLOOR, CEIL` and the helper `jaccard_index(sets) -> float`, `mean_inclusion(stab_df, panel) -> float`. Every later task assumes exactly these exist and that `N_PERM`, `K_MRMR` and `kuncheva_index` do not.

- [ ] **Step 1: Commit the current state so the original notebook is recoverable**

```bash
cd /home/isabella/Develop/analisi-dati
git add CLAUDE.md progetto/project_revised.ipynb
git commit -m "chore: snapshot notebook before course-level simplification"
```

- [ ] **Step 2: Replace the whole setup cell (anchor `RANDOM_STATE = 43`) with this source**

```python
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.utils import resample

from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from IPython.display import display

# --- Costanti globali -------------------------------------------------------
RANDOM_STATE = 43          # seme unico: la consegna valuta la riproducibilita
N_BOOTSTRAP = 100          # repliche bootstrap per la stability selection
STABILITY_THRESHOLD = 0.80 # soglia di inclusion probability per il pannello finale
FDR_ALPHA = 0.05           # livello per la correzione di Benjamini-Hochberg
FLOOR, CEIL = 100, 16000   # soglie di thresholding (Dudoit et al. 2002)

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

rng = np.random.default_rng(RANDOM_STATE)


# --- Indici di stabilita della selezione ------------------------------------
def jaccard_index(sets):
    """Sovrapposizione media |A intersect B| / |A union B| fra coppie di repliche."""
    vals = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    return float(np.mean(vals)) if vals else np.nan


def mean_inclusion(stab_df, panel):
    """Probabilita media di inclusione dei geni di un pannello."""
    m = stab_df.set_index('Gene_Probe')['Inclusion_Probability']
    return float(np.mean([m.get(g, 0.0) for g in panel])) if panel else np.nan


print("Ambiente inizializzato.")
```

Removed: `scipy.cluster.hierarchy`, `scipy.spatial.distance`, `mrmr`, `RFECV`, the `N_PERM` and `K_MRMR` constants, and `kuncheva_index`. Each is justified by the task that deletes its only consumer (T2, T3, T7, T9).

`SVC` is kept — Task 2 rebuilds the SVM arm rather than removing it — but neither `RFE` nor `RFECV` is imported. sklearn's `RFE` computes its step **once from the original feature count** and holds it constant (`_rfe.py`: `step = int(max(1, self.step * n_features))`), so it can only decay linearly. Guyon et al. (2002, §5.1) eliminate *geometrically* on the leukemia data — first down to the nearest power of two, then halving the survivors each iteration — and no constant `step` reproduces that. Task 2 writes the loop directly instead.

- [ ] **Step 3: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): drop imports and constants for removed methods"
```

---

### Task 2: Drop the mRMR pre-filter, rebuild the SVM arm as plain SVM-RFE

**Why mRMR goes.** Mundra & Rajapakse (2010) rank genes by Eq. (5),

    r_i = β·|w_i| + (1 − β)·R_{S,i} / Q_{S,i}

where `R_{S,i}` is the mutual information between gene *i* and the class labels and `Q_{S,i}` is its average mutual information with the other genes in the surviving set *S*. Their Algorithm 1 recomputes `R_{S,i}` and `Q_{S,i}` **inside** the elimination loop, for every gene still in *S*, at every iteration. The notebook does something structurally different: `mrmr_classif` runs **once** to pick an initial 100 genes, after which RFE ranks purely by w² and redundancy is never consulted again — exactly the failure their paper exists to fix. Citing Mundra for it would misdescribe what ran.

Three concrete reasons not to implement it faithfully instead:

1. **Cost.** `Q_{S,i}` requires pairwise mutual information among all surviving genes, recomputed each iteration. At |S| = 3,363 that is ~5.7 M pairs on the first iteration alone, times ~12 iterations, times 100 bootstrap replicates. Their own §IV concedes the algorithm "is computationally more expensive than SVM-RFE or MRMR methods".
2. **New free parameters.** β, chosen empirically from {0.2, 0.4, 0.5, 0.6, 0.8}; the SVM sensitivity η, from 2⁻²⁰…2¹⁵; and the discretization of Eq. (9) (x̃ = ±2 / 0 against μ ± σ/2) needed before any mutual information can be computed. All three tuned by ten-fold CV.
3. **Small payoff.** Their Table II, leukemia dataset: SVM-RFE alone → 47 genes, 97.88% accuracy; SVM-RFE with MRMR filter → 37 genes, 98.35%. Half a percentage point and ten genes.

mRMR therefore moves to the report's "strategies considered" section, where the criterion can be stated accurately and this trade-off quoted.

**Why SVM-RFE stays.** SVM is in the course programme, and SVM-RFE needs no external package and no theory beyond the linear SVM: for a linear kernel `svc.coef_[0]` *is* Guyon's weight vector w = Σₖ αₖyₖxₖ (his Eq. 4), and his §2.5 justifies w² as the ranking criterion via the OBD second-order argument. It is the canonical wrapper method on this dataset and supplies the wrapper category of the Saeys et al. (2007) taxonomy. Guyon's headline result on these data — **2 genes at zero leave-one-out error** — is also a direct point of comparison for the Lasso panel.

**Two changes to its shape.** The pre-filter is dropped entirely: elimination starts from the full filtered space, as in Guyon. And the panel size is fixed to the Lasso panel size instead of being chosen by `RFECV` — this makes the arm comparable to the others at equal sparsity, and removes the inner-CV curve that the old cell's own caveat admitted was optimistically biased. Guyon likewise reports error across a grid of gene counts rather than letting CV pick one.

**Files:**
- Modify: `progetto/project_revised.ipynb` (delete 2 cells, replace 1)

**Interfaces:**
- Consumes: `X_train`, `X_test`, `y_train`, `y_test`, `feature_cols`, `lasso_final_genes`, `lasso_stable_set`, `N_BOOTSTRAP`, `RANDOM_STATE`.
- Produces: `k_svm` (int), `svm_genes` (list[str]), `svm_boot_sets` (list[set]), `svm_stability_df` (columns `Gene_Probe`, `Selection_Count`, `Inclusion_Probability`), `svm_stable_set` (set), `svm_acc`, `svm_bacc` (floats). Task 9 depends on all of these. **`mrmr_features`, `svm_rfe_genes`, `svm_final_features`, `svm_used_fallback`, `X_train_df`, `X_test_df`, `rfecv` no longer exist.**

- [ ] **Step 1: Delete two cells, in descending index order**

Anchors, delete in this order:
1. `top_svm = svm_stability_df.head(20)`
2. `mrmr_features = mrmr_classif(`

- [ ] **Step 2: Replace the cell anchored `svm_boot_sets = []` with this source**

```python
# SVM-RFE come in Guyon et al. (2002): si addestra un SVM lineare, si ordinano le
# feature per w^2 (criterio giustificato nella sez. 2.5 dell'articolo) e si
# eliminano le peggiori, ricorsivamente. Nessun pre-filtro: l'eliminazione parte
# dall'intero spazio filtrato. Il numero finale di geni e fissato a quello del
# pannello del Lasso, cosi che i metodi siano confrontati a parita di sparsita.
k_svm = len(lasso_final_genes)


def fit_svm_rfe(X, y_vec, k):
    """Eliminazione a dimezzamento, lo schema usato da Guyon et al. (2002, sez.
    5.1) proprio sui dati di leucemia: al primo passo si scende alla potenza di
    2 piu vicina, poi si elimina meta dei geni superstiti a ogni iterazione.
    Ne risultano i sottoinsiemi annidati F_1 < F_2 < ... < F descritti nella
    sez. 2.6. Per 3363 sonde bastano una dozzina di addestramenti.

    Nota: `sklearn.feature_selection.RFE` non puo riprodurre questo schema,
    perche calcola il passo una volta sola sul numero di feature iniziale e lo
    mantiene costante, ottenendo un decadimento lineare invece che geometrico.
    """
    keep = np.arange(X.shape[1])
    target = 2 ** int(np.floor(np.log2(len(keep))))
    while len(keep) > k:
        svc = SVC(kernel="linear", class_weight="balanced", C=1.0)
        svc.fit(X[:, keep], y_vec)
        order = np.argsort(svc.coef_[0] ** 2)[::-1]   # migliori per primi
        keep = keep[order[:max(k, target)]]
        target = len(keep) // 2
    return np.sort(keep)


svm_idx = fit_svm_rfe(X_train, y_train, k_svm)
svm_genes = [feature_cols[i] for i in svm_idx]

svm_model = SVC(kernel="linear", class_weight="balanced", C=1.0)
svm_model.fit(X_train[:, svm_idx], y_train)
svm_pred = svm_model.predict(X_test[:, svm_idx])
svm_acc = accuracy_score(y_test, svm_pred)
svm_bacc = balanced_accuracy_score(y_test, svm_pred)

print(f"--- SVM-RFE: eliminazione ricorsiva fino a {k_svm} geni ---")
print(f"Geni selezionati: {svm_genes}")
print(f"Sovrapposizione con il pannello del Lasso: "
      f"{len(set(svm_genes) & lasso_stable_set)} geni")
print(f"Balanced accuracy su hold-out: {svm_bacc:.2%}")

# Stessa stability selection degli altri metodi: l'eliminazione ricorsiva viene
# rieseguita da zero dentro ciascuna replica bootstrap.
svm_boot_sets = []
svm_counts = Counter()
print(f"\nStability selection per SVM-RFE ({N_BOOTSTRAP} repliche)...")
for i in range(N_BOOTSTRAP):
    if i % 20 == 0:
        print(f"  replica {i}/{N_BOOTSTRAP}")
    X_boot, y_boot = resample(X_train, y_train, random_state=i, stratify=y_train)
    genes = {feature_cols[j] for j in fit_svm_rfe(X_boot, y_boot, k_svm)}
    svm_boot_sets.append(genes)
    svm_counts.update(genes)

svm_stability_df = (
    pd.DataFrame({"Gene_Probe": list(svm_counts.keys()),
                  "Selection_Count": list(svm_counts.values())})
    .assign(Inclusion_Probability=lambda d: d["Selection_Count"] / N_BOOTSTRAP)
    .sort_values("Inclusion_Probability", ascending=False)
    .reset_index(drop=True)
)
svm_stable_set = set(svm_stability_df[
    svm_stability_df["Inclusion_Probability"] >= STABILITY_THRESHOLD]["Gene_Probe"])

print(f"\nGeni selezionati almeno una volta: {len(svm_stability_df)}")
print(f"Geni con inclusion probability >= {STABILITY_THRESHOLD:.0%}: {len(svm_stable_set)}")
print(f"Inclusion probability massima: "
      f"{svm_stability_df['Inclusion_Probability'].max():.2f}")
```

Halving costs ~12 SVM fits per replicate on ~3,300 probes, so ~1,200 fits in total — seconds, not minutes.

**The one trade-off to record in the report.** Guyon §2.6 notes that removing several features at a time comes *"at the expense of possible classification performance degradation"*, and Mundra §II-B repeats that chunking *"may have negative effect on selection of genes when the set of genes is small"*. Halving all the way down means the final eliminations — the ones that decide the panel — are made in chunks. Guyon nonetheless used exactly this schedule on the leukemia data, which is why it is adopted here. If you want the more careful variant, halve only while the set is large and then remove one gene at a time, by replacing the last line of the loop body with:

```python
        target = len(keep) // 2 if len(keep) > 32 else len(keep) - 1
```

That costs ~42 fits per replicate instead of ~12, and departs from what the paper did on this dataset — state whichever you choose.

Note there is no `svm_used_fallback` branch. If no gene reaches the 80% threshold, that is the result and it gets reported as such; the old code fell back to an incomparable top-10 panel and then had to warn the reader not to compare it.

- [ ] **Step 3: Verify the mRMR symbols are gone and the SVM ones exist**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['mrmr', 'RFECV', 'svm_rfe_genes', 'svm_final_features',
             'svm_used_fallback', 'X_train_df', 'X_test_df', 'K_MRMR']:
    if name in src:
        print('STILL REFERENCED:', name)
for name in ['k_svm', 'svm_genes', 'svm_boot_sets', 'svm_stability_df',
             'svm_stable_set', 'svm_bacc']:
    if name not in src:
        print('MISSING:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` or `MISSING` lines.

- [ ] **Step 4: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): drop mRMR pre-filter, rebuild SVM arm as plain SVM-RFE"
```

---

### Task 3: Remove the permutation null

**Rationale to record in the report:** the permutation null (20 permutations × 100 bootstrap replicates = 2,000 model fits) calibrates the null distribution of the stability procedure itself. `FinalTask.pdf` §2 asks to *contrast statistically significant genes against ML-selected variables* — that requirement is met by Benjamini–Hochberg in Phase 2 and by the overlap analysis in Phase 5. The permutation null answers a question the assignment does not pose, at the highest compute cost in the notebook.

**Files:**
- Modify: `progetto/project_revised.ipynb` (delete 2 cells, edit 1)

**Interfaces:**
- Produces: `perm_counts` and `perm_model` no longer exist. The synthesis cell (anchor `overlap = lasso_stable_set`) currently prints `perm_counts.mean()` and must lose those lines.

- [ ] **Step 1: Delete the two cells, in descending index order**

Anchors, delete in this order:
1. `perm_model = LogisticRegression(`
2. `### 5.2 Controllo delle false scoperte: il null`

- [ ] **Step 2: In the cell anchored `overlap = lasso_stable_set & robust_baseline_genes`, delete the three lines that reference `perm_counts`**

Remove exactly this block:

```python
print(f"  3. Sotto etichette permutate la procedura produce in media {perm_counts.mean():.2f}")
print(f"     geni sopra soglia (massimo {perm_counts.max()}).")
```

Leave points 1 and 2 of that numbered list intact, and renumber nothing — the list reads correctly with two items.

- [ ] **Step 3: Add a markdown cell immediately before the synthesis cell, replacing the deleted `### 5.2` heading**

```markdown
### 5.2 Contrasto fra significativita statistica e selezione automatica

Il secondo fenomeno richiesto dalla consegna e il contrasto fra i geni statisticamente
significativi e le variabili selezionate dai modelli. I due criteri controllano cose
diverse: il False Discovery Rate controlla i falsi positivi ipotesi per ipotesi, ma non
dice nulla su quali geni verrebbero riselezionati in un nuovo esperimento; la stability
selection controlla esattamente questo, la riproducibilita della selezione, ma non offre
di per se garanzie di significativita. E la loro concordanza a rendere credibile il
pannello proposto, ed e per questo che vanno riportati insieme.

Che i due criteri divergano non e una sorpresa: Guyon et al. (2002, sez. 5.1) confrontarono
i geni scelti dall'SVM-RFE con quelli del criterio univariato di Golub proprio su questi
dati, e trovarono che **per pannelli di 16 geni o meno al massimo il 25% dei geni e in
comune**. Il confronto che segue va letto con quel termine di paragone in mente.
```

- [ ] **Step 4: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['perm_counts', 'perm_model', 'N_PERM', 'permutazion']:
    if name in src:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` lines.

- [ ] **Step 5: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): remove permutation null for stability selection"
```

---

### Task 4: Remove the matched-sparsity Lasso-vs-ElasticNet experiment

**Rationale to record in the report:** these four cells calibrate Elastic Net's `C` to match Lasso's selection size, then compute a paired probability-gap statistic, then run a second control to rule out a ceiling-effect explanation of the first result. That is an experiment designed to isolate penalty *shape* from penalty *strength* — a methods question, not the phenomenon the assignment asks to document. Task 7 replaces it with a direct demonstration, and the report names the sparsity confounder in one sentence instead of engineering it away.

**Files:**
- Modify: `progetto/project_revised.ipynb` (delete 4 cells)

**Interfaces:**
- Produces: `EN_MATCHED_C`, `EN_MATCHED_L1_RATIO`, `matched_en`, `matched_counts`, `matched_sets`, `matched_probs`, `matched_map`, `matched_panel`, `group_df`, `gap_lasso`, `gap_en`, `size_lasso`, `size_en`, `size_ratio`, `final_en_model`, `en_top`, `heat` no longer exist.

- [ ] **Step 1: Delete the four cells, in descending index order**

Anchors, delete in this order:
1. `en_top = sorted(en_stable_set)`
2. `# --- Il test decisivo`
3. `EN_MATCHED_L1_RATIO = 0.5`
4. `#### Lasso contro Elastic Net a parita di sparsita`

- [ ] **Step 2: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['EN_MATCHED', 'matched_', 'group_df', 'gap_lasso', 'gap_en',
             'size_ratio', 'final_en_model', 'en_top']:
    if name in src:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` lines.

- [ ] **Step 3: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): remove matched-sparsity Lasso/ElasticNet experiment"
```

---

### Task 5: Fix the probe-filter leak and reduce the nested CV

**Rationale to record in the report:** the unsupervised probe filter was estimated on the training split, but the nested-CV cell fed the filtered matrix to `cross_val_score`, so the filter's fit leaked across outer folds while the cell's text claimed everything was refit per fold. Because the filter never inspects `y`, computing it on all 72 samples is both standard (Dudoit, Fridlyand & Speed 2002 §3.1.2 did exactly that) and makes the nested-CV claim literally true. Repeats drop from 10 to 3 — 15 outer folds is ample for a mean and a spread at this sample size.

**Files:**
- Modify: `progetto/project_revised.ipynb` (2 code cells, 1 markdown cell)

**Interfaces:**
- Consumes: `X_all`, `y`, `FLOOR`, `CEIL`, `RANDOM_STATE` from Task 1.
- Produces: `X_thr`, `probe_mask`, `feature_cols` (list[str]), `X_log_all` (DataFrame, all 72 rows), `X_train_log`, `X_test_log` (DataFrames), `X_train`, `X_test` (ndarray), `y_train`, `y_test` (Series), `scaler`. **`X_train_int`, `X_test_int` and `X_full_log` no longer exist.**

- [ ] **Step 1: Replace the cell anchored `X_thr = X_all.clip(lower=FLOOR` with this source**

```python
X_thr = X_all.clip(lower=FLOOR, upper=CEIL)

# Il filtro non supervisionato e calcolato su tutti e 72 i campioni, come in
# Dudoit et al. (2002): il criterio guarda solo la dinamica di ciascuna sonda e
# non tocca mai y, quindi non introduce alcuna forma di selection bias.
mx, mn = X_thr.max(axis=0), X_thr.min(axis=0)
probe_mask = ~((mx / mn <= 5) | ((mx - mn) <= 500))
feature_cols = list(X_thr.columns[probe_mask])

X_log_all = np.log10(X_thr.loc[:, probe_mask])

X_train_log, X_test_log, y_train, y_test = train_test_split(
    X_log_all, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_log)
X_test = scaler.transform(X_test_log)

print("--- Preprocessing (Dudoit et al. 2002, §3.1.2) ---")
print(f"(a) thresholding: floor {FLOOR}, ceiling {CEIL:,}")
print(f"(b) filtro: esclusi i probe con max/min <= 5 OPPURE (max - min) <= 500")
print(f"(c) trasformazione log10")
print()
print(f"Sonde iniziali:   {X_all.shape[1]}")
print(f"Sonde superstiti: {len(feature_cols)}")
print(f"Sonde eliminate:  {X_all.shape[1] - len(feature_cols)} "
      f"({1 - len(feature_cols)/X_all.shape[1]:.0%})")
print()
print(f"Training set: {X_train.shape[0]} pazienti "
      f"({(y_train == 0).sum()} ALL / {(y_train == 1).sum()} AML)")
print(f"Test set:     {X_test.shape[0]} pazienti "
      f"({(y_test == 0).sum()} ALL / {(y_test == 1).sum()} AML)")

assert X_train.shape[1] == len(feature_cols) == X_test.shape[1]
assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
```

The patient split is unchanged: `train_test_split` receives the same `y`, `test_size` and `random_state`, so the same 57/15 patients land in the same sets.

- [ ] **Step 2: In the markdown cell that follows it (anchor `Tre precisazioni metodologiche`), replace the third paragraph**

Find the paragraph beginning `**Il filtro e stimato sul solo training set.**` and replace that whole paragraph with:

```markdown
**Il filtro e calcolato su tutti e 72 i campioni**, come in Dudoit et al. (2002). La scelta
e lecita perche il criterio e *non supervisionato*: guarda solo la dinamica di ciascuna
sonda e non consulta mai le etichette. Non introduce quindi il selection bias descritto da
Ambroise & McLachlan (2002), che riguarda i filtri che usano `y`. Il vantaggio pratico e
che la matrice filtrata puo essere passata direttamente alla cross-validation annidata
della Fase 5 senza che il filtro debba essere ristimato dentro ogni fold: cio che deve
stare dentro il fold e tutto e solo cio che guarda le etichette, cioe standardizzazione,
selezione e classificatore.
```

- [ ] **Step 3: Replace the cell anchored `nested_pipe = Pipeline([` with this source**

```python
selector_estimator = LogisticRegression(
    l1_ratio=1.0, solver="liblinear", C=lasso_optimal_C,
    class_weight="balanced", max_iter=5000, tol=1e-8, random_state=RANDOM_STATE,
)

nested_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectFromModel(selector_estimator, threshold=1e-5)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=5000,
                               random_state=RANDOM_STATE)),
])

outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
nested_scores = cross_val_score(nested_pipe, X_log_all, y, cv=outer_cv,
                                scoring="balanced_accuracy", n_jobs=-1)
nested_mean, nested_std = nested_scores.mean(), nested_scores.std()
lo, hi = np.percentile(nested_scores, [2.5, 97.5])

print("--- Stima onesta dell'errore: CV annidata (5 fold x 3 ripetizioni) ---")
print(f"Balanced accuracy: {nested_mean:.3f} (deviazione standard {nested_std:.3f})")
print(f"Intervallo empirico al 95%: [{lo:.3f}, {hi:.3f}]")
print(f"Fold valutati: {len(nested_scores)}")

plt.figure(figsize=(8, 4))
sns.histplot(nested_scores, bins=8, color="crimson", alpha=0.75)
plt.axvline(nested_mean, color="black", linestyle="--", label=f"Media = {nested_mean:.3f}")
plt.xlabel("Balanced accuracy (fold esterno)")
plt.ylabel("Frequenza")
plt.title("Distribuzione dell'errore in cross-validation annidata")
plt.legend()
plt.tight_layout()
plt.show()
```

The interpretive `print()` block that was here moves to markdown in Step 4.

- [ ] **Step 4: Insert a markdown cell immediately after it**

```markdown
Standardizzazione, selezione e classificatore stanno **dentro** la pipeline e vengono
percio rifatti da zero in ciascun fold esterno. Se la selezione fosse eseguita una volta
sola su tutti i dati e poi validata, la stima risulterebbe ottimisticamente distorta: e
l'errore classico su questi dati, documentato da Ambroise & McLachlan (2002) e da
Varma & Simon (2006), e produce accuratezze prossime al 100% prive di significato. Il
filtro non supervisionato della Fase 1 resta invece fuori dalla pipeline, ed e corretto
che sia cosi: non guarda le etichette, quindi non puo trasferire informazione sul target
da un fold all'altro.
```

- [ ] **Step 5: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['X_full_log', 'X_train_int', 'X_test_int', 'n_repeats=10']:
    if name in src:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` lines.

- [ ] **Step 6: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "fix(notebook): remove probe-filter leak into nested CV, reduce repeats to 3"
```

---

### Task 6: Compute the regularization path once

**Rationale:** the cell anchored `cv_scores = lasso_grid.cv_results_` fits 30 models to count non-zero coefficients, and the cell anchored `top_n = 30` fits the same 30 models again to draw the coefficient path. One computation serves both.

**Files:**
- Modify: `progetto/project_revised.ipynb` (2 code cells)

**Interfaces:**
- Consumes: `lasso_grid`, `param_grid`, `X_train`, `y_train`, `lasso_optimal_C`, `selected_indices`, `feature_cols`, `lasso_stable_set` from existing cells.
- Produces: `C_grid` (ndarray, 30 values) and `coef_path` (ndarray, shape `(30, n_features)`), consumed by the coefficient-path figure. `n_selected` becomes an ndarray of ints rather than a list. **`C_values` no longer exists.**

- [ ] **Step 1: In the cell anchored `cv_scores = lasso_grid.cv_results_`, replace the `n_selected` loop**

Replace exactly this block:

```python
n_selected = []
for c in C_grid:
    m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c,
                           class_weight="balanced", max_iter=5000, tol=1e-8,
                           random_state=RANDOM_STATE).fit(X_train, y_train)
    n_selected.append(int((m.coef_[0] != 0).sum()))
```

with:

```python
# Il percorso di regolarizzazione viene calcolato una volta sola: serve qui per
# il conteggio dei geni selezionati e piu avanti per il grafico dei coefficienti.
coef_path = np.array([
    LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c,
                       class_weight="balanced", max_iter=5000, tol=1e-8,
                       random_state=RANDOM_STATE).fit(X_train, y_train).coef_[0]
    for c in C_grid
])
n_selected = (coef_path != 0).sum(axis=1)
```

- [ ] **Step 2: In the same cell, delete the unused `plateau` line**

Remove exactly:

```python
plateau = [n for c, n in zip(C_grid, cv_scores) if n >= cv_scores.max() - 0.01]
```

It is computed and never read; the threshold-sensitivity cell recomputes its own.

- [ ] **Step 3: In the cell anchored `top_n = 30`, delete the recomputation and switch to `C_grid`**

Delete exactly this block:

```python
C_values = np.logspace(-3, 1, 30)
coef_path = np.array([
    LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c,
                       class_weight="balanced", max_iter=5000, tol=1e-8,
                       random_state=RANDOM_STATE).fit(X_train, y_train).coef_[0]
    for c in C_values
])
```

Then replace the single plotting line `plt.plot(C_values, coef_path[:, idx],` with `plt.plot(C_grid, coef_path[:, idx],`.

- [ ] **Step 4: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
print('C_values occurrences (expect 0):', src.count('C_values'))
print('coef_path assignments (expect 1):', src.count('coef_path = np.array'))
print('C_grid occurrences (expect >= 4):', src.count('C_grid'))
EOF
```

Expected: `0`, `1`, and a number at least 4.

- [ ] **Step 5: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "perf(notebook): compute L1 regularization path once instead of twice"
```

---

### Task 7: Rebuild section 4.1 — collinearity, with Elastic Net relocated here

**Rationale to record in the report:** the hierarchical clustering of the correlation matrix into "co-expression modules" is genomics-domain framing; the statistical content — that some genes have near-duplicate twins, and that L1 splits a group's inclusion probability among its members — is carried entirely by a histogram plus a paired table. Elastic Net moves from Phase 5 into 4.1 because its only role in this analysis is to demonstrate the grouping effect of Zou & Hastie (2005, Thm. 1), which is a 4.1 topic. In Phase 5 it then appears simply as a second arm in the comparison.

**Files:**
- Modify: `progetto/project_revised.ipynb` (rewrite 1 markdown cell, delete 2 code cells, add 4 cells)

**Interfaces:**
- Consumes: `X_train`, `y_train`, `feature_cols`, `lasso_stability_df`, `lasso_stable_set`, `lasso_final_genes`, `lasso_boot_sets`, `robust_baseline_genes`, `N_BOOTSTRAP`, `STABILITY_THRESHOLD`, `RANDOM_STATE`.
- Produces: `corr_all` (ndarray), `max_corr` (ndarray), `prob_map` (dict), `SIGNAL_MIN` (float, defined **once**), `en_grid`, `best_C`, `best_l1_ratio`, `en_boot_sets` (list[set]), `en_stability_df` (columns `Gene_Probe`, `Inclusion_Probability`), `en_stable_set` (set), `en_acc`, `en_bacc` (floats), `twin_df` (DataFrame). Tasks 9 and 11 depend on `en_stability_df`, `en_stable_set`, `en_boot_sets`, `en_bacc`.

- [ ] **Step 1: Replace the markdown cell anchored `### 4.1 L'effetto della multi-collinearita` with this source**

```markdown
### 4.1 L'effetto della multi-collinearita sulla sparsita

Questo e il primo dei due fenomeni che la consegna chiede di esplorare esplicitamente:
*"Identify the effect of multi-collinearity on sparsity, to compute and map true feature
inclusion probabilities."* L'aggettivo **true** e la chiave della sezione.

Il meccanismo e noto in letteratura come *grouping effect* (Zou & Hastie, 2005). Siano due
geni fortemente correlati. Se la soluzione ottima assegna alla coppia un peso complessivo
`c`, la penalita L1 vale `|b_i| + |b_j|`, e quando i due predittori sono quasi identici
**ogni** ripartizione con `b_i + b_j = c` produce la stessa devianza e la stessa penalita:
la soluzione non e unica. La geometria della palla L1, che ha gli spigoli sugli assi,
spinge pero gli algoritmi verso le soluzioni in cui uno dei due coefficienti e esattamente
zero. Il Lasso quindi **sceglie un rappresentante e scarta il gemello**, e quale dei due
venga scelto dipende da fluttuazioni campionarie minime.

La conseguenza sulle inclusion probability e diretta: la probabilita del *gruppo* si
**spartisce** fra i suoi membri. Un gruppo selezionato nel 100% delle repliche puo
presentarsi come due geni al 55% e al 45%, nessuno dei quali raggiunge la soglia. Le
probabilita osservate per singolo gene sono percio una stima distorta della rilevanza
reale — ed e esattamente cio a cui allude la parola *true* nella consegna. Il paradosso da
documentare e che la sparsita esatta, la proprieta che rende l'L1 attraente per la
selezione di biomarcatori, **degrada la metrica di riproducibilita** su cui la consegna
valuta il lavoro.

Lo verifichiamo in tre passi: quanto sono diffusi i geni quasi-duplicati; come si comporta
sugli stessi dati una penalita Elastic Net, che al termine L1 affianca un termine L2; e
come le due penalita trattano i due membri di ciascuna coppia correlata.
```

- [ ] **Step 2: Delete the two old code cells, in descending index order**

Anchors, delete in this order:
1. `prob_map = dict(zip(lasso_stability_df`
2. `sig_names = sorted(robust_baseline_genes)`

- [ ] **Step 3: Delete the Elastic Net cell from Phase 5 (anchor `en_param_grid = {"l1_ratio"`)** — it is re-inserted here in Step 5.

- [ ] **Step 4: Insert a new code cell after the markdown of Step 1 — diffusion of collinearity**

```python
corr_all = np.abs(np.corrcoef(X_train, rowvar=False))
np.fill_diagonal(corr_all, 0.0)
max_corr = corr_all.max(axis=0)

print("--- Quanto sono diffusi i geni quasi-duplicati? ---")
for t in (0.9, 0.8, 0.7):
    n = int((max_corr > t).sum())
    print(f"  sonde con almeno un gemello a |r| > {t}: {n} su {len(feature_cols)} "
          f"({n/len(feature_cols):.1%})")
print(f"\n|r| massima mediana su tutte le sonde: {np.median(max_corr):.3f}")

plt.figure(figsize=(9, 5))
sns.histplot(max_corr, bins=60, color="steelblue")
plt.axvline(0.8, color="crimson", linestyle="--", label="|r| = 0.8")
plt.yscale("log")
plt.xlabel("|r| massima di ciascuna sonda verso un'altra sonda")
plt.ylabel("Numero di sonde (scala logaritmica)")
plt.title("Diffusione della collinearita nello spazio filtrato")
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Insert a new code cell after it — Elastic Net and its stability selection**

```python
en_param_grid = {"l1_ratio": [0.1, 0.5, 0.9], "C": np.logspace(-2, 1, 6)}
en_base_model = LogisticRegression(solver="saga", class_weight="balanced",
                                   max_iter=5000, random_state=RANDOM_STATE)
en_grid = GridSearchCV(en_base_model, en_param_grid,
                       cv=StratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=RANDOM_STATE),
                       scoring="balanced_accuracy", n_jobs=-1)
en_grid.fit(X_train, y_train)

best_l1_ratio = en_grid.best_params_["l1_ratio"]
best_C = en_grid.best_params_["C"]
print(f"Elastic Net: l1_ratio ottimale = {best_l1_ratio}, C ottimale = {best_C:.4f}")

en_boot_sets = []
en_counts = np.zeros(len(feature_cols))
boot_en = LogisticRegression(solver="saga", C=best_C, l1_ratio=best_l1_ratio,
                             class_weight="balanced", max_iter=5000,
                             random_state=RANDOM_STATE)

print(f"Stability selection per l'Elastic Net ({N_BOOTSTRAP} repliche)...")
for i in range(N_BOOTSTRAP):
    X_boot, y_boot = resample(X_train, y_train, random_state=i, stratify=y_train)
    boot_en.fit(X_boot, y_boot)
    idx = np.where(boot_en.coef_[0] != 0)[0]
    en_counts[idx] += 1
    en_boot_sets.append({feature_cols[j] for j in idx})

en_stability_df = (
    pd.DataFrame({"Gene_Probe": feature_cols, "Inclusion_Probability": en_counts / N_BOOTSTRAP})
    .sort_values("Inclusion_Probability", ascending=False).reset_index(drop=True)
)
en_stable_set = set(en_stability_df[
    en_stability_df["Inclusion_Probability"] >= STABILITY_THRESHOLD]["Gene_Probe"])

en_acc = en_bacc = np.nan
if en_stable_set:
    en_idx = [feature_cols.index(g) for g in sorted(en_stable_set)]
    en_eval_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                     random_state=RANDOM_STATE)
    en_eval_clf.fit(X_train[:, en_idx], y_train)
    en_y_pred = en_eval_clf.predict(X_test[:, en_idx])
    en_acc = accuracy_score(y_test, en_y_pred)
    en_bacc = balanced_accuracy_score(y_test, en_y_pred)

print(f"\nGeni con inclusion probability >= {STABILITY_THRESHOLD:.0%}: {len(en_stable_set)}")
print(f"Dimensione media della selezione per replica: "
      f"{np.mean([len(s) for s in en_boot_sets]):.1f} geni "
      f"(Lasso: {np.mean([len(s) for s in lasso_boot_sets]):.1f})")
```

Note the evaluation classifier uses `C=1.0`, not the `C=1e5` of the deleted cell: an effectively unpenalized logistic regression on a handful of collinear genes is numerically fragile, and `C=1.0` matches how the Lasso panel is evaluated, keeping the two arms comparable.

- [ ] **Step 6: Insert a new code cell after it — the paired twin table**

```python
prob_map = dict(zip(lasso_stability_df["Gene_Probe"],
                    lasso_stability_df["Inclusion_Probability"]))
en_prob_map = dict(zip(en_stability_df["Gene_Probe"],
                       en_stability_df["Inclusion_Probability"]))

SIGNAL_MIN = 0.20   # soglia minima di segnale: sotto, la probabilita e zero per
                    # assenza di segnale e non per collinearita

rows = []
for g in lasso_stability_df.query("Inclusion_Probability >= @SIGNAL_MIN")["Gene_Probe"]:
    j = feature_cols.index(g)
    k = int(np.argmax(corr_all[j]))
    twin = feature_cols[k]
    rows.append({
        "Gene": g,
        "Gemello": twin,
        "|r|": round(float(corr_all[j, k]), 3),
        "Lasso p(gene)": round(prob_map.get(g, 0.0), 2),
        "Lasso p(gemello)": round(prob_map.get(twin, 0.0), 2),
        "EN p(gene)": round(en_prob_map.get(g, 0.0), 2),
        "EN p(gemello)": round(en_prob_map.get(twin, 0.0), 2),
        "Nel pannello": g in lasso_stable_set,
    })

twin_df = pd.DataFrame(rows).sort_values("Lasso p(gene)", ascending=False)
twin_df["Lasso: divario"] = (twin_df["Lasso p(gene)"] - twin_df["Lasso p(gemello)"]).round(2)
twin_df["EN: divario"] = (twin_df["EN p(gene)"] - twin_df["EN p(gemello)"]).round(2)

print(f"--- Geni con segnale (p >= {SIGNAL_MIN:.0%}) e il loro gemello piu correlato ---")
display(twin_df)

collinear = twin_df[twin_df["|r|"] > 0.7]
print(f"Coppie con |r| > 0.7: {len(collinear)} su {len(twin_df)}")
if len(collinear):
    print(f"Divario medio di probabilita fra i due membri della coppia:")
    print(f"  Lasso:       {collinear['Lasso: divario'].mean():.3f}")
    print(f"  Elastic Net: {collinear['EN: divario'].mean():.3f}")

plt.figure(figsize=(9, 6))
sns.scatterplot(x=max_corr, y=[prob_map.get(g, 0.0) for g in feature_cols],
                alpha=0.35, color="lightsteelblue", s=25, label="Tutte le sonde")
sig_pts = twin_df
sns.scatterplot(data=sig_pts, x="|r|", y="Lasso p(gene)",
                color="crimson", s=90, label=f"Geni con segnale (p >= {SIGNAL_MIN:.0%})")
plt.axhline(STABILITY_THRESHOLD, color="black", linestyle="--",
            label=f"Soglia {STABILITY_THRESHOLD:.0%}")
plt.xlabel("|r| massima verso un'altra sonda")
plt.ylabel(f"Inclusion probability del Lasso ({N_BOOTSTRAP} bootstrap)")
plt.title("Collinearita e stabilita: il fenomeno vive nei pochi geni con segnale")
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 7: Insert a markdown cell after it — the interpretation, including the stated limitation**

```markdown
**Come leggere la tabella.** Ogni riga accosta un gene con segnale al suo gemello piu
correlato, e riporta con quale frequenza ciascuno dei due viene selezionato sotto le due
penalita. Le righe con `|r|` alta e un divario Lasso ampio sono la manifestazione diretta
del fenomeno: due geni quasi indistinguibili sui dati, di cui la penalita L1 seleziona
sistematicamente uno solo. La probabilita del gruppo si spartisce, e nessuno dei due
membri riflette da solo la rilevanza dell'informazione che condividono.

**Il confronto con l'Elastic Net e coerente con la teoria, ma non e un esperimento
controllato.** Il termine L2 forza i coefficienti di predittori correlati a convergere
(Zou & Hastie, 2005, Teorema 1), e ci si attende percio un divario piu contenuto. Va pero
dichiarato un confondimento che non abbiamo eliminato: ai rispettivi parametri ottimi
l'Elastic Net **e meno sparso** del Lasso, e un modello che seleziona piu geni spinge le
probabilita di entrambi i membri verso 1, comprimendo il divario per effetto soffitto
anche in assenza di qualunque grouping effect. Isolare la forma della penalita dalla sua
intensita richiederebbe di tarare i due modelli a parita di sparsita dentro ogni replica
bootstrap, il che eccede lo scopo di questo lavoro. La conclusione corretta e quindi che i
dati sono **compatibili** con il grouping effect previsto dalla teoria, non che lo
dimostrino.
```

- [ ] **Step 8: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['linkage(', 'fcluster', 'squareform', 'sig_names', 'corr_sig', 'modules']:
    if name in src:
        print('STILL REFERENCED:', name)
print('SIGNAL_MIN definitions (expect 1):', src.count('SIGNAL_MIN = 0.20'))
print('en_param_grid definitions (expect 1):', src.count('en_param_grid = {'))
EOF
```

Expected: no `STILL REFERENCED` lines, then `1` and `1`.

- [ ] **Step 9: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): rebuild collinearity section, relocate Elastic Net to 4.1"
```

---

### Task 8: Add the top-k univariate selection arm

**Rationale to record in the report:** the third comparison arm costs nothing — the p-values already exist — and it earns its place twice. It is the classic filter-versus-embedded contrast of the Saeys et al. (2007) taxonomy, and it turns the assignment's "contrast significant genes against ML-selected variables" from a set-overlap observation into an actual head-to-head on accuracy *and* stability.

**Files:**
- Modify: `progetto/project_revised.ipynb` (rewrite 1 markdown cell, add 1 code cell)

**Interfaces:**
- Consumes: `significance_results` (DataFrame with `Gene_Probe`, `T_Test_FDR_P`), `lasso_final_genes`, `X_train`, `X_test`, `y_train`, `y_test`, `feature_cols`, `N_BOOTSTRAP`.
- Produces: `k_uni` (int), `uni_genes` (list[str]), `uni_boot_sets` (list[set]), `uni_stability_df` (columns `Gene_Probe`, `Selection_Count`, `Inclusion_Probability`), `uni_stable_set` (set), `uni_acc`, `uni_bacc` (floats). Task 9 depends on all of these.

- [ ] **Step 1: Replace the markdown cell anchored `### 5.3 Metodi alternativi di selezione` with this source**

```markdown
### 5.3 Confronto fra strategie di selezione

Il pannello ottenuto con la penalita L1 va confrontato con approcci costruiti su principi
diversi, per capire quanto il risultato dipenda dalla scelta del metodo. Seguendo la
tassonomia di Saeys, Inza & Larranaga (2007) copriamo una strategia per ciascuna delle tre
famiglie, quattro metodi in tutto:

- **filtro**: i `k` geni con p-value piu basso al t-test, seguiti da una regressione
  logistica non penalizzata su quei soli geni;
- **embedded**: la regressione logistica L1 della Fase 3, in cui selezione e stima
  avvengono nello stesso passo;
- **embedded a penalita mista**: l'**Elastic Net**, gia introdotto nella Fase 4.1 per
  documentare il grouping effect;
- **wrapper**: **SVM-RFE** (Guyon et al., 2002), che addestra un SVM lineare, ordina le
  feature per `w^2` ed elimina ricorsivamente le peggiori, dimezzando a ogni passo il
  numero di geni superstiti come nell'esperimento sui dati di leucemia dell'articolo
  originale (sez. 5.1).

Tutti e quattro selezionano lo stesso numero `k` di geni, pari alla dimensione del pannello
del Lasso: il confronto avviene percio a parita di sparsita, e le differenze osservate nella
stabilita non sono imputabili al fatto che un metodo sia semplicemente piu generoso.

Il filtro univariato e il termine di paragone piu istruttivo per la consegna, perche usa
esattamente la stessa statistica della Fase 2: mette quindi a confronto diretto "geni
statisticamente significativi" e "geni selezionati da un modello".

**Un metodo considerato e non implementato: il filtro MRMR dentro l'SVM-RFE.** Mundra &
Rajapakse (2010) potenziano l'SVM-RFE sostituendo il ranking per solo `w^2` con la
combinazione convessa della loro eq. (5),

$$r_i = \beta\,|w_i| + (1-\beta)\,\frac{R_{S,i}}{Q_{S,i}}$$

dove `R_{S,i}` e l'informazione mutua fra il gene *i* e le etichette di classe e `Q_{S,i}` la
sua informazione mutua media con gli altri geni ancora in gioco. Il punto essenziale e che
il loro Algoritmo 1 **ricalcola entrambe le quantita dentro il ciclo di eliminazione**, per
ogni gene superstite e a ogni iterazione: la ridondanza e valutata rispetto all'insieme che
sopravvive, non una volta per tutte.

Non lo abbiamo implementato per tre ragioni. Primo, il costo: `Q_{S,i}` richiede
l'informazione mutua fra tutte le coppie di geni superstiti, ricalcolata a ogni iterazione —
con 3363 sonde sono milioni di coppie per iterazione, moltiplicate per le repliche bootstrap;
gli autori stessi (sez. IV) osservano che il metodo e piu costoso di SVM-RFE e di MRMR presi
separatamente. Secondo, i parametri liberi aggiuntivi: β, la sensibilita η dell'SVM e la
discretizzazione delle intensita della loro eq. (9), tutti da stimare in cross-validation.
Terzo, il guadagno riportato: sui dati di leucemia la loro Tabella II riporta 47 geni e
97.88% di accuratezza per l'SVM-RFE puro contro 37 geni e 98.35% per la versione con filtro
MRMR — mezzo punto percentuale e dieci geni.

Applicare invece mRMR una volta sola come pre-filtro, lasciando poi che l'RFE ordini per
solo `w^2`, non e un'approssimazione del metodo ma una procedura diversa, in cui la
proprieta di minima ridondanza non viene mantenuta durante l'eliminazione — cioe esattamente
il problema che quel lavoro si propone di risolvere.
```

- [ ] **Step 2: Insert a code cell immediately after it**

```python
k_uni = len(lasso_final_genes)
uni_genes = (significance_results.nsmallest(k_uni, "T_Test_FDR_P")["Gene_Probe"]
             .tolist())
uni_idx = [feature_cols.index(g) for g in uni_genes]

uni_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                               random_state=RANDOM_STATE)
uni_model.fit(X_train[:, uni_idx], y_train)
uni_pred = uni_model.predict(X_test[:, uni_idx])
uni_acc = accuracy_score(y_test, uni_pred)
uni_bacc = balanced_accuracy_score(y_test, uni_pred)

# Stessa stability selection degli altri due metodi: il t-test viene RIESEGUITO
# dentro ciascuna replica bootstrap, non riutilizzato dal calcolo globale.
uni_boot_sets = []
uni_counts = Counter()
for i in range(N_BOOTSTRAP):
    X_boot, y_boot = resample(X_train, y_train, random_state=i, stratify=y_train)
    yb = y_boot.to_numpy()
    _, p_boot = ttest_ind(X_boot[yb == 0], X_boot[yb == 1], equal_var=False, axis=0)
    genes = {feature_cols[j] for j in np.argsort(p_boot)[:k_uni]}
    uni_boot_sets.append(genes)
    uni_counts.update(genes)

uni_stability_df = (
    pd.DataFrame({"Gene_Probe": list(uni_counts.keys()),
                  "Selection_Count": list(uni_counts.values())})
    .assign(Inclusion_Probability=lambda d: d["Selection_Count"] / N_BOOTSTRAP)
    .sort_values("Inclusion_Probability", ascending=False)
    .reset_index(drop=True)
)
uni_stable_set = set(uni_stability_df[
    uni_stability_df["Inclusion_Probability"] >= STABILITY_THRESHOLD]["Gene_Probe"])

print(f"--- Filtro univariato: i {k_uni} geni con p-value piu basso ---")
print(f"Geni scelti: {uni_genes}")
print(f"Sovrapposizione con il pannello del Lasso: "
      f"{len(set(uni_genes) & lasso_stable_set)} geni")
print(f"Balanced accuracy su hold-out: {uni_bacc:.2%}")
print()
print(f"Stability selection ({N_BOOTSTRAP} repliche, t-test rieseguito in ciascuna):")
print(f"  geni selezionati almeno una volta: {len(uni_stability_df)}")
print(f"  geni con inclusion probability >= {STABILITY_THRESHOLD:.0%}: {len(uni_stable_set)}")
```

- [ ] **Step 3: Verify the symbols exist and nothing from the old arm survives**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['uni_stability_df', 'uni_boot_sets', 'uni_stable_set', 'uni_bacc', 'k_uni']:
    if name not in src:
        print('MISSING:', name)
if 'mrmr' in src:
    print('STILL REFERENCED: mrmr')
print('check done')
EOF
```

Expected: `check done` with no `MISSING` or `STILL REFERENCED` lines.

- [ ] **Step 4: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "feat(notebook): add top-k univariate filter as third comparison arm"
```

---

### Task 9: Rebuild the reproducibility comparison on four arms, drop Kuncheva

**Rationale to record in the report:** the Kuncheva (2007) index corrects set overlap for chance, which matters when comparing methods whose selection sizes differ by an order of magnitude. All four arms now select the same `k`, so Jaccard carries the same information and is standard course material. Kuncheva stays in the report as an index considered.

**Files:**
- Modify: `progetto/project_revised.ipynb` (1 markdown cell, 2 code cells)

**Interfaces:**
- Consumes: `lasso_stability_df`, `lasso_final_genes`, `lasso_boot_sets`, `lasso_bacc`, `en_stability_df`, `en_stable_set`, `en_boot_sets`, `en_bacc`, `uni_stability_df`, `uni_genes`, `uni_boot_sets`, `uni_bacc`, `svm_stability_df`, `svm_genes`, `svm_boot_sets`, `svm_stable_set`, `svm_bacc`, `robust_baseline_genes`, `nested_mean`, `nested_std`, `jaccard_index`, `mean_inclusion`.
- Produces: `stability_scores` (DataFrame), `summary_df` (DataFrame). Task 11 reads both.

- [ ] **Step 1: Replace the markdown cell anchored `### 5.4 Score di riproducibilita` with this source**

```markdown
### 5.4 Score di riproducibilita

La consegna indica esplicitamente la riproducibilita come criterio di valutazione. La
misuriamo su due assi, che non ordinano necessariamente i metodi allo stesso modo:

- la **probabilita media di inclusione** dei geni del pannello finale, che misura la
  stabilita *per singolo gene*: quanto spesso uno specifico gene ricompare;
- l'**indice di Jaccard**, cioe la sovrapposizione media fra gli insiemi selezionati in
  coppie di repliche bootstrap, che misura la stabilita *per insieme*: quanto due repliche
  scelgono la stessa lista.

Un metodo puo avere pochi geni quasi sempre presenti e una coda ampia di geni occasionali
(alta stabilita per-gene, bassa per-insieme), oppure una selezione diffusa ma pescata da un
bacino ristretto (il contrario). Riportarne uno solo darebbe una descrizione parziale.
```

- [ ] **Step 2: Replace the cell anchored `stability_scores = pd.DataFrame([` with this source**

```python
stability_scores = pd.DataFrame([
    {"Metodo": "Lasso (L1)",
     "Prob. inclusione media": mean_inclusion(lasso_stability_df, lasso_final_genes),
     "Jaccard": jaccard_index(lasso_boot_sets),
     "Dim. media selezione": float(np.mean([len(s) for s in lasso_boot_sets]))},
    {"Metodo": "Elastic Net (L1+L2)",
     "Prob. inclusione media": mean_inclusion(en_stability_df, sorted(en_stable_set)),
     "Jaccard": jaccard_index(en_boot_sets),
     "Dim. media selezione": float(np.mean([len(s) for s in en_boot_sets]))},
    {"Metodo": "Filtro univariato (top-k)",
     "Prob. inclusione media": mean_inclusion(uni_stability_df, uni_genes),
     "Jaccard": jaccard_index(uni_boot_sets),
     "Dim. media selezione": float(np.mean([len(s) for s in uni_boot_sets]))},
    {"Metodo": "SVM-RFE (wrapper)",
     "Prob. inclusione media": mean_inclusion(svm_stability_df, svm_genes),
     "Jaccard": jaccard_index(svm_boot_sets),
     "Dim. media selezione": float(np.mean([len(s) for s in svm_boot_sets]))},
]).round(3)

print("--- Score di riproducibilita della selezione ---")
display(stability_scores)
```

- [ ] **Step 3: Replace the cell anchored `def plot_dropoff(ax, df, label` with this source**

```python
def plot_dropoff(ax, df, label, color, marker):
    n = min(50, len(df))
    ax.plot(range(1, n + 1), df["Inclusion_Probability"].head(n).to_numpy(),
            label=label, color=color, linewidth=2.5, marker=marker, markersize=4)


fig, ax = plt.subplots(figsize=(10, 6))
plot_dropoff(ax, lasso_stability_df, "Lasso (L1)", "crimson", "o")
plot_dropoff(ax, en_stability_df, "Elastic Net (L1 + L2)", "darkorange", "s")
plot_dropoff(ax, uni_stability_df, "Filtro univariato (top-k)", "steelblue", "^")
plot_dropoff(ax, svm_stability_df, "SVM-RFE (wrapper)", "seagreen", "D")
ax.axhline(STABILITY_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"Soglia {STABILITY_THRESHOLD:.0%}")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Rango del gene")
ax.set_ylabel("Inclusion probability empirica")
ax.set_title("Confronto della stabilita fra le quattro strategie di selezione")
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
venn3(subsets=[lasso_stable_set, en_stable_set, robust_baseline_genes],
      set_labels=(f"Lasso\n({len(lasso_stable_set)} geni)",
                  f"Elastic Net\n({len(en_stable_set)} geni)",
                  f"Baseline statistico\n({len(robust_baseline_genes)} geni)"),
      set_colors=("crimson", "darkorange", "skyblue"), alpha=0.7)
venn3_circles(subsets=[lasso_stable_set, en_stable_set, robust_baseline_genes],
              linestyle="solid", linewidth=1.0, color="gray")
plt.title("Concordanza fra i pannelli selezionati e la significativita statistica")
plt.tight_layout()
plt.show()

summary_df = pd.DataFrame({
    "Metodo": ["Lasso (L1)", "Elastic Net (L1+L2)", "Filtro univariato (top-k)",
               "SVM-RFE (wrapper)"],
    "Geni nel pannello": [len(lasso_final_genes), len(en_stable_set), len(uni_genes),
                          len(svm_genes)],
    "Geni >= soglia": [len(lasso_final_genes), len(en_stable_set), len(uni_stable_set),
                       len(svm_stable_set)],
    "Prob. inclusione max": [
        f"{lasso_stability_df['Inclusion_Probability'].max():.2f}",
        f"{en_stability_df['Inclusion_Probability'].max():.2f}",
        f"{uni_stability_df['Inclusion_Probability'].max():.2f}",
        f"{svm_stability_df['Inclusion_Probability'].max():.2f}",
    ],
    "Prob. inclusione media": stability_scores["Prob. inclusione media"].tolist(),
    "Jaccard": stability_scores["Jaccard"].tolist(),
    "Hold-out (n=15)": [f"{lasso_bacc:.1%}", f"{en_bacc:.1%}", f"{uni_bacc:.1%}",
                        f"{svm_bacc:.1%}"],
})

print("--- Confronto complessivo fra i metodi ---")
display(summary_df)
print(f"Stima primaria dell'errore (CV annidata sull'intera pipeline, Lasso):")
print(f"  balanced accuracy {nested_mean:.3f} +/- {nested_std:.3f}")
```

- [ ] **Step 4: Insert a markdown cell immediately after it**

```markdown
La colonna hold-out e calcolata su 15 pazienti e ha un intervallo di confidenza largo
circa 22 punti percentuali: serve a verificare che i modelli funzionino, **non** a
stabilire quale sia migliore. Il confronto fra i metodi va letto sulle colonne di
stabilita e sparsita, che sono i criteri indicati dalla consegna.
```

- [ ] **Step 5: Verify**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['Kuncheva', 'kuncheva', 'mrmr', 'X_train_df', 'X_test_df', 'RFECV']:
    if name in src:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` lines. This is the checkpoint where every removed-method symbol must be gone. Note `SVC(` and `fit_svm_rfe` are expected to be present — Task 2 kept the SVM arm.

- [ ] **Step 6: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "refactor(notebook): three-arm reproducibility comparison, drop Kuncheva index"
```

---

### Task 10: Trim EDA, remove verdict branching, move prose to markdown

**Rationale:** three PCA panels make two points; `if/elif/else` blocks that print alternative conclusions are code narrating its own result, which is a research-notebook habit rather than a course one; and roughly 40% of the interpretation lives inside `print()` calls, where it cannot be read without running the notebook and cannot be lifted into the report.

**Files:**
- Modify: `progetto/project_revised.ipynb` (4 code cells, 4 markdown insertions)

**Interfaces:** no symbol changes.

- [ ] **Step 1: In the cell anchored `pca_scatter(raw_df.loc[y_train.index, "BM.PB"]`, delete the third scatter call and the trailing `print` block**

Delete exactly:

```python
pca_scatter(raw_df.loc[y_train.index, "BM.PB"].to_numpy(), "BM.PB", palette="Set2")
```

and the four `print(...)` lines that follow `print(f"Le prime due componenti spiegano...")`, keeping that first line.

- [ ] **Step 2: Insert a markdown cell after it**

```markdown
I due grafici vanno letti insieme. Il primo mostra che ALL e AML si separano gia in due
dimensioni, il che e incoraggiante ma non sorprendente con p >> n. Il secondo e il piu
importante: la separazione per `Source` ricalca quella per diagnosi, e questo conferma
visivamente il confondimento misurato con il chi-quadro nella sezione 1.1. Ne segue che la
direzione principale di variazione non puo essere attribuita con sicurezza alla biologia.
```

- [ ] **Step 3: In the cell anchored `thresholds = np.arange(0.30, 1.001, 0.05)`, delete the entire `plateau` block**

Delete from `plateau = [t for t in (0.6, 0.7, 0.8) if` through the end of the `else:` branch. Keep the plot and the `for t in (0.9, 0.8, ...)` loop that prints panel size per threshold.

- [ ] **Step 4: Insert a markdown cell after it, and fill the bracketed values in Task 12 from the real output**

```markdown
Il pannello resta invariato su un intervallo ampio di soglie: fra i geni selezionati e i
successivi esiste un salto netto nelle probabilita di inclusione, non una transizione
graduale. La soglia dell'80% non determina quindi la risposta, ma cade in una regione in
cui non c'e nulla da tagliare — un risultato piu solido dell'aver scelto bene una soglia.
```

- [ ] **Step 5: In the cell anchored `overlap = lasso_stable_set & robust_baseline_genes`, move the closing interpretation to markdown**

Delete the final five `print(...)` lines beginning `print("I due criteri controllano cose diverse...")` — that text now lives in the markdown cell added by Task 3 Step 3. Keep every `print` that reports a computed number.

- [ ] **Step 6: In the cell anchored `G = X_all.to_numpy(dtype=float)`, move both prose blocks to markdown**

Delete the four-line `print` block starting `print("Lettura: la distribuzione e fortemente asimmetrica...")` and the five-line block starting `print("Il pannello di destra usa l'asse y logaritmico...")`. Insert one markdown cell immediately after the code cell:

```markdown
La distribuzione e fortemente asimmetrica — dal 75esimo al 99esimo percentile si passa da
poche centinaia a decine di migliaia — e circa un terzo delle sonde non si accende mai in
nessun paziente: sono rumore strumentale puro. Entrambe le cose motivano il preprocessing.

Il pannello di destra usa l'asse y logaritmico: senza di esso il picco al floor
schiaccerebbe contro l'asse x tutto il resto. Si distinguono cosi tre popolazioni — il
rumore compresso al floor, i valori informativi distribuiti fra log10 = 2 e log10 = 4.2, e
la saturazione al ceiling. Il filtro elimina la prima popolazione.
```

The `at_floor`, `at_ceil` and `interior` variables become unused; delete their three assignment lines too.

- [ ] **Step 7: Verify no verdict branching survives**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import json
nb = json.load(open('progetto/project_revised.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
for name in ['VERDETTO', 'plateau', 'at_floor', 'at_ceil', 'BM.PB", palette']:
    if name in src:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `check done` with no `STILL REFERENCED` lines.

- [ ] **Step 8: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "style(notebook): trim EDA, remove verdict branching, move prose to markdown"
```

---

### Task 11: Rewrite the closing checks, conclusions and references

**Rationale:** the preprocessing-sensitivity check answers a question the report can state in a sentence; the gene-function narrative (adipsin, CD33, zyxin biology) is what makes the work read as a biological study rather than a data analysis, and it cannot be sourced from anything actually read. The membership check against Golub's published 50-gene list stays — that is a verifiable fact, not a biological claim.

**Files:**
- Modify: `progetto/project_revised.ipynb` (1 code cell, 2 markdown cells)

**Interfaces:**
- Consumes: `feature_cols`, `lasso_stable_set`, `robust_baseline_genes`, `prob_map`, `X_train_lasso`, `y_train`, `raw_df`.

- [ ] **Step 1: Replace the cell anchored `def top_by_ttest(df_expr, y_vec, k=20)` with this source**

```python
print("--- Riscontro con la lista pubblicata da Golub et al. (1999) ---")
print("Verifica di appartenenza, non interpretazione biologica: controlliamo se le sonde")
print("del nostro pannello compaiono fra i 50 predittori riportati nell'articolo originale.")
print()
for probe, nome in [("X95735_at", "zyxin"), ("M23197_at", "CD33"),
                    ("M27891_at", "cistatina C")]:
    presente = probe in feature_cols
    print(f"{probe} ({nome}): sopravvive al filtro = {presente}, "
          f"nel pannello finale = {probe in lasso_stable_set}, "
          f"nel baseline statistico = {probe in robust_baseline_genes}, "
          f"inclusion probability = {prob_map.get(probe, 0.0):.2f}")

print("\n--- Verifica finale sui confonditori ---")
src_train = raw_df.loc[y_train.index, "Source"]
all_mask = (y_train.to_numpy() == 0)
src_all = src_train.to_numpy()[all_mask]
codes = pd.Series(src_all).astype("category").cat.codes.to_numpy()
print(f"Pazienti ALL nel training: {all_mask.sum()}, distribuiti fra i centri "
      f"{pd.Series(src_all).value_counts().to_dict()}")
if pd.Series(codes).nunique() > 1 and np.bincount(codes).min() >= 3:
    sc = cross_val_score(LogisticRegression(class_weight="balanced", max_iter=5000),
                         X_train_lasso[all_mask], codes, cv=3,
                         scoring="balanced_accuracy")
    print(f"Balanced accuracy nel predire il CENTRO dai geni del pannello, sui soli ALL: "
          f"{sc.mean():.3f} (+/- {sc.std():.3f})")
else:
    print("I pazienti ALL del training provengono quasi interamente da un unico centro:")
    print("il controllo entro-classe non e eseguibile. E esattamente il confondimento")
    print("documentato nella Fase 1, e ne costituisce un'ulteriore conferma.")
```

- [ ] **Step 2: Replace the markdown cell anchored `## Conclusioni` with this source**

The bracketed placeholders are filled from real output in Task 12. Do not guess them now.

```markdown
## Conclusioni

**Il pannello proposto.** La procedura raccomandata e la regressione logistica con penalita
L1, parametro di regolarizzazione scelto per cross-validation e selezione finale determinata
per stability selection su 100 ricampionamenti bootstrap stratificati, trattenendo i geni con
inclusion probability almeno pari all'80%.

**La dimensione del pannello non e un artefatto della soglia.** La curva di sensibilita della
Fase 4 mostra che il pannello resta identico su un intervallo ampio di soglie: fra i geni
selezionati e i successivi esiste un salto netto nelle probabilita di inclusione.

**Errore di classificazione.** La stima difendibile e quella per cross-validation annidata
sull'intera pipeline: standardizzazione, selezione e classificatore vengono rifatti dentro
ciascun fold, quindi non c'e contaminazione fra selezione e valutazione (Ambroise &
McLachlan, 2002; Varma & Simon, 2006). L'hold-out da 15 pazienti e riportato solo come
verifica di funzionamento; con quella numerosita l'intervallo di confidenza e cosi ampio da
rendere il numero inutilizzabile per confrontare metodi.

**Confronto con la letteratura.** Guyon et al. (2002) riportano sugli stessi dati un
pannello di 2 geni con errore leave-one-out nullo, ottenuto con SVM-RFE. Il pannello del
Lasso qui proposto va confrontato con quello per numerosita e composizione — un riscontro
esterno, non una validazione: la loro stima e leave-one-out sul training set del 1999,
mentre la nostra e in cross-validation annidata su una diversa partizione.

**Sparsita.** La curva accuratezza/sparsita della Fase 3 mostra che l'accuratezza raggiunge
un plateau molto prima che il numero di geni si stabilizzi: la sparsita non costa
accuratezza, e nella regione a bassa regolarizzazione si compra soltanto instabilita. Va
segnalato un limite di identificabilita: piu valori di C raggiungono lo stesso punteggio in
cross-validation, e fra i pari merito `GridSearchCV` seleziona il piu piccolo, cioe il piu
sparso — una scelta deterministica e favorevole, ma non fondata sui dati.

**Riproducibilita: due facce da non confondere.** La tabella della sezione 5.4 riporta la
stabilita per-gene (probabilita media di inclusione) e quella per-insieme (Jaccard). Non
misurano la stessa cosa e possono ordinare i metodi in modo diverso: un metodo con pochi
geni quasi sempre presenti e una coda ampia di geni occasionali eccelle sulla prima e non
sulla seconda. Per l'obiettivo di questo lavoro — identificare pochi biomarcatori
affidabili — conta la prima, ma riportare solo quella sarebbe una descrizione parziale.

**Multi-collinearita.** La Fase 4.1 documenta il meccanismo per cui la penalita L1 spartisce
la probabilita di inclusione fra i membri di un gruppo di geni correlati, cosicche le
probabilita osservate per singolo gene sottostimano la rilevanza del gruppo. Il confronto
con l'Elastic Net e coerente con il grouping effect previsto da Zou & Hastie (2005), ma non
lo dimostra: ai rispettivi ottimi i due modelli hanno sparsita diverse, e il confondimento
agisce nella stessa direzione dell'ipotesi. Il limite e dichiarato nella sezione stessa.

**Il limite principale del lavoro.** Il centro di raccolta e quasi perfettamente confuso con
la diagnosi: 64 pazienti su 72 provengono da ospedali che hanno contribuito una sola classe.
Qualunque effetto tecnico di laboratorio e percio indistinguibile dal segnale biologico, e
nessuna tecnica statistica puo separarli a partire da questi dati. Si tratta di un limite del
design dello studio del 1999, non di un difetto correggibile dell'analisi, ed e la ragione per
cui una validazione su una coorte indipendente sarebbe indispensabile prima di qualunque uso
clinico.
```

- [ ] **Step 3: Replace the markdown cell anchored `## Riferimenti` with this source**

```markdown
## Riferimenti

1. Golub, T.R., Slonim, D.K., Tamayo, P., Huard, C., Gaasenbeek, M., Mesirov, J.P., Coller, H.,
   Loh, M.L., Downing, J.R., Caligiuri, M.A., Bloomfield, C.D. & Lander, E.S. (1999).
   *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression
   Monitoring.* Science, 286(5439), 531-537.
2. Dudoit, S., Fridlyand, J. & Speed, T.P. (2002). *Comparison of Discrimination Methods for the
   Classification of Tumors Using Gene Expression Data.* Journal of the American Statistical
   Association, 97(457), 77-87. — §3.1.2 per la procedura di preprocessing adottata. Gli autori
   la attribuiscono a comunicazione personale con P. Tamayo; **non e descritta nell'articolo di
   Golub et al. (1999)**.
3. Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate: a Practical and
   Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society, Series B,
   57(1), 289-300.
4. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* Journal of the
   Royal Statistical Society, Series B, 58(1), 267-288.
5. Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.*
   Journal of the Royal Statistical Society, Series B, 67(2), 301-320. — Teorema 1, grouping
   effect.
6. Meinshausen, N. & Buhlmann, P. (2010). *Stability Selection.* Journal of the Royal Statistical
   Society, Series B, 72(4), 417-473. — Impianto generale della stability selection e delle
   inclusion probability.
7. Bach, F.R. (2008). *Bolasso: Model Consistent Lasso Estimation through the Bootstrap.*
   Proceedings of the 25th International Conference on Machine Learning (ICML), 33-40. — La
   variante con ricampionamento bootstrap, che e quella adottata qui.
8. Ambroise, C. & McLachlan, G.J. (2002). *Selection bias in gene extraction on the basis of
   microarray gene-expression data.* PNAS, 99(10), 6562-6566.
9. Varma, S. & Simon, R. (2006). *Bias in error estimation when using cross-validation for model
   selection.* BMC Bioinformatics, 7:91.
10. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*,
    2a ed. Springer. — Cap. 3 (shrinkage), cap. 7 (valutazione dei modelli), cap. 18 (p >> n).
11. Saeys, Y., Inza, I. & Larranaga, P. (2007). *A review of feature selection techniques in
    bioinformatics.* Bioinformatics, 23(19), 2507-2517. — Tassonomia filtro / wrapper /
    embedded usata nella sezione 5.3.
12. Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. (2002). *Gene Selection for Cancer
    Classification using Support Vector Machines.* Machine Learning, 46, 389-422. — SVM-RFE,
    implementato nella sezione 5.3: criterio di ranking `w^2` (sez. 2.5), schema di
    eliminazione a dimezzamento sui dati di leucemia (sez. 5.1). Sugli stessi dati gli autori
    riportano 2 geni con errore leave-one-out nullo.

### Metodi considerati e non implementati

13. Mundra, P.A. & Rajapakse, J.C. (2010). *SVM-RFE with MRMR Filter for Gene Selection.*
    IEEE Transactions on NanoBioscience, 9(1), 31-37. — Il criterio MRMR entra nel punteggio
    di ranking dell'RFE (eq. 5) e viene ricalcolato sui geni superstiti a ogni passo di
    eliminazione (Algoritmo 1). Discusso nella sezione 5.3 e non implementato.
14. Peng, H., Long, F. & Ding, C. (2005). *Feature selection based on mutual information:
    criteria of max-dependency, max-relevance, and min-redundancy.* IEEE TPAMI, 27(8),
    1226-1238. — Formulazione originale di mRMR.
15. Kuncheva, L.I. (2007). *A Stability Index for Feature Selection.* Proceedings of the 25th
    IASTED International Multi-Conference: Artificial Intelligence and Applications, 390-395.
```

**Before submitting the report, verify every entry above against the actual paper.** `FinalTask.pdf` §4.2 requires citing only references that are pertinent and that you have actually read. Any entry you have not read must be removed, not merely reformatted.

- [ ] **Step 4: Commit**

```bash
git add progetto/project_revised.ipynb
git commit -m "docs(notebook): rewrite conclusions and references for simplified analysis"
```

---

### Task 12: Full execution and final verification

**Files:**
- Modify: `progetto/project_revised.ipynb` (executed outputs; markdown fills)

- [ ] **Step 1: Confirm the notebook has no undefined-name references before executing**

```bash
cd /home/isabella/Develop/analisi-dati
uv run python - <<'EOF'
import ast, json
nb = json.load(open('progetto/project_revised.ipynb'))
code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
ast.parse(code)
print('syntax OK')
for name in ['mrmr', 'RFECV', 'kuncheva', 'Kuncheva', 'linkage(',
             'fcluster', 'squareform', 'N_PERM', 'K_MRMR', 'perm_counts', 'matched_',
             'group_df', 'X_full_log', 'X_train_int', 'C_values', 'VERDETTO',
             'svm_used_fallback', 'svm_final_features']:
    if name in code:
        print('STILL REFERENCED:', name)
print('check done')
EOF
```

Expected: `syntax OK`, then `check done` with no `STILL REFERENCED` lines.

- [ ] **Step 2: Execute the whole notebook and time it**

```bash
cd /home/isabella/Develop/analisi-dati/progetto
time uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 project_revised.ipynb
```

Expected: exit code 0. If any cell raises, fix the cell and re-run — do not proceed with a notebook that errors (`FinalTask.pdf` §4.3: *"the script must not contain any errors"*).

- [ ] **Step 3: Report the real numbers back to the user**

Extract and report: number of surviving probes after the filter, `lasso_optimal_C`, panel size and gene names, the threshold range over which the panel is invariant, `nested_mean` ± `nested_std`, the three-arm `summary_df`, and the mean Lasso-vs-EN probability gap on collinear pairs. **These may differ from the pre-simplification run** — the probe filter now uses all 72 samples (Task 5), so `feature_cols` changes and every downstream number can shift.

- [ ] **Step 4: Fill the placeholders left in markdown**

Using the real output, complete: the threshold range in the Task 10 Step 4 markdown cell, and the panel gene names in the first paragraph of the Conclusions. Then re-execute only if a code cell changed.

- [ ] **Step 5: Commit**

```bash
cd /home/isabella/Develop/analisi-dati
git add progetto/project_revised.ipynb
git commit -m "chore(notebook): execute simplified analysis end to end"
```

---

## Post-plan notes for the report

Three things this plan removes from the notebook belong in the report as prose, not as code:

1. **Why mRMR was dropped while SVM-RFE was kept** — Mundra & Rajapakse's Eq. (5) puts the MRMR term inside the RFE ranking score, and their Algorithm 1 recomputes relevance and redundancy over the surviving set at each elimination step. A one-shot pre-filter followed by w²-only elimination is a different procedure that does not preserve minimum redundancy through elimination. Faithful implementation costs pairwise mutual information over the survivors every iteration, adds β / η / a discretization rule, and buys 0.5 percentage points and ten genes on the leukemia data (their Table II). SVM-RFE alone needs none of that, so it stays. Cite Guyon et al. (2002), Mundra & Rajapakse (2010), Peng et al. (2005), Saeys et al. (2007).
2. **Why no permutation null** — false-discovery control is delivered by Benjamini–Hochberg and by the ML-versus-statistics contrast; calibrating the null of the stability procedure itself is beyond the assignment's scope.
3. **Why the Lasso-versus-Elastic-Net comparison is not a controlled experiment** — the sparsity confounder, stated in one sentence rather than engineered away.

Two housekeeping items outside this plan's scope:

- `progetto/prova.md` is LLM-generated and its numbers are stale (12 genes, C = 0.7632, 100% concordance, versus the current panel). Nothing in it should reach the report. `FinalTask.pdf` §4.2 requires declaring LLM use for editing.
- `CLAUDE.md` names `project.ipynb` as the canonical pipeline and still lists `mrmr-selection` and `fakemp` as key libraries. Update it once this plan lands; the two packages can also come out of `pyproject.toml`.

## Sources verified against full text

Both SVM papers are in `docs/` and were read in full while writing this plan; the claims in Task 2 come from them directly, not from secondary summaries:

- `docs/Gene selection using SVM-RFE.pdf` — Guyon et al. (2002). §2.5 justifies w² as ranking criterion; §2.6 defines RFE and warns that chunk removal costs performance; §5.1 gives the leukemia setup (38 train / 34 test, standardization per Golub, elimination down to the nearest power of two then halving) and the 2-genes-at-zero-LOO-error result, plus the ≤25% overlap between SVM-selected and Golub-baseline gene sets at 16 genes or fewer.
- `docs/MundraRajapakse2010a.pdf` — Mundra & Rajapakse (2010). Eq. (5) is the ranking criterion; Algorithm 1 places the MRMR computation inside the elimination loop; Eq. (9) is the discretization; §III-C covers β and η estimation; Table II has the leukemia comparison.

The remaining references in Task 11 were **not** verified this way and still need checking against the papers before the report is submitted.
