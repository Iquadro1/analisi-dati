# Revisione di `progetto/project.ipynb` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Portare `progetto/project.ipynb` a essere eseguibile end-to-end senza warning, metodologicamente difendibile, e conforme a tutti i requisiti espliciti del `FinalTask.pdf`.

**Architecture:** Il notebook resta un unico documento lineare in 5 fasi (prep/EDA → filtraggio statistico → selezione ML → stabilità bootstrap → sintesi). Non si riscrive da zero: si applicano modifiche chirurgiche cella per cella, si aggiungono le celle mancanti per i requisiti scoperti (collinearità, confonditori, score di stabilità, null da permutazione), e si sostituiscono i markdown — che oggi sono il testo della consegna — con conclusioni interpretative. Lo split train/test 57/15 con hold-out viene **mantenuto**, ma declassato a sanity check: la stima primaria dell'errore diventa una CV annidata sull'intera pipeline.

**Tech Stack:** Python 3.14, `uv`, scikit-learn 1.9.0, pandas 3.0.3, seaborn 0.13.2, statsmodels, scipy, `mrmr-selection`, `matplotlib-venn`.

## Global Constraints

- **API scikit-learn 1.9:** il parametro `penalty` di `LogisticRegression` è **deprecato in 1.8 e rimosso in 1.10**. Il tipo di penalizzazione si specifica **solo** con `l1_ratio`: `l1_ratio=0.0` → L2, `l1_ratio=1.0` → L1, `0 < l1_ratio < 1` → elastic net, `C=np.inf` → nessuna penalità. **Non passare mai `penalty`** in nessuna cella. `l1_ratio` ha default `0.0` (cambiato da `None` in 1.8).
- **Nessun leakage:** ogni `fit` di scaler, filtro, test statistico, selettore o modello avviene esclusivamente su `X_train` / `y_train` (o su un fold di training interno). Il test set da 15 campioni si tocca solo in fase di valutazione finale.
- **Metrica unica:** `balanced_accuracy` ovunque (GridSearchCV, RFECV, cross_val_score). Non mischiare `f1` e `neg_log_loss` come oggi.
- **Class weight:** `class_weight='balanced'` su ogni stimatore (47 ALL / 25 AML complessivi, 37/20 nel training).
- **Seed:** ogni sorgente di casualità ha un `random_state` esplicito. La consegna valuta la riproducibilità.
- **Split:** `test_size=0.2, random_state=43, stratify=y` → 57 train (37 ALL / 20 AML), 15 test (10 ALL / 5 AML).
- **Riferimento preprocessing:** Dudoit, S., Fridlyand, J. & Speed, T.P. (2002), *JASA* 97(457), 77–87, §3.1.2 pp. 80–81. Le soglie **non** sono nel paper di Golub et al. (1999): Dudoit le attribuisce a comunicazione personale con P. Tamayo. Citare Dudoit, non Golub, per questo punto.
- **Ambiente:** eseguire con `uv run jupyter lab` oppure `uv run jupyter nbconvert --execute --inplace progetto/project.ipynb`.
- **Fuori scope (deciso dall'utente):** non si costruisce una pipeline esportabile per un "hidden test set" esterno; la formulazione della consegna su quel punto è considerata mal scritta.

## Riferimento celle (numerazione iniziale, 1-based)

| # | tipo | contenuto |
|---|---|---|
| 1, 8, 12, 15, 18 | markdown | testo della consegna (da riscrivere, Task 11) |
| 2 | code | import |
| 3 | code | load + target binario |
| 4 | code | grafici distribuzione target |
| 5 | code | train_test_split |
| 6 | code | StandardScaler |
| 7 | code | PCA scores + loadings |
| 9 | code | t-test Welch + FDR (**duplicato della 10**) |
| 10 | code | t-test + Mann-Whitney + FDR + `robust_baseline_genes` |
| 11 | code | volcano + Venn t-test/MW |
| 13 | code | GridSearchCV Lasso |
| 14 | code | **tutta commentata** |
| 16 | code | bootstrap stabilità Lasso |
| 17 | code | bar chart stabilità + regularization path |
| 19 | code | modello finale 12 geni + concordanza |
| 20 | code | Venn ML vs baseline statistico |
| 21 | code | `y_train` (debug) |
| 22 | code | superata dalla 23 |
| 23 | code | mRMR + SVM-RFE |
| 24 | code | bootstrap stabilità SVM-RFE |
| 25 | code | ElasticNet + bootstrap |
| 26 | code | drop-off plot + Venn3 + tabella riassuntiva |
| 27 | code | heatmap coefficienti EN vs Lasso |
| 28 | code | deep dive fallimento SVM-RFE |
| 29, 30, 31 | code | **tutte commentate** |

**Attenzione:** il Task 1 cancella delle celle, quindi la numerazione cambia. Dal Task 2 in poi le celle si identificano dal loro contenuto (es. "la cella che contiene `mrmr_classif`"), non dal numero.

---

### Task 1: Pulizia, collisioni di nomi e baseline eseguibile

Obiettivo: ottenere un notebook che giri linearmente dall'alto in basso senza errori, prima di cambiare qualsiasi risultato numerico. Nessun numero deve cambiare in questo task.

**Files:**
- Modify: `progetto/project.ipynb`

**Interfaces:**
- Produces: notebook con celle rinominate secondo lo schema `lasso_*` / `svm_*` / `en_*`, usato da tutti i task successivi. Nomi canonici: `lasso_grid`, `lasso_best`, `lasso_optimal_C`, `lasso_stability_df`, `lasso_stable_set`, `lasso_final_genes`, `lasso_acc`, `svm_stability_df`, `svm_stable_genes`, `svm_final_features`, `svm_acc`, `en_stability_df`, `en_stable_set`, `en_acc`, `robust_baseline_genes`.

- [ ] **Step 1: Cancellare le celle morte**

Cancellare integralmente le celle 14, 21, 22, 29, 30, 31 (le 14/29/30/31 sono commentate al 100%, la 21 è un `y_train` di debug, la 22 è superata dalla 23). Nella cella 4 cancellare i due blocchi commentati dopo `plot_target_distribution_stacked(y, raw_df["Gender"], "Gender")` (circa 35 righe di duplicati).

- [ ] **Step 2: Eliminare la cella 9 (duplicato)**

La cella 10 rifà lo stesso t-test di Welch della cella 9 e ne è un superset (aggiunge Mann-Whitney e `robust_baseline_genes`). Cancellare la cella 9. Nella cella 10 rimuovere le prime quattro righe di import ridondanti (`numpy`, `pandas`, `scipy.stats`, `statsmodels`), già presenti nella cella 2.

Verificare che nessuna cella successiva usi variabili definite solo dalla 9: `ttest_results`, `significant_genes`, `reject_null`, `pvals_corrected`. La cella 11 usa `t_stats` e `pvals_corrected_t`, entrambe definite nella 10 — OK.

- [ ] **Step 3: Rinominare le variabili per eliminare le collisioni**

Sono quattro collisioni reali. La più pericolosa è `grid_search`: la cella 25 (ElasticNet) lo riassegna, e la sua griglia contiene **anch'essa** una chiave `'C'`, quindi rieseguire la cella 16 dopo la 25 fa prendere il C dell'ElasticNet **senza sollevare alcun errore**.

| vecchio nome | dove | nuovo nome |
|---|---|---|
| `grid_search` (cella 13) | Lasso | `lasso_grid` |
| `grid_search` (cella 25) | ElasticNet | `en_grid` |
| `best_lasso` | cella 13 | `lasso_best` |
| `optimal_C` | celle 16, 17 | `lasso_optimal_C` |
| `stability_df` | celle 16, 17, 26 | `lasso_stability_df` |
| `final_ml_set` | celle 16, 17, 20, 26 | `lasso_stable_set` |
| `highly_stable_genes` | celle 16, 19 | `lasso_stable_df` |
| `final_genes` | celle 19, 26 | `lasso_final_genes` |
| `final_model` (cella 19) | Lasso 12 geni | `lasso_final_model` |
| `final_model` (cella 23) | SVM | `svm_rfe_model` |
| `X_train_final`, `X_test_final` (cella 19) | Lasso | `X_train_lasso`, `X_test_lasso` |
| `X_train_final`, `X_test_final` (cella 23) | SVM | `X_train_svm`, `X_test_svm` |
| `final_acc`, `final_f1` | cella 19 | `lasso_acc`, `lasso_f1` |
| `acc`, `f1` | cella 25 | `en_acc`, `en_f1` |
| `y_pred` | celle 13, 23, 25 | `lasso_y_pred`, `svm_y_pred`, `en_y_pred` |
| `threshold` | celle 17, 25, 28 | `STABILITY_THRESHOLD` (costante unica, vedi Step 4) |

Propagare i rinomini in **tutte** le celle che leggono queste variabili, inclusa la tabella riassuntiva della cella 26 e la heatmap della cella 27 (che usa `best_lasso.coef_[0]`).

- [ ] **Step 4: Introdurre le costanti globali nella cella 2**

Aggiungere in fondo alla cella degli import:

```python
# --- Costanti globali ---
RANDOM_STATE = 43
N_BOOTSTRAP = 100
STABILITY_THRESHOLD = 0.80
K_MRMR = 100
FDR_ALPHA = 0.05
```

Sostituire i letterali corrispondenti nelle celle (`B = 100`, `n_iterations = 100`, `threshold = 0.80`, `K_MRMR = 100`, `alpha=0.05`).

- [ ] **Step 5: Correggere i nomi delle feature in mRMR (cella 23)**

Tre chiamate `pd.DataFrame(X_train)` / `pd.DataFrame(X_test)` non passano `columns`, quindi le colonne diventano interi `0..7128` e `optimal_genes` contiene **indici numerici invece di nomi di geni** — la riga `print("Final selected genes:", list(optimal_genes))` stampa una lista di numeri inutilizzabile nel report. La cella 24 lo fa già correttamente.

```python
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
y_train_series = pd.Series(np.asarray(y_train))
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
```

- [ ] **Step 6: Rendere robusta la cella 25 (`en_acc` può non esistere)**

`en_acc` è definito dentro `if num_stable_genes > 0:`. Se l'ElasticNet non produce geni stabili, la cella 26 muore con `NameError` a fine notebook. Inizializzare prima dell'`if`:

```python
en_acc, en_f1 = np.nan, np.nan
```

- [ ] **Step 7: Correggere il drop-off plot (cella 26)**

`ranks = range(1, 51)` è hardcoded: se un `stability_df` avesse meno di 50 righe, `plt.plot` fallisce per lunghezze diverse.

```python
def plot_dropoff(ax, df, label, color, marker):
    n = min(50, len(df))
    ax.plot(range(1, n + 1), df['Inclusion_Probability'].head(n).to_numpy(),
            label=label, color=color, linewidth=2.5, marker=marker, markersize=4)

fig, ax = plt.subplots(figsize=(10, 6))
plot_dropoff(ax, lasso_stability_df, 'Lasso (L1)', 'crimson', 'o')
plot_dropoff(ax, en_stability_df, 'Elastic Net (L1 + L2)', 'darkorange', 's')
plot_dropoff(ax, svm_stability_df, 'mRMR + SVM-RFE', 'steelblue', '^')
ax.axhline(STABILITY_THRESHOLD, color='black', linestyle='--', linewidth=1.5,
           label=f'{STABILITY_THRESHOLD:.0%} Stability Threshold')
```

Nota per il report: `lasso_stability_df` e `svm_stability_df` contengono **solo i geni selezionati almeno una volta**, mentre `en_stability_df` contiene tutti e 7129 con gli zeri. Sui primi 50 non fa differenza, ma non calcolare statistiche aggregate confrontando i tre così come sono.

- [ ] **Step 8: Correggere il FutureWarning di seaborn (cella 28)**

```python
ax = sns.barplot(
    data=top_svm_genes,
    x='Inclusion_Probability',
    y='Gene_Probe',
    hue='Gene_Probe',
    palette='viridis',
    legend=False,
)
```

- [ ] **Step 9: Annotare il fallback SVM-RFE (cella 24 → tabella cella 26)**

Quando nessun gene supera l'80%, il codice ripiega sui top-10 e la tabella riassuntiva mostra `Sparsity (Final Genes) = 10` accanto ai 12 del Lasso, come se fossero comparabili. Aggiungere una colonna esplicita:

```python
summary_data['Selection Rule'] = [
    f'>= {STABILITY_THRESHOLD:.0%} inclusion',
    f'>= {STABILITY_THRESHOLD:.0%} inclusion',
    'FALLBACK: top-10 (nessun gene ha raggiunto la soglia)'
    if len(svm_stable_genes) == 0 else f'>= {STABILITY_THRESHOLD:.0%} inclusion',
]
```

- [ ] **Step 10: Restart & Run All e verifica baseline**

```bash
cd /home/unipi/i.inuso/Develop/analisi-dati
uv run jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=3600 progetto/project.ipynb
```

Atteso: esecuzione completa senza eccezioni. Verificare che gli `execution_count` siano ora consecutivi da 1 in poi (prima erano fuori ordine: import a 32, celle 3-10 a 14-20, cella 21 a 15 — prova che il notebook non era mai stato eseguito linearmente).

I numeri devono restare quelli di prima: 663 geni significativi al t-test, ~586 nel baseline robusto, 92 geni Lasso, 829 selezionati almeno una volta, 12 stabili, 24 stabili per ElasticNet, 0 stabili per SVM-RFE. **Se un numero cambia, fermarsi e capire perché prima di procedere.**

- [ ] **Step 11: Commit**

```bash
git add progetto/project.ipynb
git commit -m "refactor(progetto): pulizia celle morte, collisioni di nomi, notebook eseguibile linearmente"
```

---

### Task 2: Migrazione all'API scikit-learn 1.9

Obiettivo: eliminare l'uso di `penalty`, deprecato in 1.8 e **rimosso in 1.10**. Nella cella del regularization path c'è un conflitto reale che diventerà un bug silenzioso.

**Files:**
- Modify: `progetto/project.ipynb` (celle: regularization path, ElasticNet GridSearchCV, ElasticNet bootstrap, heatmap coefficienti, eval_classifier)

**Interfaces:**
- Consumes: nomi canonici del Task 1.
- Produces: nessun `penalty=` residuo nel notebook.

- [ ] **Step 1: Verificare lo stato attuale**

```bash
cd /home/unipi/i.inuso/Develop/analisi-dati
uv run python -c "
import json
nb=json.load(open('progetto/project.ipynb'))
for i,c in enumerate(nb['cells'],1):
    for ln in ''.join(c['source']).split('\n'):
        if 'penalty' in ln and not ln.strip().startswith('#'):
            print(i, ln.strip())
"
```

Atteso: righe nelle celle del regularization path (`penalty='l1'`), ElasticNet grid (`penalty='elasticnet'`), ElasticNet bootstrap (`penalty='elasticnet'`), `eval_classifier` (`penalty='l2'`), heatmap (`penalty='elasticnet'`).

- [ ] **Step 2: Correggere il regularization path — questo è il fix urgente**

Il codice attuale passa `penalty='l1'` mentre `l1_ratio` resta al suo default `0.0`. sklearn 1.9 emette **due** warning: il `FutureWarning` di deprecazione e `UserWarning: Inconsistent values: penalty=l1 with l1_ratio=0.0`. In 1.10, quando `penalty` sparirà, questa cella diventerà **silenziosamente ridge** — nessun coefficiente esattamente zero — e il regularization path perderà ogni senso senza sollevare errori.

```python
for c in C_values:
    clf = LogisticRegression(
        l1_ratio=1.0,
        solver='saga',
        C=c,
        class_weight='balanced',
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    coef_path.append(clf.coef_[0])
```

Nota: si passa da `liblinear` a `saga`. Il path sarà leggermente diverso ma finalmente coerente con le celle Lasso, che già usano `saga`.

- [ ] **Step 3: Correggere le tre chiamate ElasticNet**

In sklearn 1.9 l'elastic net si specifica **solo** con `l1_ratio` strettamente fra 0 e 1. Rimuovere `penalty='elasticnet'` da: lo stimatore base del GridSearchCV, il modello dentro il loop bootstrap, e il `final_en_model` della heatmap. Tenere `solver='saga'` (obbligatorio per `0 < l1_ratio < 1`) e `l1_ratio` (nella griglia o dal `best_params_`).

```python
en_base_model = LogisticRegression(
    solver='saga',
    class_weight='balanced',
    max_iter=5000,
    random_state=RANDOM_STATE,
)
```

- [ ] **Step 4: Correggere `eval_classifier`**

```python
eval_classifier = LogisticRegression(
    C=1e5, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE
)
```

`l1_ratio=0.0` è già il default, quindi si può omettere: è ridge, che è ciò che serve per valutare il potere predittivo grezzo dei geni stabili.

- [ ] **Step 5: Verificare che non resti nessun `penalty` e nessun warning**

```bash
uv run python -c "
import json
nb=json.load(open('progetto/project.ipynb'))
hits=[(i,ln.strip()) for i,c in enumerate(nb['cells'],1)
      for ln in ''.join(c['source']).split('\n')
      if 'penalty' in ln and not ln.strip().startswith('#')]
print('residui:', hits)
assert not hits, hits
print('OK')
"
```

Poi rieseguire il notebook e cercare warning nell'output:

```bash
uv run jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=3600 progetto/project.ipynb
uv run python -c "
import json
nb=json.load(open('progetto/project.ipynb'))
for i,c in enumerate(nb['cells'],1):
    for o in c.get('outputs',[]):
        txt=''.join(o.get('text','')) if isinstance(o.get('text'),list) else str(o.get('text',''))
        if 'FutureWarning' in txt or 'Inconsistent values' in txt or 'DeprecationWarning' in txt:
            print('cella',i,txt[:200])
"
```

Atteso: nessun output.

- [ ] **Step 6: Commit**

```bash
git add progetto/project.ipynb
git commit -m "fix(progetto): migrazione a l1_ratio, rimozione del parametro penalty deprecato"
```

---

### Task 3: Correttezza statistica della selezione

Obiettivo: stratificazione, bilanciamento delle classi, metrica unica, griglia di C sensata. **Da qui in poi i numeri cambiano**: vanno rigenerati e riletti tutti.

**Files:**
- Modify: `progetto/project.ipynb` (celle: split, Lasso grid, Lasso bootstrap, modello finale Lasso, mRMR+SVM-RFE, SVM bootstrap, ElasticNet)

**Interfaces:**
- Consumes: nomi canonici del Task 1.
- Produces: `lasso_boot_sets: list[set[str]]`, `svm_boot_sets: list[set[str]]`, `en_boot_sets: list[set[str]]` — la lista dei geni selezionati in **ciascuna** replica bootstrap, consumata dal Task 8 per l'indice di Jaccard.

- [ ] **Step 1: Split stratificato**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(y_train)} campioni ({(y_train==0).sum()} ALL / {(y_train==1).sum()} AML)")
print(f"Test:  {len(y_test)} campioni ({(y_test==0).sum()} ALL / {(y_test==1).sum()} AML)")
```

Atteso: `Train: 57 campioni (37 ALL / 20 AML)`, `Test: 15 campioni (10 ALL / 5 AML)`.

- [ ] **Step 2: Allargare la griglia di C del Lasso**

Attualmente la griglia log-spaziata è commentata e sostituita da `np.linspace(0.1, 0.8, 20)`, e l'ottimo selezionato (0.7632) cade sul **penultimo punto della griglia**: un ottimo al bordo significa intervallo troppo stretto.

```python
param_grid = {'C': np.logspace(-3, 1, 30)}

lasso_model = LogisticRegression(
    l1_ratio=1.0,
    solver='saga',
    class_weight='balanced',
    max_iter=10000,
    random_state=RANDOM_STATE,
)
lasso_grid = GridSearchCV(
    estimator=lasso_model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring='balanced_accuracy',
    n_jobs=-1,
)
lasso_grid.fit(X_train, y_train)
lasso_optimal_C = lasso_grid.best_params_['C']
assert lasso_optimal_C not in (param_grid['C'][0], param_grid['C'][-1]), \
    "Ottimo al bordo della griglia: allargare l'intervallo di C"
```

- [ ] **Step 3: Aggiungere la curva accuratezza/sparsità**

Nuova cella subito dopo il GridSearchCV. Questa curva è direttamente pertinente al criterio "sparsity" della consegna e oggi manca del tutto.

```python
cv_scores = lasso_grid.cv_results_['mean_test_score']
cv_std = lasso_grid.cv_results_['std_test_score']
C_grid = param_grid['C']

n_selected = []
for c in C_grid:
    m = LogisticRegression(l1_ratio=1.0, solver='saga', C=c, class_weight='balanced',
                           max_iter=10000, random_state=RANDOM_STATE).fit(X_train, y_train)
    n_selected.append(int((m.coef_[0] != 0).sum()))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.errorbar(C_grid, cv_scores, yerr=cv_std, color='crimson', marker='o',
             capsize=3, label='Balanced accuracy (CV 5-fold)')
ax1.set_xscale('log')
ax1.set_xlabel('C (inverso della forza di regolarizzazione, scala log)')
ax1.set_ylabel('Balanced accuracy', color='crimson')
ax1.axvline(lasso_optimal_C, color='black', linestyle='--',
            label=f'C ottimale = {lasso_optimal_C:.4f}')

ax2 = ax1.twinx()
ax2.plot(C_grid, n_selected, color='steelblue', marker='s', linestyle=':',
         label='Geni selezionati')
ax2.set_yscale('log')
ax2.set_ylabel('Numero di geni con coefficiente non nullo', color='steelblue')

fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.15))
plt.title('Trade-off accuratezza / sparsità lungo il percorso di regolarizzazione', pad=15)
plt.tight_layout()
plt.show()

print(f"Al C ottimale ({lasso_optimal_C:.4f}): {n_selected[list(C_grid).index(lasso_optimal_C)]} geni, "
      f"balanced accuracy CV = {lasso_grid.best_score_:.3f}")
```

- [ ] **Step 4: Bootstrap stratificati e raccolta dei set per replica**

Il bootstrap attuale non è stratificato: con 37/20 nel training, le repliche hanno rapporti di classe visibilmente variabili, il che aggiunge rumore proprio alle inclusion probability su cui si è valutati. Inoltre oggi il codice accumula un'unica lista piatta, e questo rende impossibile calcolare un indice di stabilità set-based (Task 8).

Applicare a **tutte e tre** le celle bootstrap (Lasso, SVM-RFE, ElasticNet) questo pattern:

```python
lasso_boot_sets = []          # NUOVO: un set per replica
all_selected_genes = []       # mantenuto per il Counter esistente

boot_lasso = LogisticRegression(
    l1_ratio=1.0, solver='saga', C=lasso_optimal_C,
    class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE,
)

for i in range(N_BOOTSTRAP):
    X_boot, y_boot = resample(
        X_train, y_train, random_state=i, stratify=y_train
    )
    boot_lasso.fit(X_boot, y_boot)
    genes = {feature_cols[idx] for idx in np.where(boot_lasso.coef_[0] != 0)[0]}
    lasso_boot_sets.append(genes)
    all_selected_genes.extend(genes)
```

Per l'ElasticNet, che usa un array di conteggi invece di un `Counter`, aggiungere comunque `en_boot_sets.append(...)` in parallelo. Per l'SVM-RFE, `svm_boot_sets.append(set(selected_in_boot))`.

- [ ] **Step 5: `class_weight='balanced'` e metrica unica ovunque**

L'SVM ce l'ha già; Lasso ed ElasticNet no, nonostante lo sbilanciamento. E la metrica oggi è incoerente (`f1` per Lasso e SVM-RFE, `neg_log_loss` per ElasticNet), il che rende la tabella comparativa un confronto fra modelli ottimizzati per obiettivi diversi. `f1` è inoltre asimmetrica rispetto alla classe minoritaria.

- Aggiungere `class_weight='balanced'` a: `lasso_model`, `boot_lasso`, `lasso_final_model`, `en_base_model`, il modello EN nel loop, `final_en_model`, `eval_classifier`.
- Sostituire `scoring='f1'` → `scoring='balanced_accuracy'` in `lasso_grid`, in `svm_tuner`, e in **entrambi** gli `RFECV`.
- Sostituire `scoring='neg_log_loss'` → `scoring='balanced_accuracy'` in `en_grid`.
- Affiancare `balanced_accuracy_score` a `accuracy_score` in tutte le valutazioni sul test set. Aggiungere l'import nella cella 2: `from sklearn.metrics import balanced_accuracy_score, roc_auc_score`.

- [ ] **Step 6: Shuffle nella CV di RFECV**

Nella cella mRMR + SVM-RFE, `cv = StratifiedKFold(n_splits=5)` non mescola i campioni.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

- [ ] **Step 7: Dichiarare il caveat mRMR**

`mrmr_classif` gira su tutto `X_train` **fuori** dalla CV di `RFECV`, quindi la curva interna che sceglie il "numero ottimale di feature" è ottimisticamente distorta. Non c'è leakage sul test set, quindi la valutazione finale resta valida. Aggiungere in testa alla cella un commento esplicito:

```python
# CAVEAT METODOLOGICO: mRMR viene fittato su tutto X_train, quindi la curva CV
# interna di RFECV (e con essa il "numero ottimale di feature") e' ottimisticamente
# distorta. Non c'e' leakage verso il test set, che resta intoccato.
# Nella cella di stability selection mRMR viene invece rifatto dentro ogni replica
# bootstrap, che e' la costruzione corretta.
```

- [ ] **Step 8: Rieseguire e registrare i nuovi numeri**

```bash
uv run jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=7200 progetto/project.ipynb
```

Annotare i nuovi valori di: C ottimale, numero di geni Lasso, geni stabili ≥80% per i tre metodi, accuratezza bilanciata sul test set. Serviranno per riscrivere i markdown nel Task 11.

- [ ] **Step 9: Commit**

```bash
git add progetto/project.ipynb
git commit -m "fix(progetto): split e bootstrap stratificati, class_weight balanced, balanced_accuracy, griglia C log-spaziata"
```

---

### Task 4: Stima onesta dell'errore con CV annidata

Obiettivo: sostituire "100% su 15 campioni" con una stima difendibile. Con n=15 l'intervallo di Wilson al 95% è circa [78%, 100%]: tre metodi che riportano tutti 100% non sono distinguibili, e la tabella riassuntiva li presenta come se lo fossero.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella dopo la valutazione del modello finale Lasso)

**Interfaces:**
- Consumes: `X`, `y`, `feature_cols`, `lasso_optimal_C`, `RANDOM_STATE`.
- Produces: `nested_scores: np.ndarray` (50 valori di balanced accuracy), `nested_mean: float`, `nested_std: float`.

- [ ] **Step 1: Aggiungere gli import nella cella 2**

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
```

- [ ] **Step 2: Costruire la pipeline e valutarla**

Il punto essenziale: **scaling e selezione stanno dentro la `Pipeline`**, quindi vengono rifatti da zero in ogni fold. È esattamente questo che rende la stima non ottimistica, ed è ciò che oggi manca.

```python
selector_estimator = LogisticRegression(
    l1_ratio=1.0, solver='saga', C=lasso_optimal_C,
    class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE,
)

nested_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('select', SelectFromModel(selector_estimator, threshold=1e-5)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=5000,
                               random_state=RANDOM_STATE)),
])

outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
nested_scores = cross_val_score(
    nested_pipe, X, y, cv=outer_cv, scoring='balanced_accuracy', n_jobs=-1
)
nested_mean, nested_std = nested_scores.mean(), nested_scores.std()
```

`threshold=1e-5` è esplicito per non dipendere dall'euristica di default di `SelectFromModel`.

- [ ] **Step 3: Riportare il risultato con l'intervallo**

```python
lo, hi = np.percentile(nested_scores, [2.5, 97.5])
print("--- Stima onesta dell'errore: CV annidata (5-fold x 10 ripetizioni) ---")
print(f"Balanced accuracy: {nested_mean:.3f} +/- {nested_std:.3f} (sd)")
print(f"Intervallo empirico 95%: [{lo:.3f}, {hi:.3f}]")
print(f"Numero di fold valutati: {len(nested_scores)}")
print()
print("Selezione e scaling sono rifatti dentro ogni fold: la stima non e' ottimistica.")
print("Confronto con l'hold-out da 15 campioni: quest'ultimo ha un IC di Wilson al 95%")
print("di circa [0.78, 1.00] anche a fronte del 100% osservato, quindi non puo'")
print("discriminare fra i tre metodi. Lo usiamo solo come sanity check.")

plt.figure(figsize=(8, 4))
sns.histplot(nested_scores, bins=12, color='crimson', alpha=0.7)
plt.axvline(nested_mean, color='black', linestyle='--',
            label=f'Media = {nested_mean:.3f}')
plt.xlabel('Balanced accuracy (fold esterno)')
plt.ylabel('Frequenza')
plt.title('Distribuzione dell\'errore in CV annidata sull\'intera pipeline', pad=15)
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Declassare l'hold-out nella tabella riassuntiva**

Nella cella della tabella, rinominare la colonna `Test Accuracy` → `Hold-out (n=15, sanity check)` e aggiungere una riga di testo sotto il `display(summary_df)`:

```python
print(f"\nStima primaria dell'errore (CV annidata, intera pipeline): "
      f"{nested_mean:.3f} +/- {nested_std:.3f} balanced accuracy.")
print("La colonna hold-out e' calcolata su 15 campioni e non e' una stima affidabile:")
print("serve solo a verificare che il modello finale non sia rotto.")
```

- [ ] **Step 5: Eseguire e verificare**

Rieseguire il notebook. Atteso: `nested_scores` ha 50 elementi, la media è plausibilmente inferiore al 100% dell'hold-out. Se venisse esattamente 1.000 su tutti i 50 fold, verificare che il selettore stia davvero selezionando dentro il fold (`nested_pipe.fit(X, y); nested_pipe['select'].get_support().sum()` deve essere ≪ 7129).

- [ ] **Step 6: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): CV annidata come stima primaria dell'errore, hold-out declassato a sanity check"
```

---

### Task 5: Preprocessing microarray

Obiettivo: i valori del CSV sono **Average Difference** di Affymetrix MAS 4.0 — differenze PM−MM, quindi i negativi sono fisiologici, non dati corrotti. Oggi non c'è alcun preprocessing: `StandardScaler` è una z-score *per gene*, trasformazione affine che non tocca né l'asimmetria né i probe di rumore.

Numeri di riferimento del file: min −62 719, max +139 396, mediana −433, 69% dei valori negativo, **2456 probe (35%) con massimo < 100 su tutti e 72 i pazienti**. Applicando floor+log, l'overlap dei top-20 geni per p-value con la versione grezza è **6 su 20**: il pannello finale dipende sostanzialmente da questa scelta.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella fra il caricamento dati e lo split; nuova cella di confronto in fondo)

**Interfaces:**
- Consumes: `X` (DataFrame 72 × 7129), `y`.
- Produces: `X_train`, `X_test` filtrati e log-trasformati; `probe_mask: np.ndarray[bool]` di lunghezza 7129; `feature_cols` aggiornato ai soli probe superstiti.

- [ ] **Step 1: Definire le funzioni di preprocessing**

Criterio di Dudoit et al. (2002) §3.1.2. **Due dettagli che cambiano il risultato:** il criterio è `or` (non `and`) e usa `≤` (non `<`); e `max/min` va calcolato **dopo** il flooring, altrimenti il rapporto su valori negativi non ha senso.

```python
FLOOR, CEIL = 100, 16000

def threshold(df):
    """Floor/ceiling: trasformazione puntuale, nessun parametro stimato dai dati."""
    return df.clip(lower=FLOOR, upper=CEIL)

def fit_probe_mask(df_thresholded):
    """Filtro non supervisionato (non guarda mai y). Stimato SOLO sul train.
    Criterio di Dudoit et al. (2002): si valuta sulle INTENSITA' gia' soggette
    a floor/ceiling, non sui logaritmi, ed e' un OR con <= (non un AND con <)."""
    mx, mn = df_thresholded.max(axis=0), df_thresholded.min(axis=0)
    return ~((mx / mn <= 5) | ((mx - mn) <= 500))
```

L'ordine è vincolante: **floor/ceiling → filtro → log**. Il flooring deve precedere il filtro perché `max/min` su valori negativi non ha significato come rapporto, e deve precedere il log perché il logaritmo di un numero negativo non esiste — sono la stessa operazione che risolve due problemi.

- [ ] **Step 2: Applicarlo con la separazione train/test corretta**

Le tre operazioni non hanno lo stesso statuto e questa distinzione va scritta nel report: floor/ceiling/log sono **puntuali** (nessuna stima dai dati, applicabili ovunque); il filtro sui probe **stima** max e min da un insieme di campioni, quindi va calcolato sul solo training. Dudoit lo calcolava su tutti e 72 — noi siamo più conservativi, e questa deviazione va dichiarata.

Questo step **sostituisce** la cella di split modificata nel Task 3: lo split ora avviene sui dati soggetti a floor/ceiling, e il filtro sui probe si inserisce fra split e scaling.

```python
X_thr = threshold(X)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_thr, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

probe_mask = fit_probe_mask(X_train_raw)          # stimato solo sul train
feature_cols = list(X_train_raw.columns[probe_mask])

X_train_raw = np.log10(X_train_raw.loc[:, probe_mask])
X_test_raw = np.log10(X_test_raw.loc[:, probe_mask])

print(f"Probe iniziali: {X.shape[1]}")
print(f"Probe superstiti dopo floor/ceiling + filtro: {len(feature_cols)}")
print(f"Probe eliminati: {X.shape[1] - len(feature_cols)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
```

**Valori attesi, già verificati su questi dati:** con il criterio corretto (`or`, `≤`) e la stima sul solo training set restano **3363 probe** su 7129; stimando su tutti e 72 i campioni ne resterebbero 3462; usando erroneamente `and` invece di `or` ne resterebbero 3722. Dudoit riporta 3571 su 7129 — la differenza è attesa, il CSV in uso è una normalizzazione leggermente diversa della matrice originale. Dopo il log i valori cadono nell'intervallo [2.0, 4.2], senza NaN né infiniti.

- [ ] **Step 3: Documentare la scelta in una cella markdown**

Testo da inserire sopra la cella di preprocessing (adattare i numeri a quelli effettivi):

> **Preprocessing dei dati di espressione.** I valori del dataset sono *Average Difference* di Affymetrix MAS 4.0, cioè medie di differenze fra sonde *perfect match* e *mismatch*: i valori negativi (69% del totale, mediana −433) sono fisiologici e indicano geni non espressi, non dati corrotti. Applichiamo la procedura usata da Dudoit et al. (2002, §3.1.2) sul dataset di Golub — soglia inferiore 100, superiore 16 000; esclusione dei probe con `max/min ≤ 5` **oppure** `max − min ≤ 500`; logaritmo in base 10 — attribuita dagli autori a comunicazione personale con P. Tamayo e **non descritta nell'articolo originale di Golub et al. (1999)**. Il flooring è anche condizione necessaria per il logaritmo, dato che i valori grezzi sono in larga parte negativi. Ci discostiamo da Dudoit su un punto: loro calcolano il filtro su tutti e 72 i campioni, noi solo sul training set — il filtro è non supervisionato, quindi la loro scelta era difendibile, ma preferiamo la variante conservativa. `StandardScaler` da solo non sarebbe stato sufficiente: è una z-score per gene, trasformazione affine che lascia intatte asimmetria e outlier, e che anzi *amplifica* i probe di puro rumore dividendoli per la loro deviazione standard minuscola.

- [ ] **Step 4: Aggiungere il confronto con/senza preprocessing**

Questa è la prova di robustezza che vale di più. Cella nuova in fondo al notebook:

```python
from scipy.stats import ttest_ind as _tt

def top_genes_by_ttest(X_df, y_vec, k=20):
    Z = StandardScaler().fit_transform(X_df)
    _, p = _tt(Z[y_vec.values == 0], Z[y_vec.values == 1], equal_var=False, axis=0)
    order = np.argsort(p)[:k]
    return [X_df.columns[i] for i in order]

X_tr_grezzo = X.loc[X_train_raw.index]
top_grezzo = top_genes_by_ttest(X_tr_grezzo, y_train, k=20)
top_prep = top_genes_by_ttest(X_train_raw, y_train, k=20)

print("--- Sensibilita' del ranking univariato al preprocessing ---")
print(f"Overlap dei top-20 geni per p-value: {len(set(top_grezzo) & set(top_prep))}/20")
print(f"Geni presenti solo senza preprocessing: {sorted(set(top_grezzo) - set(top_prep))}")
print(f"Geni presenti solo con preprocessing:   {sorted(set(top_prep) - set(top_grezzo))}")
print()
print("Il pannello STABILE e' invece piu' robusto di quello univariato:")
print(f"dei {len(lasso_final_genes)} geni della stability selection, "
      f"{len(set(lasso_final_genes) & set(top_prep))} sono nei top-20 univariati.")
```

- [ ] **Step 5: Rieseguire e verificare la coerenza a valle**

Tutto ciò che indicizza `feature_cols` deve continuare a funzionare, in particolare `feature_cols.index(gene)` nella heatmap e `[feature_cols[i] for i in ...]` nelle celle di stabilità. Verificare:

```python
assert X_train.shape[1] == len(feature_cols)
assert X_test.shape[1] == len(feature_cols)
```

Il numero di geni significativi al t-test cambierà (il pool è passato da 7129 a ~3462, quindi anche la correzione FDR opera su meno test).

- [ ] **Step 6: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): preprocessing microarray secondo Dudoit et al. 2002 + confronto con dati grezzi"
```

---

### Task 6: Analisi della multi-collinearità

Obiettivo: la consegna nomina *"the effect of multi-collinearity on sparsity"* come **primo** dei due fenomeni statistici da esplorare. Oggi la collinearità non è mai misurata: la heatmap si intitola "Deep Dive: Multi-collinearity and Sparsity" ma mostra solo due colonne di coefficienti, e l'unico codice che ci provava (`absolute_cosine_filter`) era interamente commentato ed è stato cancellato nel Task 1.

**Calibrazione del tono:** sui dati reali la |r| mediana fra i geni significativi è **0.284** e solo lo **0.07%** delle coppie supera 0.8, ma **15 geni hanno almeno un gemello con |r| > 0.9**. La storia sta in quei 15. Non scrivere "il dataset è massicciamente collineare" — sarebbe falso e verificabile.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella dopo il bar chart di stabilità Lasso)

**Interfaces:**
- Consumes: `X_train`, `y_train`, `feature_cols`, `robust_baseline_genes`, `lasso_stability_df`, `lasso_final_genes`, `en_stable_set`.
- Produces: `corr_sig: np.ndarray` (matrice |r| fra geni significativi), `twin_df: pd.DataFrame` (per ogni gene del pannello Lasso, il gemello più correlato).

- [ ] **Step 1: Distribuzione delle correlazioni a coppie**

```python
sig_idx = [feature_cols.index(g) for g in sorted(robust_baseline_genes)]
sig_names = [feature_cols[i] for i in sig_idx]
S = X_train[:, sig_idx]

corr_sig = np.abs(np.corrcoef(S, rowvar=False))
np.fill_diagonal(corr_sig, 0.0)
iu = np.triu_indices_from(corr_sig, k=1)
pairwise = corr_sig[iu]

print("--- Struttura di correlazione fra i geni statisticamente significativi ---")
print(f"Geni considerati: {len(sig_names)}")
print(f"|r| mediana fra coppie: {np.median(pairwise):.3f}")
print(f"Frazione di coppie con |r| > 0.8: {(pairwise > 0.8).mean():.4f}")
print(f"Geni con almeno un 'gemello' a |r| > 0.9: {(corr_sig.max(axis=0) > 0.9).sum()}")

plt.figure(figsize=(9, 5))
sns.histplot(pairwise, bins=60, color='steelblue')
plt.axvline(0.8, color='crimson', linestyle='--', label='|r| = 0.8')
plt.xlabel('|r| di Pearson fra coppie di geni significativi')
plt.ylabel('Numero di coppie')
plt.title('Distribuzione della collinearita\' fra i geni significativi', pad=15)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Clustering gerarchico in moduli correlati**

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

dist = 1.0 - corr_sig
np.fill_diagonal(dist, 0.0)
Z_link = linkage(squareform(dist, checks=False), method='average')
modules = fcluster(Z_link, t=0.2, criterion='distance')   # moduli a |r| >= 0.8

sizes = pd.Series(modules).value_counts()
print(f"Moduli individuati (soglia |r| >= 0.8): {len(sizes)}")
print(f"Moduli con piu' di un gene: {(sizes > 1).sum()}")
print(f"Dimensione del modulo piu' grande: {sizes.max()} geni")
```

- [ ] **Step 3: Il gemello di ciascun gene del pannello Lasso**

Questo è il cuore dell'analisi: mostra il meccanismo per cui il Lasso tiene un membro del modulo e azzera gli altri, mentre l'ElasticNet tiene il modulo intero. **I due numeri — 12 geni Lasso contro 24 ElasticNet — ci sono già nel notebook; quello che manca è la spiegazione del perché.**

```python
rows = []
for g in lasso_final_genes:
    if g not in sig_names:
        rows.append({'Gene': g, 'Gemello': None, 'max_|r|': np.nan,
                     'Gemello_in_Lasso': None, 'Gemello_in_EN': None})
        continue
    i = sig_names.index(g)
    j = int(np.argmax(corr_sig[i]))
    twin = sig_names[j]
    rows.append({
        'Gene': g,
        'Gemello': twin,
        'max_|r|': round(float(corr_sig[i, j]), 3),
        'Gemello_in_Lasso': twin in set(lasso_final_genes),
        'Gemello_in_EN': twin in en_stable_set,
    })

twin_df = pd.DataFrame(rows).sort_values('max_|r|', ascending=False)
display(twin_df)

n_twin_en_only = int((~twin_df['Gemello_in_Lasso'].fillna(False)
                      & twin_df['Gemello_in_EN'].fillna(False)).sum())
print(f"\nGeni il cui gemello e' scartato dal Lasso ma tenuto dall'Elastic Net: {n_twin_en_only}")
print("E' l'effetto atteso: l'L1 puro rompe i gruppi collineari scegliendo un rappresentante,")
print("la penalita' L2 dell'Elastic Net li tiene insieme (grouping effect).")
```

- [ ] **Step 4: Inclusion probability contro collinearità**

La traduzione empirica diretta della frase della consegna: la probabilità di inclusione si **spartisce** fra gemelli collineari, e questo è il modo in cui la multi-collinearità degrada la stabilità.

```python
prob_map = dict(zip(lasso_stability_df['Gene_Probe'],
                    lasso_stability_df['Inclusion_Probability']))

pts = []
for i, g in enumerate(sig_names):
    if g in prob_map:
        pts.append({'Gene': g, 'Inclusion_Probability': prob_map[g],
                    'max_corr': float(corr_sig[i].max())})
pts_df = pd.DataFrame(pts)

plt.figure(figsize=(9, 6))
sns.scatterplot(data=pts_df, x='max_corr', y='Inclusion_Probability',
                alpha=0.6, color='steelblue')
plt.axhline(STABILITY_THRESHOLD, color='black', linestyle='--',
            label=f'Soglia {STABILITY_THRESHOLD:.0%}')
plt.xlabel('|r| massima verso un altro gene significativo')
plt.ylabel('Probabilita\' di inclusione (Lasso, 100 bootstrap)')
plt.title('Effetto della collinearita\' sulla stabilita\' della selezione', pad=15)
plt.legend()
plt.tight_layout()
plt.show()

rho = pts_df['max_corr'].corr(pts_df['Inclusion_Probability'], method='spearman')
print(f"Correlazione di Spearman fra collinearita' massima e inclusion probability: {rho:.3f}")
```

- [ ] **Step 5: Aggiornare il titolo della heatmap**

La heatmap EN vs Lasso resta utile, ma il suo titolo attuale promette un'analisi che non fa. Ora che l'analisi esiste davvero nelle celle precedenti, rinominarla in `'Coefficienti Elastic Net vs Lasso sui geni stabili dell\'Elastic Net'` e spostarla subito dopo il Task 6, così il "deep dive" ha effettivamente il contenuto che annuncia.

- [ ] **Step 6: Verificare**

Rieseguire. Atteso: `twin_df` ha una riga per ogni gene del pannello Lasso; almeno qualche riga con `max_|r|` alto e `Gemello_in_Lasso = False`. Se `n_twin_en_only` fosse 0, il grouping effect non è dimostrabile su questi dati e va detto onestamente nel report invece di forzare la narrazione.

- [ ] **Step 7: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): analisi della multi-collinearita' e del suo effetto su sparsita' e stabilita'"
```

---

### Task 7: Controllo dei confonditori

Obiettivo: i 5 fattori non genici sono scartati alla cella di caricamento senza un solo controllo, e solo `Gender` compare in un grafico. `Source` (ospedale di provenienza) è il confonditore di batch classico su questo dataset, ed è **quasi perfettamente collineare con la diagnosi**:

```
Source     ALL   AML
CALGB        0    15
CCG          0     5
DFCI        44     0
St-Jude      3     5
```

64 pazienti su 72 vengono da centri che hanno contribuito una sola classe. Qualunque effetto di laboratorio è matematicamente indistinguibile dal segnale biologico. Non è risolvibile con questi dati: va **dichiarato come limite strutturale**.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella dopo i grafici di distribuzione del target)

**Interfaces:**
- Consumes: `raw_df`, `y`, `X_train`, `y_train`, `feature_cols`, `lasso_final_genes`, `pca`.
- Produces: `confound_table: pd.DataFrame` (chi-quadro per ciascun fattore contro il target).

- [ ] **Step 1: Test di associazione fattore-target**

```python
from scipy.stats import chi2_contingency

rows = []
for col in ['BM.PB', 'Gender', 'Source', 'tissue.mf']:
    tab = pd.crosstab(raw_df[col], y)
    chi2, p, dof, _ = chi2_contingency(tab)
    rows.append({'Fattore': col, 'chi2': round(chi2, 2), 'dof': dof,
                 'p_value': f"{p:.3g}",
                 'Associato': 'SI' if p < 0.05 else 'no'})
    print(f"\n--- {col} vs diagnosi ---")
    display(tab)

confound_table = pd.DataFrame(rows)
print("\n--- Sintesi dei test di associazione ---")
display(confound_table)
```

- [ ] **Step 2: PCA ricolorata per Source e BM.PB**

Riusare la funzione `plot_pca_two_views` già presente nel notebook, cambiando solo l'etichetta.

```python
src_train = raw_df.loc[y_train.index, 'Source']
tissue_train = raw_df.loc[y_train.index, 'BM.PB']

for labels, name in [(src_train, 'Source'), (tissue_train, 'BM.PB')]:
    plot_df = pd.DataFrame({'PC1': X_train_pca[:, 0], 'PC2': X_train_pca[:, 1],
                            name: labels.values})
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue=name,
                    palette='Set2', alpha=0.85, s=60)
    plt.title(f'PCA del training set colorata per {name}', pad=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 3: I biomarcatori finali separano per Source dentro il gruppo ALL?**

Se i 12 geni distinguono gli ospedali anche fra pazienti con la stessa diagnosi, allora stanno catturando in parte l'effetto batch.

```python
all_mask = (y_train.values == 0)
idx_final = [feature_cols.index(g) for g in lasso_final_genes]
panel_all = X_train[np.ix_(all_mask, idx_final)]
src_all = src_train.values[all_mask]

print("--- I biomarcatori finali separano gli ospedali DENTRO il gruppo ALL? ---")
print(f"Pazienti ALL nel training: {all_mask.sum()}")
print(f"Distribuzione per centro: {pd.Series(src_all).value_counts().to_dict()}")

if pd.Series(src_all).nunique() > 1:
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.linear_model import LogisticRegression as _LR
    src_codes = pd.Series(src_all).astype('category').cat.codes.to_numpy()
    if np.bincount(src_codes).min() >= 3:
        sc = _cvs(_LR(class_weight='balanced', max_iter=5000),
                  panel_all, src_codes, cv=3, scoring='balanced_accuracy')
        print(f"Balanced accuracy nel predire il CENTRO dai 12 geni (solo ALL): "
              f"{sc.mean():.3f} +/- {sc.std():.3f}")
        print("Un valore nettamente sopra il caso indicherebbe che il pannello")
        print("cattura anche segnale di batch, non solo biologia.")
    else:
        print("Classi di centro troppo sbilanciate per una CV affidabile: "
              "il confronto resta qualitativo.")
```

- [ ] **Step 4: Cella markdown con la dichiarazione del limite**

> **Confonditori e limite strutturale del dataset.** Il dataset contiene cinque fattori non genici che abbiamo escluso dalla matrice delle feature. L'esclusione non è però innocua: `Source`, il centro di raccolta, è quasi perfettamente confuso con la diagnosi — 64 pazienti su 72 provengono da ospedali che hanno contribuito una sola classe (DFCI 44 ALL / 0 AML; CALGB 0/15; CCG 0/5; solo St-Jude è misto, 3/5). Ne segue che qualunque effetto tecnico specifico del laboratorio — protocollo, lotto di chip, operatore — è **matematicamente indistinguibile** dal segnale biologico ALL vs AML: nessun modello, per quanto ben regolarizzato, può separare le due componenti a partire da questi dati. Non si tratta di un difetto della nostra analisi ma di un limite di design dello studio originale, e va tenuto presente nell'interpretare la plausibilità biologica dei biomarcatori selezionati. Non includiamo i fattori come covariate proprio perché la loro collinearità con il target li renderebbe predittori quasi perfetti, mascherando il segnale genico invece di correggerlo.

- [ ] **Step 5: Verificare**

Rieseguire. Atteso: `Source` risulta fortemente associato al target (p molto piccolo); `Gender` e `BM.PB` con ogni probabilità no.

- [ ] **Step 6: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): controllo dei confonditori, confondimento Source/diagnosi documentato"
```

---

### Task 8: Score formale di riproducibilità

Obiettivo: la consegna elenca il *"reproducibility/stability score"* fra i criteri di valutazione e suggerisce letteralmente *"the average inclusion probability of your final biomarker subset"*. Oggi la tabella riporta solo il **massimo** e il conteggio ≥80%, il che rende il confronto fra i tre metodi aneddotico.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella prima della tabella riassuntiva; modifica della tabella)

**Interfaces:**
- Consumes: `lasso_boot_sets`, `svm_boot_sets`, `en_boot_sets` (dal Task 3), `lasso_stability_df`, `svm_stability_df`, `en_stability_df`, `lasso_final_genes`, `svm_final_features`, `en_stable_set`.
- Produces: `stability_scores: pd.DataFrame` con colonne `Method`, `Mean_Inclusion_Prob`, `Jaccard`, `Kuncheva`.

- [ ] **Step 1: Implementare gli indici set-based**

```python
from itertools import combinations

def jaccard_index(sets):
    """Media di |A∩B|/|A∪B| su tutte le coppie di repliche bootstrap."""
    vals = []
    for a, b in combinations(sets, 2):
        union = len(a | b)
        if union > 0:
            vals.append(len(a & b) / union)
    return float(np.mean(vals)) if vals else np.nan

def kuncheva_index(sets, n_total):
    """Indice di Kuncheva: corregge la sovrapposizione attesa per caso.
    Vale 1 per selezioni identiche, ~0 per selezioni indipendenti."""
    vals = []
    for a, b in combinations(sets, 2):
        k = min(len(a), len(b))
        if k == 0 or k == n_total:
            continue
        r = len(a & b)
        expected = k * k / n_total
        vals.append((r - expected) / (k - expected))
    return float(np.mean(vals)) if vals else np.nan
```

- [ ] **Step 2: Calcolare gli score per i tre metodi**

```python
def mean_inclusion(stability_df, panel):
    m = stability_df.set_index('Gene_Probe')['Inclusion_Probability']
    vals = [m.get(g, 0.0) for g in panel]
    return float(np.mean(vals)) if vals else np.nan

n_total = len(feature_cols)
stability_scores = pd.DataFrame([
    {'Method': 'Lasso (L1)',
     'Mean_Inclusion_Prob': mean_inclusion(lasso_stability_df, lasso_final_genes),
     'Jaccard': jaccard_index(lasso_boot_sets),
     'Kuncheva': kuncheva_index(lasso_boot_sets, n_total)},
    {'Method': 'Elastic Net',
     'Mean_Inclusion_Prob': mean_inclusion(en_stability_df, sorted(en_stable_set)),
     'Jaccard': jaccard_index(en_boot_sets),
     'Kuncheva': kuncheva_index(en_boot_sets, n_total)},
    {'Method': 'mRMR + SVM-RFE',
     'Mean_Inclusion_Prob': mean_inclusion(svm_stability_df, svm_final_features),
     'Jaccard': jaccard_index(svm_boot_sets),
     'Kuncheva': kuncheva_index(svm_boot_sets, n_total)},
]).round(3)

print("--- Score di riproducibilita' della selezione ---")
display(stability_scores)
print("Mean_Inclusion_Prob: probabilita' media di inclusione dei geni del pannello finale")
print("  (e' la metrica suggerita esplicitamente dalla consegna).")
print("Jaccard: sovrapposizione media fra i set selezionati in coppie di repliche bootstrap.")
print("Kuncheva: come Jaccard ma corretto per la sovrapposizione attesa per caso;")
print("  ~0 significa selezioni sostanzialmente indipendenti fra una replica e l'altra.")
```

- [ ] **Step 3: Integrare nella tabella riassuntiva**

Aggiungere a `summary_data` le colonne `Mean Inclusion Prob` e `Jaccard`, prese da `stability_scores`, accanto a `Max Inclusion Probability` e `Variables >= 80% Stable`.

- [ ] **Step 4: Verificare**

Rieseguire. Atteso: il Lasso ha Jaccard nettamente superiore a mRMR+SVM-RFE — coerente con il fatto che quest'ultimo non produce alcun gene sopra l'80%. Se non fosse così, il racconto sul "fallimento di stabilità dell'SVM-RFE" va rivisto sui numeri nuovi, non difeso.

- [ ] **Step 5: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): indici di riproducibilita' (inclusion media, Jaccard, Kuncheva)"
```

---

### Task 9: Controllo delle false scoperte — null da permutazione e interpretazione

Obiettivo: oggi la cella finale stampa `Concordance Rate (False Discovery Control): 100.00%` senza alcuna interpretazione. Un numero senza spiegazione è esattamente ciò che la consegna dichiara inutile.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella dopo la sintesi finale; modifica dei `print` di concordanza)

**Interfaces:**
- Consumes: `X_train`, `y_train`, `feature_cols`, `lasso_optimal_C`, `robust_baseline_genes`, `lasso_final_genes`, `STABILITY_THRESHOLD`.
- Produces: `perm_counts: list[int]` (geni sopra soglia per ciascuna permutazione).

- [ ] **Step 1: Null da permutazione sulla stability selection**

```python
N_PERM = 20          # ogni permutazione costa N_BOOTSTRAP fit: tenerlo basso
perm_counts = []

perm_model = LogisticRegression(
    l1_ratio=1.0, solver='saga', C=lasso_optimal_C,
    class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE,
)

rng = np.random.default_rng(RANDOM_STATE)
print(f"Null da permutazione: {N_PERM} permutazioni x {N_BOOTSTRAP} bootstrap...")

for p in range(N_PERM):
    y_perm = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
    counts = Counter()
    for i in range(N_BOOTSTRAP):
        Xb, yb = resample(X_train, y_perm, random_state=1000 * p + i, stratify=y_perm)
        perm_model.fit(Xb, yb)
        counts.update(np.where(perm_model.coef_[0] != 0)[0])
    n_stable = sum(1 for c in counts.values() if c / N_BOOTSTRAP >= STABILITY_THRESHOLD)
    perm_counts.append(n_stable)
    print(f"  permutazione {p+1}/{N_PERM}: {n_stable} geni sopra soglia")

print(f"\nGeni stabili sotto l'ipotesi nulla: media {np.mean(perm_counts):.2f}, "
      f"massimo {max(perm_counts)}")
print(f"Geni stabili sui dati reali: {len(lasso_final_genes)}")
```

- [ ] **Step 2: Grafico del null**

```python
plt.figure(figsize=(8, 5))
sns.histplot(perm_counts, bins=range(0, max(max(perm_counts), 1) + 2),
             color='lightsteelblue', label='Etichette permutate (H0)')
plt.axvline(len(lasso_final_genes), color='crimson', linestyle='--', linewidth=2,
            label=f'Dati reali ({len(lasso_final_genes)} geni)')
plt.xlabel(f'Geni con inclusion probability >= {STABILITY_THRESHOLD:.0%}')
plt.ylabel('Numero di permutazioni')
plt.title('Distribuzione nulla della stability selection', pad=15)
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Sostituire il `print` di concordanza con l'interpretazione**

```python
overlap = set(lasso_final_genes) & robust_baseline_genes
n_panel, n_base, n_tot = len(lasso_final_genes), len(robust_baseline_genes), len(feature_cols)
expected_by_chance = n_panel * n_base / n_tot
expected_false_fdr = FDR_ALPHA * len(sig_genes_t)

print("--- Controllo delle false scoperte: contrasto statistica vs machine learning ---")
print(f"Pannello ML (stability selection): {n_panel} geni")
print(f"Baseline statistico (t-test AND Mann-Whitney, FDR {FDR_ALPHA:.0%}): {n_base} geni")
print(f"Geni del pannello ML anche statisticamente significativi: {len(overlap)}"
      f" ({len(overlap)/n_panel:.0%})")
print()
print("Perche' questo numero e' informativo:")
print(f"  - Sovrapposizione attesa per puro caso: {expected_by_chance:.1f} geni")
print(f"    ({n_panel} geni estratti a caso da {n_tot} colpirebbero {n_base} bersagli"
      f" circa {expected_by_chance:.1f} volte).")
print(f"  - Il baseline statistico, a FDR {FDR_ALPHA:.0%} con {len(sig_genes_t)} rigetti,")
print(f"    contiene circa {expected_false_fdr:.0f} falsi positivi attesi. La stability")
print(f"    selection ne restituisce {n_panel}, tutti replicati in >= "
      f"{STABILITY_THRESHOLD:.0%} dei ricampionamenti.")
print(f"  - Sotto etichette permutate la stability selection produce in media")
print(f"    {np.mean(perm_counts):.2f} geni sopra soglia (max {max(perm_counts)}).")
print()
print("I due approcci controllano cose diverse: l'FDR controlla i falsi positivi per")
print("singola ipotesi, la stability selection controlla la riproducibilita' della")
print("SELEZIONE. La loro concordanza e' la vera evidenza a favore del pannello.")
```

- [ ] **Step 4: Verificare**

Atteso: `np.mean(perm_counts)` vicino a 0. Se fosse alto, significa che la procedura di selezione produce geni "stabili" anche senza segnale, e va indagato prima di scrivere qualsiasi conclusione.

Costo: `N_PERM × N_BOOTSTRAP = 2000` fit di Lasso su ~3462 feature. Se dovesse risultare troppo lento, abbassare `N_PERM` a 10 e dichiararlo, non ridurre `N_BOOTSTRAP`.

- [ ] **Step 5: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): null da permutazione e interpretazione della concordanza"
```

---

### Task 10 (opzionale): Π_max e il bound di Meinshausen–Bühlmann

Obiettivo: oggi `lasso_optimal_C` è fissato una volta sola per tutte le repliche. Prendendo il **massimo** della inclusion probability su una griglia di λ si ottiene il Π_max della formulazione originale, che è ciò che rende applicabile il bound teorico sui falsi positivi. È il modo più pulito di soddisfare "Control of False Discoveries" sul versante ML.

Da fare solo se c'è tempo dopo i Task 1–9 e 11. **Se non lo si implementa, aggiungere comunque un commento nella cella bootstrap** che dichiari che fissare λ è una scelta deliberata (è quello che fanno Meinshausen e Bühlmann) e non una svista.

**Files:**
- Modify: `progetto/project.ipynb` (nuova cella dopo il null da permutazione)

**Interfaces:**
- Consumes: `X_train`, `y_train`, `feature_cols`, `N_BOOTSTRAP`, `STABILITY_THRESHOLD`.
- Produces: `pi_max: np.ndarray` di lunghezza `len(feature_cols)`, `ev_bound: float`.

- [ ] **Step 1: Calcolare Π_max su una griglia di λ**

```python
C_stability = np.logspace(-2, 0, 5)
counts_per_C = np.zeros((len(C_stability), len(feature_cols)))

for ci, c in enumerate(C_stability):
    m = LogisticRegression(l1_ratio=1.0, solver='saga', C=c,
                           class_weight='balanced', max_iter=10000,
                           random_state=RANDOM_STATE)
    for i in range(N_BOOTSTRAP):
        Xb, yb = resample(X_train, y_train, random_state=i, stratify=y_train)
        m.fit(Xb, yb)
        counts_per_C[ci, np.where(m.coef_[0] != 0)[0]] += 1

probs_per_C = counts_per_C / N_BOOTSTRAP
pi_max = probs_per_C.max(axis=0)
q = float(probs_per_C.sum(axis=1).mean())   # numero medio di variabili selezionate
```

- [ ] **Step 2: Applicare il bound e riportarlo**

```python
p_tot = len(feature_cols)
pi_thr = STABILITY_THRESHOLD
ev_bound = (q ** 2) / ((2 * pi_thr - 1) * p_tot)
stable_pimax = [feature_cols[i] for i in np.where(pi_max >= pi_thr)[0]]

print("--- Stability selection alla Meinshausen-Buhlmann (Pi_max) ---")
print(f"Griglia di C: {np.round(C_stability, 4).tolist()}")
print(f"Numero medio di variabili selezionate per fit (q): {q:.1f}")
print(f"Geni con Pi_max >= {pi_thr:.0%}: {len(stable_pimax)}")
print(f"Bound sul numero atteso di falsi positivi: E[V] <= q^2 / ((2*pi_thr - 1) * p) "
      f"= {ev_bound:.2f}")
print()
print("Interpretazione: sotto le ipotesi del teorema (scambiabilita' del rumore e")
print("selezione non peggiore del caso), ci si attende al piu' "
      f"{ev_bound:.2f} falsi positivi")
print(f"in un pannello di {len(stable_pimax)} geni. Confronto con la selezione a C fisso:")
print(f"{len(lasso_final_genes)} geni, sovrapposizione "
      f"{len(set(stable_pimax) & set(lasso_final_genes))}.")
print()
print("Riferimento: Meinshausen, N. & Buhlmann, P. (2010), Stability Selection,")
print("J. R. Statist. Soc. B, 72(4), 417-473.")
```

- [ ] **Step 3: Verificare**

Atteso: `ev_bound` piccolo (ordine di grandezza sotto 1) e sovrapposizione alta con il pannello a C fisso. Costo: `5 × N_BOOTSTRAP = 500` fit.

- [ ] **Step 4: Commit**

```bash
git add progetto/project.ipynb
git commit -m "feat(progetto): Pi_max e bound teorico sui falsi positivi (Meinshausen-Buhlmann)"
```

---

### Task 11: Narrativa, conclusioni e verifica finale

Obiettivo: la consegna chiede *"a detailed account of your conclusions"* e avverte che *"the numbers, without any explanation about their meaning, are not really helpful"*. Oggi le celle markdown 1, 8, 12, 15, 18 sono il **testo della consegna** ("Use `pandas.read_csv()` to load your dataset..."), non conclusioni. L'unica cella con narrativa vera è quella sul fallimento dell'SVM-RFE — è il modello di registro da seguire.

**Files:**
- Modify: `progetto/project.ipynb` (tutte le celle markdown; nuova cella finale)

- [ ] **Step 1: Riscrivere i cinque markdown di fase**

Ognuno diventa: cosa fa questa fase, quale scelta metodologica è stata presa e perché, cosa è risultato. Riportare i numeri **effettivi** ottenuti dopo tutti i task precedenti — non quelli vecchi. Struttura per ciascuna fase:

- **Fase 1 (prep/EDA):** natura dei dati (AvgDiff MAS4, negativi fisiologici), preprocessing scelto e sua fonte, split stratificato, cosa mostra la PCA, e il confondimento con `Source`.
- **Fase 2 (filtraggio statistico):** perché due test invece di uno (Welch è sensibile all'asimmetria residua, Mann-Whitney no), perché l'intersezione come baseline robusto, quanti geni sopravvivono e quanti falsi positivi ci si attende a FDR 5%.
- **Fase 3 (selezione ML):** perché L1, cosa mostra la curva accuratezza/sparsità, il C scelto e quanti geni produce.
- **Fase 4 (stabilità):** cosa misura una inclusion probability, perché la soglia 80%, e il collegamento con l'analisi di collinearità (perché Lasso ed Elastic Net danno pannelli di dimensione diversa).
- **Fase 5 (sintesi):** il contrasto fra i due criteri di controllo delle false scoperte.

- [ ] **Step 2: Scrivere la cella finale di conclusioni**

Deve contenere, in prosa e non solo in numeri:

1. **Il pannello proposto** — quali geni, quanti, e come sono stati selezionati.
2. **Errore onesto** — `nested_mean ± nested_std` dalla CV annidata, con la nota che l'hold-out da 15 campioni è solo un sanity check.
3. **Sparsità** — numero di geni contro i 7129 di partenza (e i ~3462 dopo il filtro).
4. **Riproducibilità** — inclusion media, Jaccard, e il confronto con il null da permutazione.
5. **Perché il Lasso batte mRMR + SVM-RFE sulla stabilità** — collegando all'analisi di collinearità e al fatto che RFECV riottimizza il numero di feature a ogni replica, il che amplifica la varianza della selezione.
6. **Il limite del confondimento con `Source`**, dichiarato apertamente.
7. **Plausibilità biologica** — verificare la presenza nel pannello di `X95735_at` (zyxin) e `M23197_at` (CD33), i due marcatori AML canonici di Golub et al. Citarli costa una riga e vale molto; ma **verificare che ci siano davvero** prima di scriverlo, invece di darlo per scontato.

```python
print("--- Verifica dei marcatori canonici ---")
for probe, nome in [('X95735_at', 'zyxin'), ('M23197_at', 'CD33')]:
    in_panel = probe in set(lasso_final_genes)
    in_base = probe in robust_baseline_genes
    print(f"{probe} ({nome}): nel pannello finale = {in_panel}, "
          f"nel baseline statistico = {in_base}")
```

- [ ] **Step 3: Aggiungere la dichiarazione sull'uso di strumenti di editing**

La consegna (§4.2) chiede che l'uso di LLM per scopi di editing sia dichiarato nel report. Prevedere una riga in fondo al report PDF; non è materia del notebook ma è un requisito formale da non dimenticare.

- [ ] **Step 4: Aggiungere la bibliografia in una cella markdown finale**

Requisito esplicito della consegna: *"you are required to cite all sources"*, e *"include only those references that are pertinent (and that you have actually read!)"*.

```markdown
## Riferimenti

- Golub, T.R. et al. (1999). *Molecular Classification of Cancer: Class Discovery and
  Class Prediction by Gene Expression Monitoring.* Science, 286(5439), 531–537.
- Dudoit, S., Fridlyand, J. & Speed, T.P. (2002). *Comparison of Discrimination Methods
  for the Classification of Tumors Using Gene Expression Data.* JASA, 97(457), 77–87.
  (§3.1.2, pp. 80–81: procedura di preprocessing adottata; gli autori la attribuiscono
  a comunicazione personale con P. Tamayo, non e' descritta in Golub et al. 1999.)
- Meinshausen, N. & Bühlmann, P. (2010). *Stability Selection.* JRSS-B, 72(4), 417–473.
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.*
  JRSS-B, 57(1), 289–300.
- Guyon, I. et al. (2002). *Gene Selection for Cancer Classification using Support
  Vector Machines.* Machine Learning, 46, 389–422.
- Ding, C. & Peng, H. (2005). *Minimum Redundancy Feature Selection from Microarray
  Gene Expression Data.* J. Bioinform. Comput. Biol., 3(2), 185–205.
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic
  Net.* JRSS-B, 67(2), 301–320.
```

- [ ] **Step 5: Restart & Run All finale e verifica completa**

```bash
cd /home/unipi/i.inuso/Develop/analisi-dati
uv run jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=14400 progetto/project.ipynb
```

Poi verificare che non ci siano né errori né warning residui:

```bash
uv run python -c "
import json
nb=json.load(open('progetto/project.ipynb'))
errs, warns, ec = [], [], []
for i,c in enumerate(nb['cells'],1):
    if c['cell_type']!='code': continue
    ec.append(c.get('execution_count'))
    for o in c.get('outputs',[]):
        if o.get('output_type')=='error':
            errs.append((i,o.get('ename')))
        t=o.get('text','')
        t=''.join(t) if isinstance(t,list) else str(t)
        for w in ('FutureWarning','DeprecationWarning','ConvergenceWarning','Inconsistent values'):
            if w in t: warns.append((i,w))
print('errori:', errs)
print('warning:', sorted(set(warns)))
print('execution_count consecutivi:', ec == list(range(1, len(ec)+1)))
assert not errs, errs
"
```

Atteso: nessun errore, nessun warning, `execution_count` consecutivi da 1.

- [ ] **Step 6: Commit finale**

```bash
git add progetto/project.ipynb docs/superpowers/plans/
git commit -m "docs(progetto): conclusioni interpretative, bibliografia, notebook verificato end-to-end"
```

---

## Ordine di esecuzione e dipendenze

```
Task 1 (pulizia)  ──►  Task 2 (API sklearn)  ──►  Task 3 (correttezza statistica)
                                                          │
                                        ┌─────────────────┼─────────────────┐
                                        ▼                 ▼                 ▼
                                  Task 4 (CV)      Task 5 (preproc)   Task 8 (score)
                                        │                 │                 │
                                        └────────┬────────┴─────────────────┘
                                                 ▼
                                   Task 6 (collinearita')   Task 7 (confonditori)
                                                 │                 │
                                                 └────────┬────────┘
                                                          ▼
                                              Task 9 (permutation null)
                                                          ▼
                                              Task 10 (opzionale, Pi_max)
                                                          ▼
                                              Task 11 (narrativa + verifica)
```

**Se il tempo è poco:** i Task 6 e 7 colmano gli unici requisiti *esplicitamente richiesti dalla consegna e oggi non soddisfatti*. Vengono prima di tutto il resto tranne il Task 1, che è precondizione per qualunque altra modifica. Il Task 10 è l'unico davvero facoltativo.

**Attenzione ai tempi di esecuzione:** i Task 3, 9 e 10 aggiungono migliaia di fit di Lasso. Il notebook completo può richiedere ore. Eseguire i task uno alla volta e committare a ogni passo, così un'esecuzione fallita non fa perdere lavoro.
