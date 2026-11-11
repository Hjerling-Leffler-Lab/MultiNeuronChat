# MultiNeuronChat

**MultiNeuronChat** is a Python library for inferring *condition‑related changes* in synaptic cell‑cell communication from single‑cell / single‑nucleus RNA‑seq (sc/snRNA‑seq) datasets in **case vs control** study designs.

It builds on the communication model of [NeuronChat](https://github.com/Wei-BioMath/NeuronChat) ([Zhao _et al._ 2023](https://www.nature.com/articles/s41467-023-36800-w)), which scores neurotransmitter‑ and neuropeptide‑mediated communication between cell types, and turns it into a comparison‑first framework: a communication score is computed **per subject** for every *(source cell type, receiver cell type, interaction)* triple, and the control and case groups are compared with two‑sample tests across subjects. The statistics are therefore calibrated to subject‑level variation, and the subject, not the cell, is the unit of replication.

---
## Installation

From PyPI:

```bash
pip install multineuronchat
```

> Python ≥ 3.10 is required. Dependencies are pinned to exact versions for reproducibility.

---

## Quick start (minimal end-to-end)

MultiNeuronChat expects a **cell-wise log-normalized** gene expression matrix in [**Loom**](https://linnarssonlab.org/loompy/) format. You can either normalize your matrix manually or use the provided implementation:

```python
from multineuronchat.normalize import cell_wise_log_normalization

cell_wise_log_normalization(
    # Path to the input Loom file
    path_to_loom="path/to/data.loom",
    # Path to the cell-wise log-normalized output Loom file
    path_to_normalized_loom='path/to/cellwise_normalized_data.loom',
    # Whether to print computational progress
    verbose=True
)
```

This log-normalized loom file can then be used to run MultiNeuronChat:

```python
from multineuronchat import MultiNeuronChatObject

# 1) Configure your analysis
mnc = MultiNeuronChatObject(
    condition_label_column="condition",           # column attribute in Loom
    condition_names=("control", "case"),          # exactly two conditions; order matters (control first)
    subject_label_column="subject",               # column attribute in Loom
    cell_type_label_column="cell_types",          # column attribute in Loom
    db="human_extended"                           # "human", "mouse", "human_extended", or path to custom DB
)

# 2) Compute communication scores per subject
mnc.compute_communication_scores(
    # Path to the cell-wise log-normalized Loom file
    path_to_data_loom="path/to/cellwise_normalized_data.loom",
    # The row name of the gene attribute in the Loom file (default: 'Gene')
    gene_label_row='gene',
    # The number of processes to use for the parallel computations of communication scores
    n_processes=4,
    # The minimum number of cells of a specific cell type within a specific subject to include in the analysis
    min_n_cells_threshold=20,
    # How to summarise a gene across a subject's cells of one type: 'tri_mean' (default), 'mean', or 'trim_mean'
    mean_type='tri_mean',
    # Whether to print computational progress
    verbose=True
)

# 3) Significance testing across subjects
pvals = mnc.compute_significance(
    # The two-sample test: 'CVM' (default), 'KS', 'MannWhitneyU', or 'Anderson' (see below)
    statistical_test='CVM',
    # The number of resamples to use for permutation testing when appropriate
    n_resamples=10_000,
    # The random state, so that any permutation fallback is reproducible
    random_state=42
)

# 4) Control the FDR (Benjamini–Hochberg by default; use "by" for Benjamini–Yekutieli)
pvals_adj = mnc.correct_p_values(statistical_test="CVM", method="bh")

# 5) Save the results (and reload them later with MultiNeuronChatObject.load)
mnc.save("path/to/results/mnc.pkl")
```

During step 2, the genes are restricted to those in the interaction database and each subject is
max-normalized. The result is cached next to the input as `*_MultiNeuronChat.loom` and **reused if it already
exists**. Delete it if you change the input data.

### Choosing a statistical test

`compute_significance` compares the per-subject score distributions of the two conditions with one of four
two-sample tests:

| `statistical_test` | test |
|---|---|
| `'CVM'` | Cramér–von Mises — **default and recommended**, it outperformed the other tests in most settings of our simulation benchmark |
| `'KS'` | Kolmogorov–Smirnov |
| `'MannWhitneyU'` | Mann–Whitney U |
| `'Anderson'` | Anderson–Darling — computationally expensive and not part of the benchmark (see below) |

`compute_significance` defaults to `'CVM'`.

### Reading the results

Results are [`xarray.DataArray`](https://docs.xarray.dev/)s with labelled dimensions:

- `mnc.communication_score_dict[condition]` — scores with dimensions `(subject, source, receiver, interaction)`
- `mnc.p_values[test]` and `mnc.p_values_adj[test]` — p-values with dimensions `(source, receiver, interaction)`

```python
adj = mnc.p_values_adj["CVM"]

# All significant triples as a table, sorted by adjusted p-value
significant = adj.where(adj < 0.05).to_series().dropna().sort_values()
print(significant.head())

# A single triple, selected by label
adj.sel(source="Excitatory", receiver="Inhibitory", interaction="Glu_GRIA2")
```

`NaN` marks a triple that could not be tested, for example because a gene of the interaction is missing from
the data, a cell type had fewer than `min_n_cells_threshold` cells, or the scores were identical across all
subjects. `NaN` triples do not enter the multiple-testing correction.

## How many subjects do you need?

MultiNeuronChat compares **subject-level** distributions, so the number of subjects places a hard
floor on the smallest p-value the analysis can produce, no matter how strong the biological effect
is. All tests offered here ask how unusual the observed control/case labelling is among all possible
labellings, and there are only `C(n_control + n_case, n_control)` of those:

```
p_min = 2 / C(n_control + n_case, n_control)
```

More cells per subject and a larger effect size make the signal clearer, but they do not lower this
bound. Only more subjects can. Check your design before running anything:

```python
from multineuronchat import minimum_detectable_p

n_control, n_case = 10, 10   # subjects per condition
n_cell_types = 6             # cell types in your annotation
n_interactions = 193         # 190 "human", 183 "mouse", 193 "human_extended"

p_min, p_min_adj = minimum_detectable_p(
    n_control=n_control,
    n_case=n_case,
    n_hypotheses=n_cell_types * n_cell_types * n_interactions,
    method="bh"              # "bh" (default) or "by"
)

print(f"Smallest attainable p-value: {p_min:.2e} -> {p_min_adj:.2e} after correction")
# Smallest attainable p-value: 1.08e-05 -> 7.52e-02 after correction
# => a single, isolated effect cannot reach significance in this design
```

`minimum_detectable_p` assumes a single effect by default (`rank=1`). If many triples are perturbed at
once, the step-up correction is cheaper. Pass `rank=k` to see the floor when `k` triples separate the
conditions. `compute_significance` and `correct_p_values` run this check automatically and warn when the
design cannot reach significance.

Rules of thumb (balanced groups, `human_extended`, Benjamini–Hochberg, α = 0.05):

- **To detect an isolated effect you need at least 10 subjects per group with 3 cell types, 11 with 6 cell types
  and 12 with 16 cell types.** Smaller designs can only reach significance when many triples change together.
- **Do not coarsen your annotation to buy power.** Going from 16 to 3 cell types gains only 2 subjects
  per group and costs the resolution the method exists to provide.
- **Keep groups balanced.** 10 vs 10 subjects reach `p_min = 1.1e-05`, but the same 20 subjects split 2 vs 18
  reach only `1.1e-02`.

### Anderson-Darling

Anderson-Darling is offered for completeness but is **not recommended**. It was not evaluated in the
simulation benchmark, and it is by far the most expensive test, because almost every informative triple
is resolved by permutation.

SciPy clamps the analytical Anderson-Darling p-value at 0.001, so every strongly separated triple falls back to
a permutation test, whose p-value cannot go below `1/(n_resamples + 1)`. At the default 10,000 resamples
that floor cannot survive correction over a realistic number of triples. `compute_significance` therefore
re-runs floor-pinned triples once with the resample depth the correction requires (`correction_method` and
`alpha`, defaulting to `'bh'` and `0.05`), capped at `C(n_control + n_case, n_control)`, where the p-value
becomes exact. With `human_extended` and BH this second stage uses about 35,000 (3 cell types), 140,000
(6 cell types) or 990,000 (16 cell types) resamples **per re-run triple**. Pass `random_state` to make it reproducible, or `escalate_permutation=False` to skip it.
A warning then reports how many triples stayed pinned at the floor.

## Input data requirements

For MultiNeuronChat to function, your input Loom file must meet the following criteria:

- **Row attributes**:
  - gene symbols. The attribute name is set with `gene_label_row`, which defaults to `'Gene'`; pass e.g.
    `gene_label_row='gene'` if your file uses lower case.
- **Column attributes**:
  - labels assigning a subject to each cell (`subject` in the example above but can be configured)
  - condition labels with exactly two values (`condition` in the example above but can be configured)
  - cell type labels (`cell_types` in the example above but can be configured)

Gene symbols must match the selected interaction database (human, mouse, or your custom DB). Interactions
whose genes are absent from the data are returned as `NaN`.

## Interaction databases

Three databases ship with the package and are selected by name: `"human"` (190 interactions), `"mouse"`
(183) and `"human_extended"` (193). To use your own, pass the path to a pickled `pandas.DataFrame` with
one row per interaction and the columns

| column | meaning |
|---|---|
| `interaction_name` | name of the interaction, e.g. `Glu_GRIA2` |
| `lig_contributor`, `lig_contributor_group`, `lig_contributor_coeff` | `-`-joined ligand genes, their group ids and the group coefficients |
| `target_subunit`, `target_subunit_group`, `target_subunit_coeff` | the same for the target subunits |

Within a group (e.g. isoenzymes) the gene abundances are averaged. Across groups they are combined by a
coefficient-weighted geometric mean, as in NeuronChat.

## Visualization

Plotting functions live in `multineuronchat.visualize` and draw onto a Matplotlib `Axes`:

- `plot_p_value_differential_communication_circle_plot` draws the cell types as a circle, with an arrow for
  every significant triple between them.
- `plot_communication_score_distribution` shows the per-subject score distributions of one triple in both
  conditions, optionally annotated with its p-value.
- `plot_circle_plot` draws a circle plot from your own `(source, receiver) -> count` dictionary.

```python
import matplotlib.pyplot as plt
from multineuronchat.visualize import plot_p_value_differential_communication_circle_plot

fig, ax = plt.subplots(figsize=(6, 6))
plot_p_value_differential_communication_circle_plot(
    axs=ax,
    mnc_object=mnc,
    significance_test_to_use="CVM",
    significance_threshold=0.05,
    use_adj_p_values=True,
    show_labels=True
)
```

## Filtering hypotheses (deprecated)

Earlier versions recommended reducing the number of tested triples with a mask from `multineuronchat.masks`
before calling `compute_significance(mask=...)`. **All masks are now deprecated** and emit a `FutureWarning`.
They still work in this release but will be removed in a future one:

- `compute_wasserstein_mask` / `compute_earth_mover_mask` and `compute_threshold_communication_score_mask` are
  **not valid prefilters**. They read the condition labels, so selecting on them biases the retained p-values
  downward under the null hypothesis ([Bourgon, Gentleman & Huber 2010](https://www.pnas.org/doi/10.1073/pnas.0914005107)).
  For the Wasserstein mask this inflates the null rejection rate about threefold.
- `compute_variance_mask`, `compute_abundance_mask` and `compute_pooled_threshold_communication_score_mask`
  never read the condition labels and are therefore valid, but in our simulation benchmark none of them
  improved on testing all triples.

Run `compute_significance` without a mask. Untestable triples already return `NaN` and do not count
towards the multiple-testing correction. The `mask` argument itself remains available if you have an
a-priori reason to restrict the analysis to specific triples, chosen without looking at the condition labels.

## What's new since 2025.11.14

This release accompanies the revised manuscript.

- **Benjamini–Hochberg is the default correction** (previously Benjamini–Yekutieli). Pass `method="by"` to
  `correct_p_values` to keep the previous behaviour. Raw p-values are unaffected.
- **Power checks.** `minimum_detectable_p`, `n_relabelings` and `resamples_for_correction` are new, and
  `compute_significance` / `correct_p_values` warn when the design cannot reach significance.
- **Cramér–von Mises is the default test** of `compute_significance` (previously Kolmogorov–Smirnov), as it
  outperformed the other tests in most settings of the simulation benchmark. Pass `statistical_test='KS'` to keep
  the previous behaviour.
- **Anderson-Darling** no longer gets stuck at the Monte-Carlo floor `1/(n_resamples + 1)`. Floor-pinned triples are
  re-run with the depth the correction requires (see above). This changes Anderson-Darling p-values and makes
  the test considerably slower.
- **Reproducibility.** The subject order is now deterministic, and `compute_significance` warns when
  `random_state=None` could make permutation p-values non-reproducible.
- **All masks are deprecated** (see above), as are the plots built on Wasserstein distances,
  `plot_wasserstein_ranked_differential_communication_circle_plot` and `plot_aula_medica_plot`.

## Citation

If you use MultiNeuronChat, please cite the preprint below. The manuscript is currently under revision at
*Communications Biology*; this citation will be updated once the peer-reviewed version is published.

> Volkmer G, Hjerling-Leffler J, Bast L. Cell Type Specific Inference of Perturbations in Synaptic Communication
> with MultiNeuronChat. *bioRxiv* (2025). https://doi.org/10.1101/2025.11.11.687822

```bibtex
@article{volkmer2025multineuronchat,
  title   = {Cell Type Specific Inference of Perturbations in Synaptic Communication with MultiNeuronChat},
  author  = {Volkmer, Gianluca and Hjerling-Leffler, Jens and Bast, Lisa},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.11.687822}
}
```
