# MultiNeuronChat

**MultiNeuronChat** is a Python library for inferring *condition‑related changes* in synaptic cell‑cell communication from single‑cell / single‑nucleus RNA‑seq (sc/snRNA‑seq) datasets in **case vs control** study designs.  
It builds on the mathematical model of [Zhao _et al._ 2023](https://www.nature.com/articles/s41467-023-36800-w) method [NeuronChat](https://github.com/Wei-BioMath/NeuronChat) and extends it to multi‑condition comparisons with subject‑level statistics.

---
## Installation

From PyPI:

```bash
pip install multineuronchat
```

> Python ≥ 3.10 is required.
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
from multineuronchat.MultiNeuronChatObject import MultiNeuronChatObject

# 1) Configure your analysis
mnc = MultiNeuronChatObject(
    condition_label_column="condition",           # column attribute in Loom
    condition_names=("control", "case"),          # order matters
    subject_label_column="subject",               # column attribute in Loom
    cell_type_label_column="cell_types",          # column attribute in Loom
    db="human_extended"                           # "human", "mouse", "human_extended", or path to custom DB
)

# 2) Compute communication scores
mnc.compute_communication_scores(
    # Path to the cell-wise log-normalized Loom file
    path_to_data_loom="path/to/cellwise_normalized_data.loom",
    # The row name of the gene attribute in the Loom file
    gene_label_row='gene',
    # The number of processes to use for the parallel computations of communication scores
    n_processes=4,
    # The minimum number of cells of a specific cell type within a specific subject to include in the analysis
    min_n_cells_threshold=20,
    # Whether to print computational progress
    verbose=True
)

# 3) (Optional) Focus hypotheses with masks (Wasserstein / EMD)
from multineuronchat.masks import compute_wasserstein_mask
mask = compute_wasserstein_mask(mnc, top_percentile=99.0)

# 4) Significance testing and multiple testing correction
pvals = mnc.compute_significance(
    # The statistical test to use when comparing the communication score distributions between conditions
    statistical_test='KS',
    # Optional mask to focus the significance testing on specific interactions. If None, all interactions are tested.
    mask=mask,
    # The number of resamples to use for permutation testing when appropriate
    n_resamples=10_000,
    # The random state for reproducibility
    random_state=42
)

# 5) Control FDR (Benjamini–Yekutieli by default; use "bh" for Benjamini–Hochberg)
pvals_adj = mnc.correct_p_values(statistical_test="KS", method="by")
```


## How many subjects do you need?

MultiNeuronChat compares **subject-level** distributions, so the number of subjects places a hard
floor on the smallest p-value the analysis can produce, no matter how strong the biological effect
is. All tests offered here ask how unusual the observed control/case labelling is among all possible
labellings, and there are only `C(n_control + n_case, n_control)` of those:

```
p_min = 2 / C(n_control + n_case, n_control)
```

Even though more cells per subject and a larger effect size increase the clarity of the signal,
it does not affect teh minimal bound. Only more subjects can do so. Check your design before running anything:

```python
from multineuronchat import minimum_detectable_p

n_control, n_case = 10, 10   # subjects per condition
n_cell_types = 6             # cell types in your annotation
n_interactions = 193         # 190 "human", 183 "mouse", 193 "human_extended"

p_min, p_min_adj = minimum_detectable_p(
    n_control=n_control,
    n_case=n_case,
    n_hypotheses=n_cell_types * n_cell_types * n_interactions,
    method="by"              # "by" (default) or "bh"
)

print(f"Smallest attainable p-value: {p_min:.2e} -> {p_min_adj:.2e} after correction")
# Smallest attainable p-value: 1.08e-05 -> 7.09e-01 after correction
# => this design cannot produce a significant result at any effect size
```

Rules of thumb (balanced groups, `human_extended`):

- **Below ~8 subjects per group nothing is detectable**, regardless of effect size.
- **Do not coarsen your annotation to buy power** — going from 16 to 3 cell types buys only 3 subjects
  per group and costs the resolution the method exists to provide.
- **Keep groups balanced** — 10 vs 10 reaches `p_min = 1.1e-05`, but the same 20 subjects split 2 vs 18
  reach only `1.1e-02`.

## Input data requirements

For MultiNeuronChat to function, your input Loom file must meet the following criteria:

- **Row attributes**:
  - genes that were measured (`gene`)
- **Column attributes**:
  - labels assigning a subject to each cell (`subject` in the example above but can be configured)
  - condition labels (`condition` in the example above but can be configured)
  - cell type labels (`cell_types` in the example above but can be configured)

> Gene symbols should match the selected interaction database (human, mouse, or your custom DB).
