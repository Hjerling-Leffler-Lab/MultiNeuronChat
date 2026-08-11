import math

import numpy as np

from typing import Optional, Tuple


def minimum_detectable_p(
        n_control: int,
        n_case: int,
        n_hypotheses: Optional[int] = None,
        method: Optional[str] = 'by',
        rank: Optional[int] = 1
) -> Tuple[float, float]:
    """
    Compute the smallest p-value a two-sample rank-based test can attain for a given design, and the corresponding
    multiple-testing corrected p-value.

    Every test used by MultiNeuronChat ('KS', 'Anderson', 'CVM', 'MannWhitneyU') is a rank statistic: the p-value is the
    fraction of all relabelings of the subjects into control/case that are at least as extreme as the observed one. The
    reference set therefore has exactly C(n_control + n_case, n_control) elements, and the smallest attainable non-zero
    numerator is 2 (the two maximally separated splits - all cases above all controls and its mirror image, both counted
    because the tests are two-sided). Hence

        p_min = 2 / C(n_control + n_case, n_control)

    This is a property of the design, not of a particular test or implementation: no effect size, sequencing depth, or
    number of cells per subject can push the p-value below this floor, because subjects - not cells - are the unit of
    permutation.

    If n_hypotheses is provided, the corrected floor is computed as well. With all hypotheses tested at once, the
    Benjamini-Hochberg / Benjamini-Yekutieli step-up adjustment of the rank-th smallest p-value is

        p_min_adj = min(1, p_min * n_hypotheses * c(n_hypotheses) / rank)

    where c(m) = 1 for 'bh' and c(m) = sum_{i=1..m} 1/i for 'by'. rank is the number of hypotheses assumed to attain the
    floor simultaneously: rank=1 is the conservative lone-signal case, larger values describe a pervasive perturbation
    where many triples separate perfectly and the correction is correspondingly cheaper.

    Note that n_control and n_case are taken separately rather than as a single per-group n, because C(N, k) is maximised
    at the balanced split: 10 vs 10 subjects reach p_min = 1.1e-05, while the same 20 subjects split 2 vs 18 reach only
    1.1e-02.

    :param n_control: number of subjects in the control condition (>= 1)
    :param n_case: number of subjects in the case condition (>= 1)
    :param n_hypotheses: total number of tested hypotheses (source x receiver x interaction triples) used for the
                         multiple-testing correction. If None, only the uncorrected floor is computed and the corrected
                         floor is returned as np.nan.
    :param method: correction method, either 'by' (Benjamini-Yekutieli, MultiNeuronChat's default) or 'bh'
                   (Benjamini-Hochberg)
    :param rank: rank of the hypothesis in the step-up procedure, i.e. the number of hypotheses assumed to attain the
                 floor (default: 1)
    :return: tuple (p_min, p_min_adj) of the uncorrected and the corrected minimal attainable p-value. p_min_adj is
             np.nan if n_hypotheses is None.
    """
    if method not in ['by', 'bh']:
        raise ValueError('method must be either "by" or "bh"')

    if n_control < 1 or n_case < 1:
        raise ValueError('n_control and n_case must be positive integers')

    if n_hypotheses is not None and n_hypotheses < 1:
        raise ValueError('n_hypotheses must be a positive integer')

    if rank < 1:
        raise ValueError('rank must be a positive integer')

    if n_hypotheses is not None and rank > n_hypotheses:
        raise ValueError('rank must not be larger than n_hypotheses')

    # Size of the permutation reference set; exact integer arithmetic, so it does not overflow for large cohorts.
    n_relabelings: int = math.comb(n_control + n_case, n_control)

    # The two maximally separated splits (two-sided testing counts both). For 1 vs 1 subject there are only two
    # relabelings in total, so the floor is 1.0.
    p_min: float = min(2 / n_relabelings, 1.0)

    if n_hypotheses is None:
        return p_min, np.nan

    # Harmonic factor of the Benjamini-Yekutieli correction; computed exactly as in scipy.stats.false_discovery_control.
    if method == 'by':
        correction_factor: float = float(np.sum(1 / np.arange(1, n_hypotheses + 1)))
    else:
        correction_factor: float = 1.0

    p_min_adj: float = min(p_min * n_hypotheses * correction_factor / rank, 1.0)

    return p_min, p_min_adj
