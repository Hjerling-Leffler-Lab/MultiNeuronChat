import math

import numpy as np

from typing import Optional, Tuple


def minimum_detectable_p(
        n_control: int,
        n_case: int,
        n_hypotheses: Optional[int] = None,
        method: Optional[str] = 'bh',
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
    permutation. The floor binds exact and fully enumerated permutation p-values; asymptotic approximations lie above it.
    A Monte-Carlo permutation estimate can undercut it when n_resamples is below the size of the reference set but
    1 / (n_resamples + 1) < p_min (e.g. 8 vs 8 subjects with 10,000 resamples reports 1.0e-04 against p_min = 1.6e-04);
    this is an artefact of the estimate's resolution, which the Anderson escalation stage removes by enumerating.

    If n_hypotheses is provided, the corrected floor is computed as well. With all hypotheses tested at once, the
    Benjamini-Hochberg / Benjamini-Yekutieli step-up adjustment of the rank-th smallest p-value is

        p_min_adj = min(1, p_min * n_hypotheses * c(n_hypotheses) / rank)

    where c(m) = 1 for 'bh' and c(m) = sum_{i=1..m} 1/i for 'by'. rank is the number of hypotheses assumed to attain the
    floor simultaneously: rank=1 is the conservative lone-signal case, larger values describe a pervasive perturbation
    where many triples separate perfectly and the correction is correspondingly cheaper. rank=n_hypotheses gives the most
    favourable case, p_min * c(n_hypotheses): if even that exceeds alpha, no triple can be significant. Otherwise
    significant triples occur only in groups of at least k_min = ceil(p_min * n_hypotheses * c(n_hypotheses) / alpha),
    because the step-up rule rejects at rank k only if p_(k) <= k * alpha / (n_hypotheses * c(n_hypotheses)) and every
    p_(k) is at least p_min.

    Note that n_control and n_case are taken separately rather than as a single per-group n, because C(N, k) is maximised
    at the balanced split: 10 vs 10 subjects reach p_min = 1.1e-05, while the same 20 subjects split 2 vs 18 reach only
    1.1e-02.

    :param n_control: number of subjects in the control condition (>= 1)
    :param n_case: number of subjects in the case condition (>= 1)
    :param n_hypotheses: total number of tested hypotheses (source x receiver x interaction triples) used for the
                         multiple-testing correction. If None, only the uncorrected floor is computed and the corrected
                         floor is returned as np.nan.
    :param method: correction method, either 'bh' (Benjamini-Hochberg, MultiNeuronChat's default) or 'by'
                   (Benjamini-Yekutieli)
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


def n_relabelings(
        n_control: int,
        n_case: int
) -> int:
    """
    Return the size of the permutation reference set, C(n_control + n_case, n_control).

    This is the number of distinct ways the subjects can be relabelled into the two conditions, and therefore both the
    denominator of every exact rank-based p-value and the number of resamples above which a permutation test has nothing
    left to sample: once n_resamples reaches this value, SciPy enumerates the complete reference set and returns the
    exact p-value rather than a Monte-Carlo estimate of it.

    :param n_control: number of subjects in the control condition (>= 1)
    :param n_case: number of subjects in the case condition (>= 1)
    :return: the number of relabelings, computed with exact integer arithmetic so it does not overflow
    """
    if n_control < 1 or n_case < 1:
        raise ValueError('n_control and n_case must be positive integers')

    return math.comb(n_control + n_case, n_control)


def resamples_for_correction(
        n_hypotheses: int,
        method: Optional[str] = 'bh',
        alpha: Optional[float] = 0.05
) -> int:
    """
    Compute the number of permutation resamples required for a Monte-Carlo p-value to be able to survive multiple-testing
    correction.

    A permutation p-value estimated from n_resamples draws cannot fall below 1 / (n_resamples + 1), because the observed
    statistic is always counted among the resampled ones. After correction over n_hypotheses tests this floor becomes
    n_hypotheses * c(n_hypotheses) / (n_resamples + 1), which must be at most alpha for any triple to be reportable.
    Solving for the number of resamples gives

        R_min = ceil(n_hypotheses * c(n_hypotheses) / alpha)

    where c(m) = 1 for 'bh' and c(m) = sum_{i=1..m} 1/i for 'by'. Deriving the resample budget this way rather than
    hardcoding it is what keeps the permutation fallback and the correction consistent: with 6 cell types and 193
    interactions (n_hypotheses = 6,948, c = 9.42) the requirement is 1,309,490 resamples, so the conventional default of
    10,000 is roughly 130x too small and every strongly perturbed triple is pinned at the Monte-Carlo floor.

    The result is an upper bound on what is useful, not on what is affordable: it should always be capped at
    n_relabelings(n_control, n_case), beyond which the reference set is exhausted and the p-value is exact.

    :param n_hypotheses: number of tested hypotheses entering the multiple-testing correction (>= 1)
    :param method: correction method, either 'bh' (Benjamini-Hochberg, MultiNeuronChat's default) or 'by'
                   (Benjamini-Yekutieli)
    :param alpha: the significance level the p-value must be able to reach after correction (default: 0.05)
    :return: the smallest number of resamples at which the corrected Monte-Carlo floor is at most alpha
    """
    if method not in ['by', 'bh']:
        raise ValueError('method must be either "by" or "bh"')

    if n_hypotheses < 1:
        raise ValueError('n_hypotheses must be a positive integer')

    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in the interval (0, 1]')

    if method == 'by':
        correction_factor: float = float(np.sum(1 / np.arange(1, n_hypotheses + 1)))
    else:
        correction_factor: float = 1.0

    return int(math.ceil(n_hypotheses * correction_factor / alpha))
