import warnings

import numpy as np
import xarray as xr

from .MultiNeuronChatObject import MultiNeuronChatObject

from typing import Optional, Union, Tuple


def compute_wasserstein_mask(
    mnc_object: MultiNeuronChatObject,
    top_n: Optional[int] = None,
    top_percentile: Optional[float] = None,
    p: Optional[float] = 1.0,
    normalized: Optional[bool] = False,
    exclude_zero_distributions: Optional[bool] = True,
    return_wasserstein_distances: Optional[bool] = False
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Compute a mask for the hypotheses with the top_n or top_percentile largest Wasserstein distances between the
    distributions of the communication scores for each hypothesis. The mask is returned as a xarray DataArray with
    dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'.

    .. deprecated::
        All masks are deprecated and will be removed in a future release; call compute_significance without a mask
        instead. This mask is in addition not a valid prefilter: the Wasserstein distance measures the between-condition
        separation that the statistical tests are themselves testing for, and is analytically linked to the
        Kolmogorov-Smirnov statistic, so selecting on it biases the retained p-values downward under the null (Bourgon,
        Gentleman & Huber 2010). Measured under the global null at 12 vs 12 subjects, keeping the top 10 % of triples
        raises the KS rejection rate at alpha = 0.05 from 0.030 to 0.087, a 2.9-fold inflation.

    See also: compute_earth_mover_mask

    :param mnc_object: A MultiNeuronChatObject containing the communication scores
    :param top_n: The number of hypotheses with the largest Wasserstein distances that should be included in the mask.
                  If None, top_percentile must be provided.
    :param top_percentile: The percentile of the hypotheses with the largest Wasserstein distances that should be
                           included in the mask. If None, top_n must be provided.
    :param normalized: If True, the Wasserstein distances are normalized by the range of the communication scores for
                       each hypothesis. This is done to ensure that the Wasserstein distances are comparable.
                       Default: True
    :param exclude_zero_distributions: If True, hypotheses where both distributions consist solely of zeros are excluded.
    :param return_wasserstein_distances: If True, the computed Wasserstein distances are returned as well.
    :return: A boolean xarray DataArray with dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'
             or a tuple with the mask and the Wasserstein distances as xarray DataArrays
    """
    _warn_mask_deprecated('compute_wasserstein_mask', reads_labels=True)

    return _compute_earth_mover_mask(
        mnc_object=mnc_object,
        top_n=top_n,
        top_percentile=top_percentile,
        p=p,
        normalized=normalized,
        exclude_zero_distributions=exclude_zero_distributions,
        return_wasserstein_distances=return_wasserstein_distances
    )


def compute_earth_mover_mask(
    mnc_object: MultiNeuronChatObject,
    top_n: Optional[int] = None,
    top_percentile: Optional[float] = None,
    p: Optional[float] = 1.0,
    normalized: Optional[bool] = False,
    exclude_zero_distributions: Optional[bool] = True,
    return_wasserstein_distances: Optional[bool] = False
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Compute a mask for the hypotheses with the top_n or top_percentile largest Wasserstein distances between the
    distributions of the communication scores for each hypothesis. The mask is returned as a xarray DataArray with
    dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'.

    .. deprecated::
        Deprecated and will be removed in a future release, like all masks; call compute_significance without a mask
        instead. See compute_wasserstein_mask for why this mask is in addition not a valid prefilter.

    :param mnc_object: A MultiNeuronChatObject containing the communication scores
    :param top_n: The number of hypotheses with the largest Wasserstein distances that should be included in the mask.
                  If None, top_percentile must be provided.
    :param top_percentile: The percentile of the hypotheses with the largest Wasserstein distances that should be
                           included in the mask. If None, top_n must be provided.
    :param normalized: If True, the Wasserstein distances are normalized by the range of the communication scores for
                       each hypothesis. This is done to ensure that the Wasserstein distances are comparable.
                       Default: True
    :param exclude_zero_distributions: If True, hypotheses where both distributions consist solely of zeros are excluded.
    :param return_wasserstein_distances: If True, the computed Wasserstein distances are returned as well.
    :return: A boolean xarray DataArray with dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'
             or a tuple with the mask and the Wasserstein distances as xarray DataArrays
    """
    _warn_mask_deprecated('compute_earth_mover_mask', reads_labels=True)

    return _compute_earth_mover_mask(
        mnc_object=mnc_object,
        top_n=top_n,
        top_percentile=top_percentile,
        p=p,
        normalized=normalized,
        exclude_zero_distributions=exclude_zero_distributions,
        return_wasserstein_distances=return_wasserstein_distances
    )


def _compute_earth_mover_mask(
    mnc_object: MultiNeuronChatObject,
    top_n: Optional[int] = None,
    top_percentile: Optional[float] = None,
    p: Optional[float] = 1.0,
    normalized: Optional[bool] = False,
    exclude_zero_distributions: Optional[bool] = True,
    return_wasserstein_distances: Optional[bool] = False
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Implementation of compute_earth_mover_mask, shared with compute_wasserstein_mask so that each public call issues a
    single deprecation warning attributed to the caller.
    """
    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.communication_score_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed communication scores!'
                         'Please run compute_communication_scores() first.')

    n_source_cell_types = len(mnc_object.source_cell_types)
    n_receiver_cell_types = len(mnc_object.receiver_cell_types)
    n_interaction_types = len(mnc_object.interaction_names)
    n_hypotheses = n_source_cell_types * n_receiver_cell_types * n_interaction_types

    _validate_selection(top_n=top_n, top_percentile=top_percentile, n_hypotheses=n_hypotheses)

    if p <= 0:
        raise ValueError('p must be a positive value!')

    # Compute the Wasserstein distance between the distributions of the communication scores for each hypothesis
    wasserstein_distances = np.ones((n_source_cell_types, n_receiver_cell_types, n_interaction_types)) * np.nan

    for s, source in enumerate(mnc_object.source_cell_types):
        for r, receiver in enumerate(mnc_object.receiver_cell_types):
            for lt, interaction in enumerate(mnc_object.interaction_names):
                ctrl_dist = mnc_object.communication_score_dict[mnc_object.condition_names[0]][:, s, r, lt]
                scz_dist = mnc_object.communication_score_dict[mnc_object.condition_names[1]][:, s, r, lt]

                ctrl_dist = ctrl_dist[~np.isnan(ctrl_dist)]
                scz_dist = scz_dist[~np.isnan(scz_dist)]

                if len(ctrl_dist) == 0 or len(scz_dist) == 0:
                    continue
                if exclude_zero_distributions and np.all(ctrl_dist == 0) and np.all(scz_dist == 0):
                    continue

                min_value: float = min(ctrl_dist.min(), scz_dist.min())
                max_value: float = max(ctrl_dist.max(), scz_dist.max())
                value_range: float = max_value - min_value

                #wasserstein_distances[s, r, lt] = stats.wasserstein_distance(ctrl_dist, scz_dist)
                wasserstein_distances[s, r, lt] = _cdf_distance(p, ctrl_dist, scz_dist)

                if normalized and value_range > 0:
                    wasserstein_distances[s, r, lt] /= value_range

    mask: np.ndarray = _select_top(
        statistic=wasserstein_distances,
        top_n=top_n,
        top_percentile=top_percentile
    )

    mask_xr = _as_cube(mask, mnc_object)
    wasserstein_distances_xr = _as_cube(wasserstein_distances, mnc_object)

    if return_wasserstein_distances:
        return mask_xr, wasserstein_distances_xr
    return mask_xr


def compute_variance_mask(
    mnc_object: MultiNeuronChatObject,
    top_n: Optional[int] = None,
    top_percentile: Optional[float] = None,
    min_value: Optional[float] = None,
    statistic: Optional[str] = 'cv2',
    return_variances: Optional[bool] = False
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Compute a label-blind mask that keeps the hypotheses whose communication scores vary the most across subjects.

    .. deprecated::
        All masks are deprecated and will be removed in a future release; call compute_significance without a mask
        instead. This mask is label-blind and therefore a valid prefilter, but in the simulation benchmark it did not
        improve on testing all hypotheses.

    The variance is computed over the subjects of *both* conditions pooled into a single sample, with the condition
    labels discarded. Every statistical test used by MultiNeuronChat is
    a function T(x, z) of the pooled scores x and the condition labels z. Under the null hypothesis the labels are
    exchangeable, so conditioning on the pooled order statistics leaves the p-value super-uniform. A filter that is a
    symmetric function of the pooled sample alone is constant across the permutation orbit, hence

        P(p <= u | mask selects the triple) = E[ P(p <= u | pooled order statistics) | selected ] <= u

    i.e. selection cannot make the p-value anti-conservative. The independence is exact rather than approximate, and it
    also covers *joint* selection across triples (top_n / top_percentile rank all triples against each other): given the
    pooled samples the selected set is deterministic, and Benjamini-Yekutieli requires only super-uniform marginals under
    arbitrary dependence. The Wasserstein distance, by contrast, measures precisely the between-condition separation the
    tests are looking for, and is analytically linked to the Kolmogorov-Smirnov statistic.

    What the pooled variance contains follows from the variance decomposition

        (N - 1) * s^2_pooled = (n_control - 1) * s^2_control + (n_case - 1) * s^2_case
                               + (n_control * n_case / N) * (mean_control - mean_case)^2

    A genuine condition effect inflates the second (between-group) term, so the statistic is informative under the
    alternative while remaining independent under the null. The between-group term is however only one degree of freedom
    out of N - 1, so under the null it contributes on average 1 / (N - 1) of the total - about 1.3 % at 40 vs 40
    subjects. The mask is therefore a weak enrichment filter and its value lies mostly at the *bottom* of the ranking:
    triples whose scores are constant across all subjects (typically all zero, because the genes are absent or the cell
    type was below min_n_cells_threshold) carry no detectable signal at any effect size and are already returned as NaN
    by compute_significance. Removing them reduces the multiple-testing burden for free. Prefer min_value for that use;
    top_n and top_percentile exist mainly to compare filters at matched retention.

    Two statistics are available. The default 'cv2' is the squared coefficient of variation, s^2_pooled / mean_pooled^2,
    which approximates Var(log score) and is invariant under a per-triple rescaling of the scores. That invariance
    matches the tests, which are rank-based and cannot see the scale of a triple at all. It also keeps the mask distinct
    from compute_abundance_mask: communication scores are products of max-normalized abundances, so their variance
    scales with the squared mean and a raw-variance ranking is close to a ranking by mean abundance. 'variance' selects
    the unnormalized pooled variance, which is the more literal reading of "total variance".

    Triples with fewer than two non-NaN subjects in the pooled sample are set to NaN and never selected, as are triples
    with a pooled mean of zero when statistic='cv2'.

    :param mnc_object: A MultiNeuronChatObject containing the computed communication scores
    :param top_n: The number of hypotheses with the largest variance to include in the mask.
    :param top_percentile: The percentile cutoff on the variances; hypotheses at or above it are included. Note this is
                           the cutoff, not the retained fraction: top_percentile=99 keeps the top 1 %.
    :param min_value: An absolute cutoff; hypotheses with a variance strictly greater than min_value are included. Use
                      min_value=0.0 to drop only the triples that are constant across all subjects.
    :param statistic: 'cv2' for the squared coefficient of variation (default) or 'variance' for the raw pooled variance
    :param return_variances: If True, the computed variances are returned alongside the mask
    :return: A boolean xarray DataArray with dimensions 'source', 'receiver' and 'interaction', or a tuple of the mask
             and the variances as xarray DataArrays
    """
    _warn_mask_deprecated('compute_variance_mask')

    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.communication_score_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed communication scores!'
                         'Please run compute_communication_scores() first.')

    if statistic not in ['cv2', 'variance']:
        raise ValueError('statistic must be either "cv2" or "variance"')

    n_source_cell_types = len(mnc_object.source_cell_types)
    n_receiver_cell_types = len(mnc_object.receiver_cell_types)
    n_interaction_types = len(mnc_object.interaction_names)
    n_hypotheses = n_source_cell_types * n_receiver_cell_types * n_interaction_types

    _validate_selection(
        top_n=top_n,
        top_percentile=top_percentile,
        min_value=min_value,
        n_hypotheses=n_hypotheses,
        supports_min_value=True
    )

    # Pool the subjects of both conditions along the subject axis, discarding the condition labels. This concatenation
    # is what the validity of the filter rests on: every reduction below acts on the pooled sample only.
    pooled: xr.DataArray = xr.concat(
        [mnc_object.communication_score_dict[condition] for condition in mnc_object.condition_names],
        dim='subject'
    )

    pooled_np: np.ndarray = pooled.transpose('subject', 'source', 'receiver', 'interaction').to_numpy()

    n_valid_subjects: np.ndarray = np.sum(~np.isnan(pooled_np), axis=0)

    # Sample variance (ddof=1) over the pooled subject axis, ignoring NaNs. Triples with fewer than two non-NaN
    # subjects have no defined sample variance and are excluded below.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        variances: np.ndarray = np.nanvar(pooled_np, axis=0, ddof=1)
        pooled_means: np.ndarray = np.nanmean(pooled_np, axis=0)

    variances = np.where(n_valid_subjects >= 2, variances, np.nan)

    if statistic == 'cv2':
        # Squared coefficient of variation. A pooled mean of zero means every subject scored exactly zero, which is a
        # degenerate distribution compute_significance would return NaN for anyway.
        with np.errstate(divide='ignore', invalid='ignore'):
            variances = np.where(pooled_means != 0, variances / np.square(pooled_means), np.nan)

    mask: np.ndarray = _select_top(
        statistic=variances,
        top_n=top_n,
        top_percentile=top_percentile,
        min_value=min_value
    )

    mask_xr = _as_cube(mask, mnc_object)
    variances_xr = _as_cube(variances, mnc_object)

    if return_variances:
        return mask_xr, variances_xr
    return mask_xr


def compute_abundance_mask(
        mnc_object: MultiNeuronChatObject,
        top_n: Optional[int] = None,
        top_percentile: Optional[float] = None,
        min_value: Optional[float] = None,
        return_abundances: Optional[bool] = False
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Compute a label-blind mask that keeps the hypotheses whose bottleneck abundance is largest across subjects.

    .. deprecated::
        All masks are deprecated and will be removed in a future release; call compute_significance without a mask
        instead. This mask is label-blind and therefore a valid prefilter, but in the simulation benchmark it did not
        improve on testing all hypotheses.

    For each triple (source, receiver, interaction), the bottleneck abundance is defined as

        min(pooled_mean_ligand[source, interaction], pooled_mean_target[receiver, interaction])

    where the means are taken over the subjects of *both* conditions pooled together, discarding the condition labels.
    The min is used rather than the product because the communication score is ligand × target: either factor being
    absent makes the triple untestable regardless of how large the other is, and the product would let a large ligand
    compensate for a near-zero target. Using the product would also make this mask near-duplicate of a variance
    ranking (since score variance scales as (mean_ligand × mean_target)²), which would remove the point of having
    both filters.

    The filter is label-blind by the same exchangeability argument as compute_variance_mask: the pooled means are
    symmetric functions of the pooled sample, constant on the permutation orbit of the labels, so

        P(p <= u | mask selects the triple) <= u

    exactly, under arbitrary dependence across triples.

    Triples with a NaN pooled mean on either component (genes absent from the data, or all subjects below
    min_n_cells_threshold for that cell type) receive a NaN bottleneck score and are never selected.

    Use min_value=0.0 to drop only the triples where at least one abundance component is zero on average across all
    pooled subjects — those are untestable and already return NaN from compute_significance, so removing them reduces
    the multiple-testing burden for free without discarding any testable hypothesis.

    See also: compute_variance_mask, compute_pooled_threshold_communication_score_mask

    :param mnc_object: A MultiNeuronChatObject containing the computed communication scores
    :param top_n: The number of hypotheses with the largest bottleneck abundance to include in the mask.
    :param top_percentile: The percentile cutoff on the bottleneck abundances; hypotheses at or above it are included.
                           Note this is the cutoff, not the retained fraction: top_percentile=99 keeps the top 1 %.
    :param min_value: An absolute cutoff; hypotheses with a bottleneck abundance strictly greater than min_value are
                      included. Use min_value=0.0 to drop only the triples that have zero abundance on either side.
    :param return_abundances: If True, the computed bottleneck abundance values are returned alongside the mask.
    :return: A boolean xarray DataArray with dimensions 'source', 'receiver' and 'interaction', or a tuple of the
             mask and the bottleneck abundances as xarray DataArrays
    """
    _warn_mask_deprecated('compute_abundance_mask')

    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.ligand_abundance_dict is None or mnc_object.target_abundance_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed abundance scores. '
                         'Please run compute_communication_scores() first.')

    n_source_cell_types = len(mnc_object.source_cell_types)
    n_receiver_cell_types = len(mnc_object.receiver_cell_types)
    n_interaction_types = len(mnc_object.interaction_names)
    n_hypotheses = n_source_cell_types * n_receiver_cell_types * n_interaction_types

    _validate_selection(
        top_n=top_n,
        top_percentile=top_percentile,
        min_value=min_value,
        n_hypotheses=n_hypotheses,
        supports_min_value=True
    )

    # Pool subjects of both conditions, discarding labels. The validity of the filter rests on this
    # concatenation: everything below acts on the pooled sample only.
    pooled_ligand: xr.DataArray = xr.concat(
        [mnc_object.ligand_abundance_dict[condition] for condition in mnc_object.condition_names],
        dim='subject'
    )
    pooled_target: xr.DataArray = xr.concat(
        [mnc_object.target_abundance_dict[condition] for condition in mnc_object.condition_names],
        dim='subject'
    )

    # Pooled mean over subjects; dims are (source, interaction) and (receiver, interaction) respectively.
    # nanmean ignores NaNs from missing genes or cell types below min_n_cells_threshold; if every subject
    # is NaN the result is NaN, which propagates through the min below and is never selected.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ligand_mean: xr.DataArray = pooled_ligand.mean(dim='subject', skipna=True)
        target_mean: xr.DataArray = pooled_target.mean(dim='subject', skipna=True)

    # xarray broadcasts (source, interaction) against (receiver, interaction) by dim name, producing
    # (source, receiver, interaction). The min is taken element-wise over the two aligned arrays.
    bottleneck: xr.DataArray = xr.apply_ufunc(np.minimum, ligand_mean, target_mean)

    bottleneck_np: np.ndarray = bottleneck.transpose(
        'source', 'receiver', 'interaction'
    ).to_numpy()

    mask: np.ndarray = _select_top(
        statistic=bottleneck_np,
        top_n=top_n,
        top_percentile=top_percentile,
        min_value=min_value
    )

    mask_xr = _as_cube(mask, mnc_object)
    bottleneck_xr = _as_cube(bottleneck_np, mnc_object)

    if return_abundances:
        return mask_xr, bottleneck_xr
    return mask_xr


def compute_threshold_communication_score_mask(
        mnc_object: MultiNeuronChatObject,
        n_samples: Optional[int] = 1,
        threshold: Optional[float] = 0.0,
) -> xr.DataArray:
    """
    Compute a mask for the hypotheses where n_samples from each condition must have a communication score that is
    higher than the threshold. The mask is returned as a xarray DataArray with dimensions 'source_cell_type',
    'receiver_cell_type', and 'interaction_type'.

    .. deprecated::
        All masks are deprecated and will be removed in a future release; call compute_significance without a mask
        instead. This mask is in addition not a valid prefilter: it checks the threshold requirement separately per
        condition, so the selection depends on the condition labels and is not independent of the test statistics under
        the null hypothesis (Bourgon, Gentleman & Huber 2010).

    :param mnc_object: A MultiNeuronChatObject containing the communication scores
    :param n_samples: The number of samples from each condition that must have a communication score higher than the threshold.
    :param threshold: The threshold that the communication score must exceed.
    :return: A boolean xarray DataArray with dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'
    """
    _warn_mask_deprecated('compute_threshold_communication_score_mask', reads_labels=True)

    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.communication_score_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed communication scores!'
                         'Please run compute_communication_scores() first.')

    n_source_cell_types = len(mnc_object.source_cell_types)
    n_receiver_cell_types = len(mnc_object.receiver_cell_types)
    n_interaction_types = len(mnc_object.interaction_names)

    mask: np.array = np.zeros((n_source_cell_types, n_receiver_cell_types, n_interaction_types), dtype=bool)

    for s, source in enumerate(mnc_object.source_cell_types):
        for r, receiver in enumerate(mnc_object.receiver_cell_types):
            for lt, interaction in enumerate(mnc_object.interaction_names):
                ctrl_dist = mnc_object.communication_score_dict[mnc_object.condition_names[0]][:, s, r, lt]
                scz_dist = mnc_object.communication_score_dict[mnc_object.condition_names[1]][:, s, r, lt]

                ctrl_mask = ctrl_dist > threshold
                scz_mask = scz_dist > threshold

                if np.sum(ctrl_mask) >= n_samples and np.sum(scz_mask) >= n_samples:
                    mask[s, r, lt] = True

    mask_xr = xr.DataArray(
        mask,
        dims=('source', 'receiver', 'interaction'),
        coords={
            'source': mnc_object.source_cell_types,
            'receiver': mnc_object.receiver_cell_types,
            'interaction': mnc_object.interaction_names
        }
    )

    return mask_xr


def compute_pooled_threshold_communication_score_mask(
        mnc_object: MultiNeuronChatObject,
        n_samples: Optional[int] = 1,
        threshold: Optional[float] = 0.0,
) -> xr.DataArray:
    """
    Compute a label-blind mask that keeps hypotheses where at least n_samples subjects in the pooled
    (both-condition) sample have a communication score strictly greater than threshold.

    .. deprecated::
        All masks are deprecated and will be removed in a future release; call compute_significance without a mask
        instead. This mask is label-blind and therefore a valid prefilter, but in the simulation benchmark it did not
        improve on testing all hypotheses.

    The subjects of both conditions are pooled into a single sample before the threshold is applied,
    so the condition labels are never read. This makes the filter a symmetric function of the pooled
    sample, and therefore exactly independent of the test statistics under the null hypothesis: every
    test is a function T(x, z) of the pooled scores x and the condition labels z; under the null z is
    exchangeable given x, and a filter that depends only on x is constant on the permutation orbit, so

        P(p <= u | mask selects the triple) = E[ P(p <= u | pooled order statistics) | selected ] <= u

    The independence is exact rather than approximate, and it also covers joint selection across triples
    (Bourgon, Gentleman & Huber 2010; the exchangeability argument extends to the BY correction under
    arbitrary dependence).

    Note that n_samples counts subjects in the pooled sample (sum of both conditions). To approximate
    the old per-condition behaviour of compute_threshold_communication_score_mask at balanced designs,
    set n_samples to twice the per-condition count used there. NaN scores (subjects below
    min_n_cells_threshold or missing genes) are excluded from the count.

    See also: compute_threshold_communication_score_mask (deprecated), compute_variance_mask

    :param mnc_object: A MultiNeuronChatObject containing the computed communication scores
    :param n_samples: The minimum number of subjects in the pooled sample that must have a communication score
                      strictly greater than threshold for the hypothesis to be included in the mask. Default: 1.
    :param threshold: The score threshold. A subject contributes to the count only when its score strictly exceeds
                      this value. Default: 0.0.
    :return: A boolean xarray DataArray with dimensions 'source', 'receiver', and 'interaction'
    """
    _warn_mask_deprecated('compute_pooled_threshold_communication_score_mask')

    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.communication_score_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed communication scores. '
                         'Please run compute_communication_scores() first.')

    if n_samples < 1:
        raise ValueError('n_samples must be a positive integer')

    # Pool subjects of both conditions, discarding labels. The validity of the filter rests on this
    # concatenation: everything below acts on the pooled sample only.
    pooled: xr.DataArray = xr.concat(
        [mnc_object.communication_score_dict[condition] for condition in mnc_object.condition_names],
        dim='subject'
    )

    pooled_np: np.ndarray = pooled.transpose('subject', 'source', 'receiver', 'interaction').to_numpy()

    # NaN scores (missing genes, too few cells) compare as False against any threshold, so they are
    # never counted as exceeding it. np.nansum is used for clarity.
    above: np.ndarray = np.nansum(pooled_np > threshold, axis=0)

    mask: np.ndarray = above >= n_samples

    return _as_cube(mask, mnc_object)


########################################################################################################################
# Helper functions
########################################################################################################################

def _warn_mask_deprecated(
        function_name: str,
        reads_labels: bool = False
) -> None:
    """
    Issue the deprecation warning shared by all mask functions.

    Prefiltering is deprecated as a whole: in the simulation benchmark none of the masks improved on testing all
    hypotheses. Masks that read the condition labels are additionally flagged as invalid prefilters. FutureWarning is
    used rather than DeprecationWarning because the latter is hidden by default unless raised from __main__, and the
    warning is aimed at end users rather than at developers of downstream libraries.

    :param function_name: the public mask function being called
    :param reads_labels: whether the mask depends on the condition labels and is therefore not independent of the test
                         statistics under the null hypothesis
    """
    message: str = (
        f'{function_name} is deprecated and will be removed in a future release. Prefiltering hypotheses with a mask is '
        f'no longer recommended: in the simulation benchmark no mask improved on testing all hypotheses. Call '
        f'compute_significance without a mask instead.'
    )

    if reads_labels:
        message += (
            ' This mask also depends on the condition labels, so it is not independent of the test statistics under the '
            'null hypothesis and biases the retained p-values downward (Bourgon, Gentleman & Huber 2010).'
        )

    # stacklevel=3 attributes the warning to the caller of the public mask function
    warnings.warn(message, FutureWarning, stacklevel=3)


def _validate_selection(
        top_n: Optional[int],
        top_percentile: Optional[float],
        n_hypotheses: int,
        min_value: Optional[float] = None,
        supports_min_value: bool = False
) -> None:
    """
    Validate the mutually exclusive selection arguments shared by the mask functions.

    :param top_n: the requested number of hypotheses to keep, or None
    :param top_percentile: the requested percentile cutoff, or None
    :param n_hypotheses: the total number of hypotheses, i.e. the size of the mask cube
    :param min_value: the requested absolute cutoff, or None. Masks that do not offer an absolute cutoff pass None.
    :param supports_min_value: whether the calling mask exposes min_value, so the error message names only the
                               selection modes that mask actually accepts
    """
    provided = [name for name, value in
                (('top_n', top_n), ('top_percentile', top_percentile), ('min_value', min_value))
                if value is not None]

    if len(provided) == 0:
        available = 'top_n, top_percentile or min_value' if supports_min_value else 'top_n or top_percentile'
        raise ValueError(f'Either {available} must be provided')

    if len(provided) > 1:
        raise ValueError(f'Only one of {", ".join(provided)} must be provided')

    if top_n is not None:
        if top_n <= 0:
            raise ValueError('top_n must be a positive integer')

        # Check if top_n is bigger than the number of hypotheses
        # (product of number of source cell types, receiver cell, and interaction types)
        if top_n > n_hypotheses:
            raise ValueError(f'top_n must be smaller than the number of hypotheses ({n_hypotheses})')

    if top_percentile is not None:
        if top_percentile <= 0 or top_percentile >= 100:
            raise ValueError('top_percentile must be a float between 0 and 100')

    if min_value is not None and not np.isfinite(min_value):
        raise ValueError('min_value must be a finite number')


def _select_top(
        statistic: np.ndarray,
        top_n: Optional[int] = None,
        top_percentile: Optional[float] = None,
        min_value: Optional[float] = None
) -> np.ndarray:
    """
    Turn a cube of per-hypothesis statistics into a boolean mask, keeping the hypotheses with the largest values.

    NaN entries mark hypotheses the statistic could not be computed for and are never selected, whichever selection mode
    is used. If top_n exceeds the number of non-NaN entries, only the non-NaN ones are selected and a warning is issued -
    selecting NaN hypotheses would put untestable triples into the mask and inflate the multiple-testing burden with
    hypotheses that can only return NaN.

    :param statistic: the per-hypothesis statistic, with NaN for hypotheses it could not be computed for
    :param top_n: the number of hypotheses with the largest statistic to select
    :param top_percentile: the percentile cutoff; hypotheses at or above it are selected
    :param min_value: an absolute cutoff; hypotheses strictly greater than it are selected
    :return: a boolean array of the same shape as statistic
    """
    finite: np.ndarray = np.isfinite(statistic)

    if min_value is not None:
        return finite & (statistic > min_value)

    if top_percentile is not None:
        if not np.any(finite):
            return np.zeros_like(statistic, dtype=bool)
        percentile = np.nanpercentile(statistic, top_percentile)
        return finite & (statistic >= percentile)

    n_finite: int = int(np.count_nonzero(finite))

    if top_n > n_finite:
        warnings.warn(
            f'top_n={top_n} exceeds the number of hypotheses with a computable statistic ({n_finite}). '
            f'Selecting those {n_finite} hypotheses instead.',
            stacklevel=3
        )
        top_n = n_finite

    mask: np.ndarray = np.zeros_like(statistic, dtype=bool)

    if top_n == 0:
        return mask

    # Rank only the finite entries. np.argsort sorts NaNs to the end regardless of sign, so they are already never
    # selected while top_n <= n_finite; restricting the sort makes that explicit rather than incidental.
    finite_indices: np.ndarray = np.flatnonzero(finite)
    order: np.ndarray = np.argsort(-statistic.ravel()[finite_indices], kind='stable')

    mask.flat[finite_indices[order[:top_n]]] = True

    return mask


def _as_cube(
        values: np.ndarray,
        mnc_object: MultiNeuronChatObject
) -> xr.DataArray:
    """
    Wrap a (source, receiver, interaction) numpy array in a labelled DataArray with the object's coordinates.

    :param values: the array to wrap, of shape (n_source_cell_types, n_receiver_cell_types, n_interactions)
    :param mnc_object: the MultiNeuronChatObject supplying the coordinate labels
    :return: the values as an xarray DataArray
    """
    return xr.DataArray(
        values,
        dims=('source', 'receiver', 'interaction'),
        coords={
            'source': mnc_object.source_cell_types,
            'receiver': mnc_object.receiver_cell_types,
            'interaction': mnc_object.interaction_names
        }
    )


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)


def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.

    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.

    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.

    """
    # Validate the value array.
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None