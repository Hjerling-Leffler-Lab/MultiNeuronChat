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
    return compute_earth_mover_mask(
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
    if mnc_object is None:
        raise ValueError('A MultiNeuronChatObject must be provided')

    if mnc_object.communication_score_dict is None:
        raise ValueError('The MultiNeuronChatObject must contain the computed communication scores!'
                         'Please run compute_communication_scores() first.')

    n_source_cell_types = len(mnc_object.source_cell_types)
    n_receiver_cell_types = len(mnc_object.receiver_cell_types)
    n_interaction_types = len(mnc_object.interaction_names)
    n_hypotheses = n_source_cell_types * n_receiver_cell_types * n_interaction_types

    if top_n is None and top_percentile is None:
        raise ValueError('Either top_n or top_percentile must be provided')

    if top_n is not None and top_percentile is not None:
        raise ValueError('Only one of top_n or top_percentile must be provided')

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

    mask: np.array = np.zeros((n_source_cell_types, n_receiver_cell_types, n_interaction_types), dtype=bool)

    if top_n is not None:
        # Get the top_n indices with the smallest Wasserstein distances (which can include nan values)
        top_n_indices = np.unravel_index(np.argsort(-wasserstein_distances, axis=None)[:top_n], wasserstein_distances.shape)
        mask = np.zeros_like(wasserstein_distances, dtype=bool)
        mask[top_n_indices] = True
    else:
        percentile = np.nanpercentile(wasserstein_distances, top_percentile)
        mask = wasserstein_distances >= percentile

    mask_xr = xr.DataArray(
        mask,
        dims=('source', 'receiver', 'interaction'),
        coords={
            'source': mnc_object.source_cell_types,
            'receiver': mnc_object.receiver_cell_types,
            'interaction': mnc_object.interaction_names
        }
    )

    wasserstein_distances_xr = xr.DataArray(
        wasserstein_distances,
        dims=('source', 'receiver', 'interaction'),
        coords={
            'source': mnc_object.source_cell_types,
            'receiver': mnc_object.receiver_cell_types,
            'interaction': mnc_object.interaction_names
        }
    )

    if return_wasserstein_distances:
        return mask_xr, wasserstein_distances_xr
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

    :param mnc_object: A MultiNeuronChatObject containing the communication scores
    :param n_samples: The number of samples from each condition that must have a communication score higher than the threshold.
    :param threshold: The threshold that the communication score must exceed.
    :return: A boolean xarray DataArray with dimensions 'source_cell_type', 'receiver_cell_type', and 'interaction_type'
    """

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


########################################################################################################################
# Helper functions
########################################################################################################################

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