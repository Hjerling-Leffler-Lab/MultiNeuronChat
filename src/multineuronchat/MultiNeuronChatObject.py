import math
import os

import pickle

import warnings

import numpy as np
import xarray as xr

from scipy import stats
from scipy.stats import PermutationMethod

from .InteractionDB import InteractionDB

from .utils import gene_filter_and_subject_wise_normalize_dataset
from .power import minimum_detectable_p, n_relabelings, resamples_for_correction
from .MultiNeuronChat import compute_avg_expression, compute_communication_score_matrix

from typing import Tuple, Optional, List, Dict, Set, Any, Union


class MultiNeuronChatObject:
    def __init__(self,
                 condition_label_column: str,
                 condition_names: Tuple[str, str],
                 subject_label_column: str,
                 cell_type_label_column: str,
                 db: str,
                 interaction_db: Optional[InteractionDB] = None):
        if (db not in ['human', 'mouse', 'human_extended']) and (not os.path.exists(db)):
            raise ValueError('db must be either "human", "human_extended", or "mouse" '
                             'or a valid path to an interaction database')

        self.condition_label_column: str = condition_label_column
        self.condition_names: Tuple[str, str] = condition_names

        self.subject_label_column: str = subject_label_column
        self.cell_type_label_column: str = cell_type_label_column

        self.db: str = db

        if interaction_db is None:
            self.interaction_db: InteractionDB = InteractionDB(db=db)
        else:
            self.interaction_db: InteractionDB = interaction_db
        self.gene_set: Set[str] = self.interaction_db.get_set_of_genes()

        self.mean_type: Optional[str] = None
        self.trim_mean_fraction: Optional[float] = None

        # Info about the data
        self.source_cell_types: List[str] = []
        self.receiver_cell_types: List[str] = []
        self.interaction_names: List[str] = []
        self.__n_cell_types: int = 0
        self.__n_interactions: int = 0

        self.avg_expression_per_condition_and_subject_dict: Optional[Dict[str, Dict[str, xr.DataArray]]] = None

        self.communication_scores_per_condition_and_subject_dict: Optional[Dict[str, Dict[str, xr.DataArray]]] = None
        self.ligand_abundance_per_condition_and_subject_dict: Optional[Dict[str, Dict[str, xr.DataArray]]] = None
        self.target_abundance_per_condition_and_subject_dict: Optional[Dict[str, Dict[str, xr.DataArray]]] = None

        self.communication_score_dict: Optional[Dict[str, xr.DataArray]] = None
        self.ligand_abundance_dict: Optional[Dict[str, xr.DataArray]] = None
        self.target_abundance_dict: Optional[Dict[str, xr.DataArray]] = None

        self.p_values: Optional[Dict[str, xr.DataArray]] = None
        self.p_values_adj: Optional[Dict[str, xr.DataArray]] = None
        self.statistics: Optional[Dict[str, Dict[str, xr.DataArray]]] = None

    def compute_communication_scores(
            self,
            path_to_data_loom: str,
            path_to_subject_wise_max_normalized: Optional[str] = None,
            chunk_size: Optional[int] = 1024,
            n_processes: Optional[int] = 1,
            min_n_cells_threshold: Optional[int] = -1,
            mean_type: str = 'tri_mean',
            gene_label_row: str = 'Gene',
            trim_mean_fraction: Optional[float] = None,
            verbose: Optional[bool] = False
    ):
        """
        Compute the communication scores for each subject and condition.

        :param path_to_data_loom: path to the loom file containing the data
        :param path_to_subject_wise_max_normalized: path to the loom file containing the subject wise max normalized data. If this is not provided, it will be generated automatically. If the file already exists the subject wise max normalization will not be recomputed.
        :param chunk_size: chunk size to use for processing the loom file (can be optimized for memory efficiency)
        :param n_processes: number of processes to use for parallelization
        :param min_n_cells_threshold: minimum number of cells required within a donor for a cell-type to be considered
        :param mean_type: type of mean to use for averaging the expression values ('mean', 'tri_mean', or 'trim_mean')
        :param gene_label_row: name of the row in the loom file that contains the gene labels
        :param trim_mean_fraction: fraction to trim when using the trim mean (only required if mean_type is 'trim_mean')
        :param verbose: whether to print progress messages

        :return: None
        """
        if not os.path.exists(path_to_data_loom):
            raise FileNotFoundError(f"File {path_to_data_loom} not found")

        if mean_type not in ['mean', 'tri_mean', 'trim_mean']:
            raise ValueError('mean_type must be either "mean", "tri_mean", or "trim_mean"')
        if mean_type == 'trim_mean' and trim_mean_fraction is None:
            raise ValueError('If mean_type is "trim_mean", trim_mean_fraction must be specified')

        # If the path to the subject wise max normalized data is not provided, generate it
        if path_to_subject_wise_max_normalized is None:
            path_to_subject_wise_max_normalized = path_to_data_loom.replace('.loom', '_MultiNeuronChat.loom')

        # if path_to_subject_wise_max_normalized does not exist -> generate it
        if not os.path.exists(path_to_subject_wise_max_normalized):
            # Filter genes and max normalize each subject
            gene_filter_and_subject_wise_normalize_dataset(
                path_to_loom=path_to_data_loom,
                gene_set=self.gene_set,
                subject_label_column=self.subject_label_column,
                path_to_normalized_loom=path_to_subject_wise_max_normalized,
                gene_label_row=gene_label_row,
                chunk_size=chunk_size,
                verbose=verbose
            )

        # Compute avg expression for each subject
        self.avg_expression_per_condition_and_subject_dict: Dict[str, Dict[str, xr.DataArray]] = compute_avg_expression(
            path_to_loom=path_to_subject_wise_max_normalized,
            condition_label_column=self.condition_label_column,
            condition_label_a=self.condition_names[0],
            condition_label_b=self.condition_names[1],

            subject_label_column=self.subject_label_column,
            cell_type_label_column=self.cell_type_label_column,

            min_n_cells_threshold=min_n_cells_threshold,

            mean_type=mean_type,
            trim_mean_fraction=trim_mean_fraction,

            gene_label_row=gene_label_row,

            n_processes=n_processes
        )

        # Compute interaction score matrix for each subject
        res: Tuple[Dict[str, Dict[str, xr.DataArray]], Dict[str, Dict[str, xr.DataArray]], Dict[str, Dict[str, xr.DataArray]]] = compute_communication_score_matrix(
            avg_expression_dict=self.avg_expression_per_condition_and_subject_dict,
            interaction_db=self.interaction_db,
            n_processes=n_processes
        )
        self.communication_scores_per_condition_and_subject_dict = res[0]
        self.ligand_abundance_per_condition_and_subject_dict = res[1]
        self.target_abundance_per_condition_and_subject_dict = res[2]

        # TODO this is super ugly...
        # Set cell_types and interaction_names
        tmp_condition_dict = self.communication_scores_per_condition_and_subject_dict[self.condition_names[0]]
        tmp_subject = list(tmp_condition_dict.keys())[0]
        self.source_cell_types, self.receiver_cell_types, self.interaction_names = tmp_condition_dict[
            tmp_subject].coords.values()
        self.source_cell_types = self.source_cell_types.to_numpy().tolist()
        self.receiver_cell_types = self.receiver_cell_types.to_numpy().tolist()
        self.interaction_names = self.interaction_names.to_numpy().tolist()

        self.__n_cell_types = len(self.source_cell_types)
        self.__n_interactions = len(self.interaction_names)

        # Combine the distributions
        res: Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray], Dict[str, xr.DataArray]] = self.__combine_distributions()
        self.communication_score_dict = res[0]
        self.ligand_abundance_dict = res[1]
        self.target_abundance_dict = res[2]

    def __combine_distributions(self) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
        """
        Combine the communication scores, ligand abundance, and target abundance for each condition into a single xarray DataArray.

        :return: Tuple of three dictionaries containing the combined communication scores, ligand abundance, and target abundance for each condition
        """
        combined_communication_score_dict: Dict[str, xr.DataArray] = {}
        combined_ligand_abundance_dict: Dict[str, xr.DataArray] = {}
        combined_target_abundance_dict: Dict[str, xr.DataArray] = {}

        for condition in self.communication_scores_per_condition_and_subject_dict.keys():
            subject_dict_communication_scores: Dict[str, xr.DataArray] = self.communication_scores_per_condition_and_subject_dict[condition]
            subject_dict_ligand_abundance: Dict[str, xr.DataArray] = self.ligand_abundance_per_condition_and_subject_dict[condition]
            subject_dict_target_abundance: Dict[str, xr.DataArray] = self.target_abundance_per_condition_and_subject_dict[condition]

            subject_list: List[str] = list(subject_dict_communication_scores.keys())
            n_subjects: int = len(subject_list)

            condition_communication_score_np: np.array = np.zeros(shape=(n_subjects, self.__n_cell_types, self.__n_cell_types, self.__n_interactions))
            condition_ligand_abundance_np: np.array = np.zeros(shape=(n_subjects, self.__n_cell_types, self.__n_interactions))
            condition_target_abundance_np: np.array = np.zeros(shape=(n_subjects, self.__n_cell_types, self.__n_interactions))

            for i, subject in enumerate(subject_list):
                condition_communication_score_np[i, :, :, :] = subject_dict_communication_scores[subject]
                condition_ligand_abundance_np[i, :, :] = subject_dict_ligand_abundance[subject]
                condition_target_abundance_np[i, :, :] = subject_dict_target_abundance[subject]

            condition_array: xr.DataArray = xr.DataArray(
                data=condition_communication_score_np,
                dims=['subject', 'source', 'receiver', 'interaction'],
                coords=[subject_list, self.source_cell_types, self.receiver_cell_types, self.interaction_names]
            )

            condition_array_ligand_abundance: xr.DataArray = xr.DataArray(
                data=condition_ligand_abundance_np,
                dims=['subject', 'source', 'interaction'],
                coords=[subject_list, self.source_cell_types, self.interaction_names]
            )

            condition_array_target_abundance: xr.DataArray = xr.DataArray(
                data=condition_target_abundance_np,
                dims=['subject', 'receiver', 'interaction'],
                coords=[subject_list, self.receiver_cell_types, self.interaction_names]
            )

            combined_communication_score_dict[condition] = condition_array
            combined_ligand_abundance_dict[condition] = condition_array_ligand_abundance
            combined_target_abundance_dict[condition] = condition_array_target_abundance

        return combined_communication_score_dict, combined_ligand_abundance_dict, combined_target_abundance_dict

    def __get_condition_specific_communication_scores_for_p_value_test(
            self,
            condition: str,
            source: Union[str, int],
            receiver: Union[str, int],
            interaction: Union[str, int]
    ) -> np.ndarray:
        """
        Get the communication scores for a specific source, receiver, and interaction for a specific condition.
        Filter out all NaN values.

        :param condition: The condition for which the communication scores should be extracted
        :param source: the source cell-type
        :param receiver: the receiver cell-type
        :param interaction: the ligand-target interaction to extract
        :return: numpy array containing the communication scores for the specified condition
        """
        # check if dtype is all the same
        if not ((isinstance(source, str) and isinstance(receiver, str) and isinstance(interaction, str)) or \
                (isinstance(source, int) and isinstance(receiver, int) and isinstance(interaction, int)) or \
                (isinstance(source, np.integer) and isinstance(receiver, np.integer) and isinstance(interaction,
                                                                                                    np.integer))):
            raise ValueError('source, receiver, and interaction must be of the same type!')

        if isinstance(source, str):
            condition_communication_scores = self.communication_score_dict[condition].loc[:, source, receiver, interaction].to_numpy()
        elif isinstance(source, int) or isinstance(source, np.integer):
            condition_communication_scores = self.communication_score_dict[condition][:, source, receiver, interaction].to_numpy()
        else:
            raise ValueError('source must be either a string or an integer!')

        # Filter out NaN values
        condition_communication_scores = condition_communication_scores[~np.isnan(condition_communication_scores)]

        return condition_communication_scores

    def __get_communication_scores_for_p_value_test(
            self,
            source: Union[str, int],
            receiver: Union[str, int],
            interaction: Union[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the communication scores for a specific source, receiver, and interaction. Filter out all NaN values.
        :param source: the source cell-type
        :param receiver: the receiver cell-type
        :param interaction: the ligand-target interaction to extract
        :return: Tuple of two numpy arrays containing the communication scores for the two conditions
        """
        res: tuple[np.ndarray, np.ndarray] = (
            self.__get_condition_specific_communication_scores_for_p_value_test(
                condition=self.condition_names[0],
                source=source,
                receiver=receiver,
                interaction=interaction
            ),
            self.__get_condition_specific_communication_scores_for_p_value_test(
                condition=self.condition_names[1],
                source=source,
                receiver=receiver,
                interaction=interaction
            )
        )

        return res

    @staticmethod
    def __correct_p_values(
            p_values: np.ndarray,
            method: str = 'bh'
    ) -> np.ndarray:
        """
        Correct p-values using the Benjamini-Hochberg (default) or Benjamini-Yekutieli method and return the corrected
        p-values.
        Exclude any nan values from the correction.
        :param p_values: the p-values to correct
        :param method: the method to use for correction (default: 'bh')
        :return: the corrected p-values with nan values where the p-values were nan
        """
        if method not in ['by', 'bh']:
            raise ValueError('method must be either "by" or "bh"')

        # initiate array with nan values
        p_values_corrected_full = np.ones_like(p_values) * np.nan

        # select only non nan values for correction
        non_nan_p_values = p_values[~np.isnan(p_values)]

        # correct p-values
        p_values_corrected = stats.false_discovery_control(
            ps=non_nan_p_values,
            method=method
        )

        # insert corrected p-values into full array
        p_values_corrected_full[~np.isnan(p_values)] = p_values_corrected

        return p_values_corrected_full

    @property
    def _cube_shape(self) -> tuple[int, int, int]:
        """Shape of a single (source, receiver, interaction) cube."""
        return tuple(self.communication_score_dict[self.condition_names[0]].shape[1:])

    def _validate_or_make_mask(self, mask: Optional[Union[np.ndarray, xr.DataArray]]) -> np.ndarray:
        """Ensure mask has correct shape and dtype; create a full-True mask if None."""
        shape = self._cube_shape
        if mask is None:
            return np.ones(shape=shape, dtype=bool)
        if isinstance(mask, xr.DataArray):
            mask_arr = mask.values
        else:
            mask_arr = mask
        if mask_arr.shape != shape:
            raise ValueError('The shape of the mask does not match the shape of the communication scores')
        return mask_arr.astype(bool, copy=False)


    def correct_p_values(
            self,
            statistical_test: Optional[Union[str, List[str]]] = None,
            method: str = 'bh',
            alpha: float = 0.05,
            warn_if_underpowered: bool = True
    ) -> Union[xr.DataArray, Dict[str, xr.DataArray]]:
        """
        This function corrects the p-values using the Benjamini-Hochberg (default) or Benjamini-Yekutieli method. The
        corrected p-values are stored in the MultiNeuronChatObject. If statistical_test is None, all p-values are
        corrected. If statistical_test is a string, only the p-values of the specified test are corrected. If
        statistical_test is a list of strings, only the p-values of the specified tests are corrected.

        :param statistical_test: the test for which the p-values should be corrected
        :param method: the method to use for correction (default: 'bh')
        :param alpha: the significance level assumed by the power check; it does not threshold the output (default: 0.05)
        :param warn_if_underpowered: whether to warn, per test, when the number of subjects is too small for an isolated
                                     triple to reach significance over the triples that returned a p-value. Unlike the
                                     check in compute_significance this uses the final hypothesis count (default: True)
        :return: the corrected p-values as a DataArray or a dictionary of DataArrays
        """
        if not 0 < alpha <= 1:
            raise ValueError('alpha must be in the interval (0, 1]')

        if statistical_test is None or isinstance(statistical_test, list):
            # compute significance for all tests that have been computed or the specific list of tests
            if statistical_test is None:
                statistical_test = list(self.p_values.keys())

            not_included_tests = [test for test in statistical_test if test not in self.p_values.keys()]

            if len(not_included_tests) > 0:
                raise ValueError(f'Tests {not_included_tests} have not been computed yet')

            self.p_values_adj = {}

            for test in statistical_test:
                p_values = self.p_values[test].values

                if warn_if_underpowered:
                    self._warn_if_underpowered(
                        n_hypotheses=int(np.count_nonzero(~np.isnan(p_values))),
                        method=method,
                        alpha=alpha
                    )

                p_values_adj = self.__correct_p_values(
                    p_values,
                    method=method
                )

                p_values_adj_xr = xr.DataArray(
                    data=p_values_adj,
                    dims=['source', 'receiver', 'interaction'],
                    coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
                )

                self.p_values_adj[test] = p_values_adj_xr

            # if only one test was computed, return the p-values directly
            if len(statistical_test) == 1:
                return self.p_values_adj[statistical_test[0]]
            # if multiple tests were computed, return the p-values as a dictionary
            return self.p_values_adj
        elif isinstance(statistical_test, str):
            # compute significance for the specified test
            if statistical_test not in self.p_values.keys():
                raise ValueError(f'Test {statistical_test} has not been computed yet')

            p_values = self.p_values[statistical_test].values

            if warn_if_underpowered:
                self._warn_if_underpowered(
                    n_hypotheses=int(np.count_nonzero(~np.isnan(p_values))),
                    method=method,
                    alpha=alpha
                )

            p_values_adj = self.__correct_p_values(
                p_values,
                method=method
            )

            p_values_adj_xr = xr.DataArray(
                data=p_values_adj,
                dims=['source', 'receiver', 'interaction'],
                coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
            )

            self.p_values_adj[statistical_test] = p_values_adj_xr

            return p_values_adj_xr
        else:
            raise ValueError('If you define statistical_test, it must be either a string or a list of strings,'
                             'where each string is a valid test')

    @staticmethod
    def _needs_permutation(p: float, coarse_levels: Optional[set] = None) -> bool:
        """Return True if we should recompute via permutation.
        We only trigger when p is non-finite, equals 0.0, or equals a known coarse/clamped level (e.g., 0.001 for Anderson–Darling).
        """
        return not np.isfinite(p) or p == 0.0 or (coarse_levels and (p in coarse_levels))

    @staticmethod
    def _ensure_nonzero_p(p: float, n_resamples: Optional[int] = None) -> float:
        """Ensure returned p-value is strictly positive without imposing an arbitrary lower bound.
        If p==0 from a permutation test, return 1/(n_resamples+1). Otherwise, map exact 0.0 to the
        smallest positive float.
        """
        if p == 0.0:
            if n_resamples is not None and n_resamples > 0:
                return 1.0 / (n_resamples + 1)
            # fallback for analytical edge cases
            return np.nextafter(0.0, 1.0)
        return p

    @staticmethod
    def __perm_pvalue_independent(
            A: np.ndarray,
            B: np.ndarray,
            stat_fn,
            n_resamples: int = 10_000,
            alternative: str = 'two-sided',
            random_state: Optional[int] = None
    ) -> float:
        """Return permutation p-value for an independent two-sample statistic.
        Uses scipy.stats.permutation_test under the hood.
        """
        res = stats.permutation_test(
            (A, B),
            statistic=stat_fn,
            permutation_type='independent',
            n_resamples=n_resamples,
            alternative=alternative,
            random_state=random_state,
        )
        return float(res.pvalue)

    @staticmethod
    def _monte_carlo_floor(n_resamples: Optional[int]) -> Optional[float]:
        """
        Return the smallest p-value a permutation test with n_resamples draws can report, 1 / (n_resamples + 1).

        The observed statistic is always counted among the resampled ones, so no Monte-Carlo p-value can fall below this
        value. A returned p-value equal to it therefore carries no information beyond "smaller than the resolution of
        this run", which is precisely the condition for escalating the triple to a deeper stage.

        :param n_resamples: the number of resamples used, or None if the p-value did not come from a permutation test
        :return: the Monte-Carlo floor, or None if n_resamples is None or not positive
        """
        if n_resamples is None or n_resamples < 1:
            return None

        return 1.0 / (n_resamples + 1)

    @staticmethod
    def _is_at_monte_carlo_floor(p: float, n_resamples: Optional[int]) -> bool:
        """
        Return True if p is exactly the Monte-Carlo floor of a run with n_resamples draws.

        The comparison is deliberately made against the floor itself and never against a significance threshold: the
        trigger for escalation must depend only on the *resolution* of the first stage, not on how significant the
        result looks. Selecting triples for a deeper stage on the basis of their significance would condition the second
        stage on the first stage's outcome and inflate the type-I error rate.

        :param p: the p-value returned by the first stage
        :param n_resamples: the number of resamples the first stage used
        :return: True if p sits exactly at the floor and the triple should be escalated
        """
        floor: Optional[float] = MultiNeuronChatObject._monte_carlo_floor(n_resamples)

        if floor is None or not np.isfinite(p):
            return False

        return bool(np.isclose(p, floor, rtol=1e-12, atol=0.0))

    @staticmethod
    def _derive_seed(
            random_state: Optional[int],
            source: int,
            receiver: int,
            interaction: int
    ) -> Optional[int]:
        """
        Derive a per-triple seed for the escalation stage from the run-level random_state.

        The second stage must draw *fresh* resamples: reusing the first stage's seed would reproduce the same
        permutations and add no information. Deriving the seed from the run-level seed and the triple's position keeps
        the whole run reproducible while making every escalated triple use a different, independent set of draws.

        :param random_state: the run-level random state, or None for a non-reproducible run
        :param source: index of the source cell type
        :param receiver: index of the receiver cell type
        :param interaction: index of the interaction
        :return: the derived seed, or None if random_state is None
        """
        if random_state is None:
            return None

        return int(
            np.random.SeedSequence(
                entropy=int(random_state),
                spawn_key=(int(source), int(receiver), int(interaction))
            ).generate_state(1)[0]
        )

    def _resample_budget(
            self,
            n_hypotheses: int,
            method: str = 'bh',
            alpha: float = 0.05
    ) -> int:
        """
        Return the number of resamples the escalation stage should use for this design.

        Two bounds apply. The correction sets how deep the run *needs* to go, R_min = ceil(m * c(m) / alpha), because a
        Monte-Carlo p-value pinned at 1 / (R + 1) cannot survive correction over m hypotheses otherwise. The design sets
        how deep it *can* usefully go, C(n_control + n_case, n_control), because that is the size of the permutation
        reference set: at or above it SciPy enumerates every relabeling and returns the exact p-value, so further
        resamples are strictly wasted.

        Taking the minimum means small cohorts land on the exact p-value cheaply - with 6 subjects per group the whole
        reference set has 924 elements, far fewer than the 10,000 draws a fixed budget would spend - while large cohorts
        pay only what the correction demands.

        :param n_hypotheses: number of tested hypotheses entering the multiple-testing correction
        :param method: correction method assumed for the budget, 'by' or 'bh'
        :param alpha: significance level assumed for the budget
        :return: the number of resamples for the escalation stage
        """
        n_control: int = self.communication_score_dict[self.condition_names[0]].sizes['subject']
        n_case: int = self.communication_score_dict[self.condition_names[1]].sizes['subject']

        required: int = resamples_for_correction(
            n_hypotheses=n_hypotheses,
            method=method,
            alpha=alpha
        )

        return min(required, n_relabelings(n_control=n_control, n_case=n_case))

    def _warn_if_underpowered(
            self,
            n_hypotheses: int,
            method: str = 'bh',
            alpha: float = 0.05,
            counted_before_testing: bool = False
    ) -> None:
        """
        Warn if the study design limits which results can reach significance.

        The p-values of all supported tests are counts of subject relabelings, so they cannot fall below
        p_min = 2 / C(n_control + n_case, n_control) (see power.minimum_detectable_p). BH and BY are step-up procedures:
        they reject the k smallest p-values for the largest k with p_(k) <= k * alpha / (m * c(m)). Because every p_(k) is
        at least p_min, a rejection at rank k requires k >= k_min = ceil(p_min * m * c(m) / alpha). Two regimes follow:

        - k_min > m, i.e. p_min * c(m) > alpha: even if every tested triple attained the floor, none would survive the
          correction, so no result can be significant.
        - 1 < k_min <= m: a single, isolated signal cannot be significant, but significant triples can still occur as a
          group of at least k_min, e.g. k_min triples that separate the conditions perfectly and tie at p_min.

        Both statements are arithmetic on the correction and hold under any dependence between the triples.

        :param n_hypotheses: the number of hypotheses m entering the multiple-testing correction
        :param method: the correction method assumed for the check (default: 'bh', matching correct_p_values)
        :param alpha: the significance level assumed for the check (default: 0.05)
        :param counted_before_testing: True if n_hypotheses counts the triples selected for testing rather than those
                                       that returned a p-value. Triples returning NaN do not enter the correction and
                                       lower the bound, so the warning is then worded as conditional on all of them
                                       yielding a p-value (default: False)
        :return: None
        """
        if n_hypotheses < 1:
            return

        n_control: int = self.communication_score_dict[self.condition_names[0]].sizes['subject']
        n_case: int = self.communication_score_dict[self.condition_names[1]].sizes['subject']

        # A lone triple at the floor: the rank-1 step of the correction.
        p_min, p_min_adj = minimum_detectable_p(
            n_control=n_control,
            n_case=n_case,
            n_hypotheses=n_hypotheses,
            method=method
        )

        if p_min_adj <= alpha:
            return

        # All tested triples at the floor: the most favourable case the step-up procedure allows, p_min * c(m).
        _, p_floor_all = minimum_detectable_p(
            n_control=n_control,
            n_case=n_case,
            n_hypotheses=n_hypotheses,
            method=method,
            rank=n_hypotheses
        )

        design: str = (
            f'With {n_control} {self.condition_names[0]} and {n_case} {self.condition_names[1]} subjects the smallest '
            f'attainable p-value is {p_min:.2e}'
        )

        if counted_before_testing:
            scope: str = f'if all {n_hypotheses} tested triples yield a p-value'
            caveat: str = (
                ' Triples that return NaN do not enter the correction and relax this bound; correct_p_values repeats the '
                'check on the final count.'
            )
        else:
            scope: str = f'over the {n_hypotheses} triples with a p-value'
            caveat: str = ''

        if p_floor_all > alpha:
            warnings.warn(
                f'This study design cannot produce a significant result at any effect size {scope}. {design}; even if '
                f'all {n_hypotheses} triples attained it, {method.upper()} correction would raise it to '
                f'{p_floor_all:.2e} (alpha={alpha}). This is a property of the number of subjects, not of the data: only '
                f'additional subjects or fewer tested hypotheses can lower this bound.{caveat} See minimum_detectable_p '
                f'for details.',
                stacklevel=3
            )
            return

        k_min: int = math.ceil(p_floor_all * n_hypotheses / alpha)

        warnings.warn(
            f'This study design cannot detect an isolated effect {scope}. {design}, which becomes {p_min_adj:.2e} '
            f'after {method.upper()} correction over {n_hypotheses} hypotheses (alpha={alpha}) if a single triple '
            f'attains it. Because the correction is step-up, significant triples can still occur, but only as a group '
            f'of at least {k_min} simultaneously significant triples (for example {k_min} triples that separate the '
            f'conditions perfectly).{caveat} See minimum_detectable_p (rank parameter) for details.',
            stacklevel=3
        )

    def compute_significance(
            self,
            statistical_test: Optional[str] = 'CVM',
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None,
            warn_if_underpowered: Optional[bool] = True,
            correction_method: Optional[str] = 'bh',
            alpha: Optional[float] = 0.05,
            escalate_permutation: Optional[bool] = True
    ) -> xr.DataArray:
        """
        Compute the p-values for the specified statistical test. The p-values are stored in the MultiNeuronChatObject.
        If mask is None, the significance is tested for all interactions and cell-type pairs. If a mask is provided,
        the significance is only computed for the specified interactions and cell-type pairs, setting all other p-values
        to np.nan.

        correction_method and alpha describe the multiple-testing correction the resulting p-values are destined for.
        They do not apply a correction here - that remains the job of correct_p_values - but they determine how deep the
        permutation fallback has to resample for its p-values to be able to survive that correction (see
        power.resamples_for_correction). Both default to the values correct_p_values itself defaults to, so neither
        needs to be provided unless a different correction is intended. If they understate the eventual correction the
        result is conservative, never invalid: the p-values are merely resampled less deeply than they could have been.

        :param statistical_test: the statistical test to compute the p-values for (default: 'CVM', i.e. Cramér-von Mises).
        :param mask: a mask to specify for which interactions and cell-type pairs the significance should be computed.
        :param n_resamples: number of resamples for permutation fallback (default 10_000 if None)
        :param random_state: random state for reproducibility (default None)
        :param warn_if_underpowered: whether to warn when the number of subjects is too small for an isolated triple to
                                     reach significance after correction, stating whether any triple can (default: True)
        :param correction_method: the multiple-testing correction the p-values are intended for, 'by' or 'bh'
                                  (default: 'bh', matching correct_p_values)
        :param alpha: the significance level the p-values are intended to be judged at (default: 0.05)
        :param escalate_permutation: whether to re-run triples whose permutation p-value is pinned at the Monte-Carlo
                                     floor with the deeper resample budget the correction requires (default: True).
                                     Only affects the 'Anderson' test, the only one whose analytical p-value is clamped
                                     and therefore always routed through the permutation fallback.
        :return: the p-values as a DataArray.
        """
        mask = self._validate_or_make_mask(mask)

        if statistical_test not in ['KS', 'Anderson', 'CVM', 'MannWhitneyU']:
            raise ValueError('Statistical_test must be either "KS", "Anderson", "CVM", or "MannWhitneyU"')

        if correction_method not in ['by', 'bh']:
            raise ValueError('correction_method must be either "by" or "bh"')

        if not 0 < alpha <= 1:
            raise ValueError('alpha must be in the interval (0, 1]')

        n_hypotheses: int = int(np.count_nonzero(mask))

        if warn_if_underpowered:
            self._warn_if_underpowered(
                n_hypotheses=n_hypotheses,
                method=correction_method,
                alpha=alpha,
                counted_before_testing=True
            )

        if random_state is None:
            if statistical_test == 'Anderson':
                warnings.warn(
                    'random_state=None: Anderson-Darling always uses the permutation fallback, '
                    'so results will not be reproducible across runs. '
                    'Pass an integer to random_state for reproducible results.',
                    UserWarning,
                    stacklevel=2
                )
            else:
                warnings.warn(
                    f'random_state=None: if {statistical_test} falls back to a permutation test '
                    'for any triple, those p-values will not be reproducible across runs. '
                    'Pass an integer to random_state for reproducible results.',
                    UserWarning,
                    stacklevel=2
                )

        if statistical_test == 'KS':
            return self.__compute_p_values_KS(
                mask=mask,
                n_resamples=n_resamples,
                random_state=random_state
            )
        elif statistical_test == 'Anderson':
            return self.__compute_p_values_Anderson(
                mask=mask,
                n_resamples=n_resamples,
                random_state=random_state,
                escalate_permutation=escalate_permutation,
                correction_method=correction_method,
                alpha=alpha
            )
        elif statistical_test == 'CVM':
            return self.__compute_p_values_CVM(
                mask=mask,
                n_resamples=n_resamples,
                random_state=random_state
            )
        elif statistical_test == 'MannWhitneyU':
            return self.__compute_p_values_MannWhitneyU(
                mask=mask,
                n_resamples=n_resamples,
                random_state=random_state
            )

    def __compute_p_values_KS(
            self,
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None
    ) -> xr.DataArray:
        mask = self._validate_or_make_mask(mask)

        # Set up numpy arrays with NaN values
        shape = (self.__n_cell_types, self.__n_cell_types, self.__n_interactions)
        p_values = np.full(shape, np.nan, dtype=float)
        test_statistic = np.full(shape, np.nan, dtype=float)
        test_statistic_location = np.full(shape, np.nan, dtype=float)
        test_statistic_sign = np.full(shape, np.nan, dtype=float)

        # Compute p-values where mask is True
        idx = np.where(mask)
        _permutation_fallback_warned = False

        for source, receiver, interaction in zip(*idx):
            condition_a_communication_scores, condition_b_communication_scores = self.__get_communication_scores_for_p_value_test(
                source=source,
                receiver=receiver,
                interaction=interaction
            )

            # Check if any of the two conditions has no communication scores
            # If so, we have to set the p-value and the test statistic to np.nan as
            # the KS-Test cannot be computed on empty distributions

            if len(condition_a_communication_scores) == 0 or len(condition_b_communication_scores) == 0:
                continue

            # Check if both distributions are only zeros
            # If this is the case: set p-value and test statistic to np.nan
            if np.max(condition_a_communication_scores) == 0 and np.max(condition_b_communication_scores) == 0:
                continue

            ks_statistic = stats.ks_2samp(
                data1=condition_a_communication_scores,
                data2=condition_b_communication_scores,
            )

            # store observed stats
            test_statistic[source, receiver, interaction] = ks_statistic.statistic
            test_statistic_location[source, receiver, interaction] = ks_statistic.statistic_location
            test_statistic_sign[source, receiver, interaction] = ks_statistic.statistic_sign

            p = float(ks_statistic.pvalue)
            if self._needs_permutation(p):
                if random_state is None and not _permutation_fallback_warned:
                    warnings.warn(
                        'KS permutation fallback triggered: p-values for affected triples are not '
                        'reproducible. Pass an integer to random_state to fix this.',
                        UserWarning,
                        stacklevel=5
                    )
                    _permutation_fallback_warned = True
                p_perm = self.__perm_pvalue_independent(
                    condition_a_communication_scores,
                    condition_b_communication_scores,
                    stat_fn=lambda a, b: stats.ks_2samp(a, b).statistic,
                    n_resamples=n_resamples,
                    alternative='two-sided',
                    random_state=random_state
                )
                p_final = self._ensure_nonzero_p(p_perm, n_resamples)
            else:
                p_final = self._ensure_nonzero_p(p)
            p_values[source, receiver, interaction] = p_final

        p_values_xr = xr.DataArray(
            data=p_values,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        test_statistic_xr: xr.DataArray = xr.DataArray(
            data=test_statistic,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )
        test_statistic_location_xr: xr.DataArray = xr.DataArray(
            data=test_statistic_location,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )
        test_statistic_sign_xr: xr.DataArray = xr.DataArray(
            data=test_statistic_sign,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        if self.p_values is None:
            self.p_values = {}
            self.p_values_adj = {}
            self.statistics = {}

        self.p_values['KS'] = p_values_xr
        self.statistics['KS'] = {
            'statistics': test_statistic_xr,
            'location': test_statistic_location_xr,
            'sign': test_statistic_sign_xr
        }

        return p_values_xr

    def __compute_p_values_Anderson(
            self,
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            permutation_method: Optional[bool] = False,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None,
            escalate_permutation: Optional[bool] = True,
            correction_method: Optional[str] = 'bh',
            alpha: Optional[float] = 0.05
    ) -> xr.DataArray:
        """
        Compute Anderson-Darling p-values, resampling deeply enough that they can survive multiple-testing correction.

        Anderson-Darling is the only supported test whose analytical p-value is clamped by SciPy (to the interval
        [0.001, 0.25]), so every strongly separated triple is routed through the permutation fallback. That fallback
        carries its own resolution limit of 1 / (n_resamples + 1), and at the conventional 10,000 resamples the
        resulting 9.999e-05 cannot survive correction over a realistic number of triples - which makes exactly the
        strongest effects unreportable. Two rules remove that limit:

        1. The resample count is capped at the size of the permutation reference set, C(n_control + n_case, n_control).
           At or above it SciPy enumerates every relabeling and returns the *exact* p-value, so small cohorts become
           both exact and cheap rather than being sampled with a budget larger than the reference set itself.
        2. Triples whose first-stage p-value comes back pinned at the Monte-Carlo floor are re-run once with the deeper
           budget the correction requires (Besag & Clifford 1991). Only those triples pay the extra cost; under the null
           the floor is essentially never reached, so the escalation rate is a fraction of a percent.

        The escalation must satisfy three conditions to keep the type-I error rate valid, and all three are load-bearing:
        it triggers only on a p-value being *exactly* at the floor (never on a significance threshold, which would
        condition the second stage on the first stage's outcome), it draws *fresh* resamples via a derived seed, and it
        *replaces* the first-stage p-value outright. Taking the minimum of the two stages, or pooling their counts,
        inflates the error rate and must not be done.

        Mixing resolutions across triples is safe for Benjamini-Yekutieli and Benjamini-Hochberg, which require only
        that each p-value is super-uniform under its own null, not that all of them share a common resolution.

        :param mask: a mask selecting the triples to test
        :param permutation_method: whether to skip the analytical test and go straight to the permutation test
        :param n_resamples: number of resamples for the first stage (default 10_000 if None)
        :param random_state: random state for reproducibility (default None)
        :param escalate_permutation: whether to run the second stage for floor-pinned triples (default: True)
        :param correction_method: the correction the p-values are intended for, used to size the second stage
        :param alpha: the significance level the p-values are intended to be judged at
        :return: the p-values as a DataArray
        """
        mask = self._validate_or_make_mask(mask)

        if permutation_method and n_resamples < 1:
            raise ValueError('The number of resamples must be greater than 0')

        # Set up numpy arrays with NaN values
        shape = (self.__n_cell_types, self.__n_cell_types, self.__n_interactions)
        p_values = np.full(shape, np.nan, dtype=float)
        test_statistic = np.full(shape, np.nan, dtype=float)

        # Compute p-values where mask is True
        idx = np.where(mask)

        # Rule 1: never resample beyond the permutation reference set. Both stages are capped at it, which makes the
        # first stage already exact whenever the design is small enough for the whole reference set to be enumerated.
        n_control: int = self.communication_score_dict[self.condition_names[0]].sizes['subject']
        n_case: int = self.communication_score_dict[self.condition_names[1]].sizes['subject']
        reference_set_size: int = n_relabelings(n_control=n_control, n_case=n_case)

        n_resamples_stage_one: int = min(n_resamples, reference_set_size)

        # Rule 2: the second stage is sized by the correction the p-values must survive, capped again at the reference
        # set. If the first stage already enumerates the reference set its p-values are exact and nothing can be gained
        # by escalating, so the second stage is switched off in that case.
        n_resamples_stage_two: int = self._resample_budget(
            n_hypotheses=int(np.count_nonzero(mask)),
            method=correction_method,
            alpha=alpha
        )

        escalate: bool = (
            bool(escalate_permutation)
            and n_resamples_stage_two > n_resamples_stage_one
        )

        n_escalated: int = 0

        for source, receiver, interaction in zip(*idx):
            condition_a_communication_scores, condition_b_communication_scores = self.__get_communication_scores_for_p_value_test(
                source=source,
                receiver=receiver,
                interaction=interaction
            )

            # The Anderson Darling Test does work if all values of the data are the same
            # Therefore we have to check if the data is the same and if so we have to skip the test
            # IMPORTANT: this includes the case when both distributions are the same, e.g., both are completely zero
            if np.unique(np.hstack([condition_a_communication_scores, condition_b_communication_scores])).shape[0] == 1:
                continue

            # TODO Check this; I had a bug, where when both samples are only one sample long, the Anderson test does not work.
            # This seems to be the case because of an scipy implementation detail where they arrange N-1 sampes, i.e.,
            # if N=1, then they have 0 samples. This then lead to an error for the Anderson test.
            # Therefore, I have decided to restrict the test to the case where both samples have at least 2 samples.
            # This is not ideal, but I think it is the best solution for now.
            if len(condition_a_communication_scores) < 2 or len(condition_b_communication_scores) < 2:
                continue

            # Track whether the reported p-value came from a permutation run, and at what depth, so the escalation
            # trigger and the zero-guard below both refer to the resolution that actually produced it.
            n_resamples_used: Optional[int] = None

            # First try the default (analytical/approximate) unless permutation was explicitly requested
            if not permutation_method:
                anderson_statistic = stats.anderson_ksamp(
                    samples=[
                        condition_a_communication_scores,
                        condition_b_communication_scores,
                    ]
                )
                p = float(anderson_statistic.pvalue)
                # SciPy's approximate p can be clipped (~<=0.001). If clipped or non-finite, redo via permutation.
                if self._needs_permutation(p, coarse_levels={0.001}):
                    anderson_statistic = stats.anderson_ksamp(
                        samples=[
                            condition_a_communication_scores,
                            condition_b_communication_scores,
                        ],
                        method=PermutationMethod(
                            n_resamples=n_resamples_stage_one,
                            random_state=random_state
                        )
                    )
                    n_resamples_used = n_resamples_stage_one
            else:
                anderson_statistic = stats.anderson_ksamp(
                    samples=[
                        condition_a_communication_scores,
                        condition_b_communication_scores
                    ],
                    method=PermutationMethod(
                        n_resamples=n_resamples_stage_one,
                        random_state=random_state
                    )
                )
                n_resamples_used = n_resamples_stage_one

            # Second stage: this triple's p-value is pinned at the resolution of the first stage and carries no
            # information beyond "smaller than 1 / (R + 1)". Re-run it deeply enough to survive the correction, with
            # fresh draws from a derived seed, and replace the first-stage value outright.
            if escalate and self._is_at_monte_carlo_floor(float(anderson_statistic.pvalue), n_resamples_used):
                anderson_statistic = stats.anderson_ksamp(
                    samples=[
                        condition_a_communication_scores,
                        condition_b_communication_scores,
                    ],
                    method=PermutationMethod(
                        n_resamples=n_resamples_stage_two,
                        random_state=self._derive_seed(
                            random_state=random_state,
                            source=source,
                            receiver=receiver,
                            interaction=interaction
                        )
                    )
                )
                n_resamples_used = n_resamples_stage_two
                n_escalated += 1

            p_values[source, receiver, interaction] = self._ensure_nonzero_p(float(anderson_statistic.pvalue), n_resamples_used)
            test_statistic[source, receiver, interaction] = float(anderson_statistic.statistic)

        # If escalation was disabled or capped below what the correction needs, p-values can still be pinned at the
        # Monte-Carlo floor. Those triples are unreportable no matter how large their effect, so say so explicitly
        # rather than letting them silently fail the correction.
        floor_stage_two: Optional[float] = self._monte_carlo_floor(n_resamples_stage_two if escalate else n_resamples_stage_one)

        if floor_stage_two is not None and n_resamples_stage_two < reference_set_size:
            n_at_floor: int = int(np.count_nonzero(np.isclose(p_values, floor_stage_two, rtol=1e-12, atol=0.0)))

            if n_at_floor > 0:
                warnings.warn(
                    f'{n_at_floor} triple(s) returned a p-value pinned at the permutation resolution '
                    f'{floor_stage_two:.2e} = 1/(n_resamples+1). Their true p-value is smaller, but cannot be resolved '
                    f'at this resample budget, so they may fail the multiple-testing correction regardless of effect '
                    f'size. Increase n_resamples, or enable escalate_permutation, to resolve them.'
                )

        p_values_anderson_xr = xr.DataArray(
            data=p_values,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        test_statistic_anderson_xr: xr.DataArray = xr.DataArray(
            data=test_statistic,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        if self.p_values is None:
            self.p_values = {}
            self.p_values_adj = {}
            self.statistics = {}

        self.p_values['Anderson'] = p_values_anderson_xr
        self.statistics['Anderson'] = {
            'statistics': test_statistic_anderson_xr
        }

        return p_values_anderson_xr

    def __compute_p_values_CVM(
            self,
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None
    ) -> xr.DataArray:
        mask = self._validate_or_make_mask(mask)

        # Set up numpy arrays with NaN values
        shape = (self.__n_cell_types, self.__n_cell_types, self.__n_interactions)
        p_values = np.full(shape, np.nan, dtype=float)
        test_statistic = np.full(shape, np.nan, dtype=float)

        # Compute p-values where mask is True
        idx = np.where(mask)
        _permutation_fallback_warned = False

        for source, receiver, interaction in zip(*idx):
            condition_a_communication_scores, condition_b_communication_scores = self.__get_communication_scores_for_p_value_test(
                source=source,
                receiver=receiver,
                interaction=interaction
            )

            # The CVM Test does work if all values of the data are the same
            # Therefore we have to check if the data is the same and if so we have to skip the test
            # IMPORTANT: this includes the case when both distributions are the same, e.g., both are completely zero
            if np.unique(np.hstack([condition_a_communication_scores, condition_b_communication_scores])).shape[0] == 1:
                continue

            # The CVM Test requires that there are at least two observations in each array
            if len(condition_a_communication_scores) < 2 or len(condition_b_communication_scores) < 2:
                continue

            cvm_statistic = stats.cramervonmises_2samp(
                x=condition_a_communication_scores,
                y=condition_b_communication_scores,
                method='auto'
            )
            # Always store the observed statistic
            test_statistic[source, receiver, interaction] = cvm_statistic.statistic

            # Fallback criteria: exact zero, non-finite, or suspiciously tiny asymptotic p (underflow)
            p = float(cvm_statistic.pvalue)
            if self._needs_permutation(p):
                if random_state is None and not _permutation_fallback_warned:
                    warnings.warn(
                        'CVM permutation fallback triggered: p-values for affected triples are not '
                        'reproducible. Pass an integer to random_state to fix this.',
                        UserWarning,
                        stacklevel=5
                    )
                    _permutation_fallback_warned = True
                # Use permutation p-value with the same statistic definition
                p_perm = self.__perm_pvalue_independent(
                    condition_a_communication_scores,
                    condition_b_communication_scores,
                    stat_fn=lambda a, b: stats.cramervonmises_2samp(x=a, y=b, method='auto').statistic,
                    n_resamples=n_resamples,
                    alternative='two-sided',
                    random_state=random_state
                )
                p_final = self._ensure_nonzero_p(p_perm, n_resamples)
            else:
                p_final = self._ensure_nonzero_p(p)
            p_values[source, receiver, interaction] = p_final

        p_values_cvm_xr = xr.DataArray(
            data=p_values,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        test_statistic_cvm_xr: xr.DataArray = xr.DataArray(
            data=test_statistic,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        if self.p_values is None:
            self.p_values = {}
            self.p_values_adj = {}
            self.statistics = {}

        self.p_values['CVM'] = p_values_cvm_xr
        self.statistics['CVM'] = {
            'statistics': test_statistic_cvm_xr
        }

        return p_values_cvm_xr

    def __compute_p_values_MannWhitneyU(
            self,
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None
    ) -> xr.DataArray:
        mask = self._validate_or_make_mask(mask)

        shape = (self.__n_cell_types, self.__n_cell_types, self.__n_interactions)
        p_values = np.full(shape, np.nan, dtype=float)
        test_statistic = np.full(shape, np.nan, dtype=float)

        # Compute p-values where mask is True
        idx = np.where(mask)
        _permutation_fallback_warned = False

        for source, receiver, interaction in zip(*idx):
            condition_a_communication_scores, condition_b_communication_scores = self.__get_communication_scores_for_p_value_test(
                source=source,
                receiver=receiver,
                interaction=interaction
            )

            if np.unique(np.hstack([condition_a_communication_scores, condition_b_communication_scores])).shape[0] == 1:
                continue

            if len(condition_a_communication_scores) == 0 or len(condition_b_communication_scores) == 0:
                continue

            mannwhitneyu_statistic = stats.mannwhitneyu(
                x=condition_a_communication_scores,
                y=condition_b_communication_scores,
                alternative='two-sided',
            )

            p = float(mannwhitneyu_statistic.pvalue)
            test_statistic[source, receiver, interaction] = float(mannwhitneyu_statistic.statistic)

            if self._needs_permutation(p):
                if random_state is None and not _permutation_fallback_warned:
                    warnings.warn(
                        'MannWhitneyU permutation fallback triggered: p-values for affected triples are not '
                        'reproducible. Pass an integer to random_state to fix this.',
                        UserWarning,
                        stacklevel=5
                    )
                    _permutation_fallback_warned = True
                res_perm = stats.mannwhitneyu(
                    x=condition_a_communication_scores,
                    y=condition_b_communication_scores,
                    alternative='two-sided',
                    method=PermutationMethod(
                        n_resamples=n_resamples,
                        random_state=random_state
                    )
                )
                p_final = self._ensure_nonzero_p(float(res_perm.pvalue), n_resamples)
            else:
                p_final = self._ensure_nonzero_p(p)
            p_values[source, receiver, interaction] = p_final

        p_values_mannwhitneyu_xr = xr.DataArray(
            data=p_values,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        test_statistic_mannwhitneyu_xr: xr.DataArray = xr.DataArray(
            data=test_statistic,
            dims=['source', 'receiver', 'interaction'],
            coords=[self.source_cell_types, self.receiver_cell_types, self.interaction_names]
        )

        if self.p_values is None:
            self.p_values = {}
            self.p_values_adj = {}
            self.statistics = {}

        self.p_values['MannWhitneyU'] = p_values_mannwhitneyu_xr
        self.statistics['MannWhitneyU'] = {
            'statistics': test_statistic_mannwhitneyu_xr
        }

        return p_values_mannwhitneyu_xr

    def save(self, path_to_file: str):
        """
        Save the MultiNeuronChatObject to a pickle file.

        :param path_to_file: path to the pickle file
        """
        dir_name: str = os.path.dirname(path_to_file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Save all variables as a dictionary and finally as a pickle file
        save_dict: Dict[str, Any] = {
            '__n_cell_types': self.__n_cell_types,
            '__n_interactions': self.__n_interactions,

            'condition_label_column': self.condition_label_column,
            'condition_names': self.condition_names,

            'subject_label_column': self.subject_label_column,
            'cell_type_label_column': self.cell_type_label_column,

            'db': self.db,
            'interaction_db': self.interaction_db,
            'gene_set': self.gene_set,

            'mean_type': self.mean_type,
            'trim_mean_fraction': self.trim_mean_fraction,

            'source_cell_types': self.source_cell_types,
            'receiver_cell_types': self.receiver_cell_types,
            'interaction_names': self.interaction_names,

            'avg_expression_per_condition_and_subject_dict': self.avg_expression_per_condition_and_subject_dict,

            'communication_scores_per_condition_and_subject_dict': self.communication_scores_per_condition_and_subject_dict,
            'ligand_abundance_per_condition_and_subject_dict': self.ligand_abundance_per_condition_and_subject_dict,
            'target_abundance_per_condition_and_subject_dict': self.target_abundance_per_condition_and_subject_dict,

            'communication_score_dict': self.communication_score_dict,
            'ligand_abundance_dict': self.ligand_abundance_dict,
            'target_abundance_dict': self.target_abundance_dict,

            'p_values': self.p_values,
            'p_values_adj': self.p_values_adj,
            'statistics': self.statistics,
        }

        with open(path_to_file, 'wb') as file:
            pickle.dump(save_dict, file)

    @staticmethod
    def load(path_to_file: str):
        """
        Load a MultiNeuronChatObject from a pickle file.

        :param path_to_file: path to the pickle file
        :return: MultiNeuronChatObject loaded from the pickle file
        """
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f'The file with path {path_to_file} could not be found.')

        with open(path_to_file, 'rb') as file:
            load_dict: Dict[str, Any] = pickle.load(file)

        mnc_obj: MultiNeuronChatObject = MultiNeuronChatObject(
            condition_label_column=load_dict['condition_label_column'],
            condition_names=load_dict['condition_names'],
            subject_label_column=load_dict['subject_label_column'],
            cell_type_label_column=load_dict['cell_type_label_column'],
            db=load_dict['db'],
            interaction_db=load_dict['interaction_db'],
        )

        mnc_obj.__n_cell_types = load_dict['__n_cell_types']
        mnc_obj.__n_interactions = load_dict['__n_interactions']

        mnc_obj.gene_set = load_dict['gene_set']

        mnc_obj.mean_type = load_dict['mean_type']
        mnc_obj.trim_mean_fraction = load_dict['trim_mean_fraction']

        mnc_obj.source_cell_types = load_dict['source_cell_types']
        mnc_obj.receiver_cell_types = load_dict['receiver_cell_types']
        mnc_obj.interaction_names = load_dict['interaction_names']

        mnc_obj.avg_expression_per_condition_and_subject_dict = load_dict['avg_expression_per_condition_and_subject_dict']

        # Ensure compatibility with prior versions
        if 'ligand_abundance_per_condition_and_subject_dict' in load_dict.keys():
            mnc_obj.ligand_abundance_per_condition_and_subject_dict = load_dict['ligand_abundance_per_condition_and_subject_dict']
            mnc_obj.target_abundance_per_condition_and_subject_dict = load_dict['target_abundance_per_condition_and_subject_dict']

            mnc_obj.ligand_abundance_dict = load_dict['ligand_abundance_dict']
            mnc_obj.target_abundance_dict = load_dict['target_abundance_dict']

        mnc_obj.communication_scores_per_condition_and_subject_dict = load_dict['communication_scores_per_condition_and_subject_dict']
        mnc_obj.communication_score_dict = load_dict['communication_score_dict']

        mnc_obj.p_values = load_dict['p_values']
        mnc_obj.p_values_adj = load_dict['p_values_adj']
        mnc_obj.statistics = load_dict['statistics']

        return mnc_obj
