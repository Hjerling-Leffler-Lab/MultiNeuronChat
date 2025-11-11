import os

import pickle

import numpy as np
import xarray as xr

from scipy import stats
from scipy.stats import PermutationMethod

from .InteractionDB import InteractionDB

from .utils import gene_filter_and_subject_wise_normalize_dataset
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
            method: str = 'by'
    ) -> np.ndarray:
        """
        Correct p-values using the Benjamini-Yekutieli method and return the corrected p-values.
        Exclude any nan values from the correction.
        :param p_values: the p-values to correct
        :param method: the method to use for correction (default: 'by')
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
            method: str = 'by'
    ) -> Union[xr.DataArray, Dict[str, xr.DataArray]]:
        """
        This function corrects the p-values using the Benjamini-Yekutieli method. The corrected p-values are stored in
        the MultiNeuronChatObject. If statistical_test is None, all p-values are corrected. If statistical_test is a
        string, only the p-values of the specified test are corrected. If statistical_test is a list of strings, only
        the p-values of the specified tests are corrected.

        :param statistical_test: the test for which the p-values should be corrected
        :param method: the method to use for correction (default: 'by')
        :return: the corrected p-values as a DataArray or a dictionary of DataArrays
        """
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

    def compute_significance(
            self,
            statistical_test: Optional[str] = 'KS',
            mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
            n_resamples: Optional[int] = 10_000,
            random_state: Optional[int] = None
    ) -> xr.DataArray:
        """
        Compute the p-values for the specified statistical test. The p-values are stored in the MultiNeuronChatObject.
        If mask is None, the significance is tested for all interactions and cell-type pairs. If a mask is provided,
        the significance is only computed for the specified interactions and cell-type pairs, setting all other p-values
        to np.nan.

        :param statistical_test: the statistical test to compute the p-values for (default: 'KS', i.e. Kolmogorov-Smirnov).
        :param mask: a mask to specify for which interactions and cell-type pairs the significance should be computed.
        :param n_resamples: number of resamples for permutation fallback (default 10_000 if None)
        :param random_state: random state for reproducibility (default None)
        :return: the p-values as a DataArray.
        """
        mask = self._validate_or_make_mask(mask)

        if statistical_test not in ['KS', 'Anderson', 'CVM', 'MannWhitneyU']:
            raise ValueError('Statistical_test must be either "KS", "Anderson", "CVM", or "MannWhitneyU"')

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
                random_state=random_state
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
            random_state: Optional[int] = None
    ) -> xr.DataArray:
        mask = self._validate_or_make_mask(mask)

        if permutation_method and n_resamples < 1:
            raise ValueError('The number of resamples must be greater than 0')

        # Set up numpy arrays with NaN values
        shape = (self.__n_cell_types, self.__n_cell_types, self.__n_interactions)
        p_values = np.full(shape, np.nan, dtype=float)
        test_statistic = np.full(shape, np.nan, dtype=float)

        # Compute p-values where mask is True
        idx = np.where(mask)

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
                            n_resamples=n_resamples,
                            random_state=random_state
                        )
                    )
            else:
                anderson_statistic = stats.anderson_ksamp(
                    samples=[
                        condition_a_communication_scores,
                        condition_b_communication_scores
                    ],
                    method=PermutationMethod(
                        n_resamples=n_resamples,
                        random_state=random_state
                    )
                )

            p_values[source, receiver, interaction] = self._ensure_nonzero_p(float(anderson_statistic.pvalue), n_resamples if permutation_method else None)
            test_statistic[source, receiver, interaction] = float(anderson_statistic.statistic)

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
