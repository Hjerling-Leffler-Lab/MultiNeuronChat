import os

import warnings

from multiprocessing import Pool

import loompy

import numpy as np
import scipy.stats as stats
import xarray as xr

from .InteractionDB import *

from typing import List, Tuple, Set, Dict, Optional


def compute_subject_specific_avg_expression(
        path_to_loom: str,
        subject_id: str,
        subject_label_column: str,
        cell_type_label_column: str,
        gene_label_row: str = 'Gene',
        min_n_cells_threshold: int = -1,
        mean_type: str = 'tri_mean',
        trim_mean_fraction: Optional[float] = None,
) -> xr.DataArray:
    """
    Compute the average expression of all genes for a single individual and all cell-types that are present in the loom
    file. The average expression is computed using the Tukey's tri_mean.

    If the number of cells for a specific cell-type is less than min_n_cells_threshold, the average expression for this
    cell-type is set to NaN.

    :param path_to_loom: Path to the loom file from which the average expression should be computed. Important: The loom file must not be open in write mode by any other thread!
    :param subject_id: The subject id from the subject_label_column for which the average expression should be computed.
    :param subject_label_column: The column in the loom file that contains the subject id.
    :param cell_type_label_column: The column in the loom file that contains the cell-type labels.
    :param gene_label_row: Name of the row attribute in the loom file that contains the gene labels (default: 'Gene')
    :param min_n_cells_threshold: The minimum number of cells that are required for a cell-type to be included in the average expression computation. If the number of cells is less than min_n_cells_threshold, the average expression for this cell-type is set to NaN.
    :param mean_type: The type of mean that should be computed. The mean can be either 'tri_mean', 'mean', or 'trim_mean'.
    :param trim_mean_fraction: The fraction of the data that should be trimmed when the mean type is set to 'trim_mean'.
    :return: A matrix of the average expression of all genes for the individual and all cell-types.
    """

    if not os.path.isfile(path_to_loom):
        raise FileNotFoundError(f"File {path_to_loom} not found")

    if not path_to_loom.endswith('.loom'):
        raise ValueError('The file must be a loom file')

    if subject_id == '' or subject_id is None:
        raise ValueError('The subject id must be provided')

    if subject_label_column == '' or subject_label_column is None:
        raise ValueError('The subject label column must be provided')

    if cell_type_label_column == '' or cell_type_label_column is None:
        raise ValueError('The cell type label column must be provided')

    if mean_type not in ['tri_mean', 'mean', 'trim_mean']:
        raise ValueError('The mean type must be either "tri_mean", "mean", or "trim_mean"')
    if mean_type == 'trim_mean' and trim_mean_fraction is None:
        raise ValueError('The trim_mean fraction must be provided when the mean type is set to "trim_mean"')

    with loompy.connect(path_to_loom, mode='r') as data_loom:
        # Get all genes and cell-types included in the loom file
        genes: List[str] = data_loom.ra[gene_label_row].tolist()
        cell_types: List[str] = list(set(data_loom.ca[cell_type_label_column]))

        # Sort the cell-types
        cell_types.sort()

        # Initialize the average expression matrix
        avg_expression: np.array = np.zeros(shape=(len(genes), len(cell_types)))

        for i, cell_type in enumerate(cell_types):
            # Get all cells with the specific cell-type and subject id
            subject_and_cell_type_mask: np.array = (data_loom.ca[cell_type_label_column] == cell_type) & \
                                                   (data_loom.ca[subject_label_column] == subject_id)

            # If there are no cells with the specific cell-type and subject id, set the average expression to NaN
            if not np.any(subject_and_cell_type_mask):
                warnings.warn(f'There are no cell-types with label "{cell_type}" for subject "{subject_id}"')
                avg_expression[:, i] = np.nan
                continue

            # If the number of cells with the specific cell-type and subject id is less than min_n_cells_threshold,
            # set the average expression to NaN
            if min_n_cells_threshold > 0 and np.sum(subject_and_cell_type_mask) < min_n_cells_threshold:
                warnings.warn(f'There are less than {min_n_cells_threshold} cells with label "{cell_type}" for subject "{subject_id}"')
                avg_expression[:, i] = np.nan
                continue

            cell_type_matrix: np.array = data_loom[:, subject_and_cell_type_mask]

            if mean_type == 'mean':
                avg_expression[:, i] = np.mean(cell_type_matrix, axis=1)
            elif mean_type == 'trim_mean':
                avg_expression[:, i] = stats.trim_mean(cell_type_matrix, trim_mean_fraction, axis=1)
            else:
                # compute turkey's tri_mean
                q1: np.array = np.percentile(cell_type_matrix, 25, axis=1)
                q2: np.array = np.percentile(cell_type_matrix, 50, axis=1)
                q3: np.array = np.percentile(cell_type_matrix, 75, axis=1)

                tri_mean: np.array = (q1 + 2 * q2 + q3) / 4

                avg_expression[:, i] = tri_mean

    avg_expression_xr: xr.DataArray = xr.DataArray(data=avg_expression,
                                                   coords={'genes': genes, 'cell_types': cell_types},
                                                   dims=['genes', 'cell_types'])

    return avg_expression_xr


def compute_avg_expression(
        path_to_loom: str,

        condition_label_column: str,
        condition_label_a: str,
        condition_label_b: str,

        subject_label_column: str,
        cell_type_label_column: str,

        min_n_cells_threshold: int = -1,

        mean_type: str = 'trimean',
        trim_mean_fraction: Optional[float] = None,

        gene_label_row: str = 'Gene',

        n_processes: Optional[int] = 1
) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Compute the average expression of all genes for all individuals and all cell-types.

    :param path_to_loom: Path to the loom file from which the average expression should be computed. Important: The loom file must not be open in write mode by any other thread!
    :param condition_label_column: The column in the loom file that contains the condition label.
    :param condition_label_a: The label of the first condition.
    :param condition_label_b: The label of the second condition.
    :param subject_label_column: The column in the loom file that contains the subject id.
    :param cell_type_label_column: The column in the loom file that contains the cell-type labels.
    :param min_n_cells_threshold: The minimum number of cells that are required for a cell-type to be included in the average expression computation. If the number of cells is less than min_n_cells_threshold, the average expression for this cell-type is set to NaN.
    :param mean_type: The type of mean that should be computed. The mean can be either 'tri_mean', 'mean', or 'trim_mean'.
    :param trim_mean_fraction: The fraction of the data that should be trimmed when the mean type is set to 'trim_mean'.
    :param gene_label_row: Name of the row attribute in the loom file that contains the gene labels (default: 'Gene')
    :param n_processes: The number of processes that should be used for the computation. If n_processes is set to 1, the computation is done in a single process.
    :return: A dictionary with keys 'condition_label_a' and 'condition_label_b'. Each key contains a dictionary with the subject id as key and a matrix of the average expression as value.
    """

    if not os.path.isfile(path_to_loom):
        raise FileNotFoundError(f"File {path_to_loom} not found")

    if not path_to_loom.endswith('.loom'):
        raise ValueError('The file must be a loom file')

    if condition_label_a == condition_label_b:
        raise ValueError('The condition labels must be different')

    if condition_label_a == '' or condition_label_a is None:
        raise ValueError('The condition label a must be provided')

    if condition_label_b == '' or condition_label_b is None:
        raise ValueError('The condition label b must be provided')

    if subject_label_column == '' or subject_label_column is None:
        raise ValueError('The subject label column must be provided')

    if cell_type_label_column == '' or cell_type_label_column is None:
        raise ValueError('The cell type label column must be provided')

    if n_processes <= 0:
        raise ValueError('The number of processes must be greater than zero')

    with loompy.connect(path_to_loom, mode='r') as data_loom:
        # containing all subjects regardless of the conditon label that were provided
        all_subjects: List[str] = list(set(data_loom.ca[subject_label_column]))

        # Create a dictionary of all subjects to their condition
        # Important: we only keep those subjects that have either condition_label_a or condition_label_b
        subjects_to_condition_dict: Dict[str, str] = {
            subject: data_loom.ca[data_loom.ca[subject_label_column] == subject][condition_label_column][0]
            for subject in all_subjects
        }
        subjects_to_condition_dict = {
            key: value
            for key, value in subjects_to_condition_dict.items()
            if value in [condition_label_a, condition_label_b]
        }

        subjects: List[str] = list(subjects_to_condition_dict.keys())

    # Multiprocessing of avg_expression computation
    multiprocessing_pool_input: List[Tuple[str, str, str, str, str, int, str, float]] = [
        (
            path_to_loom,
            subject,
            subject_label_column,
            cell_type_label_column,
            gene_label_row,
            min_n_cells_threshold,
            mean_type,
            trim_mean_fraction
        )
        for subject in subjects
    ]

    if n_processes == 1:
        avg_expressions_list: List[xr.DataArray] = [
            compute_subject_specific_avg_expression(*method_input)
            for method_input in multiprocessing_pool_input

        ]
    else:
        with Pool(processes=n_processes) as pool:
            avg_expressions_list: List[xr.DataArray] = pool.starmap(
                compute_subject_specific_avg_expression,
                multiprocessing_pool_input
            )

    # Create a dictionary of dictionaries containing the average expression matrix for all subjects separated by
    # the condition.
    avg_expressions: Dict[str, Dict[str, xr.DataArray]] = {
        condition_label_a: {
            subject: avg_expressions_list[i] for i, subject in enumerate(subjects)
            if subjects_to_condition_dict[subject] == condition_label_a
        },
        condition_label_b: {
            subject: avg_expressions_list[i] for i, subject in enumerate(subjects)
            if subjects_to_condition_dict[subject] == condition_label_b
        }
    }

    return avg_expressions


def compute_subject_specific_communication_score_matrix(
        avg_expression: xr.DataArray,
        interaction_db: InteractionDB
) -> Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Compute the communication score matrix for a single individual based on the average expression of all genes and the
    interaction database.

    The communication score matrix is computed as follows:
    Step 1: Compute the abundance of the ligand and target subunit for each interaction. The abundance of a single
            production step is the defined as the arithmetic mean of the isoenzyme gene expression. Each production
            step is then multiplied by the coefficient of the production step. Then, the geometric mean over all
            production steps is computed.
    Step 2: Compute the communication score for each interaction pair. The communication score is computed
            as the product of the abundance of the ligand and target subunit.

    :param avg_expression: A matrix describing the average expression of all synthesizing genes for each cell-type.
    :param interaction_db: The interaction database that contains all interactions.
    :return: A three-dimensional matrix of the communication score for each cell-type pair and
             ligand-target pair interaction. Secondary, a tuple of two matrices containing the abundance of the ligand
             and target subunit for each interaction.
    """
    genes: Set[str] = set(avg_expression['genes'].values.tolist())
    cell_types: List[str] = avg_expression['cell_types'].values.tolist()
    interaction_names: List[str] = interaction_db.get_interaction_names()

    n_cell_types: int = len(cell_types)
    n_interactions: int = len(interaction_names)

    ligand_abundance: np.array = np.zeros(shape=(n_interactions, n_cell_types))
    target_abundance: np.array = np.zeros(shape=(n_interactions, n_cell_types))

    # Iterate through all interaction pairs
    for i, interaction_name in enumerate(interaction_names):
        interaction_info: InteractionDBRow = interaction_db[interaction_name]

        # If either there are no genes present for the ligand production or target subunit production:
        # skip this interaction
        if (len(set(interaction_info.ligand_contributor).intersection(genes)) == 0) or (len(set(interaction_info.target_subunit).intersection(genes)) == 0):
            # Set the communication score to NaN as the interaction cannot be computed
            ligand_abundance[i, :] = np.nan
            target_abundance[i, :] = np.nan
            continue

        group_coeff_sum: int = 0
        for ligand_group in interaction_info.ligand_groups:
            group_genes: Set[str] = set(interaction_info.ligand_group_to_gene_dict[ligand_group])
            group_coeff: int = interaction_info.ligand_group_to_coeff_dict[ligand_group]

            group_coeff_sum += group_coeff

            available_genes: List[str] = list(set(group_genes).intersection(genes))

            if len(available_genes) == 0:
                ligand_abundance[i, :] += 0
            else:
                # TODO potentially rewrite this to handle zeros in the log more elegantly?
                with warnings.catch_warnings():
                    # Ignore warnings that are raised when the mean of an empty slice is calculated
                    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                    # Ignore warnings that are raised when the log of zero is calculated
                    warnings.filterwarnings(action='ignore', message='divide by zero encountered in log')

                    group_score: np.array = np.mean(avg_expression.loc[available_genes, cell_types], axis=0)
                    ligand_abundance[i, :] += group_coeff * np.log(group_score)

        ligand_abundance[i, :] = np.exp(ligand_abundance[i, :] / group_coeff_sum)

        group_coeff_sum: int = 0
        for target_group in interaction_info.target_groups:
            group_genes: List[str] = interaction_info.target_group_to_gene_dict[target_group]
            group_coeff: int = interaction_info.target_group_to_coeff_dict[target_group]

            group_coeff_sum += group_coeff

            available_genes: List[str] = list(set(group_genes).intersection(genes))

            if len(available_genes) == 0:
                target_abundance[i, :] += 0
            else:
                # TODO potentially rewrite this to handle zeros in the log more elegantly?
                with warnings.catch_warnings():
                    # Ignore warnings that are raised when the mean of an empty slice is calculated
                    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                    # Ignore warnings that are raised when the log of zero is calculated
                    warnings.filterwarnings(action='ignore', message='divide by zero encountered in log')

                    group_score: float = np.mean(avg_expression.loc[available_genes, cell_types], axis=0)
                    target_abundance[i, :] += group_coeff * np.log(group_score)

        target_abundance[i, :] = np.exp(target_abundance[i, :] / group_coeff_sum)

    communication_score_matrix: xr.DataArray = xr.DataArray(
        data=np.zeros(shape=(n_cell_types, n_cell_types, n_interactions), dtype=float),
        dims=['source', 'receiver', 'ligand_target_interactions'],
        coords=[cell_types, cell_types, interaction_names]
    )

    ligand_abundance_matrix: xr.DataArray = xr.DataArray(
        data=ligand_abundance.T,
        dims=['cell_types', 'ligand_target_interactions'],
        coords=[cell_types, interaction_names]
    )

    target_abundance_matrix: xr.DataArray = xr.DataArray(
        data=target_abundance.T,
        dims=['cell_types', 'ligand_target_interactions'],
        coords=[cell_types, interaction_names]
    )

    for s, source in enumerate(cell_types):
        for r, receiver in enumerate(cell_types):
            communication_score_matrix.loc[source, receiver, :] = ligand_abundance[:, s] * target_abundance[:, r]

    return communication_score_matrix, (ligand_abundance_matrix, target_abundance_matrix)


def compute_communication_score_matrix(
        avg_expression_dict: Dict[str, Dict[str, xr.DataArray]],
        interaction_db: InteractionDB,
        n_processes: Optional[int] = 1
) -> Tuple[Dict[str, Dict[str, xr.DataArray]], Dict[str, Dict[str, xr.DataArray]], Dict[str, Dict[str, xr.DataArray]]]:
    """
    Compute the communication score matrices for each condition, subject, and cell-type pair.

    :param avg_expression_dict: A dictionary containing the average expression off all subjects for each condition. The first key is the condition, the second key is the subject id, and the value is a matrix of the average expression for each cell-type pair and ligand-target pair.
    :param interaction_db: The interaction database that contains all interactions.
    :param n_processes: The number of processes that should be used for the computation. If n_processes is set to 1, the computation is done in a single process.
    :return: A triplet of dictionaries containing the communication score matrix, the ligand abundance matrix, and the target abundance matrix. The first key is the condition, the second key is the subject id, and the value is a matrix of the communication score, ligand abundance, or target abundance, split by condition and subject id.
    """

    if n_processes <= 0:
        raise ValueError('The number of processes must be greater than zero')

    # Create a dictionary to look up the condition for a specific subject
    subject_to_condition_dict: Dict[str, str] = {
        subject: condition
        for condition in avg_expression_dict.keys()
        for subject in avg_expression_dict[condition].keys()
    }

    conditions: List[str] = list(avg_expression_dict.keys())
    subjects: List[str] = list(subject_to_condition_dict.keys())

    multiprocessing_pool_input: List[Tuple[xr.DataArray, InteractionDB]] = [
        (avg_expression_dict[subject_to_condition_dict[subject]][subject], interaction_db)
        for subject in subjects
    ]

    # Multiprocessing of communication score computation
    if n_processes == 1:
        communication_score_matrices: List[Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]] = [
            compute_subject_specific_communication_score_matrix(*method_input)
            for method_input in multiprocessing_pool_input
        ]
    else:
        with Pool(processes=n_processes) as pool:
            communication_score_matrices: List[Tuple[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]] = pool.starmap(
                compute_subject_specific_communication_score_matrix,
                multiprocessing_pool_input
            )

    # Create a dictionary of dictionaries containing the communication score matrix for all subjects separated by
    # the condition.
    communication_scores_dict: Dict[str, Dict[str, xr.DataArray]] = {
        conditions[0]: {
            subject: communication_score_matrices[i][0]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[0]
        },
        conditions[1]: {
            subject: communication_score_matrices[i][0]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[1]
        }
    }

    # Create a dictionary of dictionaries containing the ligand abundance matrix for all subjects separated by
    # the condition.
    ligand_abundance_dict: Dict[str, Dict[str, xr.DataArray]] = {
        conditions[0]: {
            subject: communication_score_matrices[i][1][0]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[0]
        },
        conditions[1]: {
            subject: communication_score_matrices[i][1][0]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[1]
        }
    }

    # Create a dictionary of dictionaries containing the target abundance matrix for all subjects separated by
    # the condition.
    target_abundance_dict: Dict[str, Dict[str, xr.DataArray]] = {
        conditions[0]: {
            subject: communication_score_matrices[i][1][1]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[0]
        },
        conditions[1]: {
            subject: communication_score_matrices[i][1][1]
            for i, subject in enumerate(subjects)
            if subject_to_condition_dict[subject] == conditions[1]
        }
    }

    return communication_scores_dict, ligand_abundance_dict, target_abundance_dict
