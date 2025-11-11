import os

import tempfile

import loompy

import numpy as np

from tqdm import tqdm

from .normalize import subject_wise_max_normalization
from .loompy_utils import create_empty_loom_file

from typing import Set, Optional, List


def filter_genes(
        path_to_loom: str,
        set_of_genes: Set[str],
        gene_label_row: str = 'Gene',
        path_to_filtered_loom: Optional[str] = '',
        chunk_size: Optional[int] = -1,
        verbose: Optional[bool] = False
) -> str:
    """
    Filters out all genes that are not in the set_of_genes from the loom file at path_to_loom. The filtered loom file
    is saved at path_to_filtered_loom. If path_to_filtered_loom is not provided, the filtered loom file is saved in the
    same directory as the original loom file with the suffix '_GeneSubsampled.loom'.

    If the system you are running this on has limited memory, you can set the chunk_size to a lower value to reduce the
    memory usage. If the chunk_size is set to zero or a negative number, the complete matrix is loaded into memory.

    :param path_to_loom: Path to the loom file that should be filtered
    :param set_of_genes: Set of genes that should be kept in the filtered loom file
                         (usually provided by MultiNeuronChat)
    :param gene_label_row: Name of the row attribute in the loom file that contains the gene labels (default: 'Gene')
    :param path_to_filtered_loom: Path where the filtered loom file should be saved (default: same directory as the
                                  original loom file with the suffix '_GeneSubsampled.loom')
    :param chunk_size: Number of columns that are loaded into memory at once (default: -1)
    :param verbose: If True, the progress is printed to the console (default: False)
    :return: Path to the filtered loom file. If path_to_filtered_loom is provided, the same path is returned.
    """

    if not os.path.isfile(path_to_loom):
        raise FileNotFoundError(f"File {path_to_loom} not found")

    if not path_to_loom.endswith('.loom'):
        raise ValueError('The file must be a loom file')

    # if path_to_filtered_loom is not provided, save the filtered loom file in the same directory as the original loom
    # file with the suffix '_GeneSubsampled.loom'
    if path_to_filtered_loom == '' or path_to_filtered_loom is None:
        path_to_filtered_loom: str = path_to_loom.replace('.loom', '_GeneSubsampled.loom')

    with loompy.connect(path_to_loom, mode='r') as src:
        n_rows, n_cols = src.shape

        # if the chunk_size is set to zero or a negative number, we load the complete matrix into memory
        if chunk_size <= 0:
            chunk_size = n_cols

        genes_to_keep_mask: List[bool] = [x in set_of_genes for x in src.ra[gene_label_row]]
        genes_to_keep_idx: np.array = np.arange(n_rows)[genes_to_keep_mask]

        n_rows_to_keep: int = len(genes_to_keep_idx)

        row_attrs = {k: src.ra[k][genes_to_keep_idx] for k in src.row_attrs.keys()}
        col_attrs = src.ca

        # Create a new loom file with the filtered genes
        create_empty_loom_file(
            path_to_loom=path_to_filtered_loom,
            shape=(n_rows_to_keep, n_cols),
            row_attrs=row_attrs,
            col_attrs=col_attrs
        )

        with loompy.connect(path_to_filtered_loom, mode='r+') as dst:
            if verbose:
                print('Filter the dataset for required genes:')

            for i in tqdm(range(0, n_cols, chunk_size), disable=(not verbose)):
                end_i: int = min(i + chunk_size, n_cols)

                dst[:, i:end_i] = src[genes_to_keep_idx, i:end_i]

    return path_to_filtered_loom


def gene_filter_and_subject_wise_normalize_dataset(
        path_to_loom: str,
        gene_set: Set[str],
        subject_label_column: str,
        path_to_normalized_loom: Optional[str] = '',
        gene_label_row: str = 'Gene',
        chunk_size: Optional[int] = -1,
        tmp_path: Optional[str] = None,
        verbose: Optional[bool] = False
) -> str:
    """
    Filters out all genes that are not in the gene_set from the loom file at path_to_loom and performs a subject-wise
    max normalization of the data. The normalized loom file is saved at path_to_normalized_loom.

    If path_to_normalized_loom is not provided, the normalized loom file is saved in the same directory as the original
    loom file with the suffix '_normalized.loom'.

    If the system you are running this on has limited memory, you can set the chunk_size to a lower value to reduce the
    memory usage. If the chunk_size is set to zero or a negative number, the complete matrix is loaded into memory.

    :param path_to_loom: Path to the loom file that should be filtered and normalized
    :param gene_set: Set of genes that should be kept in the filtered loom file
    :param subject_label_column: Name of the column in the column attributes of the loom file that contains the subject
                                 labels
    :param path_to_normalized_loom: Path where the normalized loom file should be saved
                                    (default: same directory as the original loom file with the suffix '_normalized.loom')
    :param gene_label_row: Name of the row attribute in the loom file that contains the gene labels (default: 'Gene')
    :param chunk_size: Number of columns that are loaded into memory at once (default: -1)
    :param tmp_path: Path to the temporary directory where the filtered loom file is saved (default: system's temp dir)
    :param verbose: If True, the progress is printed to the console (default: False)
    :return: Path to the gene-filtered and subject-wise max-normalized loom file. If path_to_normalized_loom is provided,
             the same path is returned.
    """
    if tmp_path is None:
        tmp_path: str = tempfile.gettempdir()
    else:
        os.makedirs(tmp_path, exist_ok=True)

    file_path: str = os.path.dirname(path_to_loom)
    file_name: str = os.path.basename(path_to_loom)

    gene_filtered_file_name: str = file_name.replace('.loom', '_filtered.loom')
    gene_filtered_path: str = os.path.join(tmp_path, gene_filtered_file_name)

    # if path_to_normalized_loom is not provided, save the normalized loom file in the same directory as the original
    # loom file with the suffix '_normalized.loom'
    if path_to_normalized_loom == '' or path_to_normalized_loom is None:
        normalized_file_name: str = gene_filtered_file_name.replace('.loom', '_normalized.loom')
        path_to_normalized_loom: str = os.path.join(file_path, normalized_file_name)

    filter_genes(
        path_to_loom=path_to_loom,
        set_of_genes=gene_set,
        path_to_filtered_loom=gene_filtered_path,
        gene_label_row=gene_label_row,
        chunk_size=chunk_size,
        verbose=verbose
    )

    subject_wise_max_normalization(
        path_to_loom=gene_filtered_path,
        subject_label_column=subject_label_column,
        path_to_normalized_loom=path_to_normalized_loom,
        chunk_size=chunk_size
    )

    # remove tmp files
    os.remove(gene_filtered_path)

    return path_to_normalized_loom