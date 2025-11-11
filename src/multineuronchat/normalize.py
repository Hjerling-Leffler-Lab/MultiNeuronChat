import os

import numpy as np

import loompy
from .loompy_utils import create_empty_loom_file

from tqdm import tqdm

from typing import Optional


def cell_wise_log_normalization(
        path_to_loom: str,
        path_to_normalized_loom: Optional[str] = '',
        chunk_size: Optional[int] = -1,
        verbose: Optional[bool] = False
) -> str:
    """
    Perform cell-wise log-normalization on the loom file at path_to_loom. The normalized loom file is saved at
    path_to_normalized_loom. If path_to_normalized_loom is not provided, the normalized loom file is saved in the same
    directory as the original loom file with the suffix '_CellWiseNormalized.loom'.

    If the system you are running this on has limited memory, you can set the chunk_size to a lower value to reduce the
    memory usage. If the chunk_size is set to zero or a negative number, the complete matrix is loaded into memory.

    :param path_to_loom: Path to the loom file that should be normalized
    :param path_to_normalized_loom: Path where the normalized loom file should be saved (default: same directory as the
                                    original loom file with the suffix '_CellWiseNormalized.loom')
    :param chunk_size: Number of columns that are loaded into memory at once (default: -1)
    :param verbose: If True, the progress is printed to the console (default: False)
    :return: Path to the cell-wise log-normalized loom file. If path_to_normalized_loom is provided, the same path is
             returned.
    """

    if not os.path.isfile(path_to_loom):
        raise FileNotFoundError(f"File {path_to_loom} not found")

    if not path_to_loom.endswith('.loom'):
        raise ValueError('The file must be a loom file')

    # if path_to_normalized_loom is not provided, save the normalized loom file in the same directory as the original
    # loom file with the suffix '_CellWiseNormalized.loom'
    if path_to_normalized_loom == '' or path_to_normalized_loom is None:
        path_to_normalized_loom: str = path_to_loom.replace(".loom", "_CellWiseNormalized.loom")

    with loompy.connect(path_to_loom, mode='r') as src:
        n_rows, n_cols = src.shape

        # if the chunk_size is set to zero or a negative number, we load the complete matrix into memory
        if chunk_size <= 0:
            chunk_size = n_cols

        # Create a new loom file with the same row and column attributes as the original loom file
        create_empty_loom_file(
            path_to_loom=path_to_normalized_loom,
            shape=(n_rows, n_cols),
            row_attrs=src.ra,
            col_attrs=src.ca,
            dtype_to_use=np.float32
        )

        # Connect to the new loom file in read/write mode and perform normalization
        with loompy.connect(path_to_normalized_loom, mode='r+') as dst:
            if verbose:
                print('Cell-Wise Log-Normalization:')

            for i in tqdm(range(0, n_cols, chunk_size), disable=(not verbose)):
                end_i: int = min(i + chunk_size, n_cols)

                data = src[:, i:end_i]

                column_sum: np.array = np.sum(data, axis=0)
                column_sum[column_sum == 0] = 1
                log_norm: np.array = np.log1p(10_000 * (data / column_sum[None, :]))

                dst[:, i:end_i] = log_norm

    return path_to_normalized_loom


def subject_wise_max_normalization(
        path_to_loom: str,
        subject_label_column: str,
        path_to_normalized_loom: Optional[str] = '',
        chunk_size: Optional[int] = -1
) -> str:
    """
    Perform subject-wise max-normalization on the loom file at path_to_loom. The normalized loom file is saved at
    path_to_normalized_loom. If path_to_normalized_loom is not provided, the normalized loom file is saved in the same
    directory as the original loom file with the suffix '_MaxNormalized.loom'.

    If the system you are running this on has limited memory, you can set the chunk_size to a lower value to reduce the
    memory usage. If the chunk_size is set to zero or a negative number, the complete matrix is loaded into memory.

    :param path_to_loom: Path to the loom file that should be normalized
    :param subject_label_column: Column name in the column attributes that contains the subject IDs
    :param path_to_normalized_loom: Path where the normalized loom file should be saved (default: same directory as the
                                    original loom file with the suffix '_MaxNormalized.loom')
    :param chunk_size: Number of columns that are loaded into memory at once (default: -1)
    :return: Path to the subject-wise max-normalized loom file. If path_to_normalized_loom is provided, the same path is
             returned.
    """

    if not os.path.isfile(path_to_loom):
        raise FileNotFoundError(f"File {path_to_loom} not found")

    if not path_to_loom.endswith('.loom'):
        raise ValueError('The file must be a loom file')

    # if path_to_normalized_loom is not provided, save the normalized loom file in the same directory as the original
    # loom file with the suffix '_MaxNormalized.loom'
    if path_to_normalized_loom == '' or path_to_normalized_loom is None:
        path_to_normalized_loom: str = path_to_loom.replace(".loom", "_MaxNormalized.loom")

    with loompy.connect(path_to_loom, mode='r') as src:
        n_rows, n_cols = src.shape

        if chunk_size is None or chunk_size < 0:
            chunk_size = n_cols

        # Pass 0: collect subjects and build index groups (no I/O on matrix)
        subjects = np.asarray(src.ca[subject_label_column])
        # Make deterministic order of subjects for reproducibility
        uniq_subjects, inv = np.unique(subjects, return_inverse=True)
        # inv[j] gives the index in uniq_subjects for column j

        # Pass 1: compute per-column maxima, streaming in chunks:
        col_max = np.empty(n_cols, dtype=np.float32)
        for i in tqdm(range(0, n_cols, chunk_size), desc='Computing column maxima'):
            end_i = min(i + chunk_size, n_cols)
            X = src[:, i:end_i].astype(np.float32, copy=False)
            col_max[i:end_i] = np.max(X, axis=0)
            del X

        # Reduce per-subject: max across the subject's columns
        subj_max = np.full(uniq_subjects.shape[0], -np.inf, dtype=np.float32)

        # vectorized max-by-group using bincount trick
        # We can’t max with bincount directly, so do a simple loop over subjects on col_max indices
        # but that loop is tiny (n_subjects), not n_rows*n_cols
        for s_idx in range(uniq_subjects.shape[0]):
            # columns belonging to subject s_idx
            mask = (inv == s_idx)
            if not np.any(mask):
                subj_max[s_idx] = 1 # just to be sure, but should not happen
            else:
                m = col_max[mask].max()
                subj_max[s_idx] = m if m > 0 else 1 # m <= 0 should not happen, but just in case

        # Map each column to its subject's max value
        col_scale = subj_max[inv]

        # Create the new loom file with the same row and column attributes as the original loom file
        create_empty_loom_file(
            path_to_loom=path_to_normalized_loom,
            shape=(n_rows, n_cols),
            row_attrs=src.ra,
            col_attrs=src.ca,
            dtype_to_use=np.float32
        )

        # Pass 2: write normalized data in chunks (in-place operation and in float32)
        with loompy.connect(path_to_normalized_loom, mode='r+') as dst:
            for i in tqdm(range(0, n_cols, chunk_size), desc='Normalizing data'):
                end_i = min(i + chunk_size, n_cols)
                X = src[:, i:end_i].astype(np.float32, copy=False)

                # Normalize by the subject's max value
                X /= col_scale[i:end_i][None, :].astype(np.float32, copy=False)

                # Write the normalized data to the new loom file
                dst[:, i:end_i] = X

                del X

    return path_to_normalized_loom
