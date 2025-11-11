import numpy as np

import loompy
from loompy import timestamp
import h5py

from typing import Dict, Tuple, Optional

def create_empty_loom_file(
        path_to_loom: str,
        shape: Tuple[int, int],
        row_attrs: Optional[Dict[str, np.ndarray]] = None,
        col_attrs: Optional[Dict[str, np.ndarray]] = None,
        dtype_to_use = np.float32
) -> None:
    """
    Creates an empty loom file at the specified path with the given number of rows and columns.

    Parameters
    ----------
    path_to_loom: str
        Path where the loom file should be created.
    shape: Tuple[int, int]
        Shape of the matrix to be created in the loom file, specified as (n_rows, n_cols).
    row_attrs: Optional[Dict[str, np.ndarray]]
        Dictionary of row attributes where keys are attribute names and values are arrays of attribute values.
    col_attrs: Optional[Dict[str, np.ndarray]]
        Dictionary of column attributes where keys are attribute names and values are arrays of attribute values.
    dtype_to_use: np.dtype
        Data type to use for the matrix in the loom file (default: np.float32).

    Returns
    -------
    None
    """
    n_rows, n_cols = shape

    with h5py.File(path_to_loom, 'w') as f:
        f.create_group('/attrs')  # v3.0.0

        f.create_group('/layers')
        f.create_group('/row_attrs')
        f.create_group('/col_attrs')
        f.create_group('/row_graphs')
        f.create_group('/col_graphs')

        f.create_dataset(
            '/matrix',
            shape=(n_rows, n_cols),
            dtype=dtype_to_use,
            chunks=True,
            compression='gzip',
            compression_opts=9,
        )

        f.flush()

        with loompy.connect(path_to_loom, 'r+') as ds:
            ds.attrs['CreationDate'] = timestamp()
            ds.attrs["LOOM_SPEC_VERSION"] = loompy.loom_spec_version

            for key, vals in row_attrs.items():
                ds.ra[key] = vals

            for key, vals in col_attrs.items():
                ds.ca[key] = vals