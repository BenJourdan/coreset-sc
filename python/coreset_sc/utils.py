import numpy as np
from scipy.sparse import csr_matrix


def convert_from_csr_matrix(matrix: csr_matrix):
    """
    Convert a scipy.sparse.csr_matrix into a format compatible with the Rust module.
    This function ensures that indices are of type `np.uintp` (which corresponds to `usize` in Rust)
    and data is of type `np.float64` (which corresponds to `f64` in Rust).
    """
    matrix.sort_indices()
    indices = matrix.indices.astype(np.uintp)  # Convert indices to `usize`
    indptr = matrix.indptr.astype(np.uintp)  # Convert indptr to `usize`

    # Convert data to float64 (f64 in Rust)
    if matrix.data.dtype == np.float32:
        data = matrix.data.astype(np.float64)
    elif matrix.data.dtype == np.float64:
        data = matrix.data
    else:
        raise ValueError("Data type not supported, expected float32 or float64.")

    return data, indices, indptr


def convert_to_csr_matrix(size, data, indptr, indices):
    """
    Convert a scipy.sparse.csr_matrix into a format compatible with the Rust module.
    This function ensures that indices are of type `np.uintp` (which corresponds to `usize` in Rust)
    and data is of type `np.float64` (which corresponds to `f64` in Rust).
    """
    matrix = csr_matrix((data, indices, indptr), shape=(size, size))
    return matrix
