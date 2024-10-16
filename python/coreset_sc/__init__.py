import numpy
import scipy

from . import coreset_sc, utils


def default_coreset(clusters, coreset_size, adj_mat):

    n = adj_mat.shape[0]

    assert isinstance(
        adj_mat, scipy.sparse.csr_matrix
    ), "adj_mat must be a scipy.sparse.csr_matrix"
    assert adj_mat.shape[0] == adj_mat.shape[1], "adj_mat must be a square matrix"
    assert isinstance(coreset_size, int), "coreset_size must be an integer"
    assert isinstance(clusters, int), "clusters must be an integer"
    assert coreset_size > 0, "coreset_size must be greater than 0"
    assert clusters > 0, "clusters must be greater than 0"
    assert (
        clusters <= coreset_size
    ), "clusters must be less than or equal to coreset_size"

    degrees = numpy.sum(adj_mat, axis=1).A1

    data, indices, indptr = utils.convert_csr_matrix(adj_mat)
    nnz_per_row = numpy.diff(indptr).astype(numpy.uint64)

    coreset, coreset_weights = coreset_sc.default_coreset(
        clusters, n, coreset_size, data, indices, indptr, nnz_per_row, degrees
    )

    return coreset, coreset_weights
