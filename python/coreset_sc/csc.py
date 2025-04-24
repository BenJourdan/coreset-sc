import math

import numpy
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

from . import coreset_sc, utils


def signless_laplacian_and_D_sparse(A, D=None):
    n = A.shape[0]
    if D is None:
        D = sparse.diags(A.sum(axis=1).A1)
    else:
        D = sparse.diags(D)
    D_inv_half = sparse.diags(1 / numpy.sqrt(D.diagonal()))
    L = D - A
    N = D_inv_half @ L @ D_inv_half
    M = sparse.eye(n) - (0.5) * N
    return M, D


def fast_spectral_cluster(M, D, k: int, kmeans_alg=None):
    # M is the signless laplacian: I - (1/2) * D^(-1/2) * A * D^(-1/2)

    n = M.shape[0]
    _l = min(k, math.ceil(math.log(k, 2)))
    t = 10 * math.ceil(math.log(n / k, 2))
    Y = numpy.random.normal(size=(n, _l))

    # We know the top eigenvector of the normalised laplacian.
    # It doesn't help with clustering, so we will project our power method to
    # be orthogonal to it.
    top_eigvec = numpy.sqrt(D @ numpy.full((n,), 1))
    norm = numpy.linalg.norm(top_eigvec)
    if norm > 0:
        top_eigvec /= norm

    for _ in range(t):
        Y = M @ Y

        # Project Y to be orthogonal to the top eigenvector
        for i in range(_l):
            Y[:, i] -= (top_eigvec.transpose() @ Y[:, i]) * top_eigvec

    kmeans = (
        kmeans_alg if kmeans_alg is not None else KMeans(n_clusters=k, n_init="auto")
    )
    kmeans.fit(Y)
    return kmeans.labels_


class CoresetSpectralClustering(BaseEstimator, ClusterMixin):
    """
    Coreset Spectral Clustering

    Parameters
    ----------
    num_clusters : int
        Number of clusters to form.

    coreset_ratio : float, default=0.01
        Ratio of the coreset size to the original data size. If set to 1.0,
        coreset clustering will be skipped and the full graph will be clustered
        directly.

    k_over_sampling_factor : float, default=2.0
        The factor to oversample the number of clusters for the coreset
        seeding stage. Higher values will increase the "resolution" of the
        sampling distribution, but take longer to compute.

    shift: float, default=0.0
        The shift to add to the implicit kernel matrix of the form K' = K + shift*D^{-1}.
        This is useful for graphs with large edge weights relative to degree, which can
        cause the kernel matrix to be indefinite.

    kmeans_alg : sklearn.cluster.KMeans, default=None
        The KMeans algorithm to use for clustering the coreset embeddings.
        If None, a default KMeans algorithm will be used.

    full_labels : bool, default=True
        Whether to return the full labels of the graph after fitting.
        If False, only the coreset labels will be returned.

    ignore_warnings : bool, default=False
        Whether to ignore warnings about the implicit Kernel matrix being indefinite.
        Distances that do become negative will be clipped to zero.

    Attributes
    ----------
    """

    def __init__(
        self,
        num_clusters,
        coreset_ratio=0.01,
        k_over_sampling_factor=2.0,
        shift=0.0,
        kmeans_alg=None,
        full_labels=True,
        ignore_warnings=False,
    ):

        if not isinstance(num_clusters, int):
            raise TypeError("Number of clusters must be an integer")
        if num_clusters <= 1:
            raise ValueError("Number of clusters must be greater than 1")

        if not isinstance(coreset_ratio, float):
            raise TypeError("Coreset ratio must be a float")
        if not (0.0 < coreset_ratio <= 1.0):
            raise ValueError("Coreset ratio must be in the range (0, 1)")

        if not isinstance(k_over_sampling_factor, float):
            raise TypeError("k_over_sampling_factor must be a float")
        if k_over_sampling_factor < 1.0:
            raise ValueError("k_over_sampling_factor must be greater than 1.0")

        if not isinstance(shift, float):
            raise TypeError("shift must be a float")
        if shift < 0.0:
            raise ValueError("shift must be non-negative")
        if full_labels not in [True, False]:
            raise TypeError("full_labels must be a boolean")
        if ignore_warnings not in [True, False]:
            raise TypeError("ignore_warnings must be a boolean")

        self.num_clusters = num_clusters
        self.coreset_ratio = coreset_ratio
        self.k_over_sampling_factor = k_over_sampling_factor
        self.shift = shift
        self.full_labels = full_labels

        self.kmeans_alg = (
            kmeans_alg
            if kmeans_alg is not None
            else KMeans(n_clusters=num_clusters, n_init=10)
        )
        self.ignore_warnings = ignore_warnings

        self.n_ = None
        self.data_ = None
        self.indices_ = None
        self.indptr_ = None
        self.nnz_per_row_ = None
        self.degrees_ = None

        self.coreset_size_ = None
        self.coreset_indices_ = None
        self.coreset_labels_ = None
        self.coreset_weights_ = None

        self.labels_ = None
        self.closest_cluster_distance_ = None

    def fit(self, adjacency_matrix, y=None):
        """
        Fit the coreset clustering algorithm on the sparse adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : scipy.sparse.csr_matrix, shape = (n_samples, n_samples)
            The adjacency matrix of the graph. This must contain self loops
            for each node.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(adjacency_matrix, sparse.csr_matrix):
            raise TypeError("Adjacency matrix must be a scipy.sparse.csr_matrix")
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if self.num_clusters * self.k_over_sampling_factor > adjacency_matrix.shape[0]:
            raise ValueError(
                "Number of clusters times the oversampling factor must be less than the number of samples"
            )

        # check that self loops are present:
        if not numpy.all(adjacency_matrix.diagonal() > 0):
            raise ValueError(
                "Adjacency matrix must contain self loop weights for each node. "
                "To set all self loops to 1 (recommended), use adjacency_matrix.setdiag(1)"
            )

        self.n_ = adjacency_matrix.shape[0]

        # Compute degree vector
        self.degrees_ = adjacency_matrix.sum(axis=1).A1
        # decompose the adjacency matrix into its components
        self.data_, self.indices_, self.indptr_ = utils.convert_from_csr_matrix(
            adjacency_matrix
        )
        self.nnz_per_row_ = numpy.diff(self.indptr_).astype(numpy.uint64)

        if self.coreset_ratio == 1.0:
            # If the coreset ratio is 1, we can skip the coreset stage
            M, D = signless_laplacian_and_D_sparse(adjacency_matrix, self.degrees_)
            self.labels_ = fast_spectral_cluster(
                M, D, self.num_clusters, self.kmeans_alg
            )
            return self

        rough_coreset_size = int(self.coreset_ratio * self.n_)

        self.coreset_indices_, self.coreset_weights_, coreset_embeddings_ = (
            coreset_sc.coreset_embeddings(
                self.num_clusters,
                self.n_,
                rough_coreset_size,
                self.k_over_sampling_factor,
                self.data_,
                self.indices_,
                self.indptr_,
                self.nnz_per_row_,
                self.degrees_,
                self.shift,
                self.ignore_warnings,
            )
        )
        self.coreset_size_ = self.coreset_indices_.shape[0]

        # run kmeans on the coreset embeddings
        self.kmeans_alg.fit(coreset_embeddings_)
        self.coreset_labels_ = self.kmeans_alg.labels_.astype(numpy.uint64)

        if self.full_labels:
            self.label_full_graph()
        return self

    def compute_conductances(self):
        """
        Compute the conductance of the labelled graph after fitting.

        Returns
        -------
        conductances : numpy.ndarray, shape = (num_clusters,)
            The conductance of each cluster
        """

        assert self.labels_ is not None, "Labels must be computed before conductances"
        # This also assumes all the other required attributes are set

        return coreset_sc.compute_conductances(
            self.num_clusters,
            self.n_,
            self.data_,
            self.indices_,
            self.indptr_,
            self.nnz_per_row_,
            self.degrees_,
            self.labels_,
        )

    def label_full_graph(self):
        """
        Label the full graph using the coreset labels.
        Skip this if the coreset ratio is 1.0.

        Returns
        -------
        labels : numpy.ndarray, shape = (n_samples,)
            Cluster assignments.
        """
        if self.coreset_ratio == 1.0:
            return None
        self.labels_, self.closest_cluster_distance_ = coreset_sc.label_full_graph(
            self.num_clusters,
            self.n_,
            self.data_,
            self.indices_,
            self.indptr_,
            self.nnz_per_row_,
            self.degrees_,
            self.coreset_indices_,
            self.coreset_weights_,
            self.coreset_labels_,
            self.shift,
        )

    def fit_predict(self, adjacency_matrix, y=None):
        """
        Fit the coreset clustering algorithm on the sparse adjacency matrix
        and return the cluster assignments.

        Parameters
        ----------
        adjacency_matrix : scipy.sparse.csr_matrix, shape = (n_samples, n_samples)
            The adjacency matrix of the graph. This must contain self loops
            for each node.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : numpy.ndarray, shape = (n_samples,)
            Cluster assignments.
        """
        self.fit(adjacency_matrix)
        if self.coreset_ratio == 1.0:
            # If the coreset ratio is 1, we've already labelled the full graph
            return self.labels_
        else:
            self.label_full_graph()
            return self.labels_
