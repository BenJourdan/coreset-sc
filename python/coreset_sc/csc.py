import numpy
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

from . import coreset_sc, utils


class CoresetSpectralClustering(BaseEstimator, ClusterMixin):
    """
    Coreset Spectral Clustering

    Parameters
    ----------
    num_clusters : int
        Number of clusters to form.

    coreset_ratio : float, default=0.01
        Ratio of the coreset size to the original data size.

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

        assert isinstance(num_clusters, int), "Number of clusters must be an integer"
        assert num_clusters > 1, "Number of clusters must be greater than 1"

        assert isinstance(coreset_ratio, float), "Coreset ratio must be a float"
        assert (
            coreset_ratio > 0.0 and coreset_ratio < 1.0
        ), "Coreset ratio must be in the range (0, 1)"

        assert isinstance(
            k_over_sampling_factor, float
        ), "k_over_sampling_factor must be a float"
        assert (
            k_over_sampling_factor >= 1.0
        ), "k_over_sampling_factor must be greater than 1.0"

        assert isinstance(shift, float), "shift must be a float"
        assert shift >= 0.0, "shift must be non-negative"
        assert full_labels in [True, False], "full_labels must be a boolean"
        assert ignore_warnings in [True, False], "ignore_warnings must be a boolean"

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
        assert isinstance(
            adjacency_matrix, sparse.csr_matrix
        ), "Adjacency matrix must be a scipy.sparse.csr_matrix"
        assert (
            adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        ), "Adjacency matrix must be square"
        assert (
            self.num_clusters * self.k_over_sampling_factor <= adjacency_matrix.shape[0]
        ), "Number of clusters times the oversampling factor must be less than the number of samples"

        self.n_ = adjacency_matrix.shape[0]

        # Compute degree vector
        self.degrees_ = adjacency_matrix.sum(axis=1).A1
        # decompose the adjacency matrix into its components
        self.data_, self.indices_, self.indptr_ = utils.convert_from_csr_matrix(
            adjacency_matrix
        )
        self.nnz_per_row_ = numpy.diff(self.indptr_).astype(numpy.uint64)

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

        Returns
        -------
        labels : numpy.ndarray, shape = (n_samples,)
            Cluster assignments.
        """

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
        self.label_full_graph()

        return self.labels_
