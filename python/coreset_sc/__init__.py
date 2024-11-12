from . import coreset_sc, utils
from .csc import CoresetSpectralClustering as CoresetSpectralClustering

# import maturin_import_hook
# import scipy
# import stag.random
# from maturin_import_hook.settings import MaturinSettings

# maturin_import_hook.install(
#     enable_project_importer=True,
#     enable_rs_file_importer=True,
#     settings=MaturinSettings(
#         release=True,
#         strip=True,
#     ),
#     show_warnings=False,
# )


def gen_sbm(n, k, p, q):
    """
    Generate an approximate sample from a Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        Number of nodes in each cluster.
    k : int
        Number of clusters.
    p : float
        Probability of an edge within the same cluster.
    q : float
        Probability of an edge between different clusters.

    Returns
    -------
    adj_mat : scipy.sparse.csr_matrix, shape = (n*k, n*k)
        The symmetric adjacency matrix of the generated graph with self loops added.
    labels : numpy.ndarray, shape = (n*k,)
        The ground truth cluster labels
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not isinstance(p, float):
        raise TypeError("p must be a float")
    if not isinstance(q, float):
        raise TypeError("q must be a float")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1")

    size, data, indices, indptr, labels = coreset_sc.gen_sbm(n, k, p, q)
    adj_mat = utils.convert_to_csr_matrix(size, data, indptr, indices)
    return adj_mat, labels


# def stag_sbm(n, k, p, q):
#     assert isinstance(n, int), "n must be an integer"
#     assert isinstance(k, int), "k must be an integer"
#     assert isinstance(p, float), "p must be a float"
#     assert isinstance(q, float), "q must be a float"
#     assert n > 0, "n must be greater than 0"
#     assert k > 0, "k must be greater than 0"
#     assert 0 <= p <= 1, "p must be between 0 and 1"
#     assert 0 <= q <= 1, "q must be between 0 and 1"
#     N = int(n * k)
#     g = stag.random.sbm(N, k, p, q, False)
#     adj = g.adjacency().to_scipy()
#     adj = (adj + scipy.sparse.eye(int(n * k))).tocsr()
#     labels = stag.random.sbm_gt_labels(N, k)

#     return adj, labels
