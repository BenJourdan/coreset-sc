from coreset_sc import CoresetSpectralClustering, gen_sbm
from sklearn.metrics.cluster import adjusted_rand_score

# Generate a graph from the stochastic block model
n = 1000  # number of nodes per cluster
k = 50  # number of clusters
p = 0.5  # probability of an intra-cluster edge
q = (1.0 / n) / k  # probability of an inter-cluster edge


# A is a sparse scipy CSR matrix of a symmetric adjacency graph
A, ground_truth_labels = gen_sbm(n, k, p, q)


coreset_ratio = 0.05  # fraction of the data to use for the coreset graph

csc = CoresetSpectralClustering(
    num_clusters=k,  # required
    coreset_ratio=coreset_ratio,
    # Optional parameters:
    k_over_sampling_factor=2.0,  # a (default) factor of 2 is guaranteed to get us a good coreset whp (in theory!)
    shift=0.01,  # (positive) shift to increase the "positive definiteness" of the kernel matrix
)
csc.fit(A)  # sample extract and cluster the coreset graph
csc.label_full_graph()  # label the rest of the graph given the coreset labels
pred_labels = csc.labels_  # get the full labels

# Alternatively, label the full graph in one line:
pred_labels = csc.fit_predict(A)
ari = adjusted_rand_score(ground_truth_labels, pred_labels)

print(ari)
