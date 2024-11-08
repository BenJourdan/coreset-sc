Usage
=====

Basic Usage
-----------

The following example demonstrates how to use `coreset-sc` to generate a graph using the stochastic block model (SBM) and perform spectral clustering with coreset sampling:

.. code-block:: python

   from coreset_sc import CoresetSpectralClustering, gen_sbm
   from sklearn.metrics.cluster import adjusted_rand_score

   # Generate a graph from the stochastic block model
   n = 1000            # number of nodes per cluster
   k = 50              # number of clusters
   p = 0.5             # probability of an intra-cluster edge
   q = (1.0 / n) / k   # probability of an inter-cluster edge

   # A is a sparse scipy CSR matrix of a symmetric adjacency graph
   A, ground_truth_labels = gen_sbm(n, k, p, q)

   coreset_ratio = 0.1  # fraction of the data to use for the coreset graph

   csc = CoresetSpectralClustering(
      num_clusters=k, coreset_ratio=coreset_ratio
   )
   csc.fit(A)  # sample, extract, and cluster the coreset graph
   csc.label_full_graph()  # label the rest of the graph given the coreset labels
   pred_labels = csc.labels_  # get the full labels

   # Alternatively, label the full graph in one line:
   pred_labels = csc.fit_predict(A)
   ari = adjusted_rand_score(ground_truth_labels, pred_labels)

   print(f"Adjusted Rand Index: {ari:.2f}")
