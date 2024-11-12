Usage
=====

Basic Usage
-----------

The following example demonstrates how to use :class:`coreset_sc.CoresetSpectralClustering` to generate a graph using the stochastic block model (SBM) and perform spectral clustering with coreset sampling.
Vertices are assumed to have self loops since edges represent similarity between vertices.

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

Advanced usage
---------------

There are additional parameters that can be set in the constructor of :class:`coreset_sc.CoresetSpectralClustering`.


This example shows demostrates how to swap out Kmeans for MiniBatchKMeans, shift the implicit kernel matrix by
a constant \*D^{-1}, use a custom over sampling factor for seeding the coreset distribution,
only cluster the coreset graph, and turn off warnings about negative kernel distances triggering clipping.

.. code-block:: python

   from coreset_sc import CoresetSpectralClustering, gen_sbm
   from sklearn.metrics.cluster import adjusted_rand_score
   from sklearn.cluster import MiniBatchKMeans

   # Generate a graph from the stochastic block model
   n = 1000            # number of nodes per cluster
   k = 50              # number of clusters
   p = 0.5             # probability of an intra-cluster edge
   q = (1.0 / n) / k   # probability of an inter-cluster edge

   # A is a sparse scipy CSR matrix of a symmetric adjacency graph
   A, ground_truth_labels = gen_sbm(n, k, p, q)

   coreset_ratio = 0.1  # fraction of the data to use for the coreset graph

   csc = CoresetSpectralClustering(
      num_clusters=k,
      coreset_ratio=coreset_ratio,
      k_over_sampling_factor=5.0  # increase the number of samples for seeding the coreset distribution
      shift = 0.25, # shift the implicit kernel matrix by  0.25* D^{-1}
      kmeans_alg=MiniBatchKMeans(n_clusters=k, batch_size=2048),  # use MiniBatchKMeans instead of KMeans,
      full_labels=False # only cluster the coreset graph
      ignore_warnings=True # turn off warnings about negative kernel distances triggering clipping
   )
   csc.fit(A)  # sample, extract, and cluster the coreset graph.
   coreset_labels = csc.coreset_labels_  # get the coreset labels
   csc.label_full_graph()  # label the rest of the graph given the coreset labels
   pred_labels = csc.labels_  # get the full labels

   # Alternatively, label the full graph in one line (ignores full_labels=False):
   pred_labels = csc.fit_predict(A)
   ari = adjusted_rand_score(ground_truth_labels, pred_labels)

   print(f"Adjusted Rand Index: {ari:.2f}")
