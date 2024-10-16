# test_coreset_sc.py
# test


def test_default_coreset():
    """Test the default coreset sampler"""

    """
    #[pyfn(m)]
    #[pyo3(name = "default_coreset")]
    fn improved_coreset_py<'py>(
        py: Python<'py>,
        clusters: usize,
        n: usize,
        coreset_size: usize,
        adj_matrix: &Bound<'py, PyAny>,
        degrees: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyTuple>{
    """
    import time

    import coreset_sc
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import kneighbors_graph

    # Generate some data
    X, y = make_blobs(n_samples=100_000, centers=3, n_features=2, random_state=42)
    # Create a k-nearest neighbors graph
    A = kneighbors_graph(
        X, n_neighbors=200, mode="connectivity", include_self=True, n_jobs=-1
    )
    # Call the function
    t0 = time.time()
    coreset, coreset_weights = coreset_sc.default_coreset(3, 1000, A)
    t1 = time.time()
    print(coreset)
    print(coreset_weights)
    print(f"Time: {t1-t0}")
    raise Exception("stop")
