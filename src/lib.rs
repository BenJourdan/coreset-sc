//! coreset-sc is a Rust library with python bindings
//! for Coreset Spectral Clustering.
//!
//! The library implementes the coreset spectral clustering algorithm given in the [paper](https://openreview.net/forum?id=1qgZXeMTTU).
//!
//! The (rust) API allows the user to pass as input one of the following:
//!
//! 1. A sparse symmetric adjacency matrix in the form of a CSR/CSC matrix.
//!     - Coreset spectral clustering is performed on the input graph.
//! 2. Input data points in euclidean space in the form of a 2D array.
//!     - A k-nn graph is constructed using [faiss-rs](https://github.com/Enet4/faiss-rs)
//!     - Or a kernel function can be specified and a coreset of the full similarity matrix can be computed lazily.
//! 3. A k-nn oracle that can be used to query the k-nearest neighbours of a point.
//!     - The coreset will be computed lazily using the oracle.



mod rust;

// Import necessary dependencies from PyO3
#[cfg(feature = "bindings")]
use pyo3::prelude::*;


// Python bindings: conditionally included if the "bindings" feature is enabled
#[cfg(feature = "bindings")]
#[allow(non_snake_case)]
#[pymodule]
fn coreset_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "sum_as_string")]
    fn sum_as_string_py(a: usize, b: usize) -> PyResult<String>{
        Ok((a + b).to_string())
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust::rust_sum_as_string;
    // Pure Rust test
    #[test]
    fn test_rust_sum_as_string() {
        assert_eq!(rust_sum_as_string(1, 2), "3");
    }

}
