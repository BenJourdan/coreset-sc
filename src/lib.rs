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

mod coreset;
use rust::*;
mod rust;

pub use rust::default_coreset_sampler;

// Import necessary dependencies from PyO3
#[cfg(feature = "bindings")]
use pyo3::prelude::*;

#[cfg(feature = "bindings")]
use pyo3::{
    pymodule,
    types::{PyModule, PyTuple},
    Bound, PyResult,
};
#[cfg(feature = "bindings")]
use numpy::{IntoPyArray, PyReadonlyArray1};
#[cfg(feature = "bindings")]
use ndarray::ArrayView1;
#[cfg(feature = "bindings")]
use faer::mat::from_raw_parts;






// Python bindings: conditionally included if the "bindings" feature is enabled
#[cfg(feature = "bindings")]
#[allow(non_snake_case)]
#[pymodule]
fn coreset_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use rand::{rngs::StdRng, SeedableRng};


    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(name = "default_coreset")]
    fn improved_coreset_py<'py>(
        py: Python<'py>,
        clusters: usize,
        n: usize,
        coreset_size: usize,
        data: PyReadonlyArray1<'py, f64>,
        indices: PyReadonlyArray1<'py, usize>,
        indptr: PyReadonlyArray1<'py, usize>,
        nnz_per_row: PyReadonlyArray1<'py, usize>,
        degrees: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyTuple>{

        let data_slice: &[f64] = data.as_slice().unwrap();
        let indices_slice: &[usize] = indices.as_slice().unwrap();
        let indptr_slice: &[usize] = indptr.as_slice().unwrap();
        let nnz_per_row_slice: &[usize] = nnz_per_row.as_slice().unwrap();
        let adj_mat_faer = data_indices_indptr_to_sparse_mat_ref(n, data_slice, indices_slice, indptr_slice, nnz_per_row_slice);
        let degrees_numpy: ArrayView1<f64> = degrees.as_array();
        let degrees_faer = unsafe{from_raw_parts::<f64>(degrees_numpy.as_ptr(), 1,n,1,1)};
        let (indices, weights) = default_coreset_sampler(adj_mat_faer, degrees_faer, clusters, coreset_size, StdRng::from_entropy()).unwrap();
        let indices_py = indices.into_pyarray_bound(py);
        let weights_py = weights.into_pyarray_bound(py);
        let tuple = PyTuple::new_bound(py, &[indices_py.to_object(py),weights_py.to_object(py)]);
        tuple
    }


    Ok(())
}

#[cfg(test)]
mod tests {
    // use super::*;
    // Pure Rust test


}
