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

#![allow(clippy::deprecated)]

mod coreset;
use faer::ColRef;


mod rust;
use numpy::PyArray1;
use numpy::ToPyArray;
use rust::*;
mod sbm;

pub use rust::default_coreset_sampler;
pub use rust::gen_sbm_with_self_loops;
pub use rust::label_full_graph;
pub use rust::compute_conductances;


use pyo3::prelude::*;


use pyo3::{
    pymodule,
    types::{PyModule, PyTuple},
    Bound, PyResult,
};

use numpy::{IntoPyArray, PyReadonlyArray1};

use ndarray::ArrayView1;
use faer::mat::from_raw_parts;
use faer::mat::MatRef;
use faer_ext::IntoNdarray;
use faer::sparse::SparseRowMatRef;
use crate::coreset::common::Float;



pub fn construct_from_py<'py>(
    n: usize,
    data: &'py PyReadonlyArray1<f64>,
    indices: &'py PyReadonlyArray1<usize>,
    indptr: &'py PyReadonlyArray1<usize>,
    nnz_per_row: &'py PyReadonlyArray1<usize>,
    degrees: &'py PyReadonlyArray1<f64>,
) -> (SparseRowMatRef<'py, usize, Float>, ColRef<'py,Float>){
    let adj_mat_faer: SparseRowMatRef< usize,Float> = data_indices_indptr_to_sparse_mat_ref(n,
        data.as_slice().unwrap(),
        indices.as_slice().unwrap(),
        indptr.as_slice().unwrap(),
        nnz_per_row.as_slice().unwrap());
    let degrees_numpy: ArrayView1<f64> = degrees.as_array();
    let degrees_faer: MatRef<Float> = unsafe{from_raw_parts::<f64>(degrees_numpy.as_ptr(), n,1,1,1)};
    let degrees_faer = degrees_faer.col(0);
    (adj_mat_faer, degrees_faer)
}


#[allow(clippy::deprecated)]
#[allow(non_snake_case)]
#[pymodule]
fn coreset_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // use numpy::PyArray1;
    use rand::{rngs::StdRng, SeedableRng};



    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(name = "old_coreset")]
    fn old_coreset_py<'py>(
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


        let (adj_mat_faer, degrees_faer) = construct_from_py(n, &data, &indices, &indptr, &nnz_per_row, &degrees);

        let (indices, weights) = coreset::old::old_coreset(adj_mat_faer, degrees_faer, clusters, coreset_size, &mut StdRng::from_os_rng());
        let indices_py = indices.into_pyarray(py);
        let weights_py = weights.into_pyarray(py);
        let tuple = PyTuple::new_bound(py, &[indices_py.to_object(py),weights_py.to_object(py)]);
        tuple
    }

    #[allow(clippy::too_many_arguments)]
    #[pyfn(m)]
    #[pyo3(name = "coreset_embeddings")]
    fn default_csc_py<'py>(
        py: Python<'py>,
        clusters: usize,
        n: usize,
        coreset_size: usize,
        k_over_sampling_factor: f64,
        data: PyReadonlyArray1<'py, f64>,
        indices: PyReadonlyArray1<'py, usize>,
        indptr: PyReadonlyArray1<'py, usize>,
        nnz_per_row: PyReadonlyArray1<'py, usize>,
        degrees: PyReadonlyArray1<'py, f64>,
        shift: f64,
        ignore_warnings: bool,
    ) -> Bound<'py, PyTuple>{
        let (adj_mat_faer, degrees_faer) = construct_from_py(n, &data, &indices, &indptr, &nnz_per_row, &degrees);
        let (indices, weights,numerical_warning) = default_coreset_sampler(
            adj_mat_faer, degrees_faer, (clusters as Float * k_over_sampling_factor) as usize,
            coreset_size, Some(shift), StdRng::from_os_rng()).unwrap();
        if numerical_warning && !ignore_warnings{
            let user_warning = py.get_type_bound::<pyo3::exceptions::PyUserWarning>();
            pyo3::PyErr::warn_bound(
                py,
                &user_warning,
                "Negative distance encountered while sampling coreset. If you are getting odd results, try increasing the shift parameter.",
                0
            ).unwrap();
        }
        let (indices,weights) = aggregate_coreset_weights(indices, weights);
        let coreset_size = indices.len();
        let mut rng = StdRng::from_os_rng();
        let coreset_embeddings = compute_coreset_embeddings(adj_mat_faer.as_ref(), degrees_faer.as_ref(), &indices, &weights, clusters, coreset_size,Some(shift), &mut rng);
        let coreset_embeddings = coreset_embeddings.as_ref().into_ndarray();
        let indices_py = indices.into_pyarray(py);
        let weights_py = weights.into_pyarray(py);
        let coreset_embeddings_py = coreset_embeddings.to_pyarray(py);
        let tuple = PyTuple::new(py, &[indices_py.to_object(py),weights_py.to_object(py),coreset_embeddings_py.to_object(py)]).unwrap();
        tuple
    }

    #[pyfn(m)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "label_full_graph")]
    fn label_full_graph_py<'py>(
        py: Python<'py>,
        clusters: usize,
        n: usize,
        data: PyReadonlyArray1<'py, f64>,
        indices: PyReadonlyArray1<'py, usize>,
        indptr: PyReadonlyArray1<'py, usize>,
        nnz_per_row: PyReadonlyArray1<'py, usize>,
        degrees: PyReadonlyArray1<'py, f64>,
        coreset_indices: PyReadonlyArray1<'py, usize>,
        coreset_weights: PyReadonlyArray1<'py, f64>,
        coreset_labels: PyReadonlyArray1<'py, usize>,
        shift: f64,
    ) -> Bound<'py, PyTuple>{

        let (adj_mat_faer, degrees_faer) = construct_from_py(n, &data, &indices, &indptr, &nnz_per_row, &degrees);

        let coreset_indices = coreset_indices.as_array();
        let coreset_weights = coreset_weights.as_array();
        let coreset_labels = coreset_labels.as_array();

        let coreset_indices = coreset_indices.as_slice().unwrap();
        let coreset_weights = coreset_weights.as_slice().unwrap();
        let coreset_labels = coreset_labels.as_slice().unwrap();

        let labels_and_distances2 = label_full_graph(
            adj_mat_faer,
            degrees_faer,
            coreset_indices,
            coreset_weights,
            coreset_labels,
            clusters,
            Some(shift),
        );
        let labels_py = labels_and_distances2.0.into_pyarray(py);
        let distances_py = labels_and_distances2.1.into_pyarray(py);
        let tuple = PyTuple::new_bound(py, &[labels_py.to_object(py),distances_py.to_object(py)]);
        tuple
    }

    #[pyfn(m)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "compute_conductances")]
    fn compute_conductances_py<'py>(
        py: Python<'py>,
        clusters: usize,
        n: usize,
        data: PyReadonlyArray1<'py, f64>,
        indices: PyReadonlyArray1<'py, usize>,
        indptr: PyReadonlyArray1<'py, usize>,
        nnz_per_row: PyReadonlyArray1<'py, usize>,
        degrees: PyReadonlyArray1<'py, f64>,
        labels: PyReadonlyArray1<'py, usize>,
    ) -> Bound<'py,PyArray1<f64>>{

        let (adj_mat_faer, degrees_faer) = construct_from_py(n, &data, &indices, &indptr, &nnz_per_row, &degrees);

        let conductances = compute_conductances(
            adj_mat_faer,
            degrees_faer,
            labels.as_array().as_slice().unwrap(),
            clusters,
        );

        let conductances_py = conductances.into_pyarray(py);
        conductances_py
    }


    #[pyfn(m)]
    #[pyo3(name = "gen_sbm")]
    fn gen_sbm_py(
        py: Python,
        n: usize,
        k: usize,
        p: f64,
        q: f64,
    ) -> Bound<PyTuple> {
        let (adj_mat,labels) = gen_sbm_with_self_loops(n, k, p, q);
        let (symbolic,data) = adj_mat.into_parts();
        let (row_size,col_size,indptr,_,indices) = symbolic.into_parts();

        assert!(row_size == n*k);
        assert!(col_size == n*k);
        let data = data.into_pyarray(py);
        let indices = indices.into_pyarray(py);
        let indptr = indptr.into_pyarray(py);
        let labels = labels.into_pyarray(py);
        let tuple = PyTuple::new_bound(py, &[
            (n*k).to_object(py),
            data.to_object(py),
            indices.to_object(py),
            indptr.to_object(py),
            labels.to_object(py),]);
        tuple
    }


    Ok(())
}
