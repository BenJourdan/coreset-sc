

use std::collections::HashMap;

use faer::sparse::{SparseRowMat, SparseRowMatRef, SymbolicSparseRowMat, SymbolicSparseRowMatRef};
use faer::MatRef;
use rand::rngs::StdRng;


use crate::coreset;


use coreset::common::{Float,Error};
use coreset::unstable;
use coreset::full::DefaultCoresetSampler;




pub fn data_indices_indptr_to_sparse_mat_ref<'a ,E>(
    n: usize,
    data: &'a [E],
    indices: &'a [usize],
    indptr: &'a [usize],
    nnz_per_row: &'a [usize]) -> SparseRowMatRef::<'a, usize, E>
where E: faer::Entity + faer::SimpleEntity {
    let symbolic_sparse_mat_ref =
         SymbolicSparseRowMatRef::new_checked(
            n,
            n,
            indptr,
            Some(nnz_per_row),
            indices,
        );

        SparseRowMatRef::new(
        symbolic_sparse_mat_ref,
        data
    )
}


pub fn default_coreset_sampler(adj_matrix: SparseRowMatRef<usize, Float>, degree_vector: MatRef<Float>, num_clusters: usize, coreset_size: usize, rng: StdRng) -> Result<(Vec<usize>,Vec<Float>),Error>{
    let mut default_sampler: DefaultCoresetSampler<unstable::TreeNode> = coreset::full::DefaultCoresetSampler::new(adj_matrix, degree_vector, num_clusters, coreset_size, rng);
    default_sampler.sample()
}

#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn extract_coreset_graph(adj_matrix: SparseRowMatRef<usize, Float>, degree_vector: MatRef<Float>, coreset_indices: &[usize], coreset_weights: &[Float]) -> SparseRowMat<usize,Float>{

    // first build a map from the raw index to the coreset index. We can also use this to check coreset membership.
    let raw_index_to_coreset_index = coreset_indices.iter().enumerate().map(|(i,coreset_i)|{
        (*coreset_i,i)
    }).collect::<HashMap<usize,usize>>();

    let W_D_inv = coreset_indices.iter().enumerate().map(|(coreset_idx,original_idx)|{
        coreset_weights[coreset_idx]/degree_vector[(0,*original_idx)]
    }).collect::<Vec<Float>>();

    let c = coreset_indices.len();
    // Todo: Guess the number of non-zero entrires in the coreset graph.
    let mut data = Vec::<Float>::with_capacity(c*200);
    let mut indices = Vec::<usize>::with_capacity(c*200);
    let mut indptr = Vec::<usize>::with_capacity(c+1);
    let mut nnz_per_row = Vec::<usize>::with_capacity(c);

    let mut indptr_counter = 0;
    for &i in coreset_indices.iter(){
        let adj_row_indices = adj_matrix.col_indices_of_row(i);
        let adj_row_data = adj_matrix.values_of_row(i);
        // get the neighbours of i that are in the coreset and transform the data
        // We are computing A_C = W_CD^{-1}_C A_C D^{-1}_C W_C
        // where:
        //  -A_C is the submatrix of A corresponding to the coreset indices,
        //  -W_C is the diagonal matrix of coreset weights,
        //  -D is the diagonal matrix of A and D_C is the submatrix of D corresponding to the coreset indices.

        let W_D_inv_i = W_D_inv[i];
        let good_indices_and_data_transformed = adj_row_indices.zip(adj_row_data.iter()).filter_map(|(j,&data)|{
            raw_index_to_coreset_index.get(&j).map(|&coreset_j|{
                (coreset_j, data*W_D_inv_i*W_D_inv[coreset_j])
            })
        }).collect::<Vec<(usize,Float)>>();

        //  push the transformed data:
        data.extend(good_indices_and_data_transformed.iter().map(|(_,data)| *data));
        // push the indices
        indices.extend(good_indices_and_data_transformed.iter().map(|(idx,_)| *idx));

        // push the non-zeros
        let nnz = good_indices_and_data_transformed.len();
        nnz_per_row.push(nnz);

        // push the indptr counter and increment it
        indptr.push(indptr_counter);
        indptr_counter += nnz;
    }

    // push the last indptr
    indptr.push(indptr_counter);

    // Now we have the data, indices, indptr and nnz_per_row. We can now build and return the coreset graph.
    SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(
            c,
            c,
            indptr,
            Some(nnz_per_row),
            indices,
        ),
        data
    )
}





#[allow(dead_code)]
#[allow(unused_variables)]
#[cfg(test)]
mod tests {
    // use super::*;
    use crate::coreset::common::*;

    use rand::Rng;

    fn compute_actual_total_contribution(data: &[Datapoint], min_self_affinity: SelfAffinity) -> Contribution{
        data.iter().map(|datapoint| {
            let delta = datapoint.weight.0*(datapoint.self_affinity.0 + min_self_affinity.0);
            datapoint.weight.0 * delta
        }).sum::<Float>().into()
    }

    fn compute_total_smoothed_contribution(data: &[Datapoint], min_self_affinity: SelfAffinity, coreset_star_weight: Weight) -> SmoothedContribution{
        let contribution = compute_actual_total_contribution(data, min_self_affinity);

        todo!()

    }





}
