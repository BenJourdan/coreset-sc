

use faer::sparse::{SparseRowMatRef, SymbolicSparseRowMatRef};
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
#[allow(unused_variables)]
#[cfg(test)]
mod tests {
    // use super::*;
    use crate::coreset::common::*;
    // use crate::coreset::sampling_tree::*;
    // use crate::unstable;

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
