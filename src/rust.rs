


use std::collections::{HashMap, HashSet};
// use std::time::Instant;

use faer::sparse::{SparseColMat, SparseRowMat, SparseRowMatRef, SymbolicSparseColMat, SymbolicSparseRowMat, SymbolicSparseRowMatRef};
use faer::{unzipped, zipped, Col, ColRef, Mat};
// use faer::sparse::linalg::matmul::{dense_sparse_matmul,sparse_dense_matmul};


use rand::rngs::StdRng;
// use rand::rngs::StdRng;
use rand::prelude::*;

use rand_distr::StandardNormal;

use crate::coreset;
use rayon::prelude::*;

use coreset::common::{Float,Error};
use coreset::unstable;
use coreset::full::DefaultCoresetSampler;
// use smartcore::cluster::kmeans::*;




pub use crate::sbm::gen_sbm_with_self_loops;



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

#[allow(dead_code)]
fn reinterpret_sym_csr_as_csc(mat_in: SparseRowMat<usize,f64>) -> SparseColMat<usize,f64>{
    let (symbolic,data) = mat_in.into_parts();
    let (rows,cols,indptr,nnz,indices) = symbolic.into_parts();
    SparseColMat::new(SymbolicSparseColMat::new_checked(
        rows,
        cols,
        indptr,
        nnz,
        indices
    ),data)
}

#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn compute_coreset_embeddings(
    adj_matrix: SparseRowMatRef<usize, Float>,
    degree_vector: ColRef<Float>,
    coreset_indices: &[usize],
    coreset_weights: &[Float],
    num_clusters: usize,
    coreset_size: usize,
    shift: Option<Float>,
    rng: &mut StdRng) -> Mat<Float>{
    // first extract the coreset graph
    let coreset_graph = extract_coreset_graph(adj_matrix, degree_vector, coreset_indices, coreset_weights, shift);
    let coreset_degree_vector = Col::from_fn(coreset_size, |i|{
        coreset_graph.values_of_row(i).iter().sum::<Float>()
    });
    // Then extract the M matrix:
    let M: SparseRowMat<usize, f64> = convert_to_signless_laplacian(coreset_graph, &coreset_degree_vector);
    // Now we can cluster the coreset graph using Peter's method:
    // We use the traspose because faer doesn't have native support for csr Sparse Dense multiplication.
    let l = num_clusters.min((num_clusters as Float).log2().ceil() as usize);
    let t = 10 * ((coreset_size/num_clusters) as Float).log2().ceil() as usize;
    let mut Y1: Mat<Float> = Mat::from_fn(coreset_size, l, |_,_|{
        rng.sample(StandardNormal)
    });
    let mut Y2: Mat<Float> = Mat::zeros(coreset_size,l);
    let mut top_eigvec = zipped!(&coreset_degree_vector).map(|unzipped!(d)| d.sqrt());
    let norm = top_eigvec.norm_l2();
    if norm >0.0{
        zipped!(&mut top_eigvec).for_each(|unzipped!(mut v)| *v /= norm);
    }
    for _ in 0..t/2{
        Y2 = M.as_ref()*Y1;
        // Now remove the contribution of the top eigenvector from Y2
        let contributions_to_top_eig = top_eigvec.transpose()*Y2.as_ref();
        Y2.col_iter_mut().zip(contributions_to_top_eig.iter()).for_each(|(mut y2,contrib)|{
            y2 -= *contrib*top_eigvec.as_ref();
        });
        Y1 = M.as_ref()*Y2;
        // Now remove the contribution of the top eigenvector from Y1
        let contributions_to_top_eig = top_eigvec.transpose()*Y1.as_ref();
        Y1.col_iter_mut().zip(contributions_to_top_eig.iter()).for_each(|(mut y1,contrib)|{
            y1 -= *contrib*top_eigvec.as_ref();
        });

    }
    Y1
}

pub fn convert_to_signless_laplacian(adj_matrix: SparseRowMat<usize, Float>, degree_vector: &Col<Float>)->SparseRowMat<usize, Float>{
    // Fuse everything together. We only loop over the non-zero entries of the adjacency matrix once.

    // We want to replicate the following python code assuming D is np.diag(A.sum(axis=1)):

    //
    // D_inv_half = np.diag(1 / np.sqrt(D.diagonal()))
    // L = D - A
    // N = D_inv_half @ L @ D_inv_half
    // M = np.eye(n) - (0.5)*N

    // Expanding M we see that

    // M = np.eye(n) - (0.5)*D_inv_half@(D-A)@D_inv_half
    // = (0.5)*np.eye(n) + (0.5)*D_inv_half@A@D_inv_half
    // = (0.5)*(np.eye(n) + D_inv_half@A@D_inv_half)
    let n = adj_matrix.nrows();

    let degree_inv_half = zipped!(degree_vector).map(|unzipped!(d)| 1.0/(d.sqrt()));


    // We first deconstruct the sparse matrix into it's component parts
    // We need to do this

    let (sparse_rep,mut data) = adj_matrix.into_parts();
    let (rows,cols,indptr,nnz,indices) = sparse_rep.into_parts();

    let nnz = nnz.unwrap();
    let mut data_chunks: Vec<&mut [Float]> = Vec::with_capacity(n);
    let mut indices_chunks: Vec<&[usize]> = Vec::with_capacity(n);
    let mut remaining_data: &mut [Float] = &mut data;
    let mut remaining_indices: &[usize] = & indices;

    for &nnz_i in nnz.iter() {
        let (data_chunk, data_remaining) = remaining_data.split_at_mut(nnz_i);
        let (indices_chunk, indices_remaining) = remaining_indices.split_at(nnz_i);
        data_chunks.push(data_chunk);
        indices_chunks.push(indices_chunk);
        remaining_data = data_remaining;
        remaining_indices = indices_remaining;
    }

    data_chunks.into_par_iter()
    .zip(indices_chunks.par_iter())
    .enumerate().for_each(|(i,(data_chunk,indices_chunk))|{
        let d_inv_half_i = degree_inv_half[i];
        for (data, &j) in data_chunk.iter_mut().zip(indices_chunk.iter()){
            *data = 0.5*(((i==j) as usize as Float) + d_inv_half_i**data*degree_inv_half[j]);
            // This is the line where we fuse everything together :)
        }
    });

    // reassemble the matrix:
    SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(
            rows,
            cols,
            indptr,
            Some(nnz),
            indices),
            data
    )
}

pub fn default_coreset_sampler(
    adj_matrix: SparseRowMatRef<usize, Float>,
    degree_vector: ColRef<Float>,
    num_clusters: usize,
    coreset_size: usize,
    shift: Option<Float>,
    rng: StdRng) -> Result<(Vec<usize>,Vec<Float>,bool),Error>{
    let mut default_sampler: DefaultCoresetSampler<unstable::TreeNode> = coreset::full::DefaultCoresetSampler::new(adj_matrix, degree_vector, num_clusters, coreset_size, shift, rng);
    default_sampler.sample()
}

#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn extract_coreset_graph(
    adj_matrix: SparseRowMatRef<usize, Float>,
    degree_vector: ColRef<Float>,
    coreset_indices: &[usize],
    coreset_weights: &[Float],
    shift: Option<Float>
    ) -> SparseRowMat<usize,Float>{

    // first build a map from the raw index to the coreset index. We can also use this to check coreset membership.
    let raw_index_to_coreset_index = coreset_indices.iter().enumerate().map(|(i,coreset_i)|{
        (*coreset_i,i)
    }).collect::<HashMap<usize,usize>>();

    let shift = shift.unwrap_or(0.0);

    let W_D_inv = coreset_indices.iter().enumerate().map(|(coreset_idx,original_idx)|{
        coreset_weights[coreset_idx]/degree_vector[*original_idx]
    }).collect::<Vec<Float>>();

    let coreset_size = coreset_indices.len();
    // Todo: Guess the number of non-zero entrires in the coreset graph.
    let mut data = Vec::<Float>::with_capacity(coreset_size*200);
    let mut indices = Vec::<usize>::with_capacity(coreset_size*200);
    let mut indptr = Vec::<usize>::with_capacity(coreset_size+1);
    let mut nnz_per_row = Vec::<usize>::with_capacity(coreset_size);

    let mut indptr_counter = 0;
    for (i,&index) in coreset_indices.iter().enumerate(){
        let adj_row_indices = adj_matrix.col_indices_of_row(index);
        let adj_row_data = adj_matrix.values_of_row(index);
        // get the neighbours of index that are in the coreset and transform the data
        // We are computing
        // A_C = W_CD^{-1}_C A_C D^{-1}_C W_C + W_C shift*D^{-1}_C W_C
        //     = W_CD^{-1}_C A_C D^{-1}_C W_C + shift* W_C*D^{-1}_C W_C
        // where:
        //  -A_C is the submatrix of A corresponding to the coreset indices,
        //  -W_C is the diagonal matrix of coreset weights,
        //  -D is the diagonal matrix of A and D_C is the submatrix of D corresponding to the coreset indices.

        let W_D_inv_i = W_D_inv[i];
        let mut good_indices_and_data_transformed = adj_row_indices.zip(adj_row_data.iter()).filter_map(|(j,&data)|{
            if index == j{
                raw_index_to_coreset_index.get(&j).map(|&coreset_j|{
                (coreset_j, data*W_D_inv_i*W_D_inv[i] + shift*coreset_weights[i]*W_D_inv_i)
            })}
            else{
                raw_index_to_coreset_index.get(&j).map(|&coreset_j|{
                (coreset_j, data*W_D_inv_i*W_D_inv[coreset_j])
            })}

        }).collect::<Vec<(usize,Float)>>();
        good_indices_and_data_transformed.sort_unstable_by_key(|&(idx,_)| idx);
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
            coreset_size,
            coreset_size,
            indptr,
            Some(nnz_per_row),
            indices,
        ),
        data
    )
}

#[allow(dead_code)]
pub fn aggregate_coreset_weights(coreset_indices: Vec<usize>, coreset_weights: Vec<f64>)-> (Vec<usize>,Vec<f64>){
    let mut coreset_map: HashMap<usize,f64> = HashMap::with_capacity(coreset_indices.len());
    coreset_indices.into_iter().zip(coreset_weights).for_each(|(i,w)|{
        *coreset_map.entry(i).or_insert(0.0) += w;
    });
    coreset_map.into_iter().unzip()
}

pub fn label_full_graph(
    adj_mat: SparseRowMatRef<usize, Float>,
    degree_vector: ColRef<Float>,
    coreset_indices: &[usize],
    coreset_weights: &[Float],
    coreset_labels: &[usize],
    num_clusters: usize,
    shift: Option<Float>
) -> (Vec<usize>, Vec<Float>){


    let shift = shift.unwrap_or(0.0);
    let num_points = adj_mat.nrows();

    // group the coreset and coreset weights by label and process each group in parallel:

    let coreset_grouped = coreset_indices.iter().zip(coreset_labels).zip(coreset_weights).fold(
        vec![(vec![],vec![]);num_clusters], |mut acc, ((&i, &label), &weight)|{
            acc[label].0.push(i);
            acc[label].1.push(weight);
            acc
        }
    );

    // Now we compute the center norms and center denoms for each cluster
    let result = coreset_grouped.into_par_iter().enumerate().map(|(_, (indices,weights))|{
        // return zero if the cluster is empty
        if indices.is_empty(){
            return (0.0,0.0);
        }

        let indices_set: HashSet<&usize> = indices.iter().collect();
        let index_to_weight: HashMap<_,_> = indices.iter().zip(weights.iter()).collect();

        // compute the denominator:
        let denom = weights.iter().sum::<Float>();
        // compute the center norm sum
        let mut center_norm_sum = 0.0;
        indices.iter().for_each(|i|{
            let weight = index_to_weight[i];
            let neighbour_indices = adj_mat.col_indices_of_row(*i);
            let neighbour_values = adj_mat.values_of_row(*i).iter().enumerate().map(|(j,v)|{
                if *i!=j{
                    v/(degree_vector[*i]*degree_vector[j])
                }else{
                    v/(degree_vector[*i]*degree_vector[j]) + shift/(degree_vector[*i])
                }

            });
            center_norm_sum += neighbour_indices.zip(neighbour_values).fold(
                0.0f64, |acc, (j, value)|{
                    match indices_set.contains(&j){
                        true => acc + value*weight*index_to_weight[&j],
                        false => acc
                    }
                });
        });
        center_norm_sum /= denom*denom;
        (center_norm_sum,denom)
    }).collect::<Vec<(Float,Float)>>();

    let (center_norms, center_denoms): (Vec<f64>,Vec<f64>) = result.into_iter().unzip();

    // Now find the cluster with the smallest center norm - this will be the "default" cluster

    let smallest_center_by_norm = center_norms.iter().enumerate()
        .min_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let smallest_center_by_norm_value = center_norms[smallest_center_by_norm];

    // Now prepare to label everything in parallel:

    let coreset_set = coreset_indices.iter().collect::<HashSet<_>>();
    let label_map = coreset_indices.iter().zip(coreset_labels).collect::<HashMap<_,_>>();
    let weight_map = coreset_indices.iter().zip(coreset_weights).collect::<HashMap<_,_>>();

    let labels_and_distances2: (Vec<usize>,Vec<Float>) = (0..num_points).into_par_iter().map(|i|{

        let vertex_degree = degree_vector[i];
        // store the inner product to all the centers
        let mut x_to_c_is = HashMap::new();


        let neighbour_indices = adj_mat.col_indices_of_row(i);
        let neighbour_edge_weights = adj_mat.values_of_row(i);
        // let neighbour_values = adj_mat.values_of_row(i).iter().enumerate().map(|(j,v)|{
        //     v/(degree_vector[i]*degree_vector[j])
        // });

        neighbour_indices.enumerate().for_each(|(j,indx)|{
            if coreset_set.contains(&indx){
                    let label = label_map[&indx];
                    let neighbour_weight = weight_map[&indx];
                    let inner_prod_with_vertex = {
                        if i!=indx{
                            neighbour_edge_weights[j]/(vertex_degree*degree_vector[indx])
                        }else{
                            neighbour_edge_weights[j]/(vertex_degree*degree_vector[indx]) + shift/(vertex_degree)
                        }
                    };
                    x_to_c_is.entry(label).and_modify(|e|{
                        *e += neighbour_weight*inner_prod_with_vertex;
                    }).or_insert(neighbour_weight*inner_prod_with_vertex);
                }
        });

        // normalize the inner products to each cluster by each center denominator
        x_to_c_is.iter_mut().for_each(|(k,v)| *v /= center_denoms[**k]);

        let mut best_center = smallest_center_by_norm;
        let mut best_center_value = smallest_center_by_norm_value;

        x_to_c_is.iter().for_each(|(center,v)|{
            // right now v is just the inner product to each center, not the distance

            // When we compute the (smallest) distance, we can ignore the contribution of the vertex
            let distance = center_norms[**center] - 2.0*v;
            if distance < best_center_value{
                best_center = **center;
                best_center_value = distance;
            }
        });
        (best_center,best_center_value + adj_mat[(i,i)]/(vertex_degree*vertex_degree) + shift/(vertex_degree))
    }).unzip();
    labels_and_distances2
}

pub fn compute_conductances(
    adj_mat: SparseRowMatRef<usize, Float>,
    degrees: ColRef<Float>,
    labels: &[usize],
    num_clusters: usize,
) -> Vec<Float>{
    let mut volumes = vec![0.0; num_clusters];
    let mut cuts = vec![0.0; num_clusters];
    let mut grouped_labels = vec![Vec::new(); num_clusters];
    labels.iter().enumerate().for_each(|(i,&label)|{
        volumes[label] += degrees[i];
        grouped_labels[label].push(i);
    });
    grouped_labels.par_iter().zip(cuts.par_iter_mut())
        .enumerate().for_each(|(_,(cluster_indices, cut))|{
            cluster_indices.iter().for_each(|&i|{
                let neighbour_indices = adj_mat.col_indices_of_row(i);
                let neighbour_values = adj_mat.values_of_row(i);
                neighbour_indices.zip(neighbour_values).for_each(|(j,&v)|{
                    if labels[i] != labels[j] && j>i{
                        *cut += v;
                    }
                });
            });
    });
    cuts.iter().zip(volumes.iter()).map(|(&cut,&vol)|{
        if vol > 0.0{
            cut/vol
        }else{
            0.0
        }
    }).collect()
}


#[allow(dead_code)]
#[allow(unused_variables)]
#[cfg(test)]
mod tests {

    use faer::Col;
    use rand::rngs::StdRng;
    use rand::prelude::*;

    // use super::*;
    use crate::{aggregate_coreset_weights, coreset::common::*};

    use super::{default_coreset_sampler, gen_sbm_with_self_loops};

    // use rand::Rng;

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

    #[test]
    fn test_coreset_clustering(){

        let n = 100;
        let num_clusters = 5;
        let p = 0.99;
        let q = 0.001;

        let num_nodes = n*num_clusters;

        let (adj_mat,labels) = gen_sbm_with_self_loops(n, num_clusters, p, q);
        assert!(adj_mat.nrows() == num_nodes);
        assert!(adj_mat.ncols() == num_nodes);

        let degree_vector = Col::from_fn(num_nodes, |i|{
            adj_mat.values_of_row(i).iter().sum::<Float>()
        });
        assert!(degree_vector.nrows() == num_nodes);

        let coreset_size = n*num_clusters/10;

        let rng = StdRng::from_os_rng();

        let (coreset_indices, coreset_weights,warning) = default_coreset_sampler(adj_mat.as_ref(), degree_vector.as_ref(), 2*num_clusters, coreset_size,Some(0.0), rng).unwrap();
        let (coreset_indices,coreset_weights) = aggregate_coreset_weights(coreset_indices, coreset_weights);
        let coreset_size = coreset_indices.len();
        let mut rng = StdRng::from_os_rng();

        let coreset_embeddings = super::compute_coreset_embeddings(adj_mat.as_ref(), degree_vector.as_ref(), &coreset_indices, &coreset_weights, num_clusters, coreset_size,Some(0.0) ,&mut rng);


        // panic!();
    }




}
