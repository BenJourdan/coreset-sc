use faer::Col;
use faer::{sparse::SparseRowMatRef, unzipped, zipped, ColRef};
use faer::sparse::SparseRowMat;
use rand::rngs::StdRng;
use rand::Rng;

use super::common::Float;




#[allow(non_snake_case)]
pub fn old_coreset<'a>(
    adj_matrix: SparseRowMatRef<'a,usize, Float>,
    degree_vector: ColRef<'a,Float>,
    num_clusters: usize,
    coreset_size: usize,
    rng: &mut StdRng)
    -> (Vec<usize>, Vec<Float>){

    // First construct K with D^{-1}AD^{-1}
    let n = adj_matrix.nrows();
    assert_eq!(n, adj_matrix.ncols());
    assert_eq!(n, degree_vector.nrows());

    let W = degree_vector;
    let d_inv = zipped!(degree_vector).map(|unzipped!(d)| 1.0/d.read());
    let triplets: Vec<(usize,usize,f64)> = (0..n).map(|i| (i,i, d_inv[i])).collect();
    let diagonal_d_inv = SparseRowMat::try_new_from_triplets(n, n, triplets.as_slice()).unwrap();

    let K = diagonal_d_inv.as_ref()*adj_matrix.as_ref()*diagonal_d_inv.as_ref();


    let self_affinities = Col::from_fn(n, |i| K[(i,i)]);
    let mut probs = Col::zeros(n);

    let mut initial_coreset = Vec::with_capacity(coreset_size);
    let mut initial_coreset_weight = 0.0;


    // Select the first point uniformly at random
    let mut index_to_add = rng.random_range(0..n);
    initial_coreset.push(index_to_add);
    initial_coreset_weight += W[index_to_add];

    let mut dist_2_to_new_index = Col::from_fn(n, |i| (K[(index_to_add,index_to_add)] + K[(i,i)] - 2.0*K.get(index_to_add, i).unwrap_or(&0.0)));

    (0..num_clusters-1).for_each(|_|{
        // Sample the next point proportional to the distance to the initial coreset
        zipped!(&mut probs,&dist_2_to_new_index).for_each_with_index(|i,unzipped!(mut p,d)| *p = W[i]*d.read());
        index_to_add = rng.sample(rand::distr::weighted::WeightedIndex::new(probs.as_slice()).unwrap());
        initial_coreset.push(index_to_add);
        initial_coreset_weight += W[index_to_add];

        let self_affinity = self_affinities[index_to_add];
        zipped!(&mut dist_2_to_new_index).for_each_with_index(|i,unzipped!(mut d)|{
            let new_dist = self_affinity + self_affinities[i] - 2.0*K.get(index_to_add, i).unwrap_or(&0.0);
            *d = (*d).min(new_dist);
        });
    });
    let costs = zipped!(&dist_2_to_new_index).map_with_index(|i,unzipped!(d)| W[i]*d.read());
    // Now we have the initial coreset, we sample the coreset with smoothed sampling
    let total_cost = costs.iter().sum::<Float>();

    let mut coreset_dist = Col::from_fn(n, |i| costs[i]/total_cost + W[i]/initial_coreset_weight);
    let coreset_dist_sum = coreset_dist.iter().sum::<Float>();
    zipped!(&mut coreset_dist).for_each(|unzipped!(mut d)| *d /= coreset_dist_sum);


    let (mut coreset,mut coreset_weights): (Vec<_>,Vec<_>) = (0..coreset_size).map(|_|{
        let index = rng.sample(rand::distr::weighted::WeightedIndex::new(coreset_dist.as_slice()).unwrap());
        let prob = coreset_dist[index];
        let weight = W[index]/(prob*coreset_size as Float);
        (index, weight)
    }).unzip();



    // sort the indices and weights in ascending order of indices
    let mut combined: Vec<_> = coreset.iter_mut().zip(coreset_weights.iter_mut()).collect();
    combined.sort_by(|(a,_),(b,_)| a.cmp(b));

    (coreset,coreset_weights)
}
