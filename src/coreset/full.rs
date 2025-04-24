use crate::coreset::{common::*, sampling_tree::SamplingTree};

use rand::rngs::StdRng;

use faer::sparse::SparseRowMatRef;
use faer::ColRef;
use rand::Rng;


#[derive(Debug)]
pub struct DefaultCoresetSampler<'a,T>{
    sampling_tree: SamplingTree<T>,
    num_clusters: usize,
    coreset_star_weight: Weight,
    coreset_size: usize,
    rng: StdRng,
    number_of_data_points: usize,
    adj_matrix: SparseRowMatRef<'a, usize, Float>,
    degree_vector: ColRef::<'a, Float>,
    self_affinities: Vec<SelfAffinity>,
    x_star_index: Index,
    numerical_warning: bool,
}




/// Assumes a undirected graph where self loops are all 1
impl <'a,T> DefaultCoresetSampler<'a,T>
    where T: Node
{

    pub fn new(
        adj_matrix: SparseRowMatRef<'a,usize, Float>,
        degree_vector: ColRef<'a,Float>,
        num_clusters: usize,
        coreset_size: usize,
        shift: Option<Float>,
        rng: StdRng) -> Self{

        let n = adj_matrix.nrows();
        assert_eq!(n, adj_matrix.ncols());
        assert_eq!(n, degree_vector.nrows());

        let mut sampling_tree = SamplingTree::<T>::new();

        let shift = shift.unwrap_or(0.0);

        // Find the node with the lowest self affinity. Aka lowest value of A[i,i]/d[i]^2
        let self_affinities: Vec<SelfAffinity> = degree_vector
            .iter().enumerate().map(|(i,d)| SelfAffinity((shift/d) +adj_matrix[(i,i)]/(d*d))).collect();
        let x_star = self_affinities.iter().enumerate().min_by(|a,b| a.1.0.partial_cmp(&b.1.0).unwrap()).unwrap().0;
        let min_self_affinity = self_affinities[x_star];

        // Populate the sampling  tree with weights and self affinities
        sampling_tree.insert_from_iterator(degree_vector.iter().zip(self_affinities.iter()).map(|(d,self_affinity)|{
            (Weight(*d),*self_affinity)
        }),
             min_self_affinity
        );

        DefaultCoresetSampler{
            sampling_tree,
            num_clusters,
            coreset_star_weight: Weight(0.0),
            coreset_size,
            rng,
            number_of_data_points: n,
            adj_matrix,
            degree_vector,
            self_affinities,
            x_star_index: Index(x_star),
            numerical_warning: false,
        }
    }


    fn repair(&mut self, point_added: Index){
        // We implicitly add the point to the init set and update it's neighbours:
        let point_added_degree: Float = self.degree_vector[point_added.0];
        let point_added_weight: Weight = point_added_degree.into();
        let point_added_self_affinity: SelfAffinity = self.self_affinities[point_added.0];

        self.coreset_star_weight += point_added_weight;

        // set the contribution of the added point to zero:
        self.sampling_tree.update_delta(point_added, Delta(0.0)).unwrap();
        // Now we update the neighbours of the added point:
        self.adj_matrix.col_indices_of_row(point_added.0).map(Index)
            .zip(self.adj_matrix.values_of_row(point_added.0))
            .for_each(|(neighbour_index,edge_weight)|{
            // If the neighbour is the added point, skip it
            if neighbour_index == self.x_star_index{
                return;
            }
            // compute the distance^2 between the added point and the neighbour:
            let neighbour_degree: Float = self.degree_vector[neighbour_index.0];
            let neighbour_self_affinity: SelfAffinity = self.self_affinities[neighbour_index.0];
            let cross_term: CoresetCrossTerm = (edge_weight/(point_added_degree*neighbour_degree)).into();
            let mut distance2 =  point_added_self_affinity.0 + neighbour_self_affinity.0 - 2.0*cross_term.0;
            // update the delta of the neighbour:
            if distance2 < 0.0{
                self.numerical_warning = true;
                distance2 = 0.0;
            }
            self.sampling_tree.update_delta(neighbour_index, Delta(distance2)).unwrap();
        })
    }

    fn sample_first_point(&mut self){
        self.repair(self.x_star_index);
    }

    fn sample_next_k(&mut self) -> Result<(),Error>{
        // Now we run k-means++ to sample the next k points (total k+1 points)
        // first we uniformly sample the first point and repair:
        let uniform_sampled_index = Index(self.rng.random_range(0..self.number_of_data_points));
        self.repair(uniform_sampled_index);
        // Now we sample the next k-1 points and repair:
        for i in 0..self.num_clusters-1{
            let mut maybe_index = self.sampling_tree.sample(&mut self.rng);

            while let Err(Error::NumericalError) = maybe_index{
                // If we fail to sample, we rebuild the tree and try again:
                println!("Numerical error detected on round {}. Rebuilding tree", i);
                self.sampling_tree.rebuild_from_leaves();
                maybe_index = self.sampling_tree.sample(&mut self.rng);
            }
            let index = maybe_index.unwrap();
            self.repair(index);
        }
        Ok(())
    }

    fn sample_rest(&mut self) -> Result<(Vec<usize>, Vec<Float>),Error>{
        // Now we have seeded the sampling distribution, we sample the actual coreset:
        let mut coreset_indices: Vec<usize> = Vec::with_capacity(self.coreset_size);
        let mut coreset_weights: Vec<Float> = Vec::with_capacity(self.coreset_size);

        let cost = self.sampling_tree.storage.first().unwrap().contribution();

        for _ in 0..self.coreset_size{
            let (index,prob) = self.sampling_tree.sample_smoothed(
                &mut self.rng, cost, self.coreset_star_weight)?;
            let weight = self.sampling_tree.storage.get(self.sampling_tree.get_shifted_node_index(index).unwrap().0).unwrap().weight();
            coreset_indices.push(index.0);
            coreset_weights.push(weight.0/(prob*self.coreset_size as Float));
        }

        // sort the indices and weights in ascending order of indices
        let mut combined: Vec<_> = coreset_indices.iter_mut().zip(coreset_weights.iter_mut()).collect();
        combined.sort_by(|a,b| a.0.cmp(&b.0));


        Ok((coreset_indices,coreset_weights))
    }

    pub fn sample(&mut self) -> Result<(Vec<usize>, Vec<Float>,bool),Error> {
        self.sample_first_point();
        self.sample_next_k()?;
        self.sample_rest().map(|(a,b)| (a,b,self.numerical_warning))
    }

}

#[cfg(test)]
mod tests{
    use faer::Col;
    use faer::sparse::SparseRowMat;
    use rand::SeedableRng;
    use super::super::unstable;

    use super::*;

    #[test]
    fn basic(){
        // create a sparse kernel matrix corresponding to the following dense matrix:
        // 1.0      0       0.4     0.8
        // 0        1.0     0.5     0
        // 0.4      0.5     1.0     0
        // 0.8      0       0       1.0
        let adj_matrix = SparseRowMat::<usize, Float>::try_new_from_triplets(
            4,
            4,
            &[
                (0, 0, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
                (2, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (3, 0, 1.0),
                (3, 3, 1.0),
            ],
        ).unwrap();

        let dense_mat = adj_matrix.to_dense();
        let degree_vector = Col::from_fn(4,|i| dense_mat.row(i).iter().sum::<Float>());


        // display the kernel matrix nicely
        println!("{:?}", &adj_matrix.to_dense());


        let mut coreset_sampler = DefaultCoresetSampler::<unstable::TreeNode>::new(
            adj_matrix.as_ref(),
            degree_vector.as_ref(),
            2,
            3,
            Some(0.0),
            rand::rngs::StdRng::from_rng(&mut rand::rng()),
        );
        println!("{:?}", &coreset_sampler);


        let (coreset_indices, coreset_weights,_) = coreset_sampler.sample().unwrap();
        println!("{:?}", coreset_indices);
        println!("{:?}", coreset_weights);
        // panic!("stop");
    }
}
