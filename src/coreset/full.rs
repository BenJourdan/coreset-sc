use crate::coreset::{common::*, sampling_tree::SamplingTree};

use rand::rngs::StdRng;

use faer::sparse::SparseRowMatRef;
use faer::MatRef;
use rand::Rng;
// use faer::prelude::*;


#[derive(Debug)]
pub struct DefaultCoresetSampler<'a,T>{
    sampling_tree: SamplingTree<T>,
    num_clusters: usize,
    coreset_star_weight: Weight,
    coreset_size: usize,
    rng: StdRng,
    number_of_data_points: usize,
    adj_matrix: SparseRowMatRef<'a, usize, Float>,
    degree_vector: MatRef::<'a, Float>,
    x_star_index: Index,
}




/// Assumes a undirected graph where self loops are all 1
impl <'a,T> DefaultCoresetSampler<'a,T>
    where T: Node
{
    pub fn new(adj_matrix: SparseRowMatRef<'a,usize, Float>, degree_vector: MatRef<'a,Float>, num_clusters: usize, coreset_size: usize, rng: StdRng) -> Self{

        let n = adj_matrix.nrows();
        assert_eq!(n, adj_matrix.ncols());
        assert_eq!(n, degree_vector.ncols());

        let mut sampling_tree = SamplingTree::<T>::new();
        // Find the node with the highest degree and hence lowest self affinity
        let x_star = degree_vector.col(0).iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let min_self_affinity = 1.0f64/(degree_vector[(0,x_star)]*degree_vector[(0,x_star)]);
        // Populate the sampling  tree
        sampling_tree.insert_from_iterator((0..n).map(|i|{
            let d = degree_vector[(0,i)];
            let d_2 = d*d;
            (d.into(), (1.0/d_2).into())
            }),
             min_self_affinity.into()
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
            x_star_index: Index(x_star),
        }
    }


    fn repair(&mut self, point_added: Index){
        // We implicitly add the point to the init set and update it's neighbours:
        let point_added_degree: Float = self.degree_vector[(0,point_added.0)];
        let point_added_weight: Weight = point_added_degree.into();
        self.coreset_star_weight += point_added_weight;

        let point_added_self_affinity: SelfAffinity =  (1.0f64/(point_added_degree*point_added_degree)).into();
        // set the contribution of the added point to zero:
        self.sampling_tree.update_delta(point_added, Delta(0.0)).unwrap();
        // Now we update the neighbours of the added point:
        self.adj_matrix.col_indices_of_row(point_added.0).map(Index).for_each(|neighbour_index|{
            // If the neighbour is the added point, skip it
            if neighbour_index == self.x_star_index{
                return;
            }
            // compute the distance^2 between the added point and the neighbour:
            let neighbour_degree: Float = self.degree_vector[(0,neighbour_index.0)];
            let neighbour_self_affinity: SelfAffinity = (1.0f64/(neighbour_degree*neighbour_degree)).into();
            let cross_term: CoresetCrossTerm = (1.0f64/(point_added_degree*neighbour_degree)).into();
            let distance2 =  point_added_self_affinity.0 + neighbour_self_affinity.0 - 2.0*cross_term.0;
            // update the delta of the neighbour:
            self.sampling_tree.update_delta(neighbour_index, Delta(distance2)).unwrap();
        })
    }

    fn sample_first_point(&mut self){
        self.repair(self.x_star_index);
    }

    fn sample_next_k(&mut self) -> Result<(),Error>{
        // Now we run k-means++ to sample the next k points (total k+1 points)
        // first we uniformly sample the first point and repair:
        let uniform_sampled_index = Index(self.rng.gen_range(0..self.number_of_data_points));
        self.repair(uniform_sampled_index);
        // Now we sample the next k-1 points and repair:
        for _ in 0..self.num_clusters-1{
            let index = self.sampling_tree.sample(&mut self.rng)?;
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

    pub fn sample(&mut self) -> Result<(Vec<usize>, Vec<Float>),Error> {
        println!("starting");
        self.sample_first_point();
        println!("sampled first point");
        self.sample_next_k()?;
        println!("sampled next k");
        self.sample_rest()
    }

}

#[cfg(test)]
mod tests{
    use faer::{sparse::SparseRowMat, Mat};
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
        let degree_vector = Mat::from_fn(1,4,|_,i| dense_mat.row(i).iter().sum::<Float>());


        // display the kernel matrix nicely
        println!("{:?}", &adj_matrix.to_dense());

        let mut coreset_sampler = DefaultCoresetSampler::<unstable::TreeNode>::new(
            adj_matrix.as_ref(),
            degree_vector.as_ref(),
            2,
            3,
            rand::SeedableRng::from_rng(rand::rngs::StdRng::from_entropy()).unwrap()
        );
        println!("{:?}", &coreset_sampler);


        let (coreset_indices, coreset_weights) = coreset_sampler.sample().unwrap();
        println!("{:?}", coreset_indices);
        println!("{:?}", coreset_weights);
        panic!("stop");
    }
}
