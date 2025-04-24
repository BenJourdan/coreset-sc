#![allow(clippy::deprecated)]

use rand::seq::IteratorRandom;
use sampling_tree::SimpleSamplingTree;
use rand_distr::{Binomial,Distribution};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use rand_old::{self, SeedableRng};
// use aligned_vec::{AVec,avec};
use rayon::prelude::*;


pub fn largest_triangleq_num_below(x: usize)-> usize{
    //  Floor((-1 + sqrt(1+8x))/2)
    ((((1+8*x) as f64).sqrt()-1.0)/2.0).floor() as usize
}

pub fn triangle_number(n: usize)-> usize{
    (n*(n+1))/2
}

pub fn extract_edge_indices(idx: usize) -> (usize,usize){
    let tria_num_largest = largest_triangleq_num_below(idx);
    let col = tria_num_largest + 1;
    let row = idx - triangle_number(tria_num_largest);
    (row,col)
}


pub fn shift_edge_indices_same_cluster((row,col): (usize,usize),cluster_index:usize,cluster_size:usize) -> (usize,usize){
    (row + cluster_index*cluster_size, col + cluster_index*cluster_size)
}


pub fn gen_sbm_with_self_loops(
    nodes_per_cluster: usize,
    num_clusters: usize,
    p: f64,
    q: f64,
) -> (SparseRowMat<usize,f64>,Vec<usize>){

    let n = nodes_per_cluster;
    let k = num_clusters;


    // First we compute how many intra and inter cluster edges we will sample:
    let mut rng = rand::thread_rng();
    // intra:
    let intra_bin = Binomial::new(((n*(n-1))/2) as u64,p).unwrap();
    let num_intra_cluster_edges = (0..k).map(|_| intra_bin.sample(&mut rng) as usize).collect::<Vec<usize>>();
    let total_intra_cluster_edges = num_intra_cluster_edges.iter().sum::<usize>();

    // inter:
    let inter_bin = Binomial::new(((k*n*(k-1)*n)/2) as u64,q).unwrap();
    let num_inter_cluster_edges =inter_bin.sample(&mut rand::thread_rng()) as usize;
    // double count the inter and intra edges since we want (u,v) and (v,u). Then add self loops (counted once)
    let total_edges = 2*total_intra_cluster_edges + 2*num_inter_cluster_edges + n*k;
    // Allocate space for the edge list using the average number of edges per node


    // TODO: Do some fancy maths to properly compute the optimal preallocated size of each edge list:
    // Trade off between memory and time. Get it wrong and reallocations will kill multi-threading
    // as every thread will be fighting to use the system allocator.
    let overhead = 1.1f64;

    let mut edge_list = (0..n*k)
    // .into_par_iter()
        .map(|_|{
            Vec::<usize>::with_capacity(((total_edges/(n*k))as f64 *overhead) as usize)
        }).collect::<Vec<Vec<usize>>>();


    // Now we can start sampling edges

    // First sample inter-cluster edges:
    // Each cluster has a contribution equal to the total number of inter-cluster edges it could have:
    // This is n* (k-1)*n (double counts by 2)
    // n pointer in the cluster, potentially connected to the (k-1) other clusters of size n.
    let mut cluster_sampling_tree = SimpleSamplingTree::<usize>::from_iterable(
        (0..k).map(|_| n*n*(k-1))
    ).unwrap();

    // Now we create another sampling tree for each cluster,
    // where each vertex has a leaf with initial contribution equal to the total number of inter-cluster edges it could have:
    // This is n*(k-1)
    // 1 point in the cluster, potentially connected to the (k-1) other clusters of size n.
    let mut vertex_sampling_trees = (0..k)
        .map(|_| SimpleSamplingTree::<usize>::from_iterable((0..n).map(|_| n*(k-1))).unwrap()
    ).collect::<Vec<SimpleSamplingTree<usize>>>();


    // Now we sample the number of inter-cluster edges across the whole graph at once:
    let mut rng = rand_old::rngs::StdRng::from_entropy();
    (0..num_inter_cluster_edges).for_each(|_|{
        // sample first cluster index, store current contribution and temporarily set it to 0:
        let cluster_i = cluster_sampling_tree.sample(&mut rng).unwrap();
        let cluster_i_temp_contrib = cluster_sampling_tree.get_contribution(cluster_i).unwrap();
        cluster_sampling_tree.update(cluster_i, 0).unwrap();

        // now sample the first vertex index and get its contribution:
        let cluster_i_vertex = vertex_sampling_trees[cluster_i.0].sample(&mut rng).unwrap();
        let cluster_i_vertex_contribution = vertex_sampling_trees[cluster_i.0].get_contribution(cluster_i_vertex).unwrap();

        // Now get the existing neighbours of i, store their contributions and temporarily set them to 0
        // Reflect this in their cluster contributions as well:
        let vertex_i_neighbours = edge_list[cluster_i.0*n + cluster_i_vertex.0].clone();
        let vertex_i_neighbour_contributions_and_indices = vertex_i_neighbours.iter().map(|&v|{
            let cluster_neighbour = v/n;
            let vertex_neighbour = v%n;
            let contribution = vertex_sampling_trees[cluster_neighbour].get_contribution(vertex_neighbour.into()).unwrap();
            // temporarily set the contribution to 0:
            vertex_sampling_trees[cluster_neighbour].update(vertex_neighbour.into(),0).unwrap();
            // reflect this in the cluster contribution:
            cluster_sampling_tree.update(
                cluster_neighbour.into(),
                cluster_sampling_tree.get_contribution(cluster_neighbour.into()).unwrap()-contribution).unwrap();

            (contribution,v)
        }).collect::<Vec<(usize,usize)>>();

        // Sample second cluster index, store current conribution.
        let cluster_j = cluster_sampling_tree.sample(&mut rng).unwrap();
        let cluster_j_temp_contrib = cluster_sampling_tree.get_contribution(cluster_j).unwrap();
        // Now sample the vertex indices and get their contributions:
        let cluster_j_vertex = vertex_sampling_trees[cluster_j.0].sample(&mut rng).unwrap();
        let cluster_j_vertex_contribution = vertex_sampling_trees[cluster_j.0].get_contribution(cluster_j_vertex).unwrap();
        // Now update the vertex tree contributions:
        vertex_sampling_trees[cluster_i.0].update(cluster_i_vertex, cluster_i_vertex_contribution-1).unwrap();
        vertex_sampling_trees[cluster_j.0].update(cluster_j_vertex, cluster_j_vertex_contribution-1).unwrap();
        // Update the cluster tree contributions:
        cluster_sampling_tree.update(cluster_i, cluster_i_temp_contrib-1).unwrap();
        cluster_sampling_tree.update(cluster_j, cluster_j_temp_contrib-1).unwrap();
        // Now we have the cluster and vertex indices, we can add the edge to the set by shifting the indices:
        let (u,v) = (cluster_i.0*n + cluster_i_vertex.0, cluster_j.0*n + cluster_j_vertex.0);
        edge_list[u].push(v);
        edge_list[v].push(u);

        // Now we need to repair the contributions of the neighbours of i:
        vertex_i_neighbour_contributions_and_indices.into_iter().for_each(|(contribution,v)|{
            let cluster_neighbour = v/n;
            let vertex_neighbour = v%n;
            // repair the contribution:
            vertex_sampling_trees[cluster_neighbour].update(vertex_neighbour.into(),contribution).unwrap();
            // repair the cluster contribution:
            cluster_sampling_tree.update(
                cluster_neighbour.into(),
                cluster_sampling_tree.get_contribution(cluster_neighbour.into()).unwrap()+contribution).unwrap();
        });
    });



    edge_list.par_iter_mut().enumerate().for_each(|(i, edges)|{
        edges.push(i);
    });

    // Now we move onto sampling intra-cluster edges:


    // First we sample the indices of the intra-cluster edges:
    let indices_per_cluster = (0..k).into_par_iter().map(|cluster_i|{
        (0..((n*(n-1))/2)).choose_multiple(&mut rand::rng(), num_intra_cluster_edges[cluster_i])
    });
    // Now we shift the indices to the correct cluster and convert them to edge pairs.
    let shifted_indices_per_cluster = indices_per_cluster.into_par_iter().enumerate().map(
        |(cluster_i,indices)|{
            indices.into_iter().map(|i|{
                let (u,v) = extract_edge_indices(i);
                let (u,v) = shift_edge_indices_same_cluster((u,v),cluster_i,n);
                (u,v)
            }).collect::<Vec<(usize,usize)>>()
    });
    edge_list.par_chunks_exact_mut(n)
        .zip(shifted_indices_per_cluster)
        .for_each(
        |(edge_list_local,indices)|{
            indices.iter().for_each(|&(u,v)|{
                edge_list_local[u%n].push(v);
                edge_list_local[v%n].push(u);
            });
        }
    );


    // build the indices and indptr quickly from the edge list:
    let mut indptr = vec![0;n*k+1];
    let mut indices = vec![0;total_edges];



    // populate indptr initially with the number of edges in each row. We will cumsum this later.
    indptr[1..].par_iter_mut().enumerate().for_each(|(i,p)|{
        *p = edge_list[i].len();
    });
    let nnz = indptr[1..].to_vec();

    // Now we can split the indices array into n*k mutable slices which we will pass to rayon
    // to populate with the edges in parallel.
    let mut splits = Vec::with_capacity(n*k);
    let mut remaining: &mut[usize] = &mut indices;

    for x in &indptr[1..]{
        let (a,b) = remaining.split_at_mut(*x);
        splits.push(a);
        remaining = b;
    }

    // Now copy the (sorted) edges into the indices array:
    edge_list.par_iter_mut().zip(splits).for_each(|(edges,indices)|{
        edges.sort_unstable(); // we don't care about sort stability
        indices.copy_from_slice(edges);
    });

    // now finish cumsum on indptr:
    for i in 1..indptr.len(){
        indptr[i] += indptr[i-1];
    }

    // Now we have the indices and indptr, we can build the adjacency matrix:
        let structure = SymbolicSparseRowMat::new_checked(
            n*k,
            n*k,
            indptr,
            Some(nnz),
            indices,
        );

    let mut data = vec![0.0f64;total_edges];
    data.par_iter_mut().for_each(|x| *x = 1.0);
    let sparse_mat = SparseRowMat::<usize,f64>::new(structure,data);


    let labels = (0..n*k).map(|i| i/n).collect::<Vec<usize>>();

    (sparse_mat,labels)
}



#[cfg(test)]
mod tests{



    use super::*;

    #[test]
    fn test_small_sbm(){
        let adj_mat = gen_sbm_with_self_loops(5,2,0.5,0.1);

        println!("{:?}",adj_mat);

    }
    #[test]
    fn test_sbm(){
        let n = 1000;
        let k = 100;
        let p = 0.5;
        let q = (1.0/(n as f64))/(k as f64);
        let t0 = std::time::Instant::now();
        let (adj_mat,_labels) = gen_sbm_with_self_loops(n,k,p,q);
        println!("total time: {:?}",t0.elapsed());
        println!("{:?} shape",adj_mat.shape());
        // panic!();
    }




    #[test]
    fn splat(){

        let adjacency: Vec<Box<[usize]>> = vec![
            vec![1,2,3].into_boxed_slice(),
            vec![4,6].into_boxed_slice(),
            vec![9].into_boxed_slice(),
        ];

        println!("{:?}",adjacency);
    }
}
