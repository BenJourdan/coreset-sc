use std::fmt::{Debug, Formatter};
use crate::coreset::common::*;
use std::mem::MaybeUninit;


#[allow(dead_code)]
pub struct SamplingTree<T>
{
    // Leaves are stored at the end of the storage vector. The root is at index 0.
    pub storage: Vec<T>,
}


impl <T> Debug for SamplingTree<T>
where T: Debug
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.storage.first(){
            None =>{
                f.debug_struct("SamplingTree")
                .field("root", &Option::<()>::None)
                .finish()
            },
            Some(root) =>{
                let tree_node = root;
                f.debug_struct("SamplingTree")
                .field("root", tree_node)
                .finish()
            }
        }
    }
}

impl <T> SamplingTree<T>
where T: Node
{
    pub fn new() -> Self{
        SamplingTree{
            storage: Vec::new()}
    }

    pub fn get_shifted_node_index(&self, node_index: Index) -> Result<ShiftedIndex,Error>{
        let storage_size = self.storage.len();
        let num_leaves = (storage_size + 1).div_ceil(2);
        let shift = num_leaves - 1;
        let shifted_node_index = node_index.0 + shift;
        match shifted_node_index < storage_size{
            false => Err(Error::NodeNotFound(node_index)),
            true => Ok(ShiftedIndex(shifted_node_index))
        }
    }
    pub fn get_node_index(&self, shifted_node_index: ShiftedIndex) -> Result<Index, Error>{
        let storage_size = self.storage.len();
        let num_leaves = (storage_size + 1).div_ceil(2);
        let shift = num_leaves - 1;
        let node_index = shifted_node_index.0 - shift;
        match node_index < num_leaves{
            false => Err(Error::NodeNotFound(Index(node_index))),
            true => Ok(Index(node_index))
        }
    }

    pub fn rebuild_from_leaves(&mut self){
        let num_nodes = self.storage.len();
        let num_leaves = (num_nodes + 1).div_ceil(2);
        // Leaves are stored in the last num_leaves elements of the storage vector.
        // We proceed by updating the first num_leaves -1 elements in reverse order.
        (0..num_leaves-1).rev().for_each(|i|{
            let (left_child_idx,right_child_idx) = (2*i+1,2*i+2);
            let left_child_ref = &self.storage[left_child_idx];
            let right_child_ref = &self.storage[right_child_idx];
            self.storage[i] = T::from_children(left_child_ref, right_child_ref);
        });
    }

    pub fn insert_from_iterator<I>(&mut self, mut iterator: I, min_self_affinity:SelfAffinity) ->std::ops::Range<ShiftedIndex>
    where I: Iterator<Item = (Weight,SelfAffinity)> + std::iter::ExactSizeIterator
    {
        // Given an iterator of leaf node data, we create a balanced binary tree in a bottom up fashion.
        // This is to help with cache locality and branch prediction.

        // We will fill up an array with the nodes. Leafs will be stored at the end of the array.

        // The total number of nodes in a binary tree with num_leaves is 2*num_leaves - 1.
        // given an index i, the left child is 2*i + 1 and the right child is 2*i + 2.
        // The parent of a node at index i is (i-1)/2.
        if iterator.len() == 0{
            self.storage = Vec::new();
            return ShiftedIndex(0)..ShiftedIndex(0);
        }
        let num_leaves = iterator.len();
        let tree_len = 2*num_leaves - 1;
        // The leaves are stored at the end of the array. We will return a range of indices that correspond to the leaves.
        let mut tree: Vec<MaybeUninit<T>> = Vec::with_capacity(tree_len);
        unsafe{
            tree.set_len(tree_len);
        }
        // println!("{:?}",(num_leaves..tree_len).collect::<Vec<usize>>());
        // We will store the leaves at the end of the array.
        tree[num_leaves-1..].iter_mut().for_each(| node|{
            let node = node.as_mut_ptr();
            unsafe{
                let (weight, self_affinity) = iterator.next().unwrap();
                node.write(T::new(weight, self_affinity, min_self_affinity));
            }
        });
        // println!("{:?}", (0..num_leaves).collect::<Vec<usize>>());
        // Now we proceed backwards from num_leaves-1 to 0, constructing the interal nodes.
        (0..num_leaves-1).rev().for_each(|i|{
            let (left_child_idx,right_child_idx) = (2*i+1,2*i+2);
            unsafe{
                let node = tree[i].as_mut_ptr();
                let left_child_ref = tree[left_child_idx].as_ptr().as_ref().unwrap();
                let right_child_ref = tree[right_child_idx].as_ptr().as_ref().unwrap();
                node.write(T::from_children(left_child_ref, right_child_ref));
            }
        });
        // The root is at index 0. Now we transmute the tree to a Vec<T> and store it in the storage field.
        let tree = unsafe{
            std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(tree)
        };
        self.storage = tree;
        ShiftedIndex(num_leaves-1)..ShiftedIndex(tree_len)
    }

    pub fn sample(&self, rng: &mut impl rand::Rng) -> Result<Index,Error>{
        self._sample(rng, false, Contribution(0.0), Weight(0.0)).map(|(index,_)| index)
    }

    pub fn sample_smoothed(&self, rng: &mut impl rand::Rng, cost: Contribution, coreset_star_weight: Weight) -> Result<(Index,Float),Error>{
        self._sample(rng, true, cost, coreset_star_weight)
    }

    pub fn _sample(&self, rng: &mut impl rand::Rng, smoothed:bool, cost:Contribution, coreset_star_weight:Weight) -> Result<(Index,Float),Error>{
        let shifted_idx_res = T::_sample(&self.storage, rng, smoothed, cost, coreset_star_weight);
        shifted_idx_res.map(|(shifted_idx,prob)|{
            let idx = self.get_node_index(shifted_idx).unwrap();
            (idx, prob)
        })
    }

    // pub fn compute_sampling_probability(&self, idx: Index) -> Float{
    //     self._computed_sampling_probability(false, idx, Contribution(0.0), Weight(0.0)).unwrap()
    // }

    // pub fn compute_smoothed_sampling_probability(&self, idx: Index, cost: Contribution, coreset_star_weight: Weight) -> Float{
    //     self._computed_sampling_probability(true, idx, cost, coreset_star_weight).unwrap()
    // }

    pub fn _computed_sampling_probability(&self, smoothed: bool, idx: Index, cost: Contribution, coreset_star_weight: Weight) -> Result<Float,Error>{
        let shifted_idx = self.get_shifted_node_index(idx).unwrap();
        T::_computed_sampling_probability(&self.storage, smoothed, shifted_idx, cost, coreset_star_weight)
    }

    pub fn update_delta(&mut self, idx: Index, new_delta:Delta) -> Result<(),Error>{
        let shifted_idx = self.get_shifted_node_index(idx)?;
        assert!(new_delta.0 >= 0.0, "Delta: {} is negative", new_delta.0);
        T::update_delta(&mut self.storage, shifted_idx, new_delta);
        Ok(())
    }

}
