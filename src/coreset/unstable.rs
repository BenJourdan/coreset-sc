use std::fmt::{Formatter, Debug};
use crate::coreset::common::*;
// MARK: - Node structs
#[derive(Debug)]
#[allow(dead_code)]
pub struct LeafNode{
    pub weight: Weight,
    pub delta: Delta,
}
impl LeafNode{
    pub fn contribution(&self) -> Contribution{
        (self.weight.0*self.delta.0).into()
    }

    // pub fn weight(&self) -> Weight{
    //     self.weight
    // }

}
pub struct InternalNode{
    pub contribution: Contribution,
    pub weight: Weight,
}

impl InternalNode{

    #[allow(dead_code)]
    pub fn contribution(&self) -> Contribution{
        self.contribution
    }

    #[allow(dead_code)]
    pub fn smoothed_contribution(&self, cost: Contribution, coreset_star_weight: Weight) -> SmoothedContribution{
        (self.contribution.0/cost.0 + self.weight.0/coreset_star_weight.0).into()
    }
}

impl Debug for InternalNode{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalNode")
            .field("contribution", &self.contribution)
            .field("weight", &self.weight)
            .finish()
    }

}

#[derive(Debug)]
#[allow(dead_code)]
pub enum TreeNode{
    Leaf(LeafNode),
    Internal(InternalNode)
}

#[allow(dead_code)]
impl TreeNode{
    pub fn contribution(&self) -> Contribution{
        match self{
            TreeNode::Leaf(leaf_node) => leaf_node.contribution(),
            TreeNode::Internal(internal_node) => internal_node.contribution()
        }
    }
    pub fn smoothed_contribution(&self, cost: Contribution, coreset_star_weight: Weight) -> SmoothedContribution{
        match self{
            TreeNode::Leaf(LeafNode{weight, ..}) =>{
                let contribution = self.contribution();
                let smoothed_contribution = contribution.0/cost.0 + weight.0/coreset_star_weight.0;
                smoothed_contribution.into()
            },
            TreeNode::Internal(internal_node) =>{
                internal_node.smoothed_contribution(cost, coreset_star_weight)
            }
        }
    }

    pub fn weight(&self) -> Weight{
        match self{
            TreeNode::Leaf(LeafNode { weight, ..}) => *weight,
            TreeNode::Internal(InternalNode{weight, ..}) => *weight
        }
    }

    pub fn delta(&self) -> Delta{
        match self{
            TreeNode::Leaf(LeafNode{delta, ..}) => *delta,
            TreeNode::Internal(_) => panic!("Internal nodes don't have deltas")
        }
    }
}


impl Node for TreeNode{
    fn contribution(&self) -> Contribution {
        self.contribution()
    }

    fn smoothed_contribution(&self, cost: Contribution, coreset_star_weight: Weight) -> SmoothedContribution {
        self.smoothed_contribution(cost, coreset_star_weight)
    }

    fn weight(&self) -> Weight {
        self.weight()
    }

    fn new(weight: Weight, self_affinity: SelfAffinity, min_self_affinity: SelfAffinity) -> Self {
        TreeNode::Leaf(LeafNode{
            weight,
            delta: (self_affinity.0 + min_self_affinity.0).into()
        })
    }

    fn update_delta(storage: &mut Vec<Self>, shifted_index: ShiftedIndex, new_delta: Delta) {

        let mut shifted_node_index = shifted_index;

        let leaf = storage.get_mut(shifted_node_index.0).unwrap();

        match leaf{
            TreeNode::Internal(_) => panic!("should have started at a leaf node"),
            TreeNode::Leaf(leaf_node) => {
                if leaf_node.delta.0 <= new_delta.0{
                    return;
                }

                let delta_diff = leaf_node.delta.0 - new_delta.0;
                let contribution_diff = delta_diff*leaf_node.weight.0;
                leaf_node.delta = new_delta;

                let mut parent = TreeNode::parent(shifted_node_index);
                while let Ok(parent_idx) = parent{
                    let parent_node = storage.get_mut(parent_idx.0).unwrap();
                    parent_node.update_contribution(contribution_diff.into());
                    shifted_node_index = parent_idx;
                    parent = TreeNode::parent(shifted_node_index);

                }
            }
        }
    }

    fn update_contribution(&mut self, contribution_diff: Contribution) {
        match self{
            TreeNode::Leaf(_) => panic!("Leaf nodes don't have contributions"),
            TreeNode::Internal(internal_node) => {
                internal_node.contribution.0 -= contribution_diff.0;
            }
        }
    }

    fn from_children(left: &Self, right: &Self) -> Self {
        let contribution = (left.contribution().0 + right.contribution().0).into();
        let weight = left.weight() + right.weight();
        TreeNode::Internal(InternalNode{
            contribution,
            weight
        })
    }

    fn _sample(storage: &[Self], rng: &mut impl rand::Rng, smoothed:bool, cost:Contribution, coreset_star_weight:Weight) -> Result<(ShiftedIndex,Float),Error> {
        if storage.is_empty(){
            return Err(Error::EmptyTree)
        }
        let mut shifted_node_index: ShiftedIndex = ShiftedIndex(0);
        let mut prob: Float = 1.0;

        let mut node = storage.get(shifted_node_index.0).unwrap();
        match smoothed{
            true =>{
                while let TreeNode::Internal(_) = node{
                    let left_node_idx = TreeNode::left_child(shifted_node_index);
                    let right_node_idx = TreeNode::right_child(shifted_node_index);
                    let left_node = storage.get(left_node_idx.0).unwrap();
                    let right_node = storage.get(right_node_idx.0).unwrap();
                    let left_node_smoothed_contribution = left_node.smoothed_contribution(cost, coreset_star_weight);
                    let right_node_smoothed_contribution = right_node.smoothed_contribution(cost, coreset_star_weight);

                    let total_smoothed_contribution = left_node_smoothed_contribution.0 + right_node_smoothed_contribution.0;
                    if total_smoothed_contribution <= 0.0{
                        return Err(Error::NumericalError)
                    }

                    let sample = rng.random_range(0.0..total_smoothed_contribution);
                    node = match sample <= left_node_smoothed_contribution.0{
                        true => {
                            prob *= left_node_smoothed_contribution.0/total_smoothed_contribution;
                            shifted_node_index = left_node_idx;
                            left_node
                        },
                        false =>{
                            prob *= right_node_smoothed_contribution.0/total_smoothed_contribution;
                            shifted_node_index = right_node_idx;
                            right_node
                        }
                    }
                }
                Ok((shifted_node_index, prob))
            },
            false =>{
                while let TreeNode::Internal(_) = node{
                    let left_node_idx = TreeNode::left_child(shifted_node_index);
                    let right_node_idx = TreeNode::right_child(shifted_node_index);
                    let left_node = storage.get(left_node_idx.0).unwrap();
                    let right_node = storage.get(right_node_idx.0).unwrap();
                    let left_node_contribution = left_node.contribution();
                    let right_node_contribution = right_node.contribution();
                    let total_smoothed_contribution = left_node_contribution.0 + right_node_contribution.0;
                    if total_smoothed_contribution <= 0.0{
                        return Err(Error::NumericalError)
                    }

                    let sample = rng.random_range(0.0..total_smoothed_contribution);
                    node = match sample <= left_node_contribution.0{
                        true => {
                            prob *= left_node_contribution.0/total_smoothed_contribution;
                            shifted_node_index = left_node_idx;
                            left_node
                        },
                        false =>{
                            prob *= right_node_contribution.0/total_smoothed_contribution;
                            shifted_node_index = right_node_idx;
                            right_node
                        }
                    }
                }
                Ok((shifted_node_index, prob))
            }
        }

    }

    fn _computed_sampling_probability(storage: &[Self], smoothed: bool, shifted_idx: ShiftedIndex, cost: Contribution, coreset_star_weight: Weight) ->Result<Float,Error> {

        let mut shifted_node_index = shifted_idx;
        let mut prob: Float = 1.0;

        match smoothed{
            true =>{
                while let Ok(parent_idx) = TreeNode::parent(shifted_node_index){
                    let parent_node = storage.get(parent_idx.0).unwrap();
                    let parent_contribution = parent_node.smoothed_contribution(cost, coreset_star_weight);
                    let child_contribution = storage.get(shifted_node_index.0).unwrap().smoothed_contribution(cost, coreset_star_weight);
                    prob *= child_contribution.0/parent_contribution.0;
                    shifted_node_index = parent_idx;
                }
            },
            false =>{
                while let Ok(parent_idx) = TreeNode::parent(shifted_node_index){
                    let parent_node = storage.get(parent_idx.0).unwrap();
                    let parent_contribution = parent_node.contribution();
                    let child_contribution = storage.get(shifted_node_index.0).unwrap().contribution();
                    prob *= child_contribution.0/parent_contribution.0;
                    shifted_node_index = parent_idx;
                }
            }
        }
        Ok(prob)
    }

}
