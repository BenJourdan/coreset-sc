use std::ops::{Add, AddAssign};
use std::fmt::Debug;


pub type Float = f64;


#[allow(clippy::enum_variant_names)]
#[allow(dead_code)]
#[derive(Debug)]
pub enum Error{
    NodeNotFound(Index),
    NodeHasNoParent(ShiftedIndex),
    NodeAlreadyInserted(Index),
    EmptyTree,
    NumericalError
}

impl std::fmt::Display for Error{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result{
        match self{
            Error::NodeNotFound(index) => write!(f, "Node with index {} not found", index.0),
            Error::NodeHasNoParent(index) => write!(f, "Node with shifted_index {} has no parent", index.0),
            Error::NodeAlreadyInserted(index) => write!(f, "Node with index {} is already inserted", index.0),
            Error::EmptyTree => write!(f, "Tree is empty"),
            Error::NumericalError => write!(f, "Numerical error"),
        }
    }
}


impl std::error::Error for Error {}
// MARK: -Newtypes

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct Index(pub usize);

impl From<usize> for Index{
    fn from(index: usize) -> Self{
        Index(index)
    }
}

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
pub struct ShiftedIndex(pub usize);

impl From<usize> for ShiftedIndex{
    fn from(index: usize) -> Self{
        ShiftedIndex(index)
    }
}


#[derive(Copy, Clone, Debug)]
pub struct Weight(pub Float);

impl From<Float> for Weight{
    fn from(weight: Float) -> Self{
        Weight(weight)
    }
}



impl Add for Weight{
    type Output = Weight;
    fn add(self, other: Weight) -> Weight{
        Weight(self.0 + other.0)
    }
}

impl AddAssign for Weight{
    fn add_assign(&mut self, other: Weight){
        self.0 += other.0;
    }
}


#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct CoresetCrossTerm(pub Float);
impl From<Float> for CoresetCrossTerm{
    fn from(coreset_cross_term: Float) -> Self{
        CoresetCrossTerm(coreset_cross_term)
    }
}



#[derive(Copy, Clone, Debug)]
pub struct SelfAffinity(pub Float);

impl From<Float> for SelfAffinity{
    fn from(self_affinity: Float) -> Self{
        SelfAffinity(self_affinity)
    }
}



#[derive(Copy, Clone, Debug)]

pub struct Delta(pub Float);
impl From<Float> for Delta{
    fn from(delta: Float) -> Self{
        Delta(delta)
    }
}


#[derive(Copy, Clone, Debug)]
pub struct Contribution(pub Float);
impl From<Float> for Contribution{
    fn from(contribution: Float) -> Self{
        Contribution(contribution)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmoothedContribution(pub Float);
impl From<Float> for SmoothedContribution{
    fn from(contribution: Float) -> Self{
        SmoothedContribution(contribution)
    }
}


// MARK: Datapoint struct

#[derive(Debug)]
pub struct Datapoint{
    pub weight: Weight,
    pub self_affinity: SelfAffinity,
}

impl Datapoint{
    // pub fn new(weight: Weight, self_affinity: SelfAffinity) -> Self{
    //     Datapoint{
    //         weight,
    //         self_affinity
    //     }
    // }
}

impl Datapoint{
    #[allow(dead_code)]
    pub fn contribution(&self, smallest_coreset_self_affinity: Float) -> Contribution{
        Contribution(self.weight.0*(self.self_affinity.0 + smallest_coreset_self_affinity))
    }

    #[allow(dead_code)]
    pub fn smoothed_contribution(&self,smallest_coreset_self_affinity: Float, cost: Float, coreset_star_weight: Weight) -> Float{
        self.contribution(smallest_coreset_self_affinity).0/cost + self.weight.0/coreset_star_weight.0
    }
}


#[derive(Debug, Copy, Clone)]
pub struct DatapointWithCoresetCrossTerm{
    pub weight: Weight,
    pub self_affinity: SelfAffinity,
    pub coreset_cross_term: CoresetCrossTerm
}

impl DatapointWithCoresetCrossTerm{
    #[allow(dead_code)]
    pub fn contribution(&self) -> Float{
        self.weight.0*(self.self_affinity.0 + self.coreset_cross_term.0)
    }

    #[allow(dead_code)]
    pub fn smoothed_contribution(&self, cost: Float, coreset_start_weight: Weight) -> Float{
        self.contribution()/cost + self.weight.0/coreset_start_weight.0
    }
}

// MARK: Node trait
pub trait Node
where Self: Sized
{
    fn left_child(shifted_index: ShiftedIndex) -> ShiftedIndex{
        ShiftedIndex(2*shifted_index.0 + 1)
    }
    fn right_child(shifted_index: ShiftedIndex) -> ShiftedIndex{
        ShiftedIndex(2*shifted_index.0 + 2)
    }

    fn parent(shifted_index: ShiftedIndex) -> Result<ShiftedIndex,Error>{
        if shifted_index.0 == 0{
            return Err(Error::NodeHasNoParent(shifted_index));
        }
        Ok(ShiftedIndex((shifted_index.0 - 1)/2))
    }

    fn contribution(&self) -> Contribution;

    #[allow(dead_code)]
    fn smoothed_contribution(&self, cost: Contribution, coreset_star_weight: Weight) -> SmoothedContribution;
    fn weight(&self) -> Weight;
    fn new(weight: Weight, self_affinity: SelfAffinity, min_self_affinity: SelfAffinity) -> Self;
    fn update_delta(storage: &mut Vec<Self>, shifted_index: ShiftedIndex, new_delta: Delta);
    fn update_contribution(&mut self, contribution_diff: Contribution);
    fn from_children(left: &Self, right: &Self) -> Self;
    fn _sample(storage: &[Self], rng: &mut impl rand::Rng, smoothed:bool, cost:Contribution, coreset_star_weight:Weight) -> Result<(ShiftedIndex,Float),Error>;
    fn _computed_sampling_probability(storage: &[Self], smoothed: bool, shifted_idx: ShiftedIndex, cost: Contribution, coreset_star_weight: Weight) ->Result<Float,Error>;
}


// MARK: tests:
#[cfg(test)]
mod tests{
    pub use super::*;

    #[test]
    fn test_datapoint_contribution(){
        let datapoint = Datapoint{
            weight: Weight(1.0),
            self_affinity: SelfAffinity(2.0),
        };
        let smallest_coreset_self_affinity = 0.5;
        let contribution = datapoint.contribution(smallest_coreset_self_affinity);
        assert_eq!(contribution.0, 2.5);
    }

    #[test]
    fn test_datapoint_smoothed_contribution(){
        let datapoint = DatapointWithCoresetCrossTerm{
            weight: Weight(1.0),
            self_affinity: SelfAffinity(2.0),
            coreset_cross_term: CoresetCrossTerm(0.5),
        };
        let cost = 5.0;
        let coreset_start_weight = Weight(10.0);
        let smoothed_contribution = datapoint.smoothed_contribution(cost, coreset_start_weight);
        assert_eq!(smoothed_contribution, 0.6);
    }
}
