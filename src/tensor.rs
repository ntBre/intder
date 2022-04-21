use std::ops::{Index, IndexMut};

pub struct Tensor3(pub Vec<Vec<Vec<f64>>>);

impl Tensor3 {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self(vec![vec![vec![0.0; k]; j]; i])
    }
}

impl Index<(usize, usize, usize)> for Tensor3 {
    type Output = f64;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1][index.2]
    }
}

impl IndexMut<(usize, usize, usize)> for Tensor3 {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1][index.2]
    }
}
