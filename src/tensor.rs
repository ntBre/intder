use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Tensor3(pub Vec<Vec<Vec<f64>>>);

// TODO could probably replace these fields with vectors and fc3 index formula
// if they're always symmetric. Then I don't have to do all the symmetry stuff
// myself, I can just sort the indices when I access them
impl Tensor3 {
    /// return a new i x j x k tensor
    pub fn zeros(i: usize, j: usize, k: usize) -> Self {
        Self(vec![vec![vec![0.0; k]; j]; i])
    }

    pub fn print(&self) {
	println!();
	for mat in &self.0 {
            for row in mat {
                for col in row {
                    print!("{:12.6}", col);
                }
                println!();
            }
            println!();
            println!();
        }
    }
    // fortran 1, 1, 2 is my 1, 0, 1 / 2, 1, 2

    // fortran 1, 1, 3 is my 1, 0, 2 / 2 1 3
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
