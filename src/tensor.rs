use std::ops::{Index, IndexMut, Sub};

pub mod tensor3;
use tensor3::Tensor3;

#[derive(Clone)]
pub struct Tensor4(Vec<Vec<Vec<Vec<f64>>>>);

// TODO could probably replace these fields with vectors and fc3 index formula
// if they're always symmetric. Then I don't have to do all the symmetry stuff
// myself, I can just sort the indices when I access them
impl Tensor4 {
    /// return a new i x j x k x l tensor
    pub fn zeros(i: usize, j: usize, k: usize, l: usize) -> Self {
        Self(vec![vec![vec![vec![0.0; l]; k]; j]; i])
    }

    pub fn print(&self) {
        println!();
        for (i, tens) in self.0.iter().enumerate() {
            println!("I = {i:5}");
            Tensor3(tens.clone()).print();
        }
    }

    /// panics if any of the latter three dimensions is empty
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (
            self.0.len(),
            self.0[0].len(),
            self.0[0][0].len(),
            self.0[0][0][0].len(),
        )
    }

    pub fn equal(&self, other: &Self, eps: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for (i, tens) in self.0.iter().enumerate() {
            for (j, mat) in tens.iter().enumerate() {
                for (k, row) in mat.iter().enumerate() {
                    for (l, col) in row.iter().enumerate() {
                        if f64::abs(col - other[(i, j, k, l)]) > eps {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn fill4a(&mut self, ny: usize) {
        for q in 0..ny {
            for p in 0..=q {
                for n in 0..=p {
                    for m in 0..=n {
                        self[(n, m, p, q)] = self[(m, n, p, q)];
                        self[(n, p, m, q)] = self[(m, n, p, q)];
                        self[(n, p, q, m)] = self[(m, n, p, q)];
                        self[(m, p, n, q)] = self[(m, n, p, q)];
                        self[(p, m, n, q)] = self[(m, n, p, q)];
                        self[(p, n, m, q)] = self[(m, n, p, q)];
                        self[(p, n, q, m)] = self[(m, n, p, q)];
                        self[(m, p, q, n)] = self[(m, n, p, q)];
                        self[(p, m, q, n)] = self[(m, n, p, q)];
                        self[(p, q, m, n)] = self[(m, n, p, q)];
                        self[(p, q, n, m)] = self[(m, n, p, q)];
                        self[(m, n, q, p)] = self[(m, n, p, q)];
                        self[(n, m, q, p)] = self[(m, n, p, q)];
                        self[(n, q, m, p)] = self[(m, n, p, q)];
                        self[(n, q, p, m)] = self[(m, n, p, q)];
                        self[(m, q, n, p)] = self[(m, n, p, q)];
                        self[(q, m, n, p)] = self[(m, n, p, q)];
                        self[(q, n, m, p)] = self[(m, n, p, q)];
                        self[(q, n, p, m)] = self[(m, n, p, q)];
                        self[(m, q, p, n)] = self[(m, n, p, q)];
                        self[(q, m, p, n)] = self[(m, n, p, q)];
                        self[(q, p, m, n)] = self[(m, n, p, q)];
                        self[(q, p, n, m)] = self[(m, n, p, q)];
                    }
                }
            }
        }
    }

    /// panics if self is empty
    pub fn max(&self) -> f64 {
        let mut max = self[(0, 0, 0, 0)];
        for tens in &self.0 {
            for mat in tens {
                for row in mat {
                    for col in row {
                        if col > &max {
                            max = *col;
                        }
                    }
                }
            }
        }
        max
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor4 {
    type Output = f64;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1][index.2][index.3]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor4 {
    fn index_mut(
        &mut self,
        index: (usize, usize, usize, usize),
    ) -> &mut Self::Output {
        &mut self.0[index.0][index.1][index.2][index.3]
    }
}

impl Sub<Tensor4> for Tensor4 {
    type Output = Tensor4;

    fn sub(self, rhs: Tensor4) -> Self::Output {
        if self.shape() != rhs.shape() {
            panic!("Tensor4::sub: dimension mismatch");
        }
        let (a, b, c, d) = self.shape();
        let mut ret = Self::Output::zeros(a, b, c, d);
        for (i, tens) in self.0.iter().enumerate() {
            for (j, mat) in tens.iter().enumerate() {
                for (k, row) in mat.iter().enumerate() {
                    for (l, col) in row.iter().enumerate() {
                        ret[(i, j, k, l)] = col - rhs[(i, j, k, l)];
                    }
                }
            }
        }
        ret
    }
}
