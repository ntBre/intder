use std::{
    io::BufRead,
    io::BufReader,
    ops::{Index, IndexMut, Neg},
};

#[derive(Clone, Debug)]
pub struct Tensor3(pub(crate) Vec<Vec<Vec<f64>>>);

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

    pub fn load(filename: &str) -> Self {
        let f =
            std::fs::File::open(filename).expect("failed to open tensor file");
        let lines = BufReader::new(f).lines();
        let mut ret = Vec::new();
        let mut buf = Vec::new();
        for line in lines.flatten() {
            let mut fields = line.split_whitespace().peekable();
            if fields.peek().is_none() {
                // in between chunks
                ret.push(buf);
                buf = Vec::new();
            } else {
                let row = fields
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                buf.push(row);
            }
        }
        ret.push(buf);
        Self(ret)
    }

    /// panics if either of the latter two dimensions is empty
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.0.len(), self.0[0].len(), self.0[0][0].len())
    }

    pub fn equal(&self, other: &Self, eps: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for (i, mat) in self.0.iter().enumerate() {
            for (j, row) in mat.iter().enumerate() {
                for k in 0..row.len() {
                    if f64::abs(self[(i, j, k)] - other[(i, j, k)]) > eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// copy values across the 3D diagonals
    pub fn fill3b(&mut self) {
        for m in 0..3 {
            for n in 0..m {
                for p in 0..n {
                    self[(n, m, p)] = self[(m, n, p)];
                    self[(n, p, m)] = self[(m, n, p)];
                    self[(m, p, n)] = self[(m, n, p)];
                    self[(p, m, n)] = self[(m, n, p)];
                    self[(p, n, m)] = self[(m, n, p)];
                }
                self[(n, m, n)] = self[(m, n, n)];
                self[(n, n, m)] = self[(m, n, n)];
            }
            for p in 0..m {
                self[(m, p, m)] = self[(m, m, p)];
                self[(p, m, m)] = self[(m, m, p)];
            }
        }
    }

    pub fn fill3a(&mut self, nsx: usize) {
        for p in 0..nsx {
            for n in 0..p {
                for m in 0..n {
                    self[(n, m, p)] = self[(m, n, p)];
                    self[(n, p, m)] = self[(m, n, p)];
                    self[(m, p, n)] = self[(m, n, p)];
                    self[(p, m, n)] = self[(m, n, p)];
                    self[(p, n, m)] = self[(m, n, p)];
                }
                self[(n, p, n)] = self[(n, n, p)];
                self[(p, n, n)] = self[(n, n, p)];
            }
            for m in 0..p {
                self[(p, m, p)] = self[(m, p, p)];
                self[(p, p, m)] = self[(m, p, p)];
            }
        }
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

impl Neg for Tensor3 {
    type Output = Tensor3;

    fn neg(self) -> Self::Output {
        let mut ret = self;
        for mat in &mut ret.0 {
            for row in mat {
                for col in row {
                    *col *= -1.0;
                }
            }
        }
        ret
    }
}
