use std::ops::Index;

use crate::{DVec, Siic, Vec3, ANGBOHR};

#[derive(Debug, PartialEq, Clone)]
pub struct Geom(pub Vec<Vec3>);

impl Geom {
    pub fn new() -> Self {
        Geom(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, it: Vec3) {
        self.0.push(it)
    }

    /// return the unit vector from atom i to atom j
    pub fn unit(&self, i: usize, j: usize) -> Vec3 {
        let diff = self[j] - self[i];
        diff / diff.magnitude()
    }

    /// distance between atoms i and j
    pub fn dist(&self, i: usize, j: usize) -> f64 {
        ANGBOHR * (self[j] - self[i]).magnitude()
    }

    /// angle in radians between atoms i, j, and k, where j is the central atom
    pub fn angle(&self, i: usize, j: usize, k: usize) -> f64 {
        let e_ji = Self::unit(self, j, i);
        let e_jk = Self::unit(self, j, k);
        (e_ji.dot(&e_jk)).acos()
    }

    pub fn s_vec(&self, ic: &Siic) -> Vec<f64> {
        let mut tmp = vec![0.0; 3 * self.len()];
        match ic {
            Siic::Stretch(a, b) => {
                let e_12 = self.unit(*a, *b);
                for i in 0..3 {
                    tmp[3 * a + i] = -e_12[i % 3];
                    tmp[3 * b + i] = e_12[i % 3];
                }
            }
            Siic::Bend(a, b, c) => {
                let e_21 = self.unit(*b, *a);
                let e_23 = self.unit(*b, *c);
                let t_12 = self.dist(*b, *a);
                let t_32 = self.dist(*b, *c);
                let w = e_21.dot(&e_23);
                let sp = (1.0 - w * w).sqrt();
                let c1 = 1.0 / (t_12 * sp);
                let c2 = 1.0 / (t_32 * sp);
                for i in 0..3 {
                    tmp[3 * a + i] = (w * e_21[i] - e_23[i]) * c1;
                    tmp[3 * c + i] = (w * e_23[i] - e_21[i]) * c2;
                    tmp[3 * b + i] = -tmp[3 * a + i] - tmp[3 * c + i];
                }
            }
            Siic::Torsion(a, b, c, d) => {
                let e_21 = self.unit(*b, *a);
                let e_32 = self.unit(*c, *b);
                let e_43 = self.unit(*d, *c);
                let t_21 = self.dist(*b, *a);
                let t_32 = self.dist(*c, *b);
                let t_43 = self.dist(*d, *c);
                let v5 = e_21.cross(&e_32);
                let v6 = e_43.cross(&e_32);
                let w2 = e_21.dot(&e_32);
                let w3 = e_43.dot(&e_32);
                let cp2 = -w2;
                let cp3 = -w3;
                let sp2 = (1.0 - cp2 * cp2).sqrt();
                let sp3 = (1.0 - cp3 * cp3).sqrt();
                // terminal atoms
                let w1 = 1.0 / (t_21 * sp2 * sp2);
                let w2 = 1.0 / (t_43 * sp3 * sp3);
                for i in 0..3 {
                    tmp[3 * a + i] = -w1 * v5[i];
                    tmp[3 * d + i] = -w2 * v6[i];
                }
                let w3 = (t_32 - t_21 * cp2) * w1 / t_32;
                let w4 = cp3 / (t_32 * sp3 * sp3);
                let w5 = (t_32 - t_43 * cp3) * w2 / t_32;
                let w6 = cp2 / (t_32 * sp2 * sp2);
                for i in 0..3 {
                    tmp[3 * b + i] = w3 * v5[i] + w4 * v6[i];
                    tmp[3 * c + i] = w5 * v6[i] + w6 * v5[i];
                }
            }
        }
        tmp
    }
}

impl From<psqs::geom::Geom> for Geom {
    /// panics if any element of `geom` has a length other than 3 or if `geom`
    /// is not Cartesian
    fn from(geom: psqs::geom::Geom) -> Self {
        let geom = geom.xyz().unwrap();
        let mut ret = Vec::new();
        for atom in geom {
            ret.push(Vec3::from_row_slice(&atom.coord()));
        }
        Self(ret)
    }
}

impl From<&DVec> for Geom {
    fn from(dvec: &DVec) -> Self {
        Self(
            dvec.as_slice()
                .chunks(3)
                .map(|x| Vec3::from_row_slice(x))
                .collect(),
        )
    }
}

impl Into<DVec> for Geom {
    fn into(self) -> DVec {
        let mut geom = Vec::with_capacity(self.len());
        for c in &self {
            geom.extend(&c);
        }
        DVec::from(geom)
    }
}

impl IntoIterator for &Geom {
    type Item = Vec3;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.clone().into_iter()
    }
}

impl Index<usize> for Geom {
    type Output = Vec3;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
