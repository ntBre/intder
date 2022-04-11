use std::ops::Index;

use crate::{DVec, Vec3};

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
