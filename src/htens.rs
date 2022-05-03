use crate::{geom::Geom, hmat::Hmat, tensor::tensor3::Tensor3, Siic};

pub struct Htens {
    pub h111: Tensor3,
    pub h112: Tensor3,
    pub h113: Tensor3,
    pub h123: Tensor3,
    pub h221: Tensor3,
    pub h222: Tensor3,
    pub h223: Tensor3,
    pub h331: Tensor3,
    pub h332: Tensor3,
    pub h333: Tensor3,
}

impl Htens {
    pub fn zeros() -> Self {
        Self {
            h111: Tensor3::zeros(3, 3, 3),
            h112: Tensor3::zeros(3, 3, 3),
            h113: Tensor3::zeros(3, 3, 3),
            h123: Tensor3::zeros(3, 3, 3),
            h221: Tensor3::zeros(3, 3, 3),
            h222: Tensor3::zeros(3, 3, 3),
            h223: Tensor3::zeros(3, 3, 3),
            h331: Tensor3::zeros(3, 3, 3),
            h332: Tensor3::zeros(3, 3, 3),
            h333: Tensor3::zeros(3, 3, 3),
        }
    }

    pub fn new(geom: &Geom, s: &Siic) -> Self {
        use Siic::*;
        let hm = Hmat::new(geom, s);
        let mut h = Htens::zeros();
        match s {
            // HIJKS1
            Stretch(j, i) => {
                let v1 = geom.unit(*i, *j);
                let t21 = geom.dist(*i, *j);
                let w1 = 1.0 / t21;
                for k in 0..3 {
                    for j in k..3 {
                        for i in j..3 {
                            h.h111[(i, j, k)] = -(v1[i] * hm.h11[(k, j)]
                                + v1[j] * hm.h11[(k, i)]
                                + v1[k] * hm.h11[(j, i)])
                                * w1;
                        }
                    }
                }
                h.h111.fill3b();
            }
            // HIJKS2
            Bend(i, j, k) => {
                // copied from h_mat Bend
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let h11a = Hmat::new(geom, &Stretch(*i, *j)).h11;
                let h33a = Hmat::new(geom, &Stretch(*k, *j)).h11;
                let phi = geom.angle(*i, *j, *k);
                // end copy
                let hijs2 = Hmat::new(geom, s);
                let h111a = Self::new(geom, &Stretch(*i, *j)).h111;
                let h333a = Self::new(geom, &Stretch(*k, *j)).h111;
                let sphi = phi.sin();
                let ctphi = phi.cos() / sphi;
                let w1 = 1.0 / t21;
                let w2 = 1.0 / t23;
                let w3 = ctphi * w1;
                let w4 = ctphi * w2;
                for k in 0..3 {
                    let w5 = v1[k] * ctphi + e21[k] * w1;
                    let w6 = e21[k] * w3;
                    let w7 = v1[k] * w1;
                    let w8 = v3[k] * ctphi + e23[k] * w2;
                    let w9 = e23[k] * w4;
                    let w10 = v3[k] * w2;
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h221[(i, j, k)] = w5 * hijs2.h11[(i, j)]
                                + v1[i] * v1[j] * w6
                                + h11a[(i, j)] * w7;
                            h.h223[(i, j, k)] = w8 * hijs2.h33[(i, j)]
                                + v3[i] * v3[j] * w9
                                + h33a[(i, j)] * w10;
                        }
                    }
                }

                for k in 0..3 {
                    for j in k..3 {
                        for i in j..3 {
                            h.h111[(i, j, k)] = -(h.h221[(i, j, k)]
                                + h.h221[(j, k, i)]
                                + h.h221[(i, k, j)])
                                + v1[i] * v1[j] * v1[k]
                                + h111a[(i, j, k)] * w3;
                            h.h333[(i, j, k)] = -(h.h223[(i, j, k)]
                                + h.h223[(j, k, i)]
                                + h.h223[(i, k, j)])
                                + v3[i] * v3[j] * v3[k]
                                + h333a[(i, j, k)] * w4;
                        }
                    }
                }
                h.h111.fill3b();
                h.h333.fill3b();

                for i in 0..3 {
                    let w3 = v1[i] * ctphi + e21[i] * w1;
                    let w4 = v3[i] * ctphi + e23[i] * w2;
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h221[(i, j, k)] = w3 * hijs2.h31[(k, j)];
                            h.h223[(i, j, k)] = w4 * hijs2.h31[(j, k)];
                        }
                    }
                }

                let w3 = 1.0 / (sphi * sphi);
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h113[(i, j, k)] = v3[k]
                                * (v1[i] * v1[j] - h11a[(i, j)] * w1)
                                * w3
                                - h.h221[(i, j, k)]
                                - h.h221[(j, i, k)];
                            h.h331[(i, j, k)] = v1[k]
                                * (v3[i] * v3[j] - h33a[(i, j)] * w2)
                                * w3
                                - h.h223[(i, j, k)]
                                - h.h223[(j, i, k)];
                        }
                    }
                }

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h123[(i, j, k)] =
                                -(h.h331[(j, k, i)] + h.h113[(i, j, k)]);
                            h.h112[(i, j, k)] =
                                -(h.h111[(i, j, k)] + h.h113[(i, j, k)]);
                            h.h332[(i, j, k)] =
                                -(h.h333[(i, j, k)] + h.h331[(i, j, k)]);
                        }
                    }
                }

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h221[(j, k, i)] =
                                -(h.h123[(i, j, k)] + h.h112[(i, k, j)]);
                            h.h223[(j, k, i)] =
                                -(h.h332[(i, j, k)] + h.h123[(j, k, i)]);
                        }
                    }
                }

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h222[(i, j, k)] =
                                -(h.h223[(j, k, i)] + h.h221[(j, k, i)]);
                        }
                    }
                }
            }
            #[allow(unused)]
            Torsion(i, j, k, l) => {
                todo!()
            }
        }
        h
    }
}
