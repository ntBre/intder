use crate::{geom::Geom, hmat::Hmat, tensor::tensor3::Tensor3, Siic, Vec3};

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
    pub h411: Tensor3,
    pub h421: Tensor3,
    pub h422: Tensor3,
    pub h431: Tensor3,
    pub h432: Tensor3,
    pub h433: Tensor3,
    pub h441: Tensor3,
    pub h442: Tensor3,
    pub h443: Tensor3,
    pub h444: Tensor3,
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
            h411: Tensor3::zeros(3, 3, 3),
            h421: Tensor3::zeros(3, 3, 3),
            h422: Tensor3::zeros(3, 3, 3),
            h431: Tensor3::zeros(3, 3, 3),
            h432: Tensor3::zeros(3, 3, 3),
            h433: Tensor3::zeros(3, 3, 3),
            h441: Tensor3::zeros(3, 3, 3),
            h442: Tensor3::zeros(3, 3, 3),
            h443: Tensor3::zeros(3, 3, 3),
            h444: Tensor3::zeros(3, 3, 3),
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
            // HIJKS6
            Torsion(i, j, k, l) => {
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
                let v2 = &tmp[3 * j..3 * j + 3];
                let v4 = &tmp[3 * l..3 * l + 3];
                // unit and non-unit vectors
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let e34 = geom.unit(*k, *l);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let t34 = geom.dist(*k, *l);
                // angles
                let p2 = geom.angle(*i, *j, *k);
                let tmp = geom.s_vec(&Bend(*i, *j, *k));
                let bp21 = &tmp[3 * i..3 * i + 3];
                let bp22 = &tmp[3 * j..3 * j + 3];
                let bp23 = &tmp[3 * k..3 * k + 3];

                let p3 = geom.angle(*j, *k, *l);
                let tmp = geom.s_vec(&Bend(*j, *k, *l));
                let bp32 = &tmp[3 * j..3 * j + 3];
                let bp33 = &tmp[3 * k..3 * k + 3];
                let bp34 = &tmp[3 * l..3 * l + 3];
                // matrices
                let h32 = Hmat::mat1(&e23);
                let h21 = Hmat::mat1(&e21);
                let h43 = Hmat::mat1(&e34);

                let c1 = 1.0 / t21;
                let c2 = 1.0 / t34;
                let c3 = 1.0 / t23;
                let c4 = f64::sin(p2);
                let c5 = f64::cos(p2);
                let c6 = c5 / c4;
                let c7 = f64::sin(p3);
                let c8 = f64::cos(p3);
                let c9 = c8 / c7;
                let c10 = 1.0 / (c4 * c4);
                let c11 = 1.0 / (c7 * c7);
                let c12 = c1 * c1;
                let c13 = c2 * c2;
                let c14 = c3 * c3;
                let c15 = t21 * c3;
                let c16 = t34 * c3;
                let w1 = 2.0 * c6;
                let w2 = 2.0 * c9;
                let w3 = 2.0 * c1;
                let w4 = 2.0 * c2;
                let w5 = c5 * c3;
                let w6 = c8 * c3;
                for k in 0..3 {
                    h.h411[(0, 0, k)] = e21[k] * c1 + bp21[k] * c6;
                    h.h411[(0, 1, k)] = e34[k] * c2 + bp34[k] * c9;
                    h.h411[(0, 2, k)] = e23[k] * c3 + bp23[k] * w1;
                    h.h411[(1, 0, k)] = -e23[k] * c3 + bp32[k] * w2;
                    h.h411[(1, 1, k)] =
                        e21[k] * w3 + e23[k] * c3 - bp22[k] * w1;
                    h.h411[(1, 2, k)] =
                        e34[k] * w4 - e23[k] * c3 - bp33[k] * w2;
                    h.h411[(2, 0, k)] = e23[k] * w5 + bp23[k] * c4;
                    h.h411[(2, 1, k)] = -e23[k] * w6 + bp32[k] * c7;
                } // end 10

                let mut v2 = v2.to_vec();
                for k in 0..3 {
                    for m in 0..3 {
                        v2[m] = 0.0;
                    }
                    v2[k] = 1.0;
                    let h22 = Hmat::mat1(&Vec3::from_row_slice(&v2));
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h421[(i, j, k)] = h22[(i, j)];
                        }
                    }
                } // end 15

                let w1 = 2.0 * c10 * c12;
                let w2 = 2.0 * c11 * c13;
                for k in 0..3 {
                    let w3 = w1 * h.h411[(0, 0, k)];
                    let w4 = w2 * h.h411[(0, 1, k)];
                    for j in 0..=k {
                        for i in 0..=j {
                            h.h111[(i, j, k)] = w3 * h32[(i, j)];
                            h.h444[(i, j, k)] = -w4 * h32[(i, j)];
                        }
                    }
                } // end 102

                let w1 = c10 * c12;
                let w2 = c11 * c13;
                let w3 = w1 * c3;
                let w4 = w2 * c3;
                for k in 0..3 {
                    let w5 = w1 * h.h411[(0, 2, k)];
                    let w6 = w2 * h.h411[(1, 0, k)];
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h113[(i, j, k)] =
                                w5 * h32[(i, j)] - w3 * h.h421[(i, j, k)];
                            h.h442[(i, j, k)] =
                                -w6 * h32[(i, j)] - w4 * h.h421[(i, j, k)];
                        }
                    }
                } // end 107

                let w1 = c1 * c3 * c10;
                let w2 = c2 * c3 * c11;
                for k in 0..3 {
                    let w5 = w1 * h.h411[(1, 1, k)];
                    let w6 = w2 * h.h411[(1, 2, k)];
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h123[(i, k, j)] =
                                -w5 * h21[(i, j)] + w3 * h.h421[(i, j, k)];
                            h.h432[(i, k, j)] =
                                -w6 * h43[(i, j)] + w4 * h.h421[(i, j, k)];
                        }
                    }
                } // end 112

                let hijs2 = Hmat::new(geom, &Bend(*i, *j, *k));
                let h11 = hijs2.h11;
                let mut h21 = hijs2.h21;
                let h31 = hijs2.h31;
                let h32 = hijs2.h32;
                let h44 = Hmat::new(geom, &Stretch(*i, *j)).h11;
                let h42 = Hmat::new(geom, &Stretch(*j, *k)).h11;
                let mut h43 = Hmat::mat1(&e34);
                let w1 = 2.0 * c1;
                let w2 = 2.0 * c12;
                for k in 0..3 {
                    for i in 0..=k {
                        h43[(i, k)] = 2.0
                            * (w1 * h44[(i, k)] + c6 * h11[(i, k)]
                                - c10 * bp21[(i)] * bp21[(k)]);
                    }
                    h43[(k, k)] = h43[(k, k)] - w2;
                }
                for k in 0..3 {
                    for j in 0..=k {
                        for i in 0..=j {
                            h.h111[(i, j, k)] -= v1[(j)] * h43[(i, k)];
                        }
                    }
                } // end 122

                let w1 = 2.0 * c6;
                let w2 = 2.0 * c10;
                let w3 = 2.0 * c3;
                for k in 0..3 {
                    for i in 0..3 {
                        h43[(i, k)] =
                            h31[(k, i)] * w1 - bp21[(i)] * bp23[(k)] * w2;
                    }
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h113[(i, j, k)] =
                                h.h113[(i, j, k)] - v1[(j)] * h43[(i, k)];
                        }
                    }
                }
                for k in 0..3 {
                    for j in 0..3 {
                        h43[(j, k)] = w3 * h42[(j, k)] - w1 * h32[(k, j)]
                            + w2 * bp22[(j)] * bp23[(k)];
                    }
                    h43[(k, k)] = h43[(k, k)] - c14;
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h123[(i, j, k)] =
                                h.h123[(i, j, k)] + v1[(i)] * h43[(j, k)];
                        }
                    }
                } // end 142

                let w1 = c4 * c3;
                let w2 = c4 * c15;
                let w3 = c5 * c15;
                let w4 = w3 * c3;
                let w5 = c3 * c15;
                for k in 0..3 {
                    for i in 0..3 {
                        let w6 = -e21[(i)] * bp23[(k)] * w1
                            + h32[(k, i)] * w2
                            + bp22[(i)] * bp23[(k)] * w3
                            - h42[(i, k)] * w4;
                        h43[(i, k)] = w6;
                    }
                }
                for k in 0..3 {
                    for i in 0..3 {
                        let w6 =
                            h43[(i, k)] + e23[(i)] * w5 * h.h411[(2, 0, k)];
                        for j in 0..3 {
                            h.h223[(i, j, k)] = -v1[(j)] * w6;
                        }
                    }
                } // end 152

                let w1 = c3 * c4 * c15;
                let w2 = c5 * c14;
                for k in 0..3 {
                    for i in 0..3 {
                        let w3 = -e23[(k)] * bp22[(i)] * w1
                            + e23[(k)] * w2 * (c15 * e23[(i)] - e21[(i)]);
                        h43[(i, k)] = h43[(i, k)] + w3;
                    }
                } // end 156

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h332[(i, j, k)] = v1[(j)] * h43[(k, i)];
                        }
                    }
                } // end 162

                let hijs2 = Hmat::new(geom, &Bend(*j, *k, *l));
                let h32 = hijs2.h21;
                let h42 = hijs2.h31;
                let h44 = hijs2.h33;
                let h11 = Hmat::new(geom, &Stretch(*l, *k)).h11;
                let h31 = Hmat::new(geom, &Stretch(*k, *j)).h11;

                let w1 = 2.0 * c2;
                let w2 = 2.0 * c13;
                for k in 0..3 {
                    for i in 0..=k {
                        h21[(i, k)] = 2.0
                            * (w1 * h11[(i, k)] + c9 * h44[(i, k)]
                                - c11 * bp34[i] * bp34[k]);
                    }
                    h21[(k, k)] -= w2;
                } // end 165

                for k in 0..3 {
                    for j in 0..=k {
                        for i in 0..=j {
                            h.h444[(i, j, k)] -= v4[j] * h21[(i, k)];
                        }
                    }
                } // end 172

                let w1 = 2.0 * c9;
                let w2 = 2.0 * c11;
                let w3 = 2.0 * c3;
                for k in 0..3 {
                    for i in 0..3 {
                        h21[(i, k)] =
                            w1 * h42[(i, k)] - w2 * bp34[(i)] * bp32[(k)];
                    }
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h442[(i, j, k)] -= v4[(j)] * h21[(i, k)];
                        }
                    }
                }

                for k in 0..3 {
                    for j in 0..3 {
                        h21[(j, k)] = w3 * h31[(j, k)] - w1 * h32[(j, k)]
                            + w2 * bp33[(j)] * bp32[(k)];
                    }
                    h21[(k, k)] -= c14;
                } // end 185

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h432[(i, j, k)] += v4[(i)] * h21[(j, k)];
                        }
                    }
                } // end 192

                let w1 = c7 * c3;
                let w2 = c7 * c16;
                let w3 = c8 * c16;
                let w4 = w3 * c3;
                let w5 = t34 * c14;
                for k in 0..3 {
                    for i in 0..3 {
                        let w6 = -e34[(i)] * bp32[(k)] * w1
                            + h32[(i, k)] * w2
                            + bp33[(i)] * bp32[(k)] * w3
                            - h31[(i, k)] * w4;
                        h21[(i, k)] = w6;
                    }
                } // end 196

                for k in 0..3 {
                    for i in 0..3 {
                        let w6 =
                            h21[(i, k)] - e23[(i)] * w5 * h.h411[(2, 1, k)];
                        for j in 0..3 {
                            h.h332[(i, j, k)] -= v4[(j)] * w6;
                        }
                    }
                } // end 202

                let w1 = c3 * c7 * c16;
                let w2 = c8 * c14;
                for k in 0..3 {
                    for i in 0..3 {
                        let w3 = e23[(k)] * bp33[(i)] * w1
                            + e23[(k)] * w2 * (e34[(i)] + c16 * e23[(i)]);
                        h21[(i, k)] += w3;
                    }
                } // end 206

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h223[(i, j, k)] += v4[j] * h21[(k, i)];
                        }
                    }
                } // end 212

                let Hmat {
                    h11,
                    h21,
                    h31,
                    h41: _,
                    h22: _,
                    h32: _,
                    h42,
                    h33: _,
                    h43,
                    h44,
                } = Hmat::new(geom, s);
                for k in 0..3 {
                    for j in 0..=k {
                        for i in 0..=j {
                            h.h111[(i, j, k)] -=
                                2.0 * h11[(j, k)] * h.h411[(0, 0, i)];
                            h.h444[(i, j, k)] -=
                                2.0 * h44[(j, k)] * h.h411[(0, 1, i)];
                        }
                    }
                } // end 222
                h.h111.fill3a(3);
                h.h444.fill3a(3);

                for i in 0..3 {
                    let w1 = 2.0 * h.h411[(0, 0, i)];
                    let w2 = 2.0 * h.h411[(0, 1, i)];
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h113[(i, j, k)] -= w1 * h31[(k, j)];
                            h.h442[(i, j, k)] -= w2 * h42[(j, k)];
                            h.h123[(i, j, k)] -=
                                h21[(j, i)] * h.h411[(0, 2, k)];
                            h.h432[(i, j, k)] -=
                                h43[(i, j)] * h.h411[(1, 0, k)];
                        }
                    }
                } // end 227

                let w4 = c5 * c15;
                let w1 = w4 - 1.0;
                let w2 = c8 * c16;
                let w3 = w2 - 1.0;
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h223[(i, j, k)] +=
                                w1 * h.h123[(j, i, k)] - w2 * h.h432[(j, k, i)];
                            h.h332[(i, j, k)] +=
                                w3 * h.h432[(j, i, k)] - w4 * h.h123[(j, k, i)];
                        }
                    }
                } // end 232

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h223[(i, j, k)] -=
                                c15 * h21[(i, j)] * h.h411[(2, 0, k)];
                            h.h332[(i, j, k)] -=
                                c16 * h43[(j, i)] * h.h411[(2, 1, k)];
                        }
                    }
                } // end 242

                for k in 0..3 {
                    for j in 0..3 {
                        let w1 = c16 * (h43[(j, k)] - c3 * v4[(j)] * e23[(k)]);
                        let w2 = c15 * (h21[(k, j)] + c3 * v1[(j)] * e23[(k)]);
                        for i in 0..3 {
                            h.h223[(i, j, k)] += w1 * h.h411[(2, 1, i)];
                            h.h332[(i, j, k)] += w2 * h.h411[(2, 0, i)];
                        }
                    }
                } // end 252

                let w1 = c5 * c3;
                let w2 = c4 * c15;
                let w3 = c5 * t21 * c14;
                let w4 = c8 * c3;
                let w5 = c7 * c16;
                let w6 = c8 * t34 * c14;
                for k in 0..3 {
                    h.h411[(0, 0, k)] =
                        w5 * bp33[(k)] + w6 * e23[(k)] + w4 * e34[(k)];
                    h.h411[(0, 1, k)] =
                        w2 * bp22[(k)] - w3 * e23[(k)] + w1 * e21[(k)];
                    h.h411[(0, 2, k)] =
                        -w1 * e21[(k)] - w2 * bp22[(k)] + w3 * e23[(k)];
                    h.h411[(1, 0, k)] =
                        -w4 * e34[(k)] - w5 * bp33[(k)] - w6 * e23[(k)];
                } // end 260

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h223[(i, j, k)] +=
                                h42[(j, i)] * h.h411[(0, 0, k)];
                            h.h332[(i, j, k)] +=
                                h31[(i, j)] * h.h411[(0, 1, k)];
                        }
                    }
                } // end 267

                for k in 0..3 {
                    for j in 0..3 {
                        let w1 = h31[(k, j)] - c3 * v1[(j)] * e23[(k)];
                        let w2 = h42[(j, k)] + c3 * v4[(j)] * e23[(k)];
                        for i in 0..3 {
                            h.h223[(i, j, k)] += w1 * h.h411[(0, 2, i)];
                            h.h332[(i, j, k)] += w2 * h.h411[(1, 0, i)];
                        }
                    }
                }

                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h411[(i, j, k)] = 0.0;
                            h.h421[(i, j, k)] = 0.0;
                            h.h431[(i, j, k)] = 0.0;
                            h.h441[(i, j, k)] = 0.0;
                            h.h443[(j, k, i)] =
                                -(h.h444[(i, j, k)] + h.h442[(j, k, i)]);
                            h.h112[(i, j, k)] =
                                -(h.h111[(i, j, k)] + h.h113[(i, j, k)]);
                        }
                    }
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h433[(k, i, j)] =
                                -(h.h443[(i, k, j)] + h.h432[(k, j, i)]);
                            h.h221[(i, j, k)] =
                                -(h.h112[(i, k, j)] + h.h123[(k, j, i)]);
                        }
                    }
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h422[(k, i, j)] =
                                -(h.h432[(k, i, j)] + h.h442[(i, k, j)]);
                            h.h331[(i, j, k)] =
                                -(h.h123[(k, i, j)] + h.h113[(i, k, j)]);
                        }
                    }
                } // end 292

                for k in 0..3 {
                    for j in 0..=k {
                        for i in 0..=j {
                            h.h222[(i, j, k)] = -(h.h221[(i, j, k)]
                                + h.h223[(i, j, k)]
                                + h.h422[(k, i, j)]);
                            h.h333[(i, j, k)] = -(h.h331[(i, j, k)]
                                + h.h332[(i, j, k)]
                                + h.h433[(k, i, j)]);
                        }
                    }
                } // end 302
                h.h222.fill3a(3);
                h.h333.fill3a(3);
            }
            // HIJKS3
            Lin1(i, j, k, _) => {
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let th = s.value(&geom);
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let h11a = Hmat::new(geom, &Stretch(*i, *j)).h11;
                let h33a = Hmat::new(geom, &Stretch(*k, *j)).h11;
                let hijs3 = Hmat::new(geom, s);
                let h11 = hijs3.h11;
                let h31 = hijs3.h31;
                let h33 = hijs3.h33;
                let h111a = Self::new(geom, &Stretch(*i, *j)).h111;
                let h333a = Self::new(geom, &Stretch(*k, *j)).h111;

                let tanth = th.tan();
                let costh = th.cos();

                let w1 = 1.0 / t21;
                let w2 = 1.0 / t23;
                let w3 = tanth * w1;
                let w4 = tanth * w2;
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h221[(i, j, k)] = h11[(i, j)]
                                * (v1[(k)] * tanth - e21[(k)] / t21);
                            h.h221[(i, j, k)] = h.h221[(i, j, k)]
                                + v1[(k)] * v1[(j)] * e21[(i)] * tanth / t21;
                            h.h221[(i, j, k)] = h.h221[(i, j, k)]
                                - (h11a[(i, j)] * v1[(k)]) / t21;
                            h.h223[(i, j, k)] = h33[(i, j)]
                                * (v3[(k)] * tanth - e23[(k)] / t23);
                            h.h223[(i, j, k)] = h.h223[(i, j, k)]
                                + v3[(k)] * v3[(j)] * e23[(i)] * tanth / t23;
                            h.h223[(i, j, k)] = h.h223[(i, j, k)]
                                - (h33a[(i, j)] * v3[(k)]) / t23;
                        }
                    }
                } // end 12
                for k in 0..3 {
                    for j in k..3 {
                        for i in j..3 {
                            h.h111[(i, j, k)] = (h.h221[(i, j, k)]
                                + h.h221[(j, k, i)]
                                + h.h221[(k, i, j)])
                                + v1[(i)] * v1[(j)] * v1[(k)]
                                - h111a[(i, j, k)] * w3;
                            h.h333[(i, j, k)] = (h.h223[(i, j, k)]
                                + h.h223[(j, k, i)]
                                + h.h223[(k, i, j)])
                                + v3[(i)] * v3[(j)] * v3[(k)]
                                - h333a[(i, j, k)] * w4;
                        }
                    }
                }
                h.h111.fill3b();
                h.h333.fill3b();
                for i in 0..3 {
                    let w5 = v1[(i)] * tanth - e21[(i)] * w1;
                    let w6 = v3[(i)] * tanth - e23[(i)] * w2;
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h221[(i, j, k)] = w5 * h31[(k, j)];
                            h.h223[(i, j, k)] = w6 * h31[(j, k)];
                        }
                    }
                }
                let w5 = 1.0 / (costh * costh);
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h113[(i, j, k)] = v3[(k)]
                                * (v1[(i)] * v1[(j)] - h11a[(i, j)] * w1)
                                * w5
                                + h.h221[(i, j, k)]
                                + h.h221[(j, i, k)];
                            h.h331[(i, j, k)] = v1[(k)]
                                * (v3[(i)] * v3[(j)] - h33a[(i, j)] * w2)
                                * w5
                                + h.h223[(i, j, k)]
                                + h.h223[(j, i, k)];
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
            // hijks7
            Out(i, j, k, l) => {
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let e24 = geom.unit(*j, *l);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let t24 = geom.dist(*j, *l);

                // vect2 call
                let phi = Siic::Bend(*k, *j, *l).value(geom);
                let svec = geom.s_vec(&Siic::Bend(*k, *j, *l));
                let bp3 = &svec[3 * k..3 * k + 3];
                let bp4 = &svec[3 * l..3 * l + 3];

                // vect5 call
                let svec = geom.s_vec(s);
                let gamma = s.value(geom);
                let v1 = &svec[3 * i..3 * i + 3];
                let v3 = &svec[3 * k..3 * k + 3];
                let v4 = &svec[3 * l..3 * l + 3];

                // hijs1 call
                let ht11 = Hmat::new(geom, &Siic::Stretch(*i, *j)).h11;

                // hijs2 call
                let Hmat {
                    h11: hp33,
                    h21: _,
                    h31: hp43,
                    h22: _,
                    h32: _,
                    h33: hp44,
                    h41: _,
                    h42: _,
                    h43: _,
                    h44: _,
                } = Hmat::new(geom, &Siic::Bend(*k, *j, *l));

                // hijs7 call
                let Hmat {
                    h11,
                    h21: _,
                    h31,
                    h22: _,
                    h32: _,
                    h33,
                    h41,
                    h42: _,
                    h43,
                    h44,
                } = Hmat::new(geom, s);

                // hijks1 call
                let ht111 = Htens::new(geom, &Siic::Stretch(*i, *j)).h111;

                // hijks2 call
                let Htens {
                    h111: _,
                    h112: _,
                    h113: hp334,
                    h123: _,
                    h221: _,
                    h222: _,
                    h223: _,
                    h331: hp443,
                    h332: _,
                    h333: _,
                    h411: _,
                    h421: _,
                    h422: _,
                    h431: _,
                    h432: _,
                    h433: _,
                    h441: _,
                    h442: _,
                    h443: _,
                    h444: _,
                } = Htens::new(geom, &Siic::Bend(*k, *j, *l));

                let cp21 = Hmat::mat1(&e21);
                let cp24 = Hmat::mat1(&e24);
                let cp2124 = e21.cross(&e24);
                let cg = gamma.cos();
                let sg = gamma.sin();
                let tg = sg / cg;
                let cp = phi.cos();
                let sp = phi.sin();
                let ctp = cp / sp;
                let c1 = 1.0 / t21;
                let s2g = 1.0 / (cg * cg);
                let c3 = 1.0 / t23;
                let c4 = 1.0 / t24;
                let c2p = 1.0 / (sp * sp);
                let c1111 = tg * c1;
                let c1112 = s2g * c1;
                let c3331 = t24 * c3 / sp;
                let c3332 = c3331 * tg;
                let c3333 = c3331 * s2g;
                let c3335 = c3 * c3;
                let c3334 = 2.0 * c3335;
                let c4411 = t23 * c4 / sp;
                let c4412 = c4411 * s2g;
                let c431 = c3 * c4 / (cg * sp);
                let c4311 = c431 * c1;
                let c4312 = c431 * tg;
                let c4313 = c3333 * c4;
                let c4431 = c4411 * tg;
                let c4442 = c4 * c4;
                let c4441 = 2.0 * c4442;
                for i in 0..3 {
                    for j in 0..=i {
                        for k in 0..=j {
                            h.h111[(i, j, k)] =
                                s2g * v1[(i)] * v1[(j)] * v1[(k)]
                                    + tg * h11[(i, j)] * v1[(k)]
                                    + tg * h11[(i, k)] * v1[(j)]
                                    + c1111
                                        * (e21[(i)] * v1[(j)] * v1[(k)]
                                            - ht111[(i, j, k)])
                                    - c1112 * ht11[(j, k)] * v1[(i)]
                                    - c1 * (e21[(i)] * h11[(j, k)]
                                        + e21[(j)] * h11[(i, k)]
                                        + e21[(k)] * h11[(i, j)]
                                        + ht11[(i, j)] * v1[(k)]
                                        + ht11[(i, k)] * v1[(j)]);
                            h.h333[(i, j, k)] = c3331
                                * (h33[(i, j)] * bp4[(k)]
                                    + hp43[(k, i)] * v3[(j)]
                                    - (v3[(j)] * bp4[(k)] + tg * hp43[(k, j)])
                                        * (c3 * e23[(i)] + ctp * bp3[(i)]))
                                + c3332 * hp334[(i, j, k)]
                                + c3333 * hp43[(k, j)] * v3[(i)]
                                + h33[(i, k)]
                                    * (tg * v3[(j)]
                                        - ctp * bp3[(j)]
                                        - c3 * e23[(j)])
                                + v3[(k)]
                                    * (v3[(i)] * v3[(j)] * s2g
                                        + h33[(i, j)] * tg
                                        + bp3[(i)] * bp3[(j)] * c2p
                                        - hp33[(i, j)] * ctp
                                        + e23[(i)] * e23[(j)] * c3334);
                            h.h444[(i, j, k)] = c4411
                                * (h44[(i, j)] * bp3[(k)]
                                    + hp43[(i, k)] * v4[(j)]
                                    - (v4[(j)] * bp3[(k)] + hp43[(j, k)] * tg)
                                        * (e24[(i)] * c4 + bp4[(i)] * ctp))
                                + c4431 * hp443[(i, j, k)]
                                + c4412 * hp43[(j, k)] * v4[(i)]
                                + h44[(i, k)]
                                    * (tg * v4[(j)]
                                        - ctp * bp4[(j)]
                                        - c4 * e24[(j)])
                                + v4[(k)]
                                    * (v4[(i)] * v4[(j)] * s2g
                                        + h44[(i, j)] * tg
                                        + bp4[(i)] * bp4[(j)] * c2p
                                        - hp44[(i, j)] * ctp
                                        + e24[(i)] * e24[(j)] * c4441);
                            if i == j {
                                h.h333[(i, j, k)] =
                                    h.h333[(i, j, k)] - v3[(k)] * c3335;
                                h.h444[(i, j, k)] =
                                    h.h444[(i, j, k)] - v4[(k)] * c4442;
                            }
                        }
                    }
                } // end 12 loop
                h.h111.fill3b();
                h.h333.fill3b();
                h.h444.fill3b();

                for k in 0..3 {
                    for j in 0..=k {
                        for i in 0..3 {
                            h.h113[(j, k, i)] =
                                s2g * v3[(i)] * v1[(j)] * v1[(k)]
                                    + tg * h31[(i, j)] * v1[(k)]
                                    + tg * h31[(i, k)] * v1[(j)]
                                    - c1 * (e21[(j)] * h31[(i, k)]
                                        + e21[(k)] * h31[(i, j)])
                                    - c1112 * ht11[(j, k)] * v3[(i)];
                            h.h113[(k, j, i)] = h.h113[(j, k, i)];
                            h.h411[(i, j, k)] =
                                s2g * v4[(i)] * v1[(j)] * v1[(k)]
                                    + tg * h41[(i, j)] * v1[(k)]
                                    + tg * h41[(i, k)] * v1[(j)]
                                    - c1 * (e21[(j)] * h41[(i, k)]
                                        + e21[(k)] * h41[(i, j)])
                                    - c1112 * ht11[(j, k)] * v4[(i)];
                            h.h411[(i, k, j)] = h.h411[(i, j, k)];
                            h.h433[(i, j, k)] = c3331
                                * (h43[(i, j)] * bp4[(k)]
                                    + hp44[(i, k)] * v3[(j)]
                                    + (c4 * e24[(i)] - ctp * bp4[(i)])
                                        * (v3[(j)] * bp4[(k)]
                                            + tg * hp43[(k, j)]))
                                + c3332 * hp443[(i, k, j)]
                                + c3333 * hp43[(k, j)] * v4[(i)]
                                + h43[(i, k)]
                                    * (tg * v3[(j)]
                                        - ctp * bp3[(j)]
                                        - c3 * e23[(j)])
                                + v3[(k)]
                                    * (tg * h43[(i, j)]
                                        + s2g * v4[(i)] * v3[(j)]
                                        - ctp * hp43[(i, j)]
                                        + c2p * bp4[(i)] * bp3[(j)]);
                            h.h433[(i, k, j)] = h.h433[(i, j, k)];
                        }
                    }
                } // end 22 loop

                for i in 0..3 {
                    for j in 0..=i {
                        for k in 0..3 {
                            h.h331[(i, j, k)] = c3331 * h31[(i, k)] * bp4[(j)]
                                + c3333 * hp43[(j, i)] * v1[(k)]
                                + (tg * v3[(i)]
                                    - ctp * bp3[(i)]
                                    - c3 * e23[(i)])
                                    * h31[(j, k)]
                                + tg * h31[(i, k)] * v3[(j)]
                                + s2g * v3[(i)] * v3[(j)] * v1[(k)];
                            h.h331[(j, i, k)] = h.h331[(i, j, k)];
                            h.h441[(i, j, k)] = c4411 * h41[(i, k)] * bp3[(j)]
                                + c4412 * hp43[(i, j)] * v1[(k)]
                                + (tg * v4[(i)]
                                    - ctp * bp4[(i)]
                                    - c4 * e24[(i)])
                                    * h41[(j, k)]
                                + tg * h41[(i, k)] * v4[(j)]
                                + s2g * v4[(i)] * v4[(j)] * v1[(k)];
                            h.h441[(j, i, k)] = h.h441[(i, j, k)];
                            h.h443[(i, j, k)] = c4411
                                * (h43[(j, k)] * bp3[(i)]
                                    + hp33[(i, k)] * v4[(j)]
                                    + (c3 * e23[(k)] - ctp * bp3[(k)])
                                        * (bp3[(i)] * v4[(j)]
                                            + tg * hp43[(j, i)]))
                                + c4431 * hp334[(i, k, j)]
                                + c4412 * hp43[(j, i)] * v3[(k)]
                                + h43[(i, k)]
                                    * (tg * v4[(j)]
                                        - ctp * bp4[(j)]
                                        - c4 * e24[(j)])
                                + v4[(i)]
                                    * (tg * h43[(j, k)]
                                        + s2g * v4[(j)] * v3[(k)]
                                        - ctp * hp43[(j, k)]
                                        + c2p * bp4[(j)] * bp3[(k)]);
                            h.h443[(j, i, k)] = h.h443[(i, j, k)];
                        }
                    }
                } // end 32 loop

                let prod = Self::tripro();
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h431[(i, j, k)] = c4311
                                * (prod[(k, j, i)]
                                    + e21[(k)] * cp21[(i, j)]
                                    + e24[(i)] * cp24[(j, k)]
                                    - e24[(i)] * e21[(k)] * cp2124[(j)])
                                + c4312
                                    * v1[(k)]
                                    * (e24[(i)] * cp2124[(j)] - cp21[(i, j)])
                                + tg * h41[(i, k)] * v3[(j)]
                                + s2g * v4[(i)] * v3[(j)] * v1[(k)]
                                + h31[(j, k)] * (tg * v4[(i)] - ctp * bp4[(i)])
                                + c4313 * e24[(i)] * bp4[(j)] * v1[(k)]
                                + c3331 * h41[(i, k)] * bp4[(j)]
                                + c3333 * hp44[(i, j)] * v1[(k)];
                        }
                    }
                } // end 42

                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h112[(i, j, k)] = -(h.h111[(i, j, k)]
                                + h.h113[(i, j, k)]
                                + h.h411[(k, i, j)]);
                            h.h421[(i, j, k)] = -(h.h411[(i, j, k)]
                                + h.h431[(i, j, k)]
                                + h.h441[(i, j, k)]);
                            h.h123[(i, j, k)] = -(h.h113[(i, j, k)]
                                + h.h331[(j, k, i)]
                                + h.h431[(j, k, i)]);
                            h.h332[(i, j, k)] = -(h.h331[(i, j, k)]
                                + h.h333[(i, j, k)]
                                + h.h433[(k, i, j)]);
                            h.h432[(i, j, k)] = -(h.h431[(i, j, k)]
                                + h.h433[(i, j, k)]
                                + h.h443[(i, k, j)]);
                            h.h442[(i, j, k)] = -(h.h441[(i, j, k)]
                                + h.h443[(i, j, k)]
                                + h.h444[(i, j, k)]);
                        }
                    }
                } // end 52

                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            h.h221[(i, j, k)] = -(h.h112[(i, k, j)]
                                + h.h123[(k, j, i)]
                                + h.h421[(i, j, k)]);
                            h.h223[(i, j, k)] = -(h.h123[(i, j, k)]
                                + h.h332[(i, k, j)]
                                + h.h432[(i, k, j)]);
                            h.h422[(i, j, k)] = -(h.h421[(i, j, k)]
                                + h.h432[(i, k, j)]
                                + h.h442[(i, k, j)]);
                        }
                    }
                } // end 62

                for i in 0..3 {
                    for j in 0..=i {
                        for k in 0..=j {
                            h.h222[(i, j, k)] = -(h.h221[(i, j, k)]
                                + h.h223[(i, j, k)]
                                + h.h422[(k, i, j)]);
                        }
                    }
                } // end 72

                h.h222.fill3b();
            }
        }
        h
    }

    /// the name suggests some kind of triple product, but I'm not really sure.
    /// I think it's a Cartesian product of some kind maybe
    fn tripro() -> Tensor3 {
        let mut ret = Tensor3::zeros(3, 3, 3);
        for k in 0..3 {
            let mut vect = nalgebra::vector![0.0, 0.0, 0.0];
            vect[k] = 1.0;
            let rmat = Hmat::mat1(&vect);
            for j in 0..3 {
                for i in 0..3 {
                    ret[(i, j, k)] = rmat[(i, j)];
                }
            }
        }
        ret
    }
}
