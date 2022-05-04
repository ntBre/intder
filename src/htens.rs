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
            #[allow(unused)]
            Torsion(i, j, k, l) => {
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
                let v2 = &tmp[3 * j..3 * j + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
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
                    let w6 = w2 * h.h411[(0, 1, k)];
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
                let h22 = hijs2.h22;
                let h32 = hijs2.h32;
                let h33 = hijs2.h33;
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
                let h22 = hijs2.h11;
                let h32 = hijs2.h21;
                let h42 = hijs2.h31;
                let h33 = hijs2.h22;
                let h43 = hijs2.h32;
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
                }
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h432[(i, j, k)] += v4[(i)] * h21[(j, k)];
                        }
                    }
                } // end 192      W1=C7*C3

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
                    h41,
                    h22,
                    h32,
                    h42,
                    h33,
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

		// TODO problem with 223 is in this loop
		// w1 and w2 are okay, check h123 and h432
                let w4 = c5 * c15;
                let w1 = w4 - 1.0;
                let w2 = c8 * c16;
                let w3 = w2 - 1.0;
		// TODO h432 is the problem
                h.h432.print();
                todo!();
                for k in 0..3 {
                    for j in 0..3 {
                        for i in 0..3 {
                            h.h223[(i, j, k)] +=
                                w1 * h.h123[(j, i, k)] - w2 * h.h432[(j, k, i)];
                            h.h332[(i, j, k)] +=
                                w3 * h.h432[(j, i, k)] - w4 * h.h123[(j, k, i)];
                        }
                    }
                }

                h.h223.print();
                todo!();
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

                h.h223.print();
                todo!();
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
                                h42[(j, i)] * h.h411[(1, 1, k)];
                            h.h332[(i, j, k)] +=
                                h31[(i, j)] * h.h411[(1, 2, k)];
                        }
                    }
                } // end 267

                for k in 0..3 {
                    for j in 0..3 {
                        let w1 = (h31[(k, j)] - c3 * v1[(j)] * e23[(k)]);
                        let w2 = (h42[(j, k)] + c3 * v4[(j)] * e23[(k)]);
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

                // h222 is bad at the end of this, but it's only written here
                // h221 is good
                // 223 and 422 are bad
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

                h.h223.print();
                todo!();
                dbg!(333);
                h.h333.print();
            }
        }
        h
    }
}
