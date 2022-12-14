use std::fmt::Display;

use crate::{geom::Geom, htens::Htens, DMat, Siic, Vec3};

use na::Matrix3;
use nalgebra as na;

#[derive(Debug)]
pub struct Hmat {
    pub h11: DMat,
    pub h21: DMat,
    pub h31: DMat,
    pub h22: DMat,
    pub h32: DMat,
    pub h33: DMat,
    pub h41: DMat,
    pub h42: DMat,
    pub h43: DMat,
    pub h44: DMat,
}

/// helper function for calling Hmat::new on `geom` with an [Siic::Stretch] made
/// from atoms `i` and `j`
pub fn hijs1(geom: &Geom, i: usize, j: usize) -> DMat {
    Hmat::new(geom, &Siic::Stretch(i, j)).h11
}

pub fn hijs2(geom: &Geom, i: usize, j: usize, k: usize) -> Hmat {
    Hmat::new(geom, &Siic::Bend(i, j, k))
}

impl Hmat {
    pub fn zeros() -> Self {
        Self {
            h11: DMat::zeros(3, 3),
            h21: DMat::zeros(3, 3),
            h31: DMat::zeros(3, 3),
            h22: DMat::zeros(3, 3),
            h32: DMat::zeros(3, 3),
            h33: DMat::zeros(3, 3),
            h41: DMat::zeros(3, 3),
            h42: DMat::zeros(3, 3),
            h43: DMat::zeros(3, 3),
            h44: DMat::zeros(3, 3),
        }
    }

    // making block matrices to pack into sr in machx
    pub fn new(geom: &Geom, s: &Siic) -> Self {
        use crate::Siic::*;
        let mut h = Self::zeros();
        match s {
            // from HIJS1
            Stretch(i, j) => {
                let v1 = geom.unit(*i, *j);
                let t21 = geom.dist(*i, *j);
                for j in 0..3 {
                    for i in 0..3 {
                        h.h11[(i, j)] = -v1[i] * v1[j];
                    }
                }
                for i in 0..3 {
                    h.h11[(i, i)] += 1.0;
                }
                h.h11 /= t21;
                for j in 0..2 {
                    for i in j + 1..3 {
                        h.h11[(j, i)] = h.h11[(i, j)];
                    }
                }
            }
            // from HIJS2
            Bend(i, j, k) => {
                let tmp = geom.s_vec(s);
                // unpack the s vector
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let h11a = Self::new(geom, &Stretch(*i, *j)).h11;
                let h33a = Self::new(geom, &Stretch(*k, *j)).h11;
                let phi = geom.angle(*i, *j, *k);
                let sphi = phi.sin();
                let ctphi = phi.cos() / sphi;
                let w1 = ctphi;
                let w2 = 1.0 / t21;
                let w3 = w1 * w2;
                let w4 = 1.0 / t23;
                let w5 = w1 * w4;
                // TODO are any of these matrix operations?
                // TODO can any of these loops be combined?
                for j in 0..3 {
                    for i in 0..3 {
                        h.h11[(i, j)] = h11a[(i, j)] * w3
                            - v1[i] * v1[j] * w1
                            - (e21[i] * v1[j] + v1[i] * e21[j]) * w2;
                        h.h33[(i, j)] = h33a[(i, j)] * w5
                            - v3[i] * v3[j] * w1
                            - (e23[i] * v3[j] + v3[i] * e23[j]) * w4;
                    }
                }
                for j in 0..2 {
                    for i in j + 1..3 {
                        h.h11[(j, i)] = h.h11[(i, j)];
                        h.h33[(j, i)] = h.h33[(i, j)];
                    }
                }
                let w3 = 1.0 / (t21 * sphi);
                for j in 0..3 {
                    let w4 = w2 * e21[j] + w1 * v1[j];
                    for i in 0..3 {
                        h.h31[(i, j)] = -h33a[(i, j)] * w3 - v3[i] * w4;
                        h.h21[(i, j)] = -(h.h11[(i, j)] + h.h31[(i, j)]);
                        h.h32[(i, j)] = -(h.h31[(i, j)] + h.h33[(i, j)]);
                    }
                }
                for j in 0..3 {
                    for i in 0..3 {
                        h.h22[(i, j)] = -(h.h21[(j, i)] + h.h32[(i, j)]);
                    }
                }
            }
            // from HIJS6
            Torsion(i, j, k, l) => {
                // unpack the s vector. mine are in the opposite order of the
                // fortran
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
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
                let bp34 = &tmp[3 * l..3 * l + 3];
                // matrices
                h.h11 = Self::mat1(&e23);
                h.h31 = Self::mat1(&e21);
                h.h44 = Self::mat1(&e23);
                h.h42 = Self::mat1(&e34);

                let xx = p2.sin();
                let xy = p3.sin();
                let xx = t21 * xx * xx;
                let xy = t34 * xy * xy;
                let w1 = 1.0 / (t21 * xx);
                let w2 = 1.0 / (t23 * xx);
                let w3 = 1.0 / (t34 * xy);
                let w4 = 1.0 / (t23 * xy);
                for j in 0..3 {
                    for i in 0..3 {
                        h.h11[(i, j)] = -h.h11[(i, j)] * w1;
                        h.h31[(i, j)] *= w2;
                        h.h44[(i, j)] *= w3;
                        h.h42[(i, j)] = -h.h42[(i, j)] * w4;
                    }
                }

                // these are cotans
                let xx = p2.cos() / p2.sin();
                let xy = p3.cos() / p3.sin();
                for i in 0..3 {
                    let w1 = 2.0 * (e21[i] / t21 + bp21[i] * xx);
                    let w2 = e23[i] / t23 + 2.0 * bp23[i] * xx;
                    let w3 = 2.0 * (e34[i] / t34 + bp34[i] * xy);
                    let w4 = e23[i] / t23 - 2.0 * bp32[i] * xy;
                    for j in 0..3 {
                        h.h11[(i, j)] -= w1 * v1[j];
                        h.h31[(i, j)] -= w2 * v1[j];
                        h.h44[(i, j)] -= w3 * v4[j];
                        h.h42[(j, i)] += w4 * v4[j];
                    }
                }

                for j in 0..3 {
                    for i in 0..3 {
                        h.h41[(i, j)] = 0.0;
                        h.h21[(i, j)] = -(h.h11[(i, j)] + h.h31[(i, j)]);
                        h.h43[(i, j)] = -(h.h44[(i, j)] + h.h42[(i, j)]);
                    }
                }

                let x1 = t21 / t23;
                let y1 = t34 / t23;
                let x2 = p2.cos();
                let y2 = p2.sin();
                let x3 = p3.cos();
                let y3 = p3.sin();
                let c1 = x1 * x2 - 1.0;
                let c2 = -x3 * y1;
                let c3 = -x2 / t23;
                let c4 = -x1 * y2;
                let c5 = x1 * x2 / t23;
                let c6 = y1 * y3;
                let c7 = -y1 * x3 / t23;
                for i in 0..3 {
                    let w1 = c3 * e21[i] + c4 * bp22[i] + c5 * e23[i];
                    let w2 = c6 * bp32[i] + c7 * e23[i];
                    for j in 0..3 {
                        h.h22[(i, j)] = c1 * h.h21[(i, j)]
                            + c2 * h.h42[(j, i)]
                            + w1 * v1[j]
                            + w2 * v4[j];
                    }
                }

                for j in 0..3 {
                    for i in 0..3 {
                        h.h32[(i, j)] =
                            -(h.h21[(j, i)] + h.h22[(i, j)] + h.h42[(i, j)]);
                    }
                }

                for j in 0..3 {
                    for i in 0..3 {
                        h.h33[(i, j)] =
                            -(h.h31[(i, j)] + h.h32[(i, j)] + h.h43[(j, i)]);
                    }
                }
            }
            // from HIJS3
            Lin1(i, j, k, l) => {
                let tmp = geom.s_vec(s);
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let th = s.value(geom);
                let e21 = geom.unit(*j, *i);
                let e23 = geom.unit(*j, *k);
                let t21 = geom.dist(*j, *i);
                let t23 = geom.dist(*j, *k);
                let h11a = Self::new(geom, &Stretch(*i, *j)).h11;
                let h33a = Self::new(geom, &Stretch(*k, *j)).h11;
                let ea = geom[*l];
                let d = {
                    let d = ea.dot(&ea);
                    1.0 / d.sqrt()
                };
                let ea = ea * d;
                let tanth = th.tan();
                let costh = th.cos();
                let em = Matrix3::new(
                    0.0, -ea[2], ea[1], ea[2], 0.0, -ea[0], -ea[1], ea[0], 0.0,
                );
                for j in 0..3 {
                    for i in 0..3 {
                        for k in 0..3 {
                            h.h22[(i, j)] += em[(i, k)] * h33a[(k, j)];
                        }
                    }
                }
                // end 2220 loop
                let w1 = 1.0 / t21;
                let w2 = 1.0 / t23;
                for j in 0..3 {
                    for i in 0..3 {
                        h.h11[(i, j)] = (-h11a[(i, j)] * w1
                            + v1[(i)] * v1[(j)])
                            * tanth
                            - (e21[(i)] * v1[(j)] + v1[(i)] * e21[(j)]) * w1;
                        h.h31[(i, j)] =
                            (h.h22[(j, i)] / costh - v3[(i)] * e21[(j)]) / t21
                                + v3[(i)] * v1[(j)] * tanth;
                        h.h33[(i, j)] = (-h33a[(i, j)] * w2
                            + v3[(i)] * v3[(j)])
                            * tanth
                            - (e23[(i)] * v3[(j)] + v3[(i)] * e23[(j)]) * w2;
                        h.h21[(i, j)] = -(h.h11[(i, j)] + h.h31[(i, j)]);
                        h.h32[(i, j)] = -(h.h31[(i, j)] + h.h33[(i, j)]);
                    }
                }
                for j in 0..3 {
                    for i in 0..3 {
                        h.h22[(i, j)] = -(h.h21[(j, i)] + h.h32[(i, j)]);
                    }
                }
            }
            // from HIJS7
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

                // hijs2 call
                let Hmat {
                    h31: hp43,
                    h33: hp44,
                    ..
                } = Hmat::new(geom, &Siic::Bend(*k, *j, *l));

                let v5 = e23.cross(&e24);
                let v6 = e24.cross(&e21);
                let cp31 = Self::mat1(&e24);
                let cp41 = Self::mat1(&e23);
                let cp43 = Self::mat1(&e21);
                let sp = phi.sin();
                let cp = phi.cos();
                let tp = sp / cp;
                let sg = gamma.sin();
                let cg = gamma.cos();
                let tg = sg / cg;
                let c21 = 1.0 / t21;
                let c23 = 1.0 / t23;
                let c24 = 1.0 / t24;
                let ctp = 1.0 / tp;
                let c11 = tg * c21 * c21;
                let c312 = c21 / (cg * sp);
                let c311 = c312 * c23;
                let c313 = c312 * ctp;
                let c411 = c312 * c24;
                let c3 = c23 / sp;
                let c4 = c24 / sp;
                let c331 = t24 * c3;
                let c332 = c331 * tg;
                let c441 = t23 * c4;
                let c442 = c441 * tg;
                let c431 = c3 * c24 / cg;
                let c432 = tg;
                let c434 = tg * c3;
                let c435 = t24 * c3;
                let c436 = c435 * tg;
                for j in 0..3 {
                    for i in j..3 {
                        h.h11[(i, j)] = v1[(j)]
                            * (tg * v1[(i)] - e21[(i)] * c21)
                            - v1[(i)] * e21[(j)] * c21;
                        h.h11[(i, j)] += e21[(i)] * e21[(j)] * c11;
                        if i == j {
                            h.h11[(i, j)] -= c11
                        }
                        h.h11[(j, i)] = h.h11[(i, j)];
                        h.h33[(i, j)] =
                            v3[(i)] * bp4[(j)] * c331 + hp43[(j, i)] * c332;
                        h.h33[(i, j)] += v3[(j)]
                            * (tg * v3[(i)] - e23[(i)] * c23 - bp3[(i)] * ctp);
                        h.h33[(j, i)] = h.h33[(i, j)];
                        h.h44[(i, j)] =
                            v4[(i)] * bp3[(j)] * c441 + hp43[(i, j)] * c442;
                        h.h44[(i, j)] += v4[(j)]
                            * (tg * v4[(i)] - e24[(i)] * c24 - bp4[(i)] * ctp);
                        h.h44[(j, i)] = h.h44[(i, j)];
                    }
                }
                for j in 0..3 {
                    let xj = tg * v1[(j)] - e21[(j)] * c21;
                    for i in 0..3 {
                        h.h31[(i, j)] = v3[(i)] * xj - cp31[(i, j)] * c311;
                        h.h31[(i, j)] = h.h31[(i, j)]
                            - e23[(i)] * v5[(j)] * c311
                            - bp3[(i)] * v5[(j)] * c313;
                        h.h41[(i, j)] = v4[(i)] * xj + cp41[(i, j)] * c411;
                        h.h41[(i, j)] = h.h41[(i, j)]
                            - e24[(i)] * v5[(j)] * c411
                            - bp4[(i)] * v5[(j)] * c313;
                        h.h21[(i, j)] =
                            -(h.h11[(i, j)] + h.h31[(i, j)] + h.h41[(i, j)]);
                        h.h43[(i, j)] = (cp43[(j, i)] - e24[(i)] * v6[(j)])
                            * c431
                            + v3[(j)] * v4[(i)] * c432;
                        h.h43[(i, j)] = h.h43[(i, j)]
                            - v3[(j)] * bp4[(i)] * ctp
                            + e24[(i)] * bp4[(j)] * c434;
                        h.h43[(i, j)] = h.h43[(i, j)]
                            + v4[(i)] * bp4[(j)] * c435
                            + hp44[(i, j)] * c436;
                    }
                }
                for i in 0..3 {
                    for j in 0..3 {
                        h.h32[(i, j)] =
                            -(h.h31[(i, j)] + h.h33[(i, j)] + h.h43[(j, i)]);
                        h.h42[(i, j)] =
                            -(h.h41[(i, j)] + h.h43[(i, j)] + h.h44[(i, j)]);
                        h.h22[(i, j)] =
                            -(h.h21[(j, i)] + h.h32[(i, j)] + h.h42[(i, j)]);
                    }
                }
            }
            // hijs8
            &Linx(i, j, k, l) => {
                let e2 = geom.unit(k, j);
                let e4 = geom.unit(k, l);
                let t32 = geom.dist(j, k);
                // vect2 call
                let s = geom.s_vec(&Siic::Bend(i, j, k));
                let q3 = na::vector![s[3 * k], s[3 * k + 1], s[3 * k + 2]];
                let t = e4.dot(&q3);
                let w = -t32 * t;
                let stre = Siic::Stretch(l, k);
                let bend = Siic::Bend(i, j, k);
                let Hmat { h11: e44, .. } = Self::new(geom, &stre);
                let Hmat {
                    h31: q31, h32: q32, ..
                } = Self::new(geom, &bend);
                let Hmat { h11: e22, .. } =
                    Self::new(geom, &Siic::Stretch(j, k));
                let Htens { h111: q444, .. } = Htens::new(geom, &stre);
                for j in 0..3 {
                    for k in 0..3 {
                        h.h44[(j, k)] = 0.0;
                        for i in 0..3 {
                            h.h44[(j, k)] -= t32 * q444[(i, j, k)] * q3[i];
                        }
                    }
                }
                let Htens {
                    h113: q113,
                    h123: q123,
                    h223: q223,
                    ..
                } = Htens::new(geom, &bend);
                for k in 0..3 {
                    for j in 0..3 {
                        h.h41[(j, k)] = 0.0;
                        h.h42[(j, k)] = 0.0;
                        h.h11[(j, k)] = 0.0;
                        h.h21[(j, k)] = 0.0;
                        h.h22[(j, k)] = w * e22[(j, k)] / t32;
                        for i in 0..3 {
                            h.h11[(j, k)] -= t32 * e4[i] * q113[(j, k, i)];
                            h.h21[(j, k)] -= e4[i]
                                * (e2[j] * q31[(i, k)] + t32 * q123[(k, j, i)]);
                            h.h22[(j, k)] -= e4[i]
                                * (e2[j] * q32[(i, k)]
                                    + e2[k] * q32[(i, j)]
                                    + t32 * q223[(j, k, i)]);
                            h.h41[(j, k)] -= t32 * e44[(i, j)] * q31[(i, k)];
                            h.h42[(j, k)] -= e44[(i, j)]
                                * (t32 * q32[(i, k)] + e2[k] * q3[i]);
                        }
                    }
                }
                for j in 0..3 {
                    for k in 0..3 {
                        h.h31[(j, k)] =
                            -h.h11[(j, k)] - h.h21[(j, k)] - h.h41[(j, k)];
                        h.h32[(j, k)] =
                            -h.h21[(k, j)] - h.h22[(j, k)] - h.h42[(j, k)];
                        h.h43[(j, k)] =
                            -h.h41[(j, k)] - h.h42[(j, k)] - h.h44[(j, k)];
                    }
                }
                for j in 0..3 {
                    for k in 0..3 {
                        h.h33[(j, k)] =
                            -h.h31[(j, k)] - h.h32[(j, k)] - h.h43[(k, j)];
                    }
                }
            }
            // hijs9
            &Liny(k1, k2, k3, k4) => {
                // vect5 call
                let out = Siic::Out(k4, k3, k2, k1);
                let tout = out.value(geom);
                let s = geom.s_vec(&out);
                let e1 = &s[3 * k1..3 * k1 + 3];
                let e2 = &s[3 * k2..3 * k2 + 3];
                let e3 = &s[3 * k3..3 * k3 + 3];
                let e4 = &s[3 * k4..3 * k4 + 3];
                let w = -tout.sin();
                let cosy = tout.cos();
                let Hmat {
                    h11: q44,
                    h21: q34,
                    h31: q24,
                    h41: q14,
                    h22: q33,
                    h32: q23,
                    h42: q13,
                    h33: q22,
                    h43: q12,
                    h44: q11,
                } = Hmat::new(geom, &out);
                for k in 0..3 {
                    for j in 0..3 {
                        h.h22[(j, k)] =
                            -w * e2[(j)] * e2[(k)] - cosy * q22[(k, j)];
                        h.h32[(j, k)] =
                            -w * e3[(j)] * e2[(k)] - cosy * q23[(k, j)];
                        h.h42[(j, k)] =
                            -w * e4[(j)] * e2[(k)] - cosy * q24[(k, j)];
                        h.h33[(j, k)] =
                            -w * e3[(j)] * e3[(k)] - cosy * q33[(k, j)];
                        h.h43[(j, k)] =
                            -w * e4[(j)] * e3[(k)] - cosy * q34[(k, j)];
                        h.h44[(j, k)] =
                            -w * e4[(j)] * e4[(k)] - cosy * q44[(k, j)];
                        h.h41[(j, k)] =
                            -w * e4[(j)] * e1[(k)] - cosy * q14[(k, j)];
                        h.h31[(j, k)] =
                            -w * e3[(j)] * e1[(k)] - cosy * q13[(k, j)];
                        h.h21[(j, k)] =
                            -w * e2[(j)] * e1[(k)] - cosy * q12[(k, j)];
                        h.h11[(j, k)] =
                            -w * e1[(j)] * e1[(k)] - cosy * q11[(k, j)];
                    }
                }
            }
        }
        h
    }

    /// helper function for constructing some kind of matrix
    pub fn mat1(v: &Vec3) -> DMat {
        let mut em = DMat::zeros(3, 3);
        em[(1, 0)] = -v[2];
        em[(2, 0)] = v[1];
        em[(2, 1)] = -v[0];
        em[(0, 1)] = -em[(1, 0)];
        em[(0, 2)] = -em[(2, 0)];
        em[(1, 2)] = -em[(2, 1)];
        em
    }
}

impl Display for Hmat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "H11 =\n{:.6}", self.h11)?;
        writeln!(f, "H21 =\n{:.6}", self.h21)?;
        writeln!(f, "H31 =\n{:.6}", self.h31)?;
        writeln!(f, "H22 =\n{:.6}", self.h22)?;
        writeln!(f, "H32 =\n{:.6}", self.h32)?;
        writeln!(f, "H33 =\n{:.6}", self.h33)?;
        writeln!(f, "H41 =\n{:.6}", self.h41)?;
        writeln!(f, "H42 =\n{:.6}", self.h42)?;
        writeln!(f, "H43 =\n{:.6}", self.h43)?;
        writeln!(f, "H44 =\n{:.6}", self.h44)?;
        Ok(())
    }
}
