use crate::{
    geom::Geom,
    hmat::{hijs1, hijs2, Hmat},
    htens::{h4th1, hijks1, hijks2, Htens},
    splat, Siic,
};
use tensor::tensor4::Tensor4;
use tensor::tensor5::Tensor5;

pub struct Htens4 {
    pub h1111: Tensor4,
    pub h1112: Tensor4,
    pub h1113: Tensor4,
    pub h1122: Tensor4,
    pub h1123: Tensor4,
    pub h1133: Tensor4,
    pub h1222: Tensor4,
    pub h1223: Tensor4,
    pub h1233: Tensor4,
    pub h1333: Tensor4,
    pub h2222: Tensor4,
    pub h2223: Tensor4,
    pub h2233: Tensor4,
    pub h2333: Tensor4,
    pub h3333: Tensor4,
}

impl Htens4 {
    fn new() -> Self {
        Self {
            h1111: Tensor4::zeros(3, 3, 3, 3),
            h1112: Tensor4::zeros(3, 3, 3, 3),
            h1113: Tensor4::zeros(3, 3, 3, 3),
            h1122: Tensor4::zeros(3, 3, 3, 3),
            h1123: Tensor4::zeros(3, 3, 3, 3),
            h1133: Tensor4::zeros(3, 3, 3, 3),
            h1222: Tensor4::zeros(3, 3, 3, 3),
            h1223: Tensor4::zeros(3, 3, 3, 3),
            h1233: Tensor4::zeros(3, 3, 3, 3),
            h1333: Tensor4::zeros(3, 3, 3, 3),
            h2222: Tensor4::zeros(3, 3, 3, 3),
            h2223: Tensor4::zeros(3, 3, 3, 3),
            h2233: Tensor4::zeros(3, 3, 3, 3),
            h2333: Tensor4::zeros(3, 3, 3, 3),
            h3333: Tensor4::zeros(3, 3, 3, 3),
        }
    }
}

#[macro_export]
macro_rules! foreach {
    ($i:ident, $j:ident, $k:ident, $($stmt:stmt;)*) => {
	for $i in 0..3 {
	    for $j in 0..3 {
		for $k in 0..3 {
		    $($stmt)*
		}
	    }
	}
    };
    ($i:ident, $j:ident, $k:ident, $l:ident, $($stmt:stmt;)*) => {
	for $i in 0..3 {
	    for $j in 0..3 {
		for $k in 0..3 {
		    for $l in 0..3 {
			$($stmt)*
		    }
		}
	    }
	}
    };
}

fn h5th1(geom: &Geom, k1: usize, k2: usize) -> Tensor5 {
    let mut h = Tensor5::zeros(3, 3, 3, 3, 3);
    let (v1, t21) = geom.vect1(k2, k1);
    let h11 = hijs1(geom, k1, k2);
    let h111 = hijks1(geom, k1, k2);
    let h1111 = h4th1(geom, k1, k2);
    for m in 0..3 {
        for l in 0..3 {
            for k in 0..=l {
                for j in 0..=k {
                    for i in 0..=j {
                        let a = h11[(i, l)] * h111[(k, j, m)]
                            + h11[(j, l)] * h111[(k, i, m)]
                            + h11[(i, j)] * h111[(k, l, m)];
                        let b = h111[(i, l, m)] * h11[(k, j)]
                            + h111[(j, l, m)] * h11[(k, i)]
                            + h111[(i, j, m)] * h11[(k, l)];
                        let c = h11[(i, m)] * h111[(k, j, l)]
                            + h11[(j, m)] * h111[(k, i, l)];
                        let d = h11[(k, m)] * h111[(i, j, l)]
                            + h11[(l, m)] * h111[(k, i, j)];
                        let e = v1[(i)] * h1111[(j, k, l, m)]
                            + v1[(j)] * h1111[(i, k, l, m)]
                            + v1[(k)] * h1111[(i, j, l, m)];
                        let f = v1[(l)] * h1111[(i, j, k, m)]
                            + v1[(m)] * h1111[(i, j, k, l)];
                        h[(i, j, k, l, m)] = -(a + b + c + d + e + f) / t21;
                    }
                }
            }
        }
    }
    // I think you could do this at the end of the loops above, but not sure

    // NOTE inlined fill4a because calling fill4a on each element of h.data
    // fills the last four indices instead of the first four as needed.
    let ny = 3;
    for mo in 0..3 {
        for q in 0..ny {
            for p in 0..=q {
                for n in 0..=p {
                    for m in 0..=n {
                        h[(n, m, p, q, mo)] = h[(m, n, p, q, mo)];
                        h[(n, p, m, q, mo)] = h[(m, n, p, q, mo)];
                        h[(n, p, q, m, mo)] = h[(m, n, p, q, mo)];
                        h[(m, p, n, q, mo)] = h[(m, n, p, q, mo)];
                        h[(p, m, n, q, mo)] = h[(m, n, p, q, mo)];
                        h[(p, n, m, q, mo)] = h[(m, n, p, q, mo)];
                        h[(p, n, q, m, mo)] = h[(m, n, p, q, mo)];
                        h[(m, p, q, n, mo)] = h[(m, n, p, q, mo)];
                        h[(p, m, q, n, mo)] = h[(m, n, p, q, mo)];
                        h[(p, q, m, n, mo)] = h[(m, n, p, q, mo)];
                        h[(p, q, n, m, mo)] = h[(m, n, p, q, mo)];
                        h[(m, n, q, p, mo)] = h[(m, n, p, q, mo)];
                        h[(n, m, q, p, mo)] = h[(m, n, p, q, mo)];
                        h[(n, q, m, p, mo)] = h[(m, n, p, q, mo)];
                        h[(n, q, p, m, mo)] = h[(m, n, p, q, mo)];
                        h[(m, q, n, p, mo)] = h[(m, n, p, q, mo)];
                        h[(q, m, n, p, mo)] = h[(m, n, p, q, mo)];
                        h[(q, n, m, p, mo)] = h[(m, n, p, q, mo)];
                        h[(q, n, p, m, mo)] = h[(m, n, p, q, mo)];
                        h[(m, q, p, n, mo)] = h[(m, n, p, q, mo)];
                        h[(q, m, p, n, mo)] = h[(m, n, p, q, mo)];
                        h[(q, p, m, n, mo)] = h[(m, n, p, q, mo)];
                        h[(q, p, n, m, mo)] = h[(m, n, p, q, mo)];
                    }
                }
            }
        }
    }
    h
}

pub(crate) fn h4th2(geom: &Geom, k1: usize, k2: usize, k3: usize) -> Htens4 {
    let bend = Siic::Bend(k1, k2, k3);
    let phi = bend.value(geom);
    let s = geom.s_vec(&bend);
    splat!(s, v1 => k1, v3 => k3);
    let Hmat { h11, h31, h33, .. } = hijs2(geom, k1, k2, k3);
    let Htens {
        h111,
        h113,
        h331,
        h333,
        ..
    } = hijks2(geom, k1, k2, k3);
    let cscp = 1.0 / phi.sin();
    let cotp = phi.cos() * cscp;
    let mut h = Htens4::new();
    for l in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    h.h1111[(i, j, k, l)] =
                        v1[(i)] * v1[(j)] * v1[(k)] * v1[(l)]
                            - h111[(j, k, l)] * v1[(i)]
                            - h111[(i, j, k)] * v1[(l)]
                            - h111[(i, j, l)] * v1[(k)]
                            - h111[(i, k, l)] * v1[(j)]
                            - h11[(i, j)] * h11[(k, l)]
                            - h11[(i, l)] * h11[(k, j)]
                            - h11[(i, k)] * h11[(j, l)];
                    h.h1111[(i, j, k, l)] = h.h1111[(i, j, k, l)] * cotp
                        + h11[(i, j)] * v1[(k)] * v1[(l)]
                        + h11[(i, k)] * v1[(j)] * v1[(l)]
                        + h11[(i, l)] * v1[(j)] * v1[(k)]
                        + h11[(j, k)] * v1[(i)] * v1[(l)]
                        + h11[(j, l)] * v1[(k)] * v1[(i)]
                        + h11[(k, l)] * v1[(j)] * v1[(i)];
                    h.h1113[(i, j, k, l)] =
                        v1[(i)] * v1[(j)] * v1[(k)] * v3[(l)]
                            - h113[(j, k, l)] * v1[(i)]
                            - h111[(i, j, k)] * v3[(l)]
                            - h113[(i, j, l)] * v1[(k)]
                            - h113[(i, k, l)] * v1[(j)]
                            - h11[(i, j)] * h31[(l, k)]
                            - h31[(l, i)] * h11[(k, j)]
                            - h11[(i, k)] * h31[(l, j)];
                    h.h1113[(i, j, k, l)] = h.h1113[(i, j, k, l)] * cotp
                        + h11[(i, j)] * v1[(k)] * v3[(l)]
                        + h11[(i, k)] * v1[(j)] * v3[(l)]
                        + h31[(l, i)] * v1[(j)] * v1[(k)]
                        + h11[(j, k)] * v1[(i)] * v3[(l)]
                        + h31[(l, j)] * v1[(k)] * v1[(i)]
                        + h31[(l, k)] * v1[(j)] * v1[(i)];
                    h.h1133[(i, j, k, l)] =
                        v1[(i)] * v1[(j)] * v3[(k)] * v3[(l)]
                            - h331[(l, k, j)] * v1[(i)]
                            - h113[(i, j, k)] * v3[(l)]
                            - h113[(i, j, l)] * v3[(k)]
                            - h331[(l, k, i)] * v1[(j)]
                            - h11[(i, j)] * h33[(l, k)]
                            - h31[(l, i)] * h31[(k, j)]
                            - h31[(k, i)] * h31[(l, j)];
                    h.h1133[(i, j, k, l)] = h.h1133[(i, j, k, l)] * cotp
                        + h11[(i, j)] * v3[(k)] * v3[(l)]
                        + h31[(k, i)] * v1[(j)] * v3[(l)]
                        + h31[(l, i)] * v1[(j)] * v3[(k)]
                        + h31[(k, j)] * v1[(i)] * v3[(l)]
                        + h31[(l, j)] * v3[(k)] * v1[(i)]
                        + h33[(l, k)] * v1[(j)] * v1[(i)];
                    h.h1333[(i, j, k, l)] =
                        v1[(i)] * v3[(j)] * v3[(k)] * v3[(l)]
                            - h333[(l, k, j)] * v1[(i)]
                            - h331[(k, j, i)] * v3[(l)]
                            - h331[(l, j, i)] * v3[(k)]
                            - h331[(l, k, i)] * v3[(j)]
                            - h31[(j, i)] * h33[(l, k)]
                            - h31[(l, i)] * h33[(k, j)]
                            - h31[(k, i)] * h33[(l, j)];
                    h.h1333[(i, j, k, l)] = h.h1333[(i, j, k, l)] * cotp
                        + h31[(j, i)] * v3[(k)] * v3[(l)]
                        + h31[(k, i)] * v3[(j)] * v3[(l)]
                        + h31[(l, i)] * v3[(j)] * v3[(k)]
                        + h33[(k, j)] * v1[(i)] * v3[(l)]
                        + h33[(l, j)] * v3[(k)] * v1[(i)]
                        + h33[(l, k)] * v3[(j)] * v1[(i)];
                    h.h3333[(i, j, k, l)] =
                        v3[(i)] * v3[(j)] * v3[(k)] * v3[(l)]
                            - h333[(l, k, j)] * v3[(i)]
                            - h333[(k, j, i)] * v3[(l)]
                            - h333[(l, j, i)] * v3[(k)]
                            - h333[(l, k, i)] * v3[(j)]
                            - h33[(j, i)] * h33[(l, k)]
                            - h33[(l, i)] * h33[(k, j)]
                            - h33[(k, i)] * h33[(l, j)];
                    h.h3333[(i, j, k, l)] = h.h3333[(i, j, k, l)] * cotp
                        + h33[(j, i)] * v3[(k)] * v3[(l)]
                        + h33[(k, i)] * v3[(j)] * v3[(l)]
                        + h33[(l, i)] * v3[(j)] * v3[(k)]
                        + h33[(k, j)] * v3[(i)] * v3[(l)]
                        + h33[(l, j)] * v3[(k)] * v3[(i)]
                        + h33[(l, k)] * v3[(j)] * v3[(i)];
                }
            }
        }
    }
    let (v1, _) = geom.vect1(k2, k1);
    let (v3, _) = geom.vect1(k2, k3);
    let h11 = hijs1(geom, k1, k2);
    let h33 = hijs1(geom, k3, k2);
    let h111 = hijks1(geom, k1, k2);
    let h333 = hijks1(geom, k3, k2);
    let q1111 = h4th1(geom, k1, k2);
    let q3333 = h4th1(geom, k3, k2);
    let q11111 = h5th1(geom, k1, k2);
    let q33333 = h5th1(geom, k3, k2);

    for m in 0..3 {
        for l in 0..3 {
            for k in 0..3 {
                for j in 0..3 {
                    for i in 0..3 {
                        h.h1111[(i, j, k, l)] -=
                            cscp * q11111[(m, i, j, k, l)] * v3[(m)];
                        h.h1113[(i, j, k, l)] -=
                            cscp * q1111[(m, i, j, k)] * h33[(l, m)];
                        h.h1133[(i, j, k, l)] -=
                            cscp * h111[(m, i, j)] * h333[(k, l, m)];
                        h.h1333[(i, j, k, l)] -=
                            cscp * h11[(m, i)] * q3333[(j, k, l, m)];
                        h.h3333[(i, j, k, l)] -=
                            cscp * v1[(m)] * q33333[(i, j, k, l, m)];
                    }
                }
            }
        }
    }

    // I guess these have to be separate since each uses the previous, but maybe
    // not
    foreach!(l, k, j, i,
         h.h1112[(i, j, k, l)] = -h.h1111[(i, j, k, l)] - h.h1113[(i, j, k, l)];
         h.h1123[(i, j, k, l)] = -h.h1113[(i, j, k, l)] - h.h1133[(i, j, k, l)];
         h.h1233[(i, j, k, l)] = -h.h1133[(i, j, k, l)] - h.h1333[(i, j, k, l)];
         h.h2333[(i, j, k, l)] = -h.h1333[(i, j, k, l)] - h.h3333[(i, j, k, l)];
    );

    foreach!(l, k, j, i,
         h.h1122[(i, j, k, l)] = -h.h1112[(i, j, k, l)] - h.h1123[(i, j, l, k)];
         h.h1223[(i, j, k, l)] = -h.h1123[(i, j, k, l)] - h.h1233[(i, k, j, l)];
         h.h2233[(i, j, k, l)] = -h.h1233[(i, j, k, l)] - h.h2333[(j, i, k, l)];
    );

    foreach!(l, k, j, i,
         h.h1222[(i,j,k,l)]=-h.h1122[(i,j,k,l)]-h.h1223[(i,k,l,j)];
         h.h2223[(i,j,k,l)]=-h.h1223[(i,j,k,l)]-h.h2233[(j,k,i,l)];
    );

    foreach!(l, k, j, i,
         h.h2222[(i, j, k, l)] = -h.h1222[(i, j, k, l)] - h.h2223[(j, k, l, i)];
    );

    h
}
