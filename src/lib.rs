use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
};

mod geom;

use geom::Geom;
use nalgebra as na;
use regex::Regex;

/// from https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
pub const ANGBOHR: f64 = 0.5291_772_109;
const DEGRAD: f64 = 180.0 / std::f64::consts::PI;
pub const DEBUG: bool = false;
const TOLDISP: f64 = 1e-14;

type Vec3 = na::Vector3<f64>;
pub type DMat = na::DMatrix<f64>;
pub type DVec = na::DVector<f64>;

#[derive(Debug, PartialEq)]
pub enum Siic {
    Stretch(usize, usize),
    /// central atom is second like normal people would expect
    Bend(usize, usize, usize),
    /// angle between planes formed by i, j, k and j, k, l
    Torsion(usize, usize, usize, usize),
}

impl Siic {
    pub fn value(&self, geom: &Geom) -> f64 {
        use Siic::*;
        match self {
            Stretch(a, b) => Intder::dist(geom, *a, *b) * ANGBOHR,
            Bend(a, b, c) => Intder::angle(geom, *a, *b, *c),
	    // the manual shows two ways to do this (p. 18), this is the second
	    // (cos) way
            Torsion(a, b, c, d) => {
                let e_ba = Intder::unit(geom, *b, *a);
                let e_cb = Intder::unit(geom, *c, *b);
                let e_cd = Intder::unit(geom, *c, *d);
                let phi_abc = Intder::angle(geom, *a, *b, *c);
                let phi_bcd = Intder::angle(geom, *b, *c, *d);
                let tau = (e_ba.cross(&-e_cb)).dot(&e_cb.cross(&e_cd))
                    / (phi_abc.sin() * phi_bcd.sin());
                let tau_size = tau.abs() - 1.0;
                // adjust tau back to 1 if it's barely off. this threshold could
                // probably be adjusted. I saw deviation of the size 4e-16 but
                // gave a slightly more generous buffer
                let tau = if tau_size > 0.0 && tau_size < 1e-14 {
                    f64::copysign(1.0, tau)
                } else {
                    tau
                };
                let it = tau.acos();
                if f64::is_nan(it) {
                    panic!("NaN found on acos of {}", tau);
                }
                it
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Intder {
    input_options: Vec<usize>,
    simple_internals: Vec<Siic>,
    pub symmetry_internals: Vec<Vec<f64>>,
    pub geom: Geom,
    pub disps: Vec<Vec<f64>>,
}

impl Intder {
    pub fn new() -> Self {
        Intder {
            input_options: Vec::new(),
            simple_internals: Vec::new(),
            symmetry_internals: Vec::new(),
            geom: Geom::new(),
            disps: Vec::new(),
        }
    }

    pub fn load(infile: &str) -> Self {
        let f = match File::open(infile) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("failed to open infile '{}'", infile);
                std::process::exit(1);
            }
        };

        let siic = Regex::new(r"STRE|BEND|TORS|OUT|LIN|SPF|RCOM").unwrap();
        // something like "    1   1   1.000000000   2   1.000000000"
        let syic =
            Regex::new(r"^\s*(\d+\s+)(\d+\s+[0-9-]\.[0-9]+(\s+|$))+").unwrap();
        let zero = Regex::new(r"^\s*0$").unwrap();
        // just a bunch of integers like 3    3    3    0    0    3 ..."
        let iops = Regex::new(r"^\s*(\d+(\s+|$))+$").unwrap();
        let geom = Regex::new(r"\s*([0-9-]+\.[0-9]+(\s+|$)){3}").unwrap();

        let mut intder = Intder::new();
        let reader = BufReader::new(f);
        let mut in_disps = false;
        let mut disp_tmp = vec![];
        for line in reader.lines().flatten() {
            if line.contains("# INTDER #") || line.len() == 0 {
                continue;
            } else if in_disps {
                // build up tmp until we hit a zero just like with syics
                if zero.is_match(&line) {
                    intder.disps.push(disp_tmp.clone());
                    disp_tmp = vec![0.0; intder.simple_internals.len()];
                    continue;
                }
                let sp: Vec<&str> = line.split_whitespace().collect();
                disp_tmp[sp[0].parse::<usize>().unwrap() - 1] =
                    sp[1].parse::<f64>().unwrap();
            } else if siic.is_match(&line) {
                let sp: Vec<&str> = line.split_whitespace().collect();
                intder.simple_internals.push(match sp[0] {
                    "STRE" => Siic::Stretch(
                        sp[1].parse::<usize>().unwrap() - 1,
                        sp[2].parse::<usize>().unwrap() - 1,
                    ),
                    "BEND" => Siic::Bend(
                        sp[1].parse::<usize>().unwrap() - 1,
                        sp[2].parse::<usize>().unwrap() - 1,
                        sp[3].parse::<usize>().unwrap() - 1,
                    ),
                    "TORS" => Siic::Torsion(
                        sp[1].parse::<usize>().unwrap() - 1,
                        sp[2].parse::<usize>().unwrap() - 1,
                        sp[3].parse::<usize>().unwrap() - 1,
                        sp[4].parse::<usize>().unwrap() - 1,
                    ),
                    e => {
                        eprintln!("unknown coordinate type '{}'", e);
                        std::process::exit(1);
                    }
                });
            } else if syic.is_match(&line) {
                // this has to come after the simple internals
                let mut tmp = vec![0.0; intder.simple_internals.len()];
                let mut sp = line.split_whitespace();
                sp.next();
                let mut idx = usize::default();
                let mut i_max = 0;
                for (i, c) in sp.enumerate() {
                    if i % 2 == 0 {
                        idx = c.parse::<usize>().unwrap() - 1;
                    } else {
                        tmp[idx] = c.parse().unwrap();
                    }
                    i_max += 1;
                }
                match i_max {
                    2 => (),
                    4 => {
                        for t in &mut tmp {
                            *t *= std::f64::consts::SQRT_2 / 2.0;
                        }
                    }
                    _ => {
                        panic!("unmatched i_max value of {}", i_max);
                    }
                }
                intder.symmetry_internals.push(tmp);
            } else if zero.is_match(&line) {
                continue;
            } else if iops.is_match(&line) {
                intder.input_options.extend(
                    line.split_whitespace()
                        .map(|x| x.parse::<usize>().unwrap()),
                );
            } else if geom.is_match(&line) {
                if let [x, y, z] = line
                    .split_whitespace()
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()[..]
                {
                    intder.geom.push(na::Vector3::new(x, y, z));
                }
            } else if line.contains("DISP") {
                in_disps = true;
                disp_tmp = vec![0.0; intder.simple_internals.len()];
            }
        }
        intder
    }

    pub fn print_geom(&self) {
        for atom in &self.geom {
            for c in &atom {
                print!("{:20.10}", c);
            }
            println!();
        }
    }

    /// print the simple internal coordinate values, assuming that vals are in
    /// the same order as self.simple_internals for unit purposes
    pub fn print_simple(&self, vals: &[f64]) {
        for (i, v) in vals.iter().enumerate() {
            if let Siic::Bend(_, _, _) = self.simple_internals[i] {
                println!("{:5}{:>18.10}", i, v * DEGRAD);
            } else {
                println!("{:5}{:>18.10}", i, v);
            }
        }
    }

    /// print the symmetry internal coordinate values
    pub fn print_symmetry(&self, vals: &[f64]) {
        for (i, v) in vals.iter().enumerate() {
            println!("{:5}{:>18.10}", i, v);
        }
    }

    /// currently returns a vector of simple internal values in Ångstroms or
    /// radians
    pub fn simple_values(&self, geom: &Geom) -> Vec<f64> {
        let mut ret = Vec::new();
        for s in &self.simple_internals {
            ret.push(s.value(&geom));
        }
        ret
    }

    /// currently returns a vector of symmetry internal values in Ångstroms or
    /// radians
    pub fn symmetry_values(&self, geom: &Geom) -> Vec<f64> {
        let mut ret = Vec::new();
        let siics = self.simple_values(&geom);
        for sic in &self.symmetry_internals {
            let mut sum = f64::default();
            for (i, s) in sic.iter().enumerate() {
                sum += s * siics[i];
            }
            ret.push(sum);
        }
        ret
    }

    /// return the unit vector from atom i to atom j
    fn unit(geom: &Geom, i: usize, j: usize) -> Vec3 {
        let diff = geom[j] - geom[i];
        diff / diff.magnitude()
    }

    /// distance between atoms i and j
    fn dist(geom: &Geom, i: usize, j: usize) -> f64 {
        (geom[j] - geom[i]).magnitude()
    }

    /// angle in radians between atoms i, j, and k, where j is the central atom
    fn angle(geom: &Geom, i: usize, j: usize, k: usize) -> f64 {
        let e_ji = Self::unit(geom, j, i);
        let e_jk = Self::unit(geom, j, k);
        (e_ji.dot(&e_jk)).acos()
    }

    pub fn s_vec(geom: &Geom, ic: &Siic, len: usize) -> Vec<f64> {
        let mut tmp = vec![0.0; len];
        // TODO write up the math from McIntosh78 and Molecular Vibrations
        match ic {
            Siic::Stretch(a, b) => {
                let e_12 = Self::unit(geom, *a, *b);
                for i in 0..3 {
                    tmp[3 * a + i] = -e_12[i % 3];
                    tmp[3 * b + i] = e_12[i % 3];
                }
            }
            Siic::Bend(a, b, c) => {
                let phi = ic.value(geom);
                // NOTE: letting 3 (the number 3, but atom `b`) be the central
                // atom in line with Mol. Vib. notation
                let e_31 = Self::unit(geom, *b, *a);
                let e_32 = Self::unit(geom, *b, *c);
                let r_31 = Self::dist(geom, *b, *a) * ANGBOHR;
                let r_32 = Self::dist(geom, *b, *c) * ANGBOHR;
                let pc = phi.cos();
                let ps = phi.sin();
                let s_t1 = (pc * e_31 - e_32) / (r_31 * ps);
                let s_t2 = (pc * e_32 - e_31) / (r_32 * ps);
                let s_t3 = ((r_31 - r_32 * pc) * e_31
                    + (r_32 - r_31 * pc) * e_32)
                    / (r_31 * r_32 * ps);
                for i in 0..3 {
                    tmp[3 * a + i] = s_t1[i % 3];
                    tmp[3 * b + i] = s_t3[i % 3];
                    tmp[3 * c + i] = s_t2[i % 3];
                }
            }
            Siic::Torsion(_a, _b, _c, _d) => {
                todo!();
                // for i in 0..3 {
                //     tmp[3 * a + i] = s_t1[i % 3];
                //     tmp[3 * b + i] = s_t3[i % 3];
                //     tmp[3 * c + i] = s_t2[i % 3];
                //     tmp[3 * d + i] = s_t2[i % 3];
                // }
            }
        }
        tmp
    }

    /// return the B matrix in simple internal coordinates
    pub fn b_matrix(&self, geom: &Geom) -> DMat {
        let geom_len = 3 * geom.len();
        let mut b_mat = Vec::new();
        for ic in &self.simple_internals {
            b_mat.extend(Self::s_vec(geom, ic, geom_len));
        }
        DMat::from_row_slice(self.simple_internals.len(), geom_len, &b_mat)
    }

    /// return the symmetry internal coordinate B matrix by computing the simple
    /// internal B and converting it
    pub fn sym_b_matrix(&self, geom: &Geom) -> DMat {
        let b = self.b_matrix(geom);
        let (r, c) = b.shape();
        let mut bs = DMat::zeros(r, c);
        for j in 0..3 * geom.len() {
            for i in 0..self.symmetry_internals.len() {
                let u_i = DVec::from(self.symmetry_internals[i].clone());
                let b_j = DVec::from(b.column(j));
                bs[(i, j)] = u_i.dot(&b_j);
            }
        }
        bs
    }

    /// Let D = BBᵀ and return A = BᵀD⁻¹
    pub fn a_matrix(b: &DMat) -> DMat {
        let d = b * b.transpose();
        let chol =
            na::Cholesky::new(d).expect("Cholesky decomposition of D failed");
        b.transpose() * chol.inverse()
    }

    /// print the initial geometry stuff
    fn print_init(&self) {
        let simple_vals = self.simple_values(&self.geom);
        let sic_vals = self.symmetry_values(&self.geom);
        println!();
        println!("NUCLEAR CARTESIAN COORDINATES (BOHR)\n");
        self.print_geom();
        println!();
        println!(
        "VALUES OF SIMPLE INTERNAL COORDINATES (ANG. or DEG.) FOR REFERENCE \
	     GEOMETRY\n"
    );
        self.print_simple(&simple_vals);
        println!();
        println!(
        "VALUES OF SYMMETRY INTERNAL COORDINATES (ANG. or RAD.) FOR REFERENCE \
	     GEOMETRY\n"
    );
        self.print_symmetry(&sic_vals);
        println!();
        println!();
    }

    pub fn convert_disps(&self) -> Vec<DVec> {
        if DEBUG {
            self.print_init();
        }
        let mut ret = Vec::new();
        for (i, disp) in self.disps.iter().enumerate() {
            let mut sic_current = DVec::from(self.symmetry_values(&self.geom));
            let mut cart_current: DVec = self.geom.clone().into();

            let disp = DVec::from(disp.clone());
            let sic_desired = &sic_current + &disp;

            if DEBUG {
                println!("DISPLACEMENT{:5}\n", i);
                println!("INTERNAL DISPLACEMENTS\n");
                for (i, d) in disp.iter().enumerate() {
                    if *d != 0.0 {
                        println!("{i:5}{d:20.10}");
                    }
                }
                println!();
                println!("SYMMETRY INTERNAL COORDINATE FINAL VALUES\n");
                self.print_symmetry(sic_desired.as_slice());
                println!();
            }

            let mut iter = 1;
            while (&sic_current - &sic_desired).abs().max() > TOLDISP {
                let b_sym = self.sym_b_matrix(&Geom::from(&cart_current));
                let d = &b_sym * b_sym.transpose();
                let a = Intder::a_matrix(&b_sym);

                if DEBUG {
                    println!(
                        "ITER={:5} MAX INTERNAL DEVIATION = {:.4e}",
                        iter,
                        (&sic_current - &sic_desired).abs().max()
                    );
                    println!("B*BT MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
                    println!("{}", d);

                    println!(
                        "DETERMINANT OF B*BT MATRIX={:8.3}",
                        d.determinant()
                    );

                    println!();
                    println!("A MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
                    println!("{:.8}", a);
                    println!();
                }

                let step = a * (&sic_desired - &sic_current);
                cart_current += step / ANGBOHR;

                sic_current = DVec::from(
                    self.symmetry_values(&Geom::from(&cart_current)),
                );

                iter += 1;
            }

            if DEBUG {
                println!("NEW CARTESIAN GEOMETRY (BOHR)\n");
                Self::print_cart(&mut std::io::stdout(), &cart_current);
                println!();
            }
            ret.push(cart_current);
        }
        ret
    }

    pub fn print_cart<W: Write>(w: &mut W, cart: &DVec) {
        for i in 0..cart.len() / 3 {
            for j in 0..3 {
                write!(w, "{:20.10}", cart[3 * i + j]).unwrap();
            }
            writeln!(w).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    const S: f64 = std::f64::consts::SQRT_2 / 2.;

    #[test]
    fn test_load() {
        let got = Intder::load("testfiles/intder.in");
        let want = Intder {
            input_options: vec![
                3, 3, 3, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 14,
            ],
            simple_internals: vec![
                Siic::Stretch(0, 1),
                Siic::Stretch(1, 2),
                Siic::Bend(0, 1, 2),
            ],
            symmetry_internals: vec![
                vec![S, S, 0.],
                vec![0., 0., 1.],
                vec![S, -S, 0.],
            ],
            geom: Geom(vec![
                na::Vector3::new(
                    0.000000000000,
                    1.431390244079,
                    0.986041163966,
                ),
                na::Vector3::new(
                    0.000000000000,
                    0.000000000000,
                    -0.124238450265,
                ),
                na::Vector3::new(
                    0.000000000000,
                    -1.431390244079,
                    0.986041163966,
                ),
            ]),
            disps: vec![
                vec![0.005, 0.0, 0.0],
                vec![0.0, 0.005, 0.0],
                vec![0.0, 0.0, 0.005],
                vec![-0.005, -0.005, -0.01],
                vec![-0.005, -0.005, 0.0],
                vec![-0.005, -0.005, 0.010],
                vec![-0.005, -0.010, 0.0],
                vec![-0.005, -0.015, 0.0],
                vec![0.0, 0.0, 0.0],
            ],
        };
        assert_eq!(got, want);
    }

    // MACHB:S is initial simple internals

    #[test]
    fn test_initial_values_simple() {
        let tests = vec![
            (
                "testfiles/intder.in",
                vec![0.9586143145, 0.9586143145, 1.8221415968],
            ),
            (
                "testfiles/c7h2.in",
                vec![
                    1.4260535407,
                    1.4260535407,
                    1.3992766813,
                    1.3992766813,
                    2.6090029486,
                    2.6090029486,
                    3.6728481977,
                    3.6728481977,
                    2.5991099760,
                    2.5991099760,
                    2.5961248359,
                    2.5961248359,
                    2.5945738184,
                    2.5945738184,
                    1.0819561376,
                    3.1415926536,
                    3.1415926536,
                    3.1415926536,
                    3.1415926536,
                    3.1415926536,
                    3.1415926536,
                ],
            ),
        ];
        for test in tests {
            let intder = Intder::load(test.0);
            let got = intder.simple_values(&intder.geom);
            assert_abs_diff_eq!(
                DVec::from(got),
                DVec::from(test.1),
                epsilon = 3e-7
            );
        }
    }

    #[test]
    fn test_initial_values_symmetry() {
        let intder = Intder::load("testfiles/intder.in");
        let got = intder.symmetry_values(&intder.geom);
        let got = got.as_slice();
        let want = vec![1.3556853647, 1.8221415968, 0.0000000000];
        let want = want.as_slice();
        assert_abs_diff_eq!(got, want, epsilon = 1e-7);
    }

    #[test]
    fn test_b_matrix() {
        let intder = Intder::load("testfiles/intder.in");
        let want = DMat::from_row_slice(
            3,
            9,
            &vec![
                // row 1
                0.0,
                0.7901604711325243,
                0.61290001620136003,
                -0.0,
                -0.7901604711325243,
                -0.61290001620136003,
                0.0,
                0.0,
                0.0,
                // row 2
                0.0,
                0.0,
                0.0,
                0.0,
                0.7901604711325243,
                -0.61290001620136003,
                -0.0,
                -0.7901604711325243,
                0.61290001620136003,
                // row 3
                -0.0,
                0.63936038937065331,
                -0.82427360602746957,
                0.0,
                0.0,
                1.6485472120549391,
                -0.0,
                -0.63936038937065331,
                -0.82427360602746957,
            ],
        );
        let got = intder.b_matrix(&intder.geom);
        assert_abs_diff_eq!(got, want, epsilon = 2e-7);
    }

    #[test]
    fn test_sym_b() {
        let intder = Intder::load("testfiles/intder.in");
        let got = intder.sym_b_matrix(&intder.geom);
        let want = DMat::from_row_slice(
            3,
            9,
            &vec![
                // row 1
                0.0,
                0.55872782736336513,
                0.4333857576453265,
                0.0,
                0.0,
                -0.86677151529065299,
                0.0,
                -0.55872782736336513,
                0.4333857576453265,
                // row 2
                0.0,
                0.63936038937065331,
                -0.82427360602746957,
                0.0,
                0.0,
                1.6485472120549391,
                0.0,
                -0.63936038937065331,
                -0.82427360602746957,
                // row 3
                0.0,
                0.55872782736336513,
                0.4333857576453265,
                0.0,
                -1.1174556547267303,
                0.0,
                0.0,
                0.55872782736336513,
                -0.4333857576453265,
            ],
        );
        assert_abs_diff_eq!(got, want, epsilon = 2e-7);
    }

    #[test]
    fn test_a_matrix() {
        let want = DMat::from_row_slice(
            9,
            3,
            &vec![
                0.0,
                0.0,
                0.0,
                0.55872782736336513,
                0.29376736196418923,
                0.24846624860790423,
                0.14446191921510884,
                -0.12624318866430007,
                0.19272663384313321,
                0.0,
                0.0,
                0.0,
                -2.0677083281253247e-17,
                -6.0369866375237933e-18,
                -0.49693249721580846,
                -0.28892383843021768,
                0.25248637732860013,
                -7.1323182117049485e-18,
                0.0,
                0.0,
                0.0,
                -0.55872782736336513,
                -0.29376736196418923,
                0.24846624860790423,
                0.14446191921510884,
                -0.12624318866430007,
                -0.19272663384313321,
            ],
        );
        let intder = Intder::load("testfiles/intder.in");
        let got = Intder::a_matrix(&intder.sym_b_matrix(&intder.geom));
        assert_abs_diff_eq!(got, want, epsilon = 3e-8);
    }

    /// load a file where each line is a DVec
    fn load_geoms(filename: &str) -> Vec<DVec> {
        let f = std::fs::File::open(filename).unwrap();
        let lines = BufReader::new(f).lines().flatten();
        let mut ret = Vec::new();
        for line in lines {
            if !line.is_empty() {
                ret.push(DVec::from(
                    line.split_whitespace()
                        .map(|x| x.parse().unwrap())
                        .collect::<Vec<_>>(),
                ));
            }
        }
        ret
    }

    #[test]
    fn test_convert_disps() {
        struct Test<'a> {
            infile: &'a str,
            wantfile: &'a str,
        }
        let tests = vec![
            Test {
                infile: "testfiles/h2o.in",
                wantfile: "testfiles/h2o.small.07",
            },
            // Test {
            //     infile: "testfiles/c7h2.in",
            //     wantfile: "testfiles/c7h2.small.07",
            // },
        ];
        for test in tests {
            let intder = Intder::load(test.infile);
            let got = intder.convert_disps();
            let want = load_geoms(test.wantfile);
            for i in 0..got.len() {
                assert_abs_diff_eq!(got[i], want[i], epsilon = 4e-8);
            }
        }
    }
}
