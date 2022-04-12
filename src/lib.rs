use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
};

mod geom;

use geom::Geom;
use nalgebra as na;
use regex::Regex;

/// from <https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0>
const ANGBOHR: f64 = 0.5291_772_109;
const DEGRAD: f64 = 180.0 / std::f64::consts::PI;

// flags
pub static mut VERBOSE: bool = false;

// TODO make these input or flag params
const TOLDISP: f64 = 1e-14;
const MAX_ITER: usize = 20;

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
            Stretch(a, b) => Intder::dist(geom, *a, *b),
            Bend(a, b, c) => Intder::angle(geom, *a, *b, *c),
            Torsion(a, b, c, d) => {
                let e_21 = Intder::unit(geom, *b, *a);
                let e_32 = Intder::unit(geom, *c, *b);
                let e_43 = Intder::unit(geom, *d, *c);
                let v5 = e_21.cross(&e_32);
                let v6 = e_43.cross(&e_32);
                let w2 = e_21.dot(&e_32);
                let w3 = e_43.dot(&e_32);
                let cp2 = -w2;
                let cp3 = -w3;
                let sp2 = (1.0 - cp2 * cp2).sqrt();
                let sp3 = (1.0 - cp3 * cp3).sqrt();
                let w2 = e_21.dot(&v6);
                let w3 = -v5.dot(&v6);
                let w = w2 / (sp2 * sp3);
                let w_size = w.abs() - 1.0;
                let w = if w_size > 0.0 && w_size < 1e-12 {
                    f64::copysign(1.0, w)
                } else {
                    w
                }
                .asin();
                if w.is_nan() {
                    panic!("nan calling sin on {}", w_size);
                }
                if w3 < 0.0 {
                    std::f64::consts::PI - w
                } else {
                    w
                }
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
        let zero = Regex::new(r"^\s*0\s*$").unwrap();
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
                disp_tmp[sp[0].parse::<usize>().unwrap() - 1] = match sp.get(1)
                {
                    Some(s) => s,
                    None => panic!("line '{}' too short", line),
                }
                .parse::<f64>()
                .unwrap();
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
        ANGBOHR * (geom[j] - geom[i]).magnitude()
    }

    /// angle in radians between atoms i, j, and k, where j is the central atom
    fn angle(geom: &Geom, i: usize, j: usize, k: usize) -> f64 {
        let e_ji = Self::unit(geom, j, i);
        let e_jk = Self::unit(geom, j, k);
        (e_ji.dot(&e_jk)).acos()
    }

    pub fn s_vec(geom: &Geom, ic: &Siic, len: usize) -> Vec<f64> {
        let mut tmp = vec![0.0; len];
        match ic {
            Siic::Stretch(a, b) => {
                let e_12 = Self::unit(geom, *a, *b);
                for i in 0..3 {
                    tmp[3 * a + i] = -e_12[i % 3];
                    tmp[3 * b + i] = e_12[i % 3];
                }
            }
            Siic::Bend(a, b, c) => {
                let e_21 = Self::unit(geom, *b, *a);
                let e_23 = Self::unit(geom, *b, *c);
                let t_12 = Self::dist(geom, *b, *a);
                let t_32 = Self::dist(geom, *b, *c);
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
                let e_21 = Self::unit(geom, *b, *a);
                let e_32 = Self::unit(geom, *c, *b);
                let e_43 = Self::unit(geom, *d, *c);
                let t_21 = Self::dist(geom, *b, *a);
                let t_32 = Self::dist(geom, *c, *b);
                let t_43 = Self::dist(geom, *d, *c);
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
        let (r, _) = b.shape();
        let mut u = Vec::new();
        for i in 0..self.symmetry_internals.len() {
            u.extend(&self.symmetry_internals[i].clone());
        }
        let u = DMat::from_row_slice(r, r, &u);
        u * b
    }

    /// Let D = BBᵀ and return A = BᵀD⁻¹
    pub fn a_matrix(b: &DMat) -> DMat {
        let d = b * b.transpose();
        let chol = match na::Cholesky::new(d) {
            Some(c) => c,
            None => {
                // compute it again to avoid cloning on the happy path
                println!("{:.8}", b * b.transpose());
                panic!("cholesky decomposition failed");
            }
        };
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
        if unsafe { VERBOSE } {
            self.print_init();
        }
        let mut ret = Vec::new();
        for (i, disp) in self.disps.iter().enumerate() {
            // initialize sics and carts to those from the input file
            let mut sic_current = DVec::from(self.symmetry_values(&self.geom));
            let mut cart_current: DVec = self.geom.clone().into();

            // get a vector from the displacement from the input file
            let disp = DVec::from(disp.clone());
            let sic_desired = &sic_current + &disp;

            if unsafe { VERBOSE } {
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

            // measure convergence by max internal deviation between the current
            // SICs and desired SICs
            let mut iter = 1;
            while (&sic_current - &sic_desired).abs().max() > TOLDISP {
                let b_sym = self.sym_b_matrix(&Geom::from(&cart_current));
                let d = &b_sym * b_sym.transpose();
                let a = Intder::a_matrix(&b_sym);

                if unsafe { VERBOSE } {
                    println!(
                        "ITER={:5} MAX INTERNAL DEVIATION = {:.4e}",
                        iter,
                        (&sic_current - &sic_desired).abs().max()
                    );
                    println!("B*BT MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
                    println!("{:.6}", d);

                    println!(
                        "DETERMINANT OF B*BT MATRIX={:8.4}",
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

                if MAX_ITER > 0 && iter > MAX_ITER {
                    panic!("max iterations exceeded");
                }
            }

            if unsafe { VERBOSE } {
                println!(
                    "ITER={:5} MAX INTERNAL DEVIATION = {:.4e}\n",
                    iter,
                    (&sic_current - &sic_desired).abs().max()
                );
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
    use std::io::Read;

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

    fn load_vec(filename: &str) -> Vec<f64> {
        let mut f = std::fs::File::open(filename).unwrap();
        let mut buf = String::new();
        f.read_to_string(&mut buf).unwrap();
        buf.split_whitespace()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    }

    struct MatTest<'a> {
        infile: &'a str,
        rows: usize,
        cols: usize,
        vecfile: &'a str,
        eps: f64,
    }

    #[test]
    fn test_b_matrix() {
        let tests = vec![
            MatTest {
                infile: "testfiles/intder.in",
                rows: 3,
                cols: 9,
                vecfile: "testfiles/h2o.bmat",
                eps: 2e-7,
            },
            MatTest {
                infile: "testfiles/c7h2.in",
                rows: 21,
                cols: 27,
                vecfile: "testfiles/c7h2.bmat",
                eps: 2.2e-7,
            },
        ];
        for test in tests {
            let intder = Intder::load(test.infile);
            let want = DMat::from_row_slice(
                test.rows,
                test.cols,
                &load_vec(test.vecfile),
            );
            let got = intder.b_matrix(&intder.geom);
            assert_abs_diff_eq!(got, want, epsilon = test.eps);
        }
    }

    #[test]
    fn test_sym_b() {
        let tests = vec![
            MatTest {
                infile: "testfiles/intder.in",
                rows: 3,
                cols: 9,
                vecfile: "testfiles/h2o.bsmat",
                eps: 2e-7,
            },
            MatTest {
                infile: "testfiles/c7h2.in",
                rows: 21,
                cols: 27,
                vecfile: "testfiles/c7h2.bsmat",
                eps: 2.2e-7,
            },
        ];
        for test in tests {
            let intder = Intder::load(test.infile);
            let got = intder.sym_b_matrix(&intder.geom);
            let want = DMat::from_row_slice(
                test.rows,
                test.cols,
                &load_vec(test.vecfile),
            );
            assert_abs_diff_eq!(got, want, epsilon = test.eps);
        }
    }

    #[allow(dead_code)]
    fn dbg_mat(a: &DMat, b: &DMat, eps: f64) {
        let a = a.as_slice();
        let b = b.as_slice();
        assert!(a.len() == b.len());
        println!();
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > eps {
                println!(
                    "{:5}{:>15.8}{:>15.8}{:>15.8e}",
                    i,
                    a[i],
                    b[i],
                    a[i] - b[i]
                );
            }
        }
    }

    #[test]
    fn test_a_matrix() {
        let tests = vec![
            MatTest {
                infile: "testfiles/intder.in",
                rows: 9,
                cols: 3,
                vecfile: "testfiles/h2o.amat",
                eps: 3e-8,
            },
            // low precision from intder.out
            MatTest {
                infile: "testfiles/c3h2.in",
                rows: 15,
                cols: 9,
                vecfile: "testfiles/c3h2.amat",
                eps: 1e-6,
            },
        ];
        for test in tests {
            let intder = Intder::load(test.infile);
            let load = load_vec(test.vecfile);
            let want = DMat::from_row_slice(test.rows, test.cols, &load);
            let got = Intder::a_matrix(&intder.sym_b_matrix(&intder.geom));
            assert_abs_diff_eq!(got, want, epsilon = test.eps);
        }
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
            Test {
                infile: "testfiles/thoco.in",
                wantfile: "testfiles/thoco.07",
            },
            Test {
                infile: "testfiles/c3h2.in",
                wantfile: "testfiles/c3h2.07",
            },
            Test {
                infile: "testfiles/c7h2.in",
                wantfile: "testfiles/c7h2.small.07",
            },
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
