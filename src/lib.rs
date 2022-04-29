use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
};

pub mod geom;
pub mod tensor;

use geom::Geom;
use nalgebra as na;
use regex::Regex;
use tensor::{Tensor3, Tensor4};

/// from <https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0>
const ANGBOHR: f64 = 0.5291_772_109;
const DEGRAD: f64 = 180.0 / std::f64::consts::PI;
/// constants from the fortran version
const HART: f64 = 4.3597482;
// const BOHR: f64 = 0.529177249;
// const DEBYE: f64 = 2.54176548;

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
pub struct Atom {
    pub label: String,
    pub weight: usize,
}

pub struct Hmats {
    pub h11: DMat,
    pub h21: DMat,
    pub h31: DMat,
    pub h22: DMat,
    pub h32: DMat,
    pub h33: DMat,
}

impl Hmats {
    fn new() -> Self {
        Self {
            h11: DMat::zeros(3, 3),
            h21: DMat::zeros(3, 3),
            h31: DMat::zeros(3, 3),
            h22: DMat::zeros(3, 3),
            h32: DMat::zeros(3, 3),
            h33: DMat::zeros(3, 3),
        }
    }
}

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
    fn new() -> Self {
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
}

#[derive(Debug, PartialEq)]
pub struct Intder {
    pub input_options: Vec<usize>,
    pub simple_internals: Vec<Siic>,
    pub symmetry_internals: Vec<Vec<f64>>,
    /// cartesian geometry in bohr
    pub geom: Geom,
    /// SIC displacements to be converted to Cartesian coordinates
    pub disps: Vec<Vec<f64>>,
    /// Atom labels and weights, for use in force constant conversions
    pub atoms: Vec<Atom>,
    /// second order SIC force constants to be converted to cartesian
    /// coordinates
    pub fc2: Vec<f64>,
    /// third order SIC force constants to be converted to cartesian
    /// coordinates
    pub fc3: Vec<f64>,
    /// fourth order SIC force constants to be converted to cartesian
    /// coordinates
    pub fc4: Vec<f64>,
}

impl Intder {
    pub fn new() -> Self {
        Intder {
            input_options: Vec::new(),
            simple_internals: Vec::new(),
            symmetry_internals: Vec::new(),
            geom: Geom::new(),
            disps: Vec::new(),
            atoms: Vec::new(),
            fc2: Vec::new(),
            fc3: Vec::new(),
            fc4: Vec::new(),
        }
    }

    pub fn load_file(infile: &str) -> Self {
        let f = match File::open(infile) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("failed to open infile '{}'", infile);
                std::process::exit(1);
            }
        };
        Self::load(f)
    }

    /// helper function for parsing a single simple internal coordinate line
    fn parse_simple_internal(sp: Vec<&str>) -> Siic {
        match sp[0] {
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
        }
    }

    pub fn load<R: Read>(r: R) -> Self {
        let siic = Regex::new(r"STRE|BEND|TORS|OUT|LIN|SPF|RCOM").unwrap();
        // something like "    1   1   1.000000000   2   1.000000000"
        let syic =
            Regex::new(r"^\s*(\d+\s+)(\d+\s+[0-9-]\.[0-9]+(\s+|$))+").unwrap();
        let zero = Regex::new(r"^\s*0\s*$").unwrap();
        // just a bunch of integers like 3    3    3    0    0    3 ..."
        let iops = Regex::new(r"^\s*(\d+(\s+|$))+$").unwrap();
        let geom = Regex::new(r"\s*([0-9-]+\.[0-9]+(\s+|$)){3}").unwrap();
        let atoms = Regex::new(r"[A-Za-z]+\d+").unwrap();
        let atom = Regex::new(r"([A-Za-z]+)(\d+)").unwrap();
        let fcs = Regex::new(r"^\s*(\d+\s+){4}[0-9-]+\.\d+\s*$").unwrap();

        let mut intder = Intder::new();
        let reader = BufReader::new(r);
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
                intder.simple_internals.push(Self::parse_simple_internal(
                    line.split_whitespace().collect(),
                ));
            } else if syic.is_match(&line) {
                // this has to come after the simple internals
                assert!(intder.simple_internals.len() > 0);
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
            } else if atoms.is_match(&line) {
                for cap in atom.captures_iter(&line) {
                    intder.atoms.push(Atom {
                        label: String::from(&cap[1]),
                        weight: cap[2].parse().unwrap(),
                    });
                }
            } else if fcs.is_match(&line) {
                let mut sp = line.split_whitespace().collect::<Vec<_>>();
                let val = sp.pop().unwrap().parse::<f64>().unwrap();
                let sp = sp
                    .iter()
                    .map(|s| s.parse::<usize>().unwrap())
                    .collect::<Vec<_>>();
                let (idx, target) = match (sp[2], sp[3]) {
                    (0, 0) => (intder.fc2_index(sp[0], sp[1]), &mut intder.fc2),
                    (_, 0) => {
                        (intder.fc3_index(sp[0], sp[1], sp[2]), &mut intder.fc3)
                    }
                    (_, _) => (
                        intder.fc4_index(sp[0], sp[1], sp[2], sp[3]),
                        &mut intder.fc4,
                    ),
                };
                if target.len() <= idx {
                    target.resize(idx + 1, 0.0);
                }
                target[idx] = val;
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

    /// return the U matrix, used for converting from simple internals to
    /// symmetry internals. dimensions are (number of symmetry internals) x
    /// (number of simple internals) since each symm. int. is a vector simp.
    /// int. long
    pub fn u_mat(&self) -> DMat {
        let r = self.symmetry_internals.len();
        let mut u = Vec::new();
        for i in 0..r {
            u.extend(&self.symmetry_internals[i].clone());
        }
        DMat::from_row_slice(r, r, &u)
    }

    /// return the symmetry internal coordinate B matrix by computing the simple
    /// internal B and converting it
    pub fn sym_b_matrix(&self, geom: &Geom) -> DMat {
        let b = self.b_matrix(geom);
        self.u_mat() * b
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

    pub fn print_cart<W: Write>(w: &mut W, cart: &DVec) {
        for i in 0..cart.len() / 3 {
            for j in 0..3 {
                write!(w, "{:20.10}", cart[3 * i + j]).unwrap();
            }
            writeln!(w).unwrap();
        }
    }

    /// convert the displacements in `self.disps` from (symmetry) internal
    /// coordinates to Cartesian coordinates
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

    fn fc2_index(&self, i: usize, j: usize) -> usize {
        let n3n = self.symmetry_internals.len();
        let mut sp = [i, j];
        sp.sort();
        n3n * (sp[0] - 1) + sp[1] - 1
    }

    fn fc3_index(&self, i: usize, j: usize, k: usize) -> usize {
        let mut sp = [i, j, k];
        sp.sort();
        sp[0] + (sp[1] - 1) * sp[1] / 2 + (sp[2] - 1) * sp[2] * (sp[2] + 1) / 6
            - 1
    }

    fn fc4_index(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        let mut sp = [i, j, k, l];
        sp.sort();
        sp[0]
            + (sp[1] - 1) * sp[1] / 2
            + (sp[2] - 1) * sp[2] * (sp[2] + 1) / 6
            + (sp[3] - 1) * sp[3] * (sp[3] + 1) * (sp[3] + 2) / 24
            - 1
    }

    // making block matrices to pack into sr in machx
    pub fn h_mat(geom: &Geom, s: &Siic) -> Hmats {
        use Siic::*;
        let mut h = Hmats::new();
        match s {
            // from HIJS1
            Stretch(i, j) => {
                let v1 = Self::unit(geom, *i, *j);
                let t21 = Self::dist(geom, *i, *j);
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
                let tmp = Self::s_vec(geom, s, 3 * geom.len());
                // unpack the s vector
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let e21 = Self::unit(geom, *j, *i);
                let e23 = Self::unit(geom, *j, *k);
                let t21 = Self::dist(geom, *j, *i);
                let t23 = Self::dist(geom, *j, *k);
                let h11a = Self::h_mat(geom, &Stretch(*i, *j)).h11;
                let h33a = Self::h_mat(geom, &Stretch(*k, *j)).h11;
                let phi = Self::angle(geom, *i, *j, *k);
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
            Torsion(_, _, _, _) => todo!(),
        }
        h
    }

    /// returns X and SR matrices in symmetry internal coordinates
    pub fn machx(&self, a_mat: &DMat) -> (Vec<DMat>, Vec<DMat>) {
        // TODO you might need the sim and sic versions, both are written to
        // different parts of the file on disk
        use Siic::*;
        let nc = 3 * self.geom.len();
        let u = self.u_mat();
        let nsym = self.symmetry_internals.len();
        if nsym == 0 {
            eprintln!("using only simple internals is unimplemented");
            todo!();
        }
        // simple internal X and SR matrices
        let mut xs_sim = Vec::new();
        let mut srs_sim = Vec::new();
        // let nsim = self.simple_internals.len();
        for s in &self.simple_internals {
            // I thought this was nc x nc but actually nc x nsym
            let mut x = DMat::zeros(nc, nsym);
            let mut sr = DMat::zeros(nc, nc);
            let h = Self::h_mat(&self.geom, &s);
            // println!("H11 = {}", &h.h11);
            // println!("H21 = {}", &h.h21);
            // println!("H31 = {}", &h.h31);
            // println!("H22 = {}", &h.h22);
            // println!("H32 = {}", &h.h32);
            // println!("H33 = {}", &h.h33);
            match s {
                Stretch(a, b) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    // TODO can you set blocks of matrices with nalgebra?
                    for j in 0..3 {
                        for i in 0..3 {
                            sr[(l1 + i, l1 + j)] = h.h11[(i, j)];
                            sr[(l2 + i, l2 + j)] = h.h11[(i, j)];
                            sr[(l1 + i, l2 + j)] = -h.h11[(i, j)];
                            sr[(l2 + i, l1 + j)] = -h.h11[(i, j)];
                        }
                    }
                    // AHX2
                    for n in 0..nsym {
                        for m in 0..=n {
                            for i in 0..3 {
                                for j in 0..3 {
                                    let w1 = (a_mat[(l1 + i, m)]
                                        - a_mat[(l2 + i, m)])
                                        * (a_mat[(l1 + j, n)]
                                            - a_mat[(l2 + j, n)]);
                                    x[(m, n)] += w1 * h.h11[(i, j)];
                                }
                            }
                        }
                    }
                    // TODO I think this can go after the i loop above
                    for n in 0..nsym {
                        for m in 0..n {
                            x[(n, m)] = x[(m, n)];
                        }
                    }
                }
                Bend(a, b, c) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    let l3 = 3 * c;
                    for j in 0..3 {
                        for i in 0..3 {
                            sr[(l1 + i, l1 + j)] = h.h11[(i, j)];
                            sr[(l2 + i, l1 + j)] = h.h21[(i, j)];
                            sr[(l3 + i, l1 + j)] = h.h31[(i, j)];
                            sr[(l1 + i, l2 + j)] = h.h21[(j, i)];
                            sr[(l2 + i, l2 + j)] = h.h22[(i, j)];
                            sr[(l3 + i, l2 + j)] = h.h32[(i, j)];
                            sr[(l1 + i, l3 + j)] = h.h31[(j, i)];
                            sr[(l2 + i, l3 + j)] = h.h32[(j, i)];
                            sr[(l3 + i, l3 + j)] = h.h33[(i, j)];
                        }
                    }
                    // AHX3
                    for n in 0..nsym {
                        for m in 0..=n {
                            for i in 0..3 {
                                for j in 0..3 {
                                    let w1 =
                                        a_mat[(l1 + i, m)] * a_mat[(l1 + j, n)];
                                    let w2 =
                                        a_mat[(l2 + i, m)] * a_mat[(l2 + j, n)];
                                    let w3 =
                                        a_mat[(l3 + i, m)] * a_mat[(l3 + j, n)];
                                    x[(m, n)] += w1 * h.h11[(i, j)]
                                        + w2 * h.h22[(i, j)]
                                        + w3 * h.h33[(i, j)];
                                    let w1 = a_mat[(l2 + i, m)]
                                        * a_mat[(l1 + j, n)]
                                        + a_mat[(l1 + j, m)]
                                            * a_mat[(l2 + i, n)];
                                    let w2 = a_mat[(l3 + i, m)]
                                        * a_mat[(l1 + j, n)]
                                        + a_mat[(l1 + j, m)]
                                            * a_mat[(l3 + i, n)];
                                    let w3 = a_mat[(l3 + i, m)]
                                        * a_mat[(l2 + j, n)]
                                        + a_mat[(l2 + j, m)]
                                            * a_mat[(l3 + i, n)];
                                    x[(m, n)] += w1 * h.h21[(i, j)]
                                        + w2 * h.h31[(i, j)]
                                        + w3 * h.h32[(i, j)];
                                }
                            }
                        }
                    }
                    // TODO move this into above loop
                    for n in 0..nsym {
                        for m in 0..=n {
                            x[(n, m)] = x[(m, n)];
                        }
                    }
                }
                Torsion(_, _, _, _) => todo!(),
            }
            // println!("SR_{} = {:12.8}", i + 1, sr);
            // println!("X = {:12.8}", x);
            // println!("U = {:12.8}", u);
            xs_sim.push(x);
            srs_sim.push(sr);
        }
        // TODO if nsym = 0, just return the sim versions
        let mut xs_sym = Vec::new();
        let mut srs_sym = Vec::new();
        for r in 0..nsym {
            let mut x_sic = DMat::zeros(nc, nsym);
            for (i, x) in xs_sim.iter().enumerate() {
                for n in 0..nsym {
                    for m in 0..nsym {
                        x_sic[(m, n)] += u[(r, i)] * x[(m, n)];
                    }
                }
            }
            xs_sym.push(x_sic);
        }
        for r in 0..nsym {
            let mut sr_sic = DMat::zeros(nc, nc);
            for (i, sr) in srs_sim.iter().enumerate() {
                for n in 0..nc {
                    for m in 0..nc {
                        sr_sic[(m, n)] += u[(r, i)] * sr[(m, n)];
                    }
                }
            }
            srs_sym.push(sr_sic);
        }
        (xs_sym, srs_sym)
    }

    pub fn h_tensor3(geom: &Geom, s: &Siic) -> Htens {
        use Siic::*;
        // TODO can reorder loops to reuse this h_mat call with machx. loop over
        // sims, for each sim, call h_mat and h_tensor so I can use that h_mat
        // in h_tensor instead of calling h_mat for each sim in machx and machy
        // separately
        let hm = Self::h_mat(geom, s);
        let mut h = Htens::new();
        // TODO see note on Tensor3 about symmetry
        match s {
            // HIJKS1
            Stretch(i, j) => {
                let v1 = Self::unit(geom, *i, *j);
                let t21 = Self::dist(geom, *i, *j);
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
                let tmp = Self::s_vec(geom, s, 3 * geom.len());
                let v1 = &tmp[3 * i..3 * i + 3];
                let v3 = &tmp[3 * k..3 * k + 3];
                let e21 = Self::unit(geom, *j, *i);
                let e23 = Self::unit(geom, *j, *k);
                let t21 = Self::dist(geom, *j, *i);
                let t23 = Self::dist(geom, *j, *k);
                let h11a = Self::h_mat(geom, &Stretch(*i, *j)).h11;
                let h33a = Self::h_mat(geom, &Stretch(*k, *j)).h11;
                let phi = Self::angle(geom, *i, *j, *k);
                // end copy
                let hijs2 = Self::h_mat(geom, s);
                let h111a = Self::h_tensor3(geom, &Stretch(*i, *j)).h111;
                let h333a = Self::h_tensor3(geom, &Stretch(*k, *j)).h111;
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
            Torsion(_, _, _, _) => todo!(),
        }
        h
    }

    /// returns the Y and SR matrices in symmetry internal coordinates
    pub fn machy(&self, a_mat: &DMat) -> (Vec<Tensor3>, Vec<Tensor3>) {
        use Siic::*;
        let nc = 3 * self.geom.len();
        let nsx = self.symmetry_internals.len();
        if nsx == 0 {
            eprintln!("using only simple internals is unimplemented");
            todo!();
        }
        let u = self.u_mat();
        let mut ys_sim = Vec::new();
        let mut srs_sim = Vec::new();
        for s in &self.simple_internals {
            let mut y = Tensor3::zeros(nc, nc, nc);
            let mut sr = Tensor3::zeros(nc, nc, nc);
            let h = Self::h_tensor3(&self.geom, &s);
            match s {
                Stretch(a, b) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    // HSRY2
                    for k in 0..3 {
                        for j in 0..3 {
                            for i in 0..3 {
                                let z = h.h111[(i, j, k)];
                                sr[(l1 + i, l1 + j, l1 + k)] = z;
                                sr[(l1 + i, l1 + j, l2 + k)] = -z;
                                sr[(l1 + i, l2 + j, l1 + k)] = -z;
                                sr[(l1 + i, l2 + j, l2 + k)] = z;
                                sr[(l2 + i, l1 + j, l1 + k)] = -z;
                                sr[(l2 + i, l1 + j, l2 + k)] = z;
                                sr[(l2 + i, l2 + j, l1 + k)] = z;
                                sr[(l2 + i, l2 + j, l2 + k)] = -z;
                            }
                        }
                    }
                    // AHY2
                    for p in 0..nsx {
                        for n in 0..=p {
                            for m in 0..=n {
                                for i in 0..3 {
                                    for j in 0..3 {
                                        for k in 0..3 {
                                            let w1 = a_mat[(l1 + j, n)]
                                                * (a_mat[(l1 + k, p)]
                                                    - a_mat[(l2 + k, p)])
                                                - a_mat[(l2 + j, n)]
                                                    * (a_mat[(l1 + k, p)]
                                                        - a_mat[(l2 + k, p)]);
                                            let w1 = (a_mat[(l1 + i, m)]
                                                - a_mat[(l2 + i, m)])
                                                * w1;
                                            y[(m, n, p)] +=
                                                w1 * h.h111[(i, j, k)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Bend(a, b, c) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    let l3 = 3 * c;
                    // HSRY3
                    for k in 0..3 {
                        for j in 0..3 {
                            for i in 0..3 {
                                sr[(l1 + i, l1 + j, l1 + k)] =
                                    h.h111[(i, j, k)];
                                sr[(l1 + i, l1 + j, l2 + k)] =
                                    h.h112[(i, j, k)];
                                sr[(l1 + i, l1 + j, l3 + k)] =
                                    h.h113[(i, j, k)];
                                sr[(l1 + i, l2 + j, l1 + k)] =
                                    h.h112[(i, k, j)];
                                sr[(l1 + i, l2 + j, l2 + k)] =
                                    h.h221[(j, k, i)];
                                sr[(l1 + i, l2 + j, l3 + k)] =
                                    h.h123[(i, j, k)];
                                sr[(l2 + i, l2 + j, l1 + k)] =
                                    h.h221[(i, j, k)];
                                sr[(l2 + i, l2 + j, l2 + k)] =
                                    h.h222[(i, j, k)];
                                sr[(l2 + i, l2 + j, l3 + k)] =
                                    h.h223[(i, j, k)];
                                sr[(l2 + i, l1 + j, l1 + k)] =
                                    h.h112[(j, k, i)];
                                sr[(l2 + i, l1 + j, l2 + k)] =
                                    h.h221[(i, k, j)];
                                sr[(l2 + i, l1 + j, l3 + k)] =
                                    h.h123[(j, i, k)];
                                sr[(l1 + i, l3 + j, l1 + k)] =
                                    h.h113[(i, k, j)];
                                sr[(l1 + i, l3 + j, l2 + k)] =
                                    h.h123[(i, k, j)];
                                sr[(l1 + i, l3 + j, l3 + k)] =
                                    h.h331[(j, k, i)];
                                sr[(l2 + i, l3 + j, l1 + k)] =
                                    h.h123[(k, i, j)];
                                sr[(l2 + i, l3 + j, l2 + k)] =
                                    h.h223[(i, k, j)];
                                sr[(l2 + i, l3 + j, l3 + k)] =
                                    h.h332[(j, k, i)];
                                sr[(l3 + i, l1 + j, l1 + k)] =
                                    h.h113[(j, k, i)];
                                sr[(l3 + i, l1 + j, l2 + k)] =
                                    h.h123[(j, k, i)];
                                sr[(l3 + i, l1 + j, l3 + k)] =
                                    h.h331[(i, k, j)];
                                sr[(l3 + i, l2 + j, l1 + k)] =
                                    h.h123[(k, j, i)];
                                sr[(l3 + i, l2 + j, l2 + k)] =
                                    h.h223[(j, k, i)];
                                sr[(l3 + i, l2 + j, l3 + k)] =
                                    h.h332[(i, k, j)];
                                sr[(l3 + i, l3 + j, l1 + k)] =
                                    h.h331[(i, j, k)];
                                sr[(l3 + i, l3 + j, l2 + k)] =
                                    h.h332[(i, j, k)];
                                sr[(l3 + i, l3 + j, l3 + k)] =
                                    h.h333[(i, j, k)];
                            }
                        }
                    }
                    // AHY3
                    for p in 0..nsx {
                        for n in 0..=p {
                            for m in 0..=n {
                                for i in 0..3 {
                                    for j in 0..3 {
                                        let v1 = a_mat[(l1 + i, m)]
                                            * a_mat[(l1 + j, n)];
                                        let v2 = a_mat[(l2 + i, m)]
                                            * a_mat[(l2 + j, n)];
                                        let v3 = a_mat[(l3 + i, m)]
                                            * a_mat[(l3 + j, n)];
                                        let v4 = a_mat[(l1 + i, m)]
                                            * a_mat[(l1 + j, p)];
                                        let v5 = a_mat[(l1 + i, n)]
                                            * a_mat[(l1 + j, p)];
                                        let v6 = a_mat[(l2 + i, m)]
                                            * a_mat[(l2 + j, p)];
                                        let v7 = a_mat[(l2 + i, n)]
                                            * a_mat[(l2 + j, p)];
                                        let v8 = a_mat[(l3 + i, m)]
                                            * a_mat[(l3 + j, p)];
                                        let v9 = a_mat[(l3 + i, n)]
                                            * a_mat[(l3 + j, p)];
                                        let v10 = a_mat[(l1 + i, m)]
                                            * a_mat[(l2 + j, n)];
                                        let v11 = a_mat[(l1 + i, m)]
                                            * a_mat[(l2 + j, p)];
                                        let v12 = a_mat[(l1 + i, n)]
                                            * a_mat[(l2 + j, m)];
                                        let v13 = a_mat[(l1 + i, n)]
                                            * a_mat[(l2 + j, p)];
                                        let v14 = a_mat[(l1 + i, p)]
                                            * a_mat[(l2 + j, m)];
                                        let v15 = a_mat[(l1 + i, p)]
                                            * a_mat[(l2 + j, n)];
                                        for k in 0..3 {
                                            let w1 = v1 * a_mat[(l1 + k, p)];
                                            let w2 = v2 * a_mat[(l2 + k, p)];
                                            let w3 = v3 * a_mat[(l3 + k, p)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h111[(i, j, k)]
                                                + w2 * h.h222[(i, j, k)]
                                                + w3 * h.h333[(i, j, k)];
                                            let w1 = v1 * a_mat[(l2 + k, p)]
                                                + v4 * a_mat[(l2 + k, n)]
                                                + v5 * a_mat[(l2 + k, m)];
                                            let w2 = v1 * a_mat[(l3 + k, p)]
                                                + v4 * a_mat[(l3 + k, n)]
                                                + v5 * a_mat[(l3 + k, m)];
                                            let w3 = v3 * a_mat[(l2 + k, p)]
                                                + v8 * a_mat[(l2 + k, n)]
                                                + v9 * a_mat[(l2 + k, m)];
                                            let w4 = v3 * a_mat[(l1 + k, p)]
                                                + v8 * a_mat[(l1 + k, n)]
                                                + v9 * a_mat[(l1 + k, m)];
                                            let w5 = v2 * a_mat[(l1 + k, p)]
                                                + v6 * a_mat[(l1 + k, n)]
                                                + v7 * a_mat[(l1 + k, m)];
                                            let w6 = v2 * a_mat[(l3 + k, p)]
                                                + v6 * a_mat[(l3 + k, n)]
                                                + v7 * a_mat[(l3 + k, m)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h112[(i, j, k)]
                                                + w2 * h.h113[(i, j, k)]
                                                + w3 * h.h332[(i, j, k)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w4 * h.h331[(i, j, k)]
                                                + w5 * h.h221[(i, j, k)]
                                                + w6 * h.h223[(i, j, k)];
                                            let w1 = a_mat[(l3 + k, p)]
                                                * (v10 + v12)
                                                + a_mat[(l3 + k, n)]
                                                    * (v11 + v14)
                                                + a_mat[(l3 + k, m)]
                                                    * (v13 + v15);
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h123[(i, j, k)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Torsion(_, _, _, _) => todo!(),
            }
            y.fill3a(nsx);
            ys_sim.push(y);
            srs_sim.push(sr);
        }

        // convert Y to symmetry internals
        let ys_sym = {
            let mut ys_sym = Vec::new();
            for r in 0..nsx {
                let mut y_sic = Tensor3::zeros(nc, nc, nc);
                for (i, y) in ys_sim.iter().enumerate() {
                    let w1 = u[(r, i)];
                    for p in 0..nsx {
                        for n in 0..=p {
                            for m in 0..=n {
                                y_sic[(m, n, p)] += w1 * y[(m, n, p)];
                            }
                        }
                    }
                }
                y_sic.fill3a(nsx);
                ys_sym.push(y_sic);
            }
            ys_sym
        };

        // convert SR to symmetry internals
        let srs_sym = {
            let mut ret = Vec::new();
            for r in 0..nsx {
                let mut sr_sic = Tensor3::zeros(nc, nc, nc);
                for (i, sr) in srs_sim.iter().enumerate() {
                    let w1 = u[(r, i)];
                    for p in 0..nc {
                        for n in 0..=p {
                            for m in 0..=n {
                                sr_sic[(m, n, p)] += w1 * sr[(m, n, p)];
                            }
                        }
                    }
                }
                sr_sic.fill3a(nc);
                ret.push(sr_sic);
            }
            ret
        };
        (ys_sym, srs_sym)
    }

    /// flatten fc2 so it can be accessed as the Fortran code expects
    fn flatten_fc2(&self, nsy: usize) -> DVec {
        let mut v = Vec::new();
        for i in 0..nsy {
            for j in 0..nsy {
                // multiply by a coefficient here if you need to convert
                // units
                v.push(self.fc2[self.fc2_index(i + 1, j + 1)]);
            }
        }
        DVec::from(v)
    }

    /// represent fc2 as a symmetric matrix
    fn mat_fc2(&self, nsy: usize) -> DMat {
        let mut v = DMat::from_row_slice(nsy, nsy, &self.fc2);
        for row in 0..nsy {
            for col in row..nsy {
                v[(col, row)] = v[(row, col)];
            }
        }
        v
    }

    fn lintr_fc2(&self, a: &DMat) -> DMat {
        let nsx = 3 * self.geom.len();
        let nsy = self.symmetry_internals.len();
        let v = self.flatten_fc2(nsy);
        let xs = {
            let mut xs = DMat::zeros(nsy, nsx);
            let mut kk = 0;
            for ik in 0..nsy * nsy {
                kk += 1;
                let j = (kk - 1) / nsy;
                let i = kk - nsy * j - 1;
                for n in 0..nsx {
                    xs[(i, n)] += a[(j, n)] * v[ik];
                }
            }
            xs
        };
        let mut f2 = a.transpose() * xs;
        for m in 1..nsx {
            for n in 0..m {
                f2[(m, n)] = (f2[(m, n)] + f2[(n, m)]) / 2.0;
                f2[(n, m)] = f2[(m, n)];
            }
        }
        f2 * ANGBOHR * ANGBOHR / HART
    }

    pub fn lintr_fc3(&self, a: &DMat) -> Tensor3 {
        let nsx = 3 * self.geom.len();
        let nsy = self.symmetry_internals.len();
        let v = &self.fc3;
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        let mut f3 = Tensor3::zeros(nsx, nsx, nsx);
        for vik in v {
            if i != j {
                if j != k {
                    for p in 0..nsx {
                        f3[(i, j, p)] += vik * a[(k, p)];
                        f3[(i, k, p)] += vik * a[(j, p)];
                        f3[(j, k, p)] += vik * a[(i, p)];
                    }
                } else {
                    for p in 0..nsx {
                        f3[(i, j, p)] += vik * a[(j, p)];
                        f3[(j, j, p)] += vik * a[(i, p)];
                    }
                }
            } else {
                if j != k {
                    for p in 0..nsx {
                        f3[(i, i, p)] += vik * a[(k, p)];
                        f3[(i, k, p)] += vik * a[(i, p)];
                    }
                } else {
                    for p in 0..nsx {
                        f3[(i, i, p)] += vik * a[(i, p)];
                    }
                }
            }
            if k < j {
                k += 1;
            } else if j < i {
                j += 1;
                k = 0;
            } else {
                i += 1;
                j = 0;
                k = 0;
            }
        }
        // end of 1138 loop, looking good so far

        // TODO can I just make a new f3 here? There is 1 untouched number left
        // in there...
        let f3_disk = f3.clone();
        for p in 0..nsx {
            for n in 0..=p {
                for i in 0..nsy {
                    f3[(i, n, p)] = 0.0;
                }
            }
        }
        // flatten f3_disk into the vector the fortran uses
        let v = {
            let mut v = Vec::new();
            for i in 0..nsy {
                for j in 0..=i {
                    for p in 0..nsx {
                        v.push(f3_disk[(i, j, p)]);
                    }
                }
            }
            v
        };
        let mut i = 0;
        let mut j = 0;
        let mut p = 0;
        for vik in v {
            if i != j {
                for n in 0..=p {
                    f3[(i, n, p)] += vik * a[(j, n)];
                    f3[(j, n, p)] += vik * a[(i, n)];
                }
            } else {
                for n in 0..=p {
                    f3[(i, n, p)] += vik * a[(i, n)];
                }
            }
            if p < nsx - 1 {
                p += 1;
            } else if j < i {
                j += 1;
                p = 0;
            } else {
                i += 1;
                j = 0;
                p = 0;
            }
        }
        // end of 1146 loop, looking good again

        let f3_disk2 = f3.clone();
        for p in 0..nsx {
            for n in 0..=p {
                for m in 0..=n {
                    f3[(m, n, p)] = 0.0
                }
            }
        }
        // flatten f3_disk2 into the vector the fortran uses
        let v = {
            let mut v = Vec::new();
            for p in 0..nsx {
                for n in 0..=p {
                    for i in 0..nsy {
                        v.push(f3_disk2[(i, n, p)]);
                    }
                }
            }
            v
        };
        let mut i = 0;
        let mut n = 0;
        let mut p = 0;
        for vik in v {
            for m in 0..=n {
                f3[(m, n, p)] += vik * a[(i, m)];
            }
            if i < nsy - 1 {
                i += 1;
            } else if n < p {
                n += 1;
                i = 0;
            } else {
                p += 1;
                n = 0;
                i = 0;
            }
        }
        // end of 1152 loop, still looking good

        f3.fill3a(nsx);
        f3
    }

    pub fn lintr_fc4(&self, a: &DMat) -> () {
        let nsx = 3 * self.geom.len();
        let nsy = self.symmetry_internals.len();
        let v = &self.fc4;
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        let mut l = 0;
        let mut f4 = Tensor4::zeros(nsx, nsx, nsx, nsx);
        for vik in v {
            if i != j {
                if j != k {
                    if k != l {
                        for q in 0..nsx {
                            f4[(i, j, k, q)] += vik * a[(l, q)];
                            f4[(i, j, l, q)] += vik * a[(k, q)];
                            f4[(i, k, l, q)] += vik * a[(j, q)];
                            f4[(j, k, l, q)] += vik * a[(i, q)];
                        }
                    } else {
                        for q in 0..nsx {
                            f4[(i, j, k, q)] += vik * a[(k, q)];
                            f4[(i, k, k, q)] += vik * a[(j, q)];
                            f4[(j, k, k, q)] += vik * a[(i, q)];
                        }
                    }
                } else {
                    if k != l {
                        for q in 0..nsx {
                            f4[(i, j, j, q)] += vik * a[(l, q)];
                            f4[(i, j, l, q)] += vik * a[(j, q)];
                            f4[(j, j, l, q)] += vik * a[(i, q)];
                        }
                    } else {
                        for q in 0..nsx {
                            f4[(i, j, j, q)] += vik * a[(j, q)];
                            f4[(j, j, j, q)] += vik * a[(i, q)];
                        }
                    }
                }
            } else {
                if j != k {
                    if k != l {
                        for q in 0..nsx {
                            f4[(i, i, k, q)] += vik * a[(l, q)];
                            f4[(i, i, l, q)] += vik * a[(k, q)];
                            f4[(i, k, l, q)] += vik * a[(i, q)];
                        }
                    } else {
                        for q in 0..nsx {
                            f4[(i, i, k, q)] += vik * a[(k, q)];
                            f4[(i, k, k, q)] += vik * a[(i, q)];
                        }
                    }
                } else {
                    if k != l {
                        for q in 0..nsx {
                            f4[(i, i, i, q)] += vik * a[(l, q)];
                            f4[(i, i, l, q)] += vik * a[(i, q)];
                        }
                    } else {
                        for q in 0..nsx {
                            f4[(i, i, i, q)] += vik * a[(i, q)];
                        }
                    }
                }
            }
            if l < k {
                l += 1;
            } else if k < j {
                k += 1;
                l = 0;
            } else if j < i {
                j += 1;
                k = 0;
                l = 0;
            } else {
                i += 1;
                j = 0;
                k = 0;
                l = 0;
            }
        }
        // end 179 loop, not looking good so far

        let f4_disk = f4.clone();
        for q in 0..nsx {
            for p in 0..=q {
                for i in 0..nsy {
                    for j in 0..=i {
                        f4[(i, j, p, q)] = 0.0;
                    }
                }
            }
        }

        // TODO resume here
    }

    fn xf2(&self, f3_raw: &Tensor3, bs: &DMat, xrs: &Vec<DMat>) -> Tensor3 {
        let ns = self.symmetry_internals.len();
        let nc = 3 * self.geom.len();
        let v = self.mat_fc2(ns);
        let xs = v * bs;
        let mut f3 = f3_raw.clone();
        for (r, xr) in xrs.iter().enumerate() {
            for k in 0..nc {
                for j in 0..=k {
                    for i in 0..=j {
                        let w = xr[(i, j)] * xs[(r, k)]
                            + xr[(i, k)] * xs[(r, j)]
                            + xr[(j, k)] * xs[(r, i)];
                        f3[(i, j, k)] += w;
                    }
                }
            }
        }
        f3.fill3a(nc);
        f3
    }

    /// Perform the linear transformation of the force constants and convert the
    /// units to those desired by SPECTRO. what they call A here is actually the
    /// SIC B matrix.
    pub fn lintr(
        &self,
        a: &DMat,
        bs: &DMat,
        xr: &Vec<DMat>,
    ) -> (DMat, Vec<f64>) {
        let f2 = self.lintr_fc2(a);
        let f3 = self.xf2(&self.lintr_fc3(a), bs, xr);
        self.lintr_fc4(a);

        // convert f3 to the proper units and return it as a Vec in the order
        // desired by spectro
        let nsx = 3 * self.geom.len();
        const CF3: f64 = ANGBOHR * ANGBOHR * ANGBOHR / HART;
        let mut f3_out = Vec::new();
        for m in 0..nsx {
            for n in 0..=m {
                for p in 0..=n {
                    f3_out.push(CF3 * f3[(m, n, p)]);
                }
            }
        }
        (f2, f3_out)
    }

    /// convert the force constants in `self.fc[234]` from (symmetry) internal
    /// coordinates to Cartesian coordinates. returns (fc2, fc3, fc4) in the
    /// order printed in the fort.{15,30,40} files for spectro. TODO - for now
    /// it just returns (fc2)
    pub fn convert_fcs(&self) -> (DMat, Vec<f64>) {
        if unsafe { VERBOSE } {
            self.print_init();
        }
        // let sics = DVec::from(self.symmetry_values(&self.geom));
        let b_sym = self.sym_b_matrix(&self.geom);
        let a = Intder::a_matrix(&b_sym);
        let (_xs, srs) = self.machx(&a);
        // let (_ys, _srsy) = self.machy(&a);
        let (f2, f3) = self.lintr(&b_sym, &b_sym, &srs);

        (f2, f3)
    }
}
