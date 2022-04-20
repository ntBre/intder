use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
};

pub mod geom;

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
    /// symmetry internals
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

    // TODO figure out what this returns, just copying the fortran for now
    pub fn machx(&self, a_mat: &DMat) {
        use Siic::*;
        let nc = 3 * self.geom.len();
        for s in &self.simple_internals {
            let mut x = DMat::zeros(nc, nc);
            let mut sr = DMat::zeros(nc, nc);
            let h = Self::h_mat(&self.geom, &s);
            println!("H11 = {}", &h.h11);
            println!("H21 = {}", &h.h21);
            println!("H31 = {}", &h.h31);
            println!("H22 = {}", &h.h22);
            println!("H32 = {}", &h.h32);
            println!("H33 = {}", &h.h33);
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
                    for n in 0..self.symmetry_internals.len() {
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
                    for n in 0..self.symmetry_internals.len() {
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
                    for n in 0..self.symmetry_internals.len() {
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
                    for n in 0..self.symmetry_internals.len() {
                        for m in 0..=n {
                            x[(n, m)] = x[(m, n)];
                        }
                    }
                }
                _ => todo!(),
            }
            println!("SR = {:12.8}", sr);
            println!("X = {:12.8}", x);
        }
    }

    /// convert the force constants in `self.fc[234]` from (symmetry) internal
    /// coordinates to Cartesian coordinates
    pub fn convert_fcs(&self) {
        if unsafe { VERBOSE } {
            self.print_init();
        }
        // let sics = DVec::from(self.symmetry_values(&self.geom));
        let b_sym = self.sym_b_matrix(&self.geom);
        let _d = &b_sym * b_sym.transpose();
        let a = Intder::a_matrix(&b_sym);
        println!("a = {:12.8}", a);
        self.machx(&a);
        // A looks good

        // fortran flow is through BINVRT, which I think is our A matrix. Then
        // it loads the fcs into arrays using spectro indexing formulas in
        // INPFKM.

        // [x] I handle this in `load`

        // Then it runs
        // [ ] MACHX - I think UGF and XS are the out params for this
        //     [ ] HIJS1 - H matrix elements for STRE
        //     [ ] AHX2 - I think these are doing some kind of math with A and H
        // [ ] MACHY - Z are work arrays
        // [ ] LINTR - linear transformation

        // TODO short break and then figure out MACHX - break at start, break at
        // end, see what goes in and what comes out

        // Then
        // [ ] XF2
        // [ ] XF3
        // [ ] YF2

        // Then
        // [ ] FCOUT to dump the force constants

        // This is where the files I need get written, so I should be able to
        // stop here.

        // after that, it goes on to do the NORMAL MODE ANALYSIS IN INTERNAL
        // COORDINATES with GFMAT

        // then NORMAL MODE ANALYSIS IN CARTESIAN COORDINATES in NORMCO
    }
}
