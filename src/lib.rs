use std::{
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader, Read, Write},
};

pub mod geom;
pub mod hmat;
pub mod htens;
pub mod tensor;

use geom::Geom;
use hmat::Hmat;
use htens::Htens;
use nalgebra as na;
use regex::Regex;
use symm::Irrep;
use tensor::{Tensor3, Tensor4};

/// from <https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0>
pub const ANGBOHR: f64 = 0.5291_772_109;
/// constants from the fortran version
const HART: f64 = 4.3597482;
// const DEBYE: f64 = 2.54176548;

// flags
pub static mut VERBOSE: bool = false;

// TODO make these input or flag params
const TOLDISP: f64 = 1e-14;
const MAX_ITER: usize = 20;

type Vec3 = na::Vector3<f64>;
pub type DMat = na::DMatrix<f64>;
pub type DVec = na::DVector<f64>;

#[derive(Debug, PartialEq, Clone)]
pub enum Siic {
    /// bond stretch between two atoms
    Stretch(usize, usize),

    /// central atom is second like normal people would expect
    Bend(usize, usize, usize),

    /// angle between planes formed by i, j, k and j, k, l
    Torsion(usize, usize, usize, usize),

    /// linear bend of atoms `i`, `j`, and `k`, about `d`, a dummy atom
    /// perpendicular to the line `i`-`j`-`k` and extending from atom `j`
    Lin1(usize, usize, usize, usize),
}

impl Display for Siic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Siic::Stretch(i, j) => write!(f, "r({}-{})", i + 1, j + 1),
            Siic::Bend(i, j, k) => {
                write!(f, "∠({}-{}-{})", i + 1, j + 1, k + 1)
            }
            Siic::Torsion(i, j, k, l) => {
                write!(f, "τ({}-{}-{}-{})", i + 1, j + 1, k + 1, l + 1)
            }
            Siic::Lin1(i, j, k, l) => {
                write!(f, "LIN({}-{}-{}-{})", i + 1, j + 1, k + 1, l + 1)
            }
        }
    }
}

impl Siic {
    pub fn value(&self, geom: &Geom) -> f64 {
        use Siic::*;
        match self {
            Stretch(a, b) => geom.dist(*a, *b),
            Bend(a, b, c) => geom.angle(*a, *b, *c),
            // vect6
            Torsion(a, b, c, d) => {
                let e_21 = geom.unit(*b, *a);
                let e_32 = geom.unit(*c, *b);
                let e_43 = geom.unit(*d, *c);
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
            // vect3
            Lin1(a, b, c, d) => {
                let e21 = geom.unit(*b, *a);
                let e23 = geom.unit(*c, *b);
                let ea = geom[*d];
                let d = {
                    let d = ea.dot(&ea);
                    1.0 / d.sqrt()
                };
                let ea = d * ea;
                let e2m = e23.cross(&e21);
                let stheta = ea.dot(&e2m);
                let w = f64::asin(stheta);
                -w
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Atom {
    pub label: String,
    pub weight: usize,
}

impl Display for Intder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Siic::*;
        writeln!(f, "# INTDER ###############")?;
        for (i, op) in self.input_options.iter().enumerate() {
            if i == 16 {
                writeln!(f)?;
            }
            write!(f, "{:5}", op)?;
        }
        writeln!(f)?;
        for siic in &self.simple_internals {
            match siic {
                Stretch(i, j) => {
                    writeln!(f, "{:<5}{:5}{:5}", "STRE", i + 1, j + 1)?
                }
                Bend(i, j, k) => writeln!(
                    f,
                    "{:<5}{:5}{:5}{:5}",
                    "BEND",
                    i + 1,
                    j + 1,
                    k + 1
                )?,
                Torsion(i, j, k, l) => writeln!(
                    f,
                    "{:<5}{:5}{:5}{:5}{:5}",
                    "TORS",
                    i + 1,
                    j + 1,
                    k + 1,
                    l + 1
                )?,
                Lin1(i, j, k, l) => writeln!(
                    f,
                    "{:<5}{:5}{:5}{:5}{:5}",
                    "LIN1",
                    i + 1,
                    j + 1,
                    k + 1,
                    l + 1
                )?,
            }
        }
        for (i, sic) in self.symmetry_internals.iter().enumerate() {
            write!(f, "{:5}", i + 1)?;
            for (j, s) in sic.iter().enumerate() {
                if *s != 0.0 {
                    let sign = s.signum();
                    write!(f, "{:4}{:14.9}", j + 1, sign)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "{:5}", 0)?;
        write!(f, "{}", self.geom)?;
        if !self.disps.is_empty() {
            writeln!(f, "DISP{:5}", self.disps.len())?;
            for disp in &self.disps {
                for (i, d) in disp.iter().enumerate() {
                    if *d != 0.0 {
                        writeln!(f, "{:5}{:20.10}", i + 1, d)?;
                    }
                }
                writeln!(f, "{:5}", 0)?;
            }
        } else {
            // assume freqs
            let nsic = self.symmetry_internals.len();
            for i in 1..=nsic {
                for j in 1..=i {
                    if let Some(v) = self.fc2.get(fc2_index(nsic, i, j)) {
                        writeln!(f, "{:5}{:5}{:5}{:5}{:20.12}", i, j, 0, 0, v)?;
                    }
                }
            }
            writeln!(f, "{:5}", 0)?;
            for i in 1..=nsic {
                for j in 1..=i {
                    for k in 1..=j {
                        if let Some(v) = self.fc3.get(fc3_index(i, j, k)) {
                            writeln!(
                                f,
                                "{:5}{:5}{:5}{:5}{:20.12}",
                                i, j, k, 0, v,
                            )?;
                        }
                    }
                }
            }
            writeln!(f, "{:5}", 0)?;
            for i in 1..=nsic {
                for j in 1..=i {
                    for k in 1..=j {
                        for l in 1..=k {
                            if let Some(v) = self.fc4.get(fc4_index(i, j, k, l))
                            {
                                writeln!(
                                    f,
                                    "{:5}{:5}{:5}{:5}{:20.12}",
                                    i, j, k, l, v
                                )?;
                            }
                        }
                    }
                }
            }
            writeln!(f, "{:5}", 0)?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
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

/// compute the index in the second-order force constant array, assuming `i`
/// and `j` have minimum values of 1
pub fn fc2_index(ncoords: usize, i: usize, j: usize) -> usize {
    let mut sp = [i, j];
    sp.sort();
    ncoords * (sp[0] - 1) + sp[1] - 1
}

pub fn fc3_index(i: usize, j: usize, k: usize) -> usize {
    let mut sp = [i, j, k];
    sp.sort();
    sp[0] + (sp[1] - 1) * sp[1] / 2 + (sp[2] - 1) * sp[2] * (sp[2] + 1) / 6 - 1
}

pub fn fc4_index(i: usize, j: usize, k: usize, l: usize) -> usize {
    let mut sp = [i, j, k, l];
    sp.sort();
    sp[0]
        + (sp[1] - 1) * sp[1] / 2
        + (sp[2] - 1) * sp[2] * (sp[2] + 1) / 6
        + (sp[3] - 1) * sp[3] * (sp[3] + 1) * (sp[3] + 2) / 24
        - 1
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
            "LIN1" => Siic::Lin1(
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
                    disp_tmp = vec![0.0; intder.symmetry_internals.len()];
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
                disp_tmp = vec![0.0; intder.symmetry_internals.len()];
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
                // here
                intder.add_fc(sp, val);
            }
        }
        intder
    }

    pub fn add_fc(&mut self, sp: Vec<usize>, val: f64) {
        let (idx, target) = match (sp[2], sp[3]) {
            (0, 0) => (
                fc2_index(self.symmetry_internals.len(), sp[0], sp[1]),
                &mut self.fc2,
            ),
            (_, 0) => (fc3_index(sp[0], sp[1], sp[2]), &mut self.fc3),
            (_, _) => (fc4_index(sp[0], sp[1], sp[2], sp[3]), &mut self.fc4),
        };
        if target.len() <= idx {
            target.resize(idx + 1, 0.0);
        }
        target[idx] = val;
    }

    /// return the number of symmetry internal coordinates
    pub fn nsym(&self) -> usize {
        self.symmetry_internals.len()
    }

    /// return the number of cartesian coordinates
    pub fn ncart(&self) -> usize {
        3 * self.geom.len()
    }

    /// return the number of dummy atoms
    pub fn ndum(&self) -> usize {
        self.input_options[7]
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
    pub fn print_simple_values(&self, vals: &[f64]) {
        for (i, v) in vals.iter().enumerate() {
            if let Siic::Bend(_, _, _) = self.simple_internals[i] {
                println!("{:5}{:>18.10}", i, v.to_degrees());
            } else {
                println!("{:5}{:>18.10}", i, v);
            }
        }
    }

    /// print the symmetry internal coordinate values
    pub fn print_symmetry_values(&self, vals: &[f64]) {
        for (i, v) in vals.iter().enumerate() {
            println!("{:5}{:>18.10}", i, v);
        }
    }

    pub fn print_sics<W: std::io::Write>(&self, w: &mut W, irreps: &[Irrep]) {
        assert_eq!(self.symmetry_internals.len(), irreps.len());
        for (i, sic) in self.symmetry_internals.iter().enumerate() {
            write!(w, "S{i:<2}({}) = ", irreps[i]).unwrap();
            // number of siics printed so far
            let mut nprt = 0;
            for (j, s) in sic.iter().enumerate() {
                if *s != 0.0 {
                    if nprt > 0 {
                        let sign = match s.signum() as isize {
                            -1 => "-",
                            1 => "+",
                            _ => panic!("it's NaN"),
                        };
                        write!(w, " {sign} ").unwrap();
                    }
                    write!(w, "{}", &self.simple_internals[j]).unwrap();
                    nprt += 1;
                }
            }
            writeln!(w).unwrap();
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

    /// return the B matrix in simple internal coordinates
    pub fn b_matrix(&self, geom: &Geom) -> DMat {
        let mut b_mat = Vec::new();
        for ic in &self.simple_internals {
            b_mat.extend(geom.s_vec(ic));
        }
        DMat::from_row_slice(
            self.simple_internals.len(),
            3 * geom.len(),
            &b_mat,
        )
    }

    /// return the U matrix, used for converting from simple internals to
    /// symmetry internals. dimensions are (number of symmetry internals) x
    /// (number of simple internals) since each symm. int. is a vector simp.
    /// int. long
    pub fn u_mat(&self) -> DMat {
        let r = self.symmetry_internals.len();
        let c = self.simple_internals.len();
        let mut u = Vec::new();
        for i in 0..r {
            u.extend(&self.symmetry_internals[i].clone());
        }
        DMat::from_row_slice(r, c, &u)
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
        self.print_simple_values(&simple_vals);
        println!();
        println!(
        "VALUES OF SYMMETRY INTERNAL COORDINATES (ANG. or RAD.) FOR REFERENCE \
	     GEOMETRY\n"
    );
        self.print_symmetry_values(&sic_vals);
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
                self.print_symmetry_values(sic_desired.as_slice());
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

    /// returns X and SR matrices in symmetry internal coordinates
    pub fn machx(&self, a_mat: &DMat) -> (Vec<DMat>, Vec<DMat>) {
        use Siic::*;
        let nc = self.ncart();
        let nsym = self.nsym();
        let u = self.u_mat();
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
            let h = Hmat::new(&self.geom, &s);
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
                    ahx3(nsym, a_mat, l1, l2, l3, &mut x, &h);
                }
                Torsion(a, b, c, d) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    let l3 = 3 * c;
                    let l4 = 3 * d;
                    for j in 0..3 {
                        for i in 0..3 {
                            sr[(l1 + i, l1 + j)] = h.h11[(i, j)];
                            sr[(l2 + i, l1 + j)] = h.h21[(i, j)];
                            sr[(l3 + i, l1 + j)] = h.h31[(i, j)];
                            sr[(l4 + i, l1 + j)] = h.h41[(i, j)];
                            sr[(l1 + i, l2 + j)] = h.h21[(j, i)];
                            sr[(l2 + i, l2 + j)] = h.h22[(i, j)];
                            sr[(l3 + i, l2 + j)] = h.h32[(i, j)];
                            sr[(l4 + i, l2 + j)] = h.h42[(i, j)];
                            sr[(l1 + i, l3 + j)] = h.h31[(j, i)];
                            sr[(l2 + i, l3 + j)] = h.h32[(j, i)];
                            sr[(l3 + i, l3 + j)] = h.h33[(i, j)];
                            sr[(l4 + i, l3 + j)] = h.h43[(i, j)];
                            sr[(l1 + i, l4 + j)] = h.h41[(j, i)];
                            sr[(l2 + i, l4 + j)] = h.h42[(j, i)];
                            sr[(l3 + i, l4 + j)] = h.h43[(j, i)];
                            sr[(l4 + i, l4 + j)] = h.h44[(i, j)];
                        }
                    }
                    // AHX4
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
                                    let w4 =
                                        a_mat[(l4 + i, m)] * a_mat[(l4 + j, n)];
                                    x[(m, n)] = x[(m, n)]
                                        + w1 * h.h11[(i, j)]
                                        + w2 * h.h22[(i, j)]
                                        + w3 * h.h33[(i, j)]
                                        + w4 * h.h44[(i, j)];
                                    let w1 = a_mat[(l2 + i, m)]
                                        * a_mat[(l1 + j, n)]
                                        + a_mat[(l1 + j, m)]
                                            * a_mat[(l2 + i, n)];
                                    let w2 = a_mat[(l3 + i, m)]
                                        * a_mat[(l1 + j, n)]
                                        + a_mat[(l1 + j, m)]
                                            * a_mat[(l3 + i, n)];
                                    let w3 = a_mat[(l4 + i, m)]
                                        * a_mat[(l1 + j, n)]
                                        + a_mat[(l1 + j, m)]
                                            * a_mat[(l4 + i, n)];
                                    x[(m, n)] = x[(m, n)]
                                        + w1 * h.h21[(i, j)]
                                        + w2 * h.h31[(i, j)]
                                        + w3 * h.h41[(i, j)];
                                    let w1 = a_mat[(l3 + i, m)]
                                        * a_mat[(l2 + j, n)]
                                        + a_mat[(l2 + j, m)]
                                            * a_mat[(l3 + i, n)];
                                    let w2 = a_mat[(l4 + i, m)]
                                        * a_mat[(l2 + j, n)]
                                        + a_mat[(l2 + j, m)]
                                            * a_mat[(l4 + i, n)];
                                    let w3 = a_mat[(l4 + i, m)]
                                        * a_mat[(l3 + j, n)]
                                        + a_mat[(l3 + j, m)]
                                            * a_mat[(l4 + i, n)];
                                    x[(m, n)] = x[(m, n)]
                                        + w1 * h.h32[(i, j)]
                                        + w2 * h.h42[(i, j)]
                                        + w3 * h.h43[(i, j)];
                                }
                            }
                        }
                    }
                }
                Lin1(a, b, c, _) => {
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
                    ahx3(nsym, a_mat, l1, l2, l3, &mut x, &h);
                }
            }
            for n in 0..nsym {
                for m in 0..=n {
                    x[(n, m)] = x[(m, n)];
                }
            }
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

    /// returns the Y and SR matrices in symmetry internal coordinates
    pub fn machy(&self, a_mat: &DMat) -> (Vec<Tensor3>, Vec<Tensor3>) {
        use Siic::*;
        let nc = self.ncart();
        let nsx = self.nsym();
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
            let h = Htens::new(&self.geom, &s);
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
                    hsry3(&mut sr, l1, &h, l2, l3);
                    // AHY3
                    ahy3(nsx, a_mat, l1, l2, l3, &mut y, &h);
                }
                Torsion(a, b, c, d) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    let l3 = 3 * c;
                    let l4 = 3 * d;
                    // HSRY4
                    for k in 0..3 {
                        for j in 0..3 {
                            for i in 0..3 {
                                sr[(l1 + i, l1 + j, l1 + k)] =
                                    h.h111[(i, j, k)];
                                sr[(l1 + i, l1 + j, l2 + k)] =
                                    h.h112[(i, j, k)];
                                sr[(l1 + i, l1 + j, l3 + k)] =
                                    h.h113[(i, j, k)];
                                sr[(l1 + i, l1 + j, l4 + k)] =
                                    h.h411[(k, j, i)];
                                sr[(l1 + i, l2 + j, l1 + k)] =
                                    h.h112[(i, k, j)];
                                sr[(l1 + i, l2 + j, l2 + k)] =
                                    h.h221[(j, k, i)];
                                sr[(l1 + i, l2 + j, l3 + k)] =
                                    h.h123[(i, j, k)];
                                sr[(l1 + i, l2 + j, l4 + k)] =
                                    h.h421[(k, j, i)];
                                sr[(l1 + i, l3 + j, l1 + k)] =
                                    h.h113[(i, k, j)];
                                sr[(l1 + i, l3 + j, l2 + k)] =
                                    h.h123[(i, k, j)];
                                sr[(l1 + i, l3 + j, l3 + k)] =
                                    h.h331[(j, k, i)];
                                sr[(l1 + i, l3 + j, l4 + k)] =
                                    h.h431[(k, j, i)];
                                sr[(l1 + i, l4 + j, l1 + k)] =
                                    h.h411[(j, k, i)];
                                sr[(l1 + i, l4 + j, l2 + k)] =
                                    h.h421[(j, k, i)];
                                sr[(l1 + i, l4 + j, l3 + k)] =
                                    h.h431[(j, k, i)];
                                sr[(l1 + i, l4 + j, l4 + k)] =
                                    h.h441[(j, k, i)];
                                sr[(l2 + i, l1 + j, l1 + k)] =
                                    h.h112[(j, k, i)];
                                sr[(l2 + i, l1 + j, l2 + k)] =
                                    h.h221[(i, k, j)];
                                sr[(l2 + i, l1 + j, l3 + k)] =
                                    h.h123[(j, i, k)];
                                sr[(l2 + i, l1 + j, l4 + k)] =
                                    h.h421[(k, i, j)];
                                sr[(l2 + i, l2 + j, l1 + k)] =
                                    h.h221[(i, j, k)];
                                sr[(l2 + i, l2 + j, l2 + k)] =
                                    h.h222[(i, j, k)];
                                sr[(l2 + i, l2 + j, l3 + k)] =
                                    h.h223[(i, j, k)];
                                sr[(l2 + i, l2 + j, l4 + k)] =
                                    h.h422[(k, j, i)];
                                sr[(l2 + i, l3 + j, l1 + k)] =
                                    h.h123[(k, i, j)];
                                sr[(l2 + i, l3 + j, l2 + k)] =
                                    h.h223[(i, k, j)];
                                sr[(l2 + i, l3 + j, l3 + k)] =
                                    h.h332[(j, k, i)];
                                sr[(l2 + i, l3 + j, l4 + k)] =
                                    h.h432[(k, j, i)];
                                sr[(l2 + i, l4 + j, l1 + k)] =
                                    h.h421[(j, i, k)];
                                sr[(l2 + i, l4 + j, l2 + k)] =
                                    h.h422[(j, i, k)];
                                sr[(l2 + i, l4 + j, l3 + k)] =
                                    h.h432[(j, k, i)];
                                sr[(l2 + i, l4 + j, l4 + k)] =
                                    h.h442[(j, k, i)];
                                sr[(l3 + i, l1 + j, l1 + k)] =
                                    h.h113[(j, k, i)];
                                sr[(l3 + i, l1 + j, l2 + k)] =
                                    h.h123[(j, k, i)];
                                sr[(l3 + i, l1 + j, l3 + k)] =
                                    h.h331[(i, k, j)];
                                sr[(l3 + i, l1 + j, l4 + k)] =
                                    h.h431[(k, i, j)];
                                sr[(l3 + i, l2 + j, l1 + k)] =
                                    h.h123[(k, j, i)];
                                sr[(l3 + i, l2 + j, l2 + k)] =
                                    h.h223[(j, k, i)];
                                sr[(l3 + i, l2 + j, l3 + k)] =
                                    h.h332[(i, k, j)];
                                sr[(l3 + i, l2 + j, l4 + k)] =
                                    h.h432[(k, i, j)];
                                sr[(l3 + i, l3 + j, l1 + k)] =
                                    h.h331[(i, j, k)];
                                sr[(l3 + i, l3 + j, l2 + k)] =
                                    h.h332[(i, j, k)];
                                sr[(l3 + i, l3 + j, l3 + k)] =
                                    h.h333[(i, j, k)];
                                sr[(l3 + i, l3 + j, l4 + k)] =
                                    h.h433[(k, i, j)];
                                sr[(l3 + i, l4 + j, l1 + k)] =
                                    h.h431[(j, i, k)];
                                sr[(l3 + i, l4 + j, l2 + k)] =
                                    h.h432[(j, i, k)];
                                sr[(l3 + i, l4 + j, l3 + k)] =
                                    h.h433[(j, i, k)];
                                sr[(l3 + i, l4 + j, l4 + k)] =
                                    h.h443[(j, k, i)];
                                sr[(l4 + i, l1 + j, l1 + k)] =
                                    h.h411[(i, j, k)];
                                sr[(l4 + i, l1 + j, l2 + k)] =
                                    h.h421[(i, k, j)];
                                sr[(l4 + i, l1 + j, l3 + k)] =
                                    h.h431[(i, k, j)];
                                sr[(l4 + i, l1 + j, l4 + k)] =
                                    h.h441[(i, k, j)];
                                sr[(l4 + i, l2 + j, l1 + k)] =
                                    h.h421[(i, j, k)];
                                sr[(l4 + i, l2 + j, l2 + k)] =
                                    h.h422[(i, j, k)];
                                sr[(l4 + i, l2 + j, l3 + k)] =
                                    h.h432[(i, k, j)];
                                sr[(l4 + i, l2 + j, l4 + k)] =
                                    h.h442[(i, k, j)];
                                sr[(l4 + i, l3 + j, l1 + k)] =
                                    h.h431[(i, j, k)];
                                sr[(l4 + i, l3 + j, l2 + k)] =
                                    h.h432[(i, j, k)];
                                sr[(l4 + i, l3 + j, l3 + k)] =
                                    h.h433[(i, j, k)];
                                sr[(l4 + i, l3 + j, l4 + k)] =
                                    h.h443[(i, k, j)];
                                sr[(l4 + i, l4 + j, l1 + k)] =
                                    h.h441[(i, j, k)];
                                sr[(l4 + i, l4 + j, l2 + k)] =
                                    h.h442[(i, j, k)];
                                sr[(l4 + i, l4 + j, l3 + k)] =
                                    h.h443[(i, j, k)];
                                sr[(l4 + i, l4 + j, l4 + k)] =
                                    h.h444[(i, j, k)];
                            }
                        }
                    }
                    // AHY4
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
                                        let v4 = a_mat[(l4 + i, m)]
                                            * a_mat[(l4 + j, n)];
                                        let v5 = a_mat[(l1 + i, m)]
                                            * a_mat[(l1 + j, p)];
                                        let v6 = a_mat[(l1 + i, n)]
                                            * a_mat[(l1 + j, p)];
                                        let v7 = a_mat[(l2 + i, m)]
                                            * a_mat[(l2 + j, p)];
                                        let v8 = a_mat[(l2 + i, n)]
                                            * a_mat[(l2 + j, p)];
                                        let v9 = a_mat[(l3 + i, m)]
                                            * a_mat[(l3 + j, p)];
                                        let v10 = a_mat[(l3 + i, n)]
                                            * a_mat[(l3 + j, p)];
                                        let v11 = a_mat[(l4 + i, m)]
                                            * a_mat[(l4 + j, p)];
                                        let v12 = a_mat[(l4 + i, n)]
                                            * a_mat[(l4 + j, p)];
                                        let v13 = a_mat[(l1 + i, m)]
                                            * a_mat[(l2 + j, n)];
                                        let v14 = a_mat[(l1 + i, m)]
                                            * a_mat[(l2 + j, p)];
                                        let v15 = a_mat[(l1 + i, n)]
                                            * a_mat[(l2 + j, m)];
                                        let v16 = a_mat[(l1 + i, n)]
                                            * a_mat[(l2 + j, p)];
                                        let v17 = a_mat[(l1 + i, p)]
                                            * a_mat[(l2 + j, m)];
                                        let v18 = a_mat[(l1 + i, p)]
                                            * a_mat[(l2 + j, n)];
                                        let v19 = a_mat[(l4 + i, m)]
                                            * a_mat[(l2 + j, n)];
                                        let v20 = a_mat[(l4 + i, m)]
                                            * a_mat[(l2 + j, p)];
                                        let v21 = a_mat[(l4 + i, n)]
                                            * a_mat[(l2 + j, m)];
                                        let v22 = a_mat[(l4 + i, n)]
                                            * a_mat[(l2 + j, p)];
                                        let v23 = a_mat[(l4 + i, p)]
                                            * a_mat[(l2 + j, m)];
                                        let v24 = a_mat[(l4 + i, p)]
                                            * a_mat[(l2 + j, n)];
                                        let v25 = a_mat[(l4 + i, m)]
                                            * a_mat[(l3 + j, n)];
                                        let v26 = a_mat[(l4 + i, m)]
                                            * a_mat[(l3 + j, p)];
                                        let v27 = a_mat[(l4 + i, n)]
                                            * a_mat[(l3 + j, m)];
                                        let v28 = a_mat[(l4 + i, n)]
                                            * a_mat[(l3 + j, p)];
                                        let v29 = a_mat[(l4 + i, p)]
                                            * a_mat[(l3 + j, m)];
                                        let v30 = a_mat[(l4 + i, p)]
                                            * a_mat[(l3 + j, n)];
                                        for k in 0..3 {
                                            let w1 = v1 * a_mat[(l1 + k, p)];
                                            let w2 = v2 * a_mat[(l2 + k, p)];
                                            let w3 = v3 * a_mat[(l3 + k, p)];
                                            let w4 = v4 * a_mat[(l4 + k, p)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h111[(i, j, k)]
                                                + w2 * h.h222[(i, j, k)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w3 * h.h333[(i, j, k)]
                                                + w4 * h.h444[(i, j, k)];
                                            let w1 = v1 * a_mat[(l2 + k, p)]
                                                + v5 * a_mat[(l2 + k, n)]
                                                + v6 * a_mat[(l2 + k, m)];
                                            let w2 = v1 * a_mat[(l3 + k, p)]
                                                + v5 * a_mat[(l3 + k, n)]
                                                + v6 * a_mat[(l3 + k, m)];
                                            let w3 = v3 * a_mat[(l2 + k, p)]
                                                + v9 * a_mat[(l2 + k, n)]
                                                + v10 * a_mat[(l2 + k, m)];
                                            let w4 = v3 * a_mat[(l1 + k, p)]
                                                + v9 * a_mat[(l1 + k, n)]
                                                + v10 * a_mat[(l1 + k, m)];
                                            let w5 = v2 * a_mat[(l1 + k, p)]
                                                + v7 * a_mat[(l1 + k, n)]
                                                + v8 * a_mat[(l1 + k, m)];
                                            let w6 = v2 * a_mat[(l3 + k, p)]
                                                + v7 * a_mat[(l3 + k, n)]
                                                + v8 * a_mat[(l3 + k, m)];
                                            let w7 = v1 * a_mat[(l4 + k, p)]
                                                + v5 * a_mat[(l4 + k, n)]
                                                + v6 * a_mat[(l4 + k, m)];
                                            let w8 = v2 * a_mat[(l4 + k, p)]
                                                + v7 * a_mat[(l4 + k, n)]
                                                + v8 * a_mat[(l4 + k, m)];
                                            let w9 = v3 * a_mat[(l4 + k, p)]
                                                + v9 * a_mat[(l4 + k, n)]
                                                + v10 * a_mat[(l4 + k, m)];
                                            let w10 = v4 * a_mat[(l1 + k, p)]
                                                + v11 * a_mat[(l1 + k, n)]
                                                + v12 * a_mat[(l1 + k, m)];
                                            let w11 = v4 * a_mat[(l2 + k, p)]
                                                + v11 * a_mat[(l2 + k, n)]
                                                + v12 * a_mat[(l2 + k, m)];
                                            let w12 = v4 * a_mat[(l3 + k, p)]
                                                + v11 * a_mat[(l3 + k, n)]
                                                + v12 * a_mat[(l3 + k, m)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h112[(i, j, k)]
                                                + w2 * h.h113[(i, j, k)]
                                                + w3 * h.h332[(i, j, k)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w4 * h.h331[(i, j, k)]
                                                + w5 * h.h221[(i, j, k)]
                                                + w6 * h.h223[(i, j, k)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w7 * h.h411[(k, i, j)]
                                                + w8 * h.h422[(k, i, j)]
                                                + w9 * h.h433[(k, i, j)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w10 * h.h441[(i, j, k)]
                                                + w11 * h.h442[(i, j, k)]
                                                + w12 * h.h443[(i, j, k)];
                                            let w1 = a_mat[(l3 + k, p)]
                                                * (v13 + v15)
                                                + a_mat[(l3 + k, n)]
                                                    * (v14 + v17)
                                                + a_mat[(l3 + k, m)]
                                                    * (v16 + v18);
                                            let w2 = a_mat[(l1 + k, p)]
                                                * (v19 + v21)
                                                + a_mat[(l1 + k, n)]
                                                    * (v20 + v23)
                                                + a_mat[(l1 + k, m)]
                                                    * (v22 + v24);
                                            let w3 = a_mat[(l1 + k, p)]
                                                * (v25 + v27)
                                                + a_mat[(l1 + k, n)]
                                                    * (v26 + v29)
                                                + a_mat[(l1 + k, m)]
                                                    * (v28 + v30);
                                            let w4 = a_mat[(l2 + k, p)]
                                                * (v25 + v27)
                                                + a_mat[(l2 + k, n)]
                                                    * (v26 + v29)
                                                + a_mat[(l2 + k, m)]
                                                    * (v28 + v30);
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w1 * h.h123[(i, j, k)]
                                                + w2 * h.h421[(i, j, k)];
                                            y[(m, n, p)] = y[(m, n, p)]
                                                + w3 * h.h431[(i, j, k)]
                                                + w4 * h.h432[(i, j, k)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Lin1(a, b, c, _) => {
                    let l1 = 3 * a;
                    let l2 = 3 * b;
                    let l3 = 3 * c;
                    // HSRY3
                    hsry3(&mut sr, l1, &h, l2, l3);
                    // AHY3
                    ahy3(nsx, a_mat, l1, l2, l3, &mut y, &h);
                }
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
                v.push(
                    self.fc2[fc2_index(
                        self.symmetry_internals.len(),
                        i + 1,
                        j + 1,
                    )],
                );
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
        let nsx = self.ncart() - 3 * self.ndum();
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
        let f2 = f2 * ANGBOHR * ANGBOHR / HART;
        f2.resize(nsx, nsx, 0.0)
    }

    pub fn lintr_fc3(&self, a: &DMat) -> Tensor3 {
        let nsx = self.ncart() - 3 * self.ndum();
        let nsy = self.nsym();
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

    pub fn lintr_fc4(&self, a: &DMat) -> Tensor4 {
        let nsx = self.ncart() - 3 * self.ndum();
        let nsy = self.nsym();
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
        // end 179 loop, looking good so far

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
        // flatten f4_disk into the vector the fortran uses
        let v = {
            let mut v = Vec::new();
            for i in 0..nsy {
                for j in 0..=i {
                    for k in 0..=j {
                        for q in 0..nsx {
                            v.push(f4_disk[(i, j, k, q)]);
                        }
                    }
                }
            }
            v
        };

        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        let mut q = 0;
        // loop starting at line 444
        for vik in v {
            if i != j {
                if j != k {
                    for p in 0..=q {
                        f4[(i, j, p, q)] += vik * a[(k, p)];
                        f4[(i, k, p, q)] += vik * a[(j, p)];
                        f4[(j, k, p, q)] += vik * a[(i, p)];
                    }
                } else {
                    for p in 0..=q {
                        f4[(i, j, p, q)] += vik * a[(j, p)];
                        f4[(j, j, p, q)] += vik * a[(i, p)];
                    }
                }
            } else {
                if j != k {
                    for p in 0..=q {
                        f4[(i, i, p, q)] += vik * a[(k, p)];
                        f4[(i, k, p, q)] += vik * a[(i, p)];
                    }
                } else {
                    for p in 0..=q {
                        f4[(i, i, p, q)] += vik * a[(i, p)];
                    }
                }
            }
            if q < nsx - 1 {
                q += 1;
            } else if k < j {
                k += 1;
                q = 0;
            } else if j < i {
                j += 1;
                k = 0;
                q = 0;
            } else {
                i += 1;
                j = 0;
                k = 0;
                q = 0;
            }
        }
        // end 200 loop

        let f4_disk2 = f4.clone();
        for q in 0..nsx {
            for p in 0..=q {
                for n in 0..=p {
                    for i in 0..nsy {
                        f4[(i, n, p, q)] = 0.0;
                    }
                }
            }
        }
        // flatten f4_disk2 into the vector the fortran uses
        let v = {
            let mut v = Vec::new();
            for i in 0..nsy {
                for j in 0..=i {
                    for q in 0..nsx {
                        for p in 0..=q {
                            v.push(f4_disk2[(i, j, p, q)]);
                        }
                    }
                }
            }
            v
        };

        let mut p = 0;
        let mut q = 0;
        let mut i = 0;
        let mut j = 0;
        // start of loop at 514
        for vik in v {
            if i != j {
                for n in 0..=p {
                    f4[(i, n, p, q)] += vik * a[(j, n)];
                    f4[(j, n, p, q)] += vik * a[(i, n)];
                }
            } else {
                for n in 0..=p {
                    f4[(i, n, p, q)] += vik * a[(i, n)];
                }
            }
            if p < q {
                p += 1;
            } else if q < nsx - 1 {
                q += 1;
                p = 0;
            } else if j < i {
                j += 1;
                p = 0;
                q = 0;
            } else {
                i += 1;
                j = 0;
                p = 0;
                q = 0;
            }
        }
        // end of 214 loop, looking good

        let v = {
            let mut v = Vec::new();
            for q in 0..nsx {
                for p in 0..=q {
                    for n in 0..=p {
                        for i in 0..nsy {
                            v.push(f4[(i, n, p, q)]);
                        }
                    }
                }
            }
            v
        };

        for q in 0..nsx {
            for p in 0..=q {
                for n in 0..=p {
                    for m in 0..=n {
                        f4[(m, n, p, q)] = 0.0;
                    }
                }
            }
        }

        let mut n = 0;
        let mut p = 0;
        let mut q = 0;
        let mut i = 0;
        // begin 224 loop
        for vik in v {
            for m in 0..=n {
                f4[(m, n, p, q)] += vik * a[(i, m)];
            }
            if i < nsy - 1 {
                i += 1;
            } else if n < p {
                n += 1;
                i = 0;
            } else if p < q {
                p += 1;
                n = 0;
                i = 0;
            } else {
                q += 1;
                p = 0;
                n = 0;
                i = 0;
            }
        }
        // end 224 loop

        f4.fill4a(nsx);
        f4
    }

    fn xf2(
        &self,
        mut f3: Tensor3,
        mut f4: Tensor4,
        bs: &DMat,
        xrs: &Vec<DMat>,
    ) -> (Tensor3, Tensor4) {
        let ns = self.symmetry_internals.len();
        let nc = self.ncart() - 3 * self.ndum();
        let v = self.mat_fc2(ns);
        let xs = v * bs;
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

        // begin F4 part

        // this should never be used as zeros. it will be set on the first
        // iteration when r == s == 0
        let mut xt = DMat::zeros(nc, nc);
        let v = self.mat_fc2(ns);
        for r in 0..ns {
            let mut xr = DMat::zeros(nc, nc);
            for (s, xs) in xrs.iter().enumerate() {
                if r == s {
                    xt = xs.clone();
                }
                for j in 0..nc {
                    for i in 0..nc {
                        xr[(i, j)] += v[(r, s)] * xs[(i, j)];
                    }
                }
            }
            for l in 0..nc {
                for k in 0..=l {
                    for j in 0..=k {
                        for i in 0..=j {
                            let w = xt[(i, j)] * xr[(k, l)]
                                + xt[(i, k)] * xr[(j, l)]
                                + xt[(i, l)] * xr[(j, k)];
                            f4[(i, j, k, l)] += w;
                        }
                    }
                }
            }
        }
        f4.fill4a(nc);
        (f3, f4)
    }

    fn xf3(&self, mut f4: Tensor4, bs: &DMat, xrs: &Vec<DMat>) -> Tensor4 {
        let ns = self.nsym();
        let nc = self.ncart() - 3 * self.ndum();
        // might need to turn this into a Tensor3 and call FILL3B on it, but
        // we'll see
        let yr = &self.fc3;
        for (r, xr) in xrs.iter().enumerate() {
            let mut xt = DMat::zeros(ns, nc);
            for l in 0..nc {
                for n in 0..ns {
                    for p in 0..ns {
                        if let Some(x) = yr.get(fc3_index(r + 1, n + 1, p + 1))
                        {
                            xt[(n, l)] += x * bs[(p, l)]
                        }
                        // else that element of fc3 was zero so the addition
                        // would be zero as well
                    }
                }
            }
            let mut xs = DMat::zeros(nc, nc);
            for l in 0..nc {
                for k in 0..nc {
                    for n in 0..ns {
                        // I think this is xt.transpose() * bs
                        xs[(k, l)] += xt[(n, l)] * bs[(n, k)];
                    }
                }
            }
            for l in 0..nc {
                for k in 0..=l {
                    for j in 0..=k {
                        for i in 0..=j {
                            let w = xr[(i, j)] * xs[(k, l)]
                                + xr[(i, k)] * xs[(j, l)]
                                + xr[(j, k)] * xs[(i, l)]
                                + xr[(k, l)] * xs[(i, j)]
                                + xr[(j, l)] * xs[(i, k)]
                                + xr[(i, l)] * xs[(j, k)];
                            f4[(i, j, k, l)] += w;
                        }
                    }
                }
            }
        }
        f4.fill4a(nc);
        f4
    }

    fn yf2(&self, mut f4: Tensor4, bs: &DMat, yrs: &Vec<Tensor3>) -> Tensor4 {
        let nc = self.ncart() - 3 * self.ndum();
        let ns = self.nsym();
        let xs = self.mat_fc2(ns);
        let xr = xs * bs;
        for (r, yr) in yrs.iter().enumerate() {
            for l in 0..nc {
                for k in 0..=l {
                    for j in 0..=k {
                        for i in 0..=j {
                            let w = yr[(i, j, k)] * xr[(r, l)]
                                + yr[(i, j, l)] * xr[(r, k)]
                                + yr[(i, k, l)] * xr[(r, j)]
                                + yr[(j, k, l)] * xr[(r, i)];
                            f4[(i, j, k, l)] += w;
                        }
                    }
                }
            }
        }
        f4.fill4a(nc);
        f4
    }

    /// Perform the linear transformation of the force constants and convert the
    /// units to those desired by SPECTRO. what they call A here is actually the
    /// SIC B matrix.
    pub fn lintr(
        &self,
        a: &DMat,
        bs: &DMat,
        xr: &Vec<DMat>,
        yr: &Vec<Tensor3>,
    ) -> (DMat, Vec<f64>, Vec<f64>) {
        let f2 = self.lintr_fc2(a);
        let (f3, f4) = self.xf2(self.lintr_fc3(a), self.lintr_fc4(a), bs, xr);
        let f4 = self.xf3(f4, bs, xr);
        let f4 = self.yf2(f4, bs, yr);

        // convert f3 to the proper units and return it as a Vec in the order
        // desired by spectro
        let nsx = self.ncart() - 3 * self.ndum();
        const CF3: f64 = ANGBOHR * ANGBOHR * ANGBOHR / HART;
        let mut f3_out = Vec::new();
        for m in 0..nsx {
            for n in 0..=m {
                for p in 0..=n {
                    f3_out.push(CF3 * f3[(m, n, p)]);
                }
            }
        }

        // convert f4 to the proper units and return it as a Vec in the order
        // desired by spectro
        const CF4: f64 = ANGBOHR * ANGBOHR * ANGBOHR * ANGBOHR / HART;
        let mut f4_out = Vec::new();
        for m in 0..nsx {
            for n in 0..=m {
                for p in 0..=n {
                    for q in 0..=p {
                        f4_out.push(CF4 * f4[(m, n, p, q)]);
                    }
                }
            }
        }
        (f2, f3_out, f4_out)
    }

    /// convert the force constants in `self.fc[234]` from (symmetry) internal
    /// coordinates to Cartesian coordinates. returns (fc2, fc3, fc4) in the
    /// order printed in the fort.{15,30,40} files for spectro.
    pub fn convert_fcs(&self) -> (DMat, Vec<f64>, Vec<f64>) {
        if unsafe { VERBOSE } {
            self.print_init();
        }
        // let sics = DVec::from(self.symmetry_values(&self.geom));
        let b_sym = self.sym_b_matrix(&self.geom);
        let a = Intder::a_matrix(&b_sym);
        let (_xs, srs) = self.machx(&a);
        let (_ys, srsy) = self.machy(&a);
        let (f2, f3, f4) = self.lintr(&b_sym, &b_sym, &srs, &srsy);

        (f2, f3, f4)
    }

    pub fn dump_fcs(dir: &str, f2: &DMat, f3: &[f64], f4: &[f64]) {
        let f2 = f2.as_slice();
        let pairs = [(f2, "fort.15"), (&f3, "fort.30"), (&f4, "fort.40")];
        for p in pairs {
            let mut f = File::create(format!("{dir}/{}", p.1))
                .expect("failed to create fort.15");
            for chunk in p.0.chunks(3) {
                for c in chunk {
                    write!(f, "{:>20.10}", c).unwrap();
                }
                writeln!(f).unwrap();
            }
        }
    }
}

fn ahy3(
    nsx: usize,
    a_mat: &DMat,
    l1: usize,
    l2: usize,
    l3: usize,
    y: &mut Tensor3,
    h: &Htens,
) {
    for p in 0..nsx {
        for n in 0..=p {
            for m in 0..=n {
                for i in 0..3 {
                    for j in 0..3 {
                        let v1 = a_mat[(l1 + i, m)] * a_mat[(l1 + j, n)];
                        let v2 = a_mat[(l2 + i, m)] * a_mat[(l2 + j, n)];
                        let v3 = a_mat[(l3 + i, m)] * a_mat[(l3 + j, n)];
                        let v4 = a_mat[(l1 + i, m)] * a_mat[(l1 + j, p)];
                        let v5 = a_mat[(l1 + i, n)] * a_mat[(l1 + j, p)];
                        let v6 = a_mat[(l2 + i, m)] * a_mat[(l2 + j, p)];
                        let v7 = a_mat[(l2 + i, n)] * a_mat[(l2 + j, p)];
                        let v8 = a_mat[(l3 + i, m)] * a_mat[(l3 + j, p)];
                        let v9 = a_mat[(l3 + i, n)] * a_mat[(l3 + j, p)];
                        let v10 = a_mat[(l1 + i, m)] * a_mat[(l2 + j, n)];
                        let v11 = a_mat[(l1 + i, m)] * a_mat[(l2 + j, p)];
                        let v12 = a_mat[(l1 + i, n)] * a_mat[(l2 + j, m)];
                        let v13 = a_mat[(l1 + i, n)] * a_mat[(l2 + j, p)];
                        let v14 = a_mat[(l1 + i, p)] * a_mat[(l2 + j, m)];
                        let v15 = a_mat[(l1 + i, p)] * a_mat[(l2 + j, n)];
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
                            let w1 = a_mat[(l3 + k, p)] * (v10 + v12)
                                + a_mat[(l3 + k, n)] * (v11 + v14)
                                + a_mat[(l3 + k, m)] * (v13 + v15);
                            y[(m, n, p)] =
                                y[(m, n, p)] + w1 * h.h123[(i, j, k)];
                        }
                    }
                }
            }
        }
    }
}

fn hsry3(sr: &mut Tensor3, l1: usize, h: &Htens, l2: usize, l3: usize) {
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                sr[(l1 + i, l1 + j, l1 + k)] = h.h111[(i, j, k)];
                sr[(l1 + i, l1 + j, l2 + k)] = h.h112[(i, j, k)];
                sr[(l1 + i, l1 + j, l3 + k)] = h.h113[(i, j, k)];
                sr[(l1 + i, l2 + j, l1 + k)] = h.h112[(i, k, j)];
                sr[(l1 + i, l2 + j, l2 + k)] = h.h221[(j, k, i)];
                sr[(l1 + i, l2 + j, l3 + k)] = h.h123[(i, j, k)];
                sr[(l2 + i, l2 + j, l1 + k)] = h.h221[(i, j, k)];
                sr[(l2 + i, l2 + j, l2 + k)] = h.h222[(i, j, k)];
                sr[(l2 + i, l2 + j, l3 + k)] = h.h223[(i, j, k)];
                sr[(l2 + i, l1 + j, l1 + k)] = h.h112[(j, k, i)];
                sr[(l2 + i, l1 + j, l2 + k)] = h.h221[(i, k, j)];
                sr[(l2 + i, l1 + j, l3 + k)] = h.h123[(j, i, k)];
                sr[(l1 + i, l3 + j, l1 + k)] = h.h113[(i, k, j)];
                sr[(l1 + i, l3 + j, l2 + k)] = h.h123[(i, k, j)];
                sr[(l1 + i, l3 + j, l3 + k)] = h.h331[(j, k, i)];
                sr[(l2 + i, l3 + j, l1 + k)] = h.h123[(k, i, j)];
                sr[(l2 + i, l3 + j, l2 + k)] = h.h223[(i, k, j)];
                sr[(l2 + i, l3 + j, l3 + k)] = h.h332[(j, k, i)];
                sr[(l3 + i, l1 + j, l1 + k)] = h.h113[(j, k, i)];
                sr[(l3 + i, l1 + j, l2 + k)] = h.h123[(j, k, i)];
                sr[(l3 + i, l1 + j, l3 + k)] = h.h331[(i, k, j)];
                sr[(l3 + i, l2 + j, l1 + k)] = h.h123[(k, j, i)];
                sr[(l3 + i, l2 + j, l2 + k)] = h.h223[(j, k, i)];
                sr[(l3 + i, l2 + j, l3 + k)] = h.h332[(i, k, j)];
                sr[(l3 + i, l3 + j, l1 + k)] = h.h331[(i, j, k)];
                sr[(l3 + i, l3 + j, l2 + k)] = h.h332[(i, j, k)];
                sr[(l3 + i, l3 + j, l3 + k)] = h.h333[(i, j, k)];
            }
        }
    }
}

fn ahx3(
    nsym: usize,
    a_mat: &DMat,
    l1: usize,
    l2: usize,
    l3: usize,
    x: &mut DMat,
    h: &Hmat,
) {
    for n in 0..nsym {
        for m in 0..=n {
            for i in 0..3 {
                for j in 0..3 {
                    let w1 = a_mat[(l1 + i, m)] * a_mat[(l1 + j, n)];
                    let w2 = a_mat[(l2 + i, m)] * a_mat[(l2 + j, n)];
                    let w3 = a_mat[(l3 + i, m)] * a_mat[(l3 + j, n)];
                    x[(m, n)] += w1 * h.h11[(i, j)]
                        + w2 * h.h22[(i, j)]
                        + w3 * h.h33[(i, j)];
                    let w1 = a_mat[(l2 + i, m)] * a_mat[(l1 + j, n)]
                        + a_mat[(l1 + j, m)] * a_mat[(l2 + i, n)];
                    let w2 = a_mat[(l3 + i, m)] * a_mat[(l1 + j, n)]
                        + a_mat[(l1 + j, m)] * a_mat[(l3 + i, n)];
                    let w3 = a_mat[(l3 + i, m)] * a_mat[(l2 + j, n)]
                        + a_mat[(l2 + j, m)] * a_mat[(l3 + i, n)];
                    x[(m, n)] += w1 * h.h21[(i, j)]
                        + w2 * h.h31[(i, j)]
                        + w3 * h.h32[(i, j)];
                }
            }
        }
    }
}
