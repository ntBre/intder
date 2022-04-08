use approx::assert_abs_diff_eq;
use rust_intder::{DVec, Intder, ANGBOHR};

const TOLDISP: f64 = 1e-14;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let infile = match args.get(1) {
        Some(f) => f,
        None => {
            eprintln!("expected input filename");
            std::process::exit(1);
        }
    };
    let intder = Intder::load(infile);
    let simple_vals = intder.simple_values();
    let sic_vals = intder.symmetry_values();

    println!();

    println!("NUCLEAR CARTESIAN COORDINATES (BOHR)\n");
    intder.print_geom();
    println!();

    println!(
        "VALUES OF SIMPLE INTERNAL COORDINATES (ANG. or DEG.) FOR REFERENCE \
	 GEOMETRY\n"
    );
    intder.print_simple(&simple_vals);
    println!();

    println!(
        "VALUES OF SYMMETRY INTERNAL COORDINATES (ANG. or RAD.) FOR REFERENCE \
	 GEOMETRY\n"
    );
    intder.print_symmetry(&sic_vals);
    println!();

    println!();
    for (i, disp) in intder.disps[..1].iter().enumerate() {
        let sic_current = DVec::from(sic_vals.clone());
        let mut cart_current = intder.geom_vec();
        let b_sym = intder.sym_b_matrix();

        println!("DISPLACEMENT{:5}\n", i);
        println!("INTERNAL DISPLACEMENTS\n");
        for (i, d) in disp.iter().enumerate() {
            if *d != 0.0 {
                println!("{i:5}{d:20.10}");
            }
        }
        let disp = DVec::from(disp.clone());
        println!();
        println!("SYMMETRY INTERNAL COORDINATE FINAL VALUES\n");
        let sic_desired = &sic_current + &disp;
        intder.print_symmetry(sic_desired.as_slice());
        println!();

        let mut iter = 1;
        while (&sic_current - &sic_desired).abs().max() > TOLDISP {
            println!(
                "ITER={:5} MAX INTERNAL DEVIATION = {:.4e}",
                iter,
                (&sic_current - &sic_desired).abs().max()
            );
            println!("B*BT MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
            let d = &b_sym * b_sym.transpose();
            println!("{}", d);

            println!("DETERMINANT OF B*BT MATRIX={:8.3}", d.determinant());

            println!();
            println!("A MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
            let a = Intder::a_matrix(&b_sym);
            println!("{}", a);
            println!();

            let step = a * (&sic_desired - &sic_current);
            cart_current += step;

            iter += 1;
	    // TODO compute new SIC values and update sic_current
        }

        println!("NEW CARTESIAN GEOMETRY (BOHR)\n");
        for i in 0..cart_current.len() / 3 {
            for j in 0..3 {
                print!("{:20.10}", cart_current[3 * i + j] / ANGBOHR);
            }
            println!();
        }
        println!();

        // todo!();
    }
    println!();
}
