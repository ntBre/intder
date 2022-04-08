use rust_intder::{DVec, Geom, Intder, ANGBOHR};

const TOLDISP: f64 = 1e-14;
const DEBUG: bool = false;

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
    let simple_vals = intder.simple_values(&intder.geom);
    let sic_vals = intder.symmetry_values(&intder.geom);

    if DEBUG {
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
    }
    for (i, disp) in intder.disps[..2].iter().enumerate() {
        let mut sic_current = DVec::from(sic_vals.clone());
        let mut cart_current: DVec = intder.geom.clone().into();

        let disp = DVec::from(disp.clone());
        let sic_desired = &sic_current + &disp;

        println!("DISPLACEMENT{:5}\n", i);
        if DEBUG {
            println!("INTERNAL DISPLACEMENTS\n");
            for (i, d) in disp.iter().enumerate() {
                if *d != 0.0 {
                    println!("{i:5}{d:20.10}");
                }
            }
            println!();
            println!("SYMMETRY INTERNAL COORDINATE FINAL VALUES\n");
            intder.print_symmetry(sic_desired.as_slice());
            println!();
        }

        let mut iter = 1;
        while (&sic_current - &sic_desired).abs().max() > TOLDISP {
            println!(
                "ITER={:5} MAX INTERNAL DEVIATION = {:.4e}",
                iter,
                (&sic_current - &sic_desired).abs().max()
            );
            let b_sym = intder.sym_b_matrix(&Geom::from(&cart_current));
            let d = &b_sym * b_sym.transpose();
            let a = Intder::a_matrix(&b_sym);

            if DEBUG {
                println!("B*BT MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
                println!("{}", d);

                println!("DETERMINANT OF B*BT MATRIX={:8.3}", d.determinant());

                println!();
                println!("A MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
                println!("{:.8}", a);
                println!();
            }

            let step = a * (&sic_desired - &sic_current);
            cart_current += step / ANGBOHR;

            sic_current =
                DVec::from(intder.symmetry_values(&Geom::from(&cart_current)));

            iter += 1;
        }

        println!("NEW CARTESIAN GEOMETRY (BOHR)\n");
        for i in 0..cart_current.len() / 3 {
            for j in 0..3 {
                print!("{:20.10}", cart_current[3 * i + j]);
            }
            println!();
        }
        println!();

        // todo!();
    }
    println!();
}
