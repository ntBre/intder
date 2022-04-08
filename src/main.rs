use rust_intder::{DVec, Intder};

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

    let sic_vals = DVec::from(sic_vals);

    let bs = intder.sym_b_matrix();
    println!();
    for (i, disp) in intder.disps.iter().enumerate() {
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
        let new_sics = &sic_vals + &disp;
        intder.print_symmetry(new_sics.as_slice());
        println!();

        println!("B*BT MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
        let d = &bs * bs.transpose();
        println!("{}", d);

        println!("DETERMINANT OF B*BT MATRIX={:8.3}", d.determinant());

        println!();
        println!("A MATRIX FOR (SYMMETRY) INTERNAL COORDINATES");
        println!("{}", Intder::a_matrix(&bs));
        println!();

        todo!();
    }
    println!();
}
