use rust_intder::{Intder, DEBUG};

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
    intder.convert_disps();
}
