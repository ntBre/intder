use std::fs::File;
use std::io::Write;

use rust_intder::{Intder, VERBOSE};

fn main() {
    let mut args = std::env::args();
    if let Some(_) = args.find(|x| x == "-v") {
        unsafe { VERBOSE = true };
    }
    let infile = match args.last() {
        Some(f) => f,
        None => {
            eprintln!("expected input file name");
            std::process::exit(1);
        }
    };
    let intder = Intder::load(&infile);
    let new_carts = intder.convert_disps();
    let mut file07 = File::create("file07").expect("failed to create file07");
    for cart in new_carts {
        writeln!(file07, "# GEOMUP #################").unwrap();
        Intder::print_cart(&mut file07, &cart);
    }
}
