use std::fs::File;
use std::io::Write;

use rust_intder::{Intder, VERBOSE};

fn parse_args<I: Iterator<Item = String>>(args: &mut I) -> Option<String> {
    let mut hold = Vec::new();
    for arg in args {
        if arg == "-v" {
            unsafe { VERBOSE = true };
        } else {
            hold.push(arg);
        }
    }
    match hold.get(1) {
        Some(s) => Some(String::from(s)),
        None => None,
    }
}

#[test]
fn test_parse_args() {
    assert_eq!(
        parse_args(
            &mut vec!["rust-intder", "-v", "intder.in"]
                .into_iter()
                .map(|s| s.to_string())
        ),
        Some("intder.in".to_string())
    );

    assert_eq!(
        parse_args(
            &mut vec!["rust-intder", "intder.in"]
                .into_iter()
                .map(|s| s.to_string())
        ),
        Some("intder.in".to_string())
    );

    assert_eq!(
        parse_args(&mut vec!["rust-intder"].into_iter().map(|s| s.to_string())),
        None,
    );
}

fn main() {
    let mut args = std::env::args();
    let infile = parse_args(&mut args);
    let intder = match infile {
        Some(s) => Intder::load_file(&s),
        None => Intder::load(std::io::stdin()),
    };
    let new_carts = intder.convert_disps();
    let mut file07 = File::create("file07").expect("failed to create file07");
    for cart in new_carts {
        writeln!(file07, "# GEOMUP #################").unwrap();
        Intder::print_cart(&mut file07, &cart);
    }
}
