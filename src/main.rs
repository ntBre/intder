use std::fs::File;
use std::io::Write;

use intder::{Intder, VERBOSE};

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
            &mut vec!["intder", "-v", "intder.in"]
                .into_iter()
                .map(|s| s.to_string())
        ),
        Some("intder.in".to_string())
    );

    assert_eq!(
        parse_args(
            &mut vec!["intder", "intder.in"]
                .into_iter()
                .map(|s| s.to_string())
        ),
        Some("intder.in".to_string())
    );

    assert_eq!(
        parse_args(&mut vec!["intder"].into_iter().map(|s| s.to_string())),
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
    if intder.input_options[14] != 0 {
        let new_carts = intder.convert_disps();
        let mut file07 =
            File::create("file07").expect("failed to create file07");
        for cart in new_carts {
            writeln!(file07, "# GEOMUP #################").unwrap();
            Intder::print_cart(&mut file07, &cart);
        }
    } else {
        let (f2, f3, f4) = intder.convert_fcs();
        let f2 = f2.as_slice();
        let pairs = [(f2, "fort.15"), (&f3, "fort.30"), (&f4, "fort.40")];
        for p in pairs {
            let mut f = File::create(p.1).expect("failed to create fort.15");
            for chunk in p.0.chunks(3) {
                for c in chunk {
                    write!(f, "{:>20.10}", c).unwrap();
                }
                writeln!(f).unwrap();
            }
        }
    }
}
