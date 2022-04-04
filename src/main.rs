use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let infile = match args.get(1) {
        Some(f) => f,
        None => {
            eprintln!("expected input filename");
            std::process::exit(1);
        }
    };

    let f = match File::open(infile) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("failed to open infile '{}'", infile);
            std::process::exit(1);
        }
    };

    let reader = BufReader::new(f);
    for line in reader.lines() {
	let line = line.unwrap();
        dbg!(line);
    }
}
