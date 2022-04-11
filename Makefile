BASE = /home/brent/Projects/rust-intder
TESTFLAGS = --test-threads=1 --nocapture

test:
	RUST_BACKTRACE=1 cargo test -- ${TESTFLAGS}

run:
	cargo run ../testfiles/intder.in

deploy:
	RUSTFLAGS='-C target-feature=+crt-static' cargo build --release	\
		--target x86_64-unknown-linux-gnu
	scp -C ${BASE}/target/x86_64-unknown-linux-gnu/release/rust-intder \
		'woods:Programs/brentder/.'
