TESTFLAGS = --test-threads=1 --nocapture

test:
	cargo test -- ${TESTFLAGS}

run:
	cargo run ../testfiles/intder.in
