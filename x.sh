#!/bin/bash
set -ex
cargo fmt --all -- --check
cargo check --workspace --all-features --all-targets --release
cargo clippy --workspace --all-features --all-targets --release  -- --deny warnings
cargo build --workspace --all-features --all-targets --release
cargo test --workspace --all-features --all-targets --release
cargo doc --workspace --all-features --no-deps --release
cargo test --workspace --all-features --doc
mdbook build --dest-dir target book
mdbook test --dest-dir target book
