//! Criterion benchmarks for `algea` and comparable math crates.

mod common;
mod integer_vector;
mod matmul;
mod matrix_square;
mod vector;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    vector::register,
    integer_vector::register,
    matmul::register,
    matrix_square::register
);
criterion_main!(benches);
