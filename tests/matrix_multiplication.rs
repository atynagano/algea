//! Tests for row-major matrix multiplication.

use algea::{
    Vector,
    row_major::{Matrix, MatrixProduct, OuterProduct, VectorMatrixProduct},
};

macro_rules! matrix_from_rows {
    ($rows:expr) => {
        Matrix::from($rows)
    };
}
macro_rules! matrix_to_rows {
    ($matrix:expr) => {
        <_>::from($matrix)
    };
}

trait VectorProduct<const R: usize, const C: usize> {
    fn product(lhs: [f32; R], rhs: [[f32; C]; R]) -> [f32; C];
}
impl<const R: usize, const C: usize> VectorProduct<R, C> for f32
where
    f32: VectorMatrixProduct<R, C>,
{
    fn product(lhs: [f32; R], rhs: [[f32; C]; R]) -> [f32; C] {
        (Vector::from(lhs) * Matrix::from(rhs)).into()
    }
}

trait OuterProductAdapter<const R: usize, const C: usize> {
    fn product(lhs: [f32; R], rhs: [f32; C]) -> [[f32; C]; R];
}
impl<const R: usize, const C: usize> OuterProductAdapter<R, C> for f32
where
    f32: OuterProduct<R, C>,
{
    fn product(lhs: [f32; R], rhs: [f32; C]) -> [[f32; C]; R] {
        let lhs = Matrix::from(core::array::from_fn(|i| [lhs[i]]));
        (lhs * Vector::from(rhs)).into()
    }
}

include!("common/matrix_multiplication.rs");
