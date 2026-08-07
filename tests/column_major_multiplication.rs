//! Tests for column-major matrix multiplication.

use algea::{
    Vector,
    column_major::{Matrix, MatrixProduct, MatrixVectorProduct, OuterProduct},
};

fn columns<T: Copy, const R: usize, const C: usize>(rows: [[T; C]; R]) -> [[T; R]; C] {
    core::array::from_fn(|j| core::array::from_fn(|i| rows[i][j]))
}

macro_rules! matrix_from_rows {
    ($rows:expr) => {
        Matrix::from(columns($rows))
    };
}
macro_rules! matrix_to_rows {
    ($matrix:expr) => {{
        let native = <_>::from($matrix);
        columns(native)
    }};
}

trait VectorProduct<const R: usize, const C: usize> {
    fn product(lhs: [f32; R], rhs: [[f32; C]; R]) -> [f32; C];
}
impl<const R: usize, const C: usize> VectorProduct<R, C> for f32
where
    f32: MatrixVectorProduct<C, R>,
{
    fn product(lhs: [f32; R], rhs: [[f32; C]; R]) -> [f32; C] {
        let transposed: [[f32; R]; C] =
            core::array::from_fn(|j| core::array::from_fn(|i| rhs[i][j]));
        let matrix: Matrix<f32, C, R> = Matrix::from(columns(transposed));
        (matrix * Vector::from(lhs)).into()
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
        let matrix: Matrix<f32, 1, C> = Matrix::from(core::array::from_fn(|j| [rhs[j]]));
        columns((Vector::from(lhs) * matrix).into())
    }
}

include!("common/matrix_multiplication.rs");
