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

trait VectorProduct<const R: usize, const C: usize>: Sized {
    fn product(lhs: [Self; R], rhs: [[Self; C]; R]) -> [Self; C];
}

trait OuterProductAdapter<const R: usize, const C: usize>: Sized {
    fn product(lhs: [Self; R], rhs: [Self; C]) -> [[Self; C]; R];
}

macro_rules! impl_adapters {
    ($t:ty) => {
        impl<const R: usize, const C: usize> VectorProduct<R, C> for $t
        where
            $t: MatrixVectorProduct<C, R>,
        {
            fn product(lhs: [$t; R], rhs: [[$t; C]; R]) -> [$t; C] {
                let transposed: [[$t; R]; C] =
                    core::array::from_fn(|j| core::array::from_fn(|i| rhs[i][j]));
                let matrix: Matrix<$t, C, R> = Matrix::from(columns(transposed));
                (matrix * Vector::from(lhs)).into()
            }
        }

        impl<const R: usize, const C: usize> OuterProductAdapter<R, C> for $t
        where
            $t: OuterProduct<R, C>,
        {
            fn product(lhs: [$t; R], rhs: [$t; C]) -> [[$t; C]; R] {
                let matrix: Matrix<$t, 1, C> = Matrix::from(core::array::from_fn(|j| [rhs[j]]));
                columns((Vector::from(lhs) * matrix).into())
            }
        }
    };
}
impl_adapters!(f32);
impl_adapters!(f64);

include!("common/matrix_multiplication.rs");
