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
            $t: VectorMatrixProduct<R, C>,
        {
            fn product(lhs: [$t; R], rhs: [[$t; C]; R]) -> [$t; C] {
                (Vector::from(lhs) * Matrix::from(rhs)).into()
            }
        }

        impl<const R: usize, const C: usize> OuterProductAdapter<R, C> for $t
        where
            $t: OuterProduct<R, C>,
        {
            fn product(lhs: [$t; R], rhs: [$t; C]) -> [[$t; C]; R] {
                let lhs = Matrix::from(core::array::from_fn(|i| [lhs[i]]));
                (lhs * Vector::from(rhs)).into()
            }
        }
    };
}
impl_adapters!(f32);
impl_adapters!(f64);

include!("common/matrix_multiplication.rs");
