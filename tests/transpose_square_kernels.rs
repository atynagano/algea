//! Tests for square-matrix transpose kernels.

use algea::row_major::Matrix;

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

include!("common/matrix_square_kernels.rs");
