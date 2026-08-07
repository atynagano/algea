//! Tests for column-major square-matrix kernels.

use algea::column_major::Matrix;

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

include!("common/matrix_square_kernels.rs");
