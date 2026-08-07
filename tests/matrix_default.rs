//! Tests for the default values of row-major and column-major matrices.

use algea::{column_major, row_major};

#[test]
fn row_major_matrix_default_is_zero_for_square_and_rectangular_shapes() {
    assert_eq!(row_major::Matrix::<i32, 2, 2>::default().to_rows(), [[0; 2]; 2]);
    assert_eq!(row_major::Matrix::<i32, 2, 3>::default().to_rows(), [[0; 3]; 2]);
}

#[test]
fn column_major_matrix_default_is_zero_for_square_and_rectangular_shapes() {
    assert_eq!(column_major::Matrix::<i32, 2, 2>::default().to_columns(), [[0; 2]; 2]);
    assert_eq!(column_major::Matrix::<i32, 2, 3>::default().to_columns(), [[0; 2]; 3]);
}
