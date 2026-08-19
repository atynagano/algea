//! Tests for the default values of row-major and column-major matrices.

use algea::{column_major, row_major};

macro_rules! integer_default_tests {
    ($module:ident, $t:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn row_major_default_is_zero_for_square_and_rectangular_shapes() {
                assert_eq!(row_major::Matrix::<$t, 2, 2>::default().to_rows(), [[0; 2]; 2]);
                assert_eq!(row_major::Matrix::<$t, 2, 3>::default().to_rows(), [[0; 3]; 2]);
            }

            #[test]
            fn column_major_default_is_zero_for_square_and_rectangular_shapes() {
                assert_eq!(column_major::Matrix::<$t, 2, 2>::default().to_columns(), [[0; 2]; 2]);
                assert_eq!(column_major::Matrix::<$t, 2, 3>::default().to_columns(), [[0; 2]; 3]);
            }
        }
    };
}

integer_default_tests!(i32_default, i32);
integer_default_tests!(i64_default, i64);
integer_default_tests!(u64_default, u64);

#[test]
fn float_matrix_default_is_zero_for_square_and_rectangular_shapes() {
    assert_eq!(row_major::Matrix::<f64, 2, 2>::default().to_rows(), [[0.0; 2]; 2]);
    assert_eq!(row_major::Matrix::<f64, 2, 3>::default().to_rows(), [[0.0; 3]; 2]);
    assert_eq!(column_major::Matrix::<f64, 2, 2>::default().to_columns(), [[0.0; 2]; 2]);
    assert_eq!(column_major::Matrix::<f64, 2, 3>::default().to_columns(), [[0.0; 2]; 3]);
}
