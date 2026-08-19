//! Tests for vector, mask, and matrix debug output.

use algea::{Mask, Vector, column_major, row_major};

macro_rules! float_vector_debug {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let vector = Vector::<$t, 4>::from([1.0, -2.5, <$t>::NAN, <$t>::INFINITY]);

            assert_eq!(format!("{vector:?}"), "[1.0, -2.5, NaN, inf]");
            assert_eq!(format!("{vector:#?}"), "[1.0, -2.5, NaN, inf]");
        }
    };
}

float_vector_debug!(vector_debug_is_compact_in_normal_and_pretty_modes, f32);
float_vector_debug!(f64_vector_debug_is_compact_in_normal_and_pretty_modes, f64);

#[test]
fn integer_vector_debug_covers_both_widths() {
    assert_eq!(
        format!("{:?}", Vector::<i64, 3>::from([i64::MIN, 0, i64::MAX])),
        format!("[{}, 0, {}]", i64::MIN, i64::MAX)
    );
    assert_eq!(
        format!("{:?}", Vector::<u64, 2>::from([0, u64::MAX])),
        format!("[0, {}]", u64::MAX)
    );
}

macro_rules! mask_debug {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let mask = Mask::<$t, 4>::from([true, false, true, false]);

            assert_eq!(format!("{mask:?}"), "[true, false, true, false]");
            assert_eq!(format!("{mask:#?}"), "[true, false, true, false]");
        }
    };
}

mask_debug!(mask_debug_is_logical_and_compact_in_normal_and_pretty_modes, i32);
mask_debug!(i64_mask_debug_is_logical_and_compact_in_normal_and_pretty_modes, i64);

macro_rules! matrix_debug {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let row_major = row_major::Matrix::<$t, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);
            let column_major =
                column_major::Matrix::<$t, 2, 3>::from_columns([[1, 4], [2, 5], [3, 6]]);

            assert_eq!(format!("{row_major:?}"), "[[1, 2, 3], [4, 5, 6]]");
            assert_eq!(format!("{column_major:?}"), "[[1, 4], [2, 5], [3, 6]]");
            assert_eq!(format!("{row_major:#?}"), "[\n    [1, 2, 3],\n    [4, 5, 6],\n]");
            assert_eq!(format!("{column_major:#?}"), "[\n    [1, 4],\n    [2, 5],\n    [3, 6],\n]");
        }
    };
}

matrix_debug!(matrix_debug_uses_each_matrix_type_storage_order, i32);
matrix_debug!(i64_matrix_debug_uses_each_matrix_type_storage_order, i64);
matrix_debug!(u64_matrix_debug_uses_each_matrix_type_storage_order, u64);
