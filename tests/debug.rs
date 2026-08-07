//! Tests for vector, mask, and matrix debug output.

use algea::{Mask, Vector, column_major, row_major};

#[test]
fn vector_debug_is_compact_in_normal_and_pretty_modes() {
    let vector = Vector::<f32, 4>::from([1.0, -2.5, f32::NAN, f32::INFINITY]);

    assert_eq!(format!("{vector:?}"), "[1.0, -2.5, NaN, inf]");
    assert_eq!(format!("{vector:#?}"), "[1.0, -2.5, NaN, inf]");
}

#[test]
fn mask_debug_is_logical_and_compact_in_normal_and_pretty_modes() {
    let mask = Mask::<i32, 4>::from([true, false, true, false]);

    assert_eq!(format!("{mask:?}"), "[true, false, true, false]");
    assert_eq!(format!("{mask:#?}"), "[true, false, true, false]");
}

#[test]
fn matrix_debug_uses_each_matrix_type_storage_order() {
    let row_major = row_major::Matrix::<i32, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);
    let column_major = column_major::Matrix::<i32, 2, 3>::from_columns([[1, 4], [2, 5], [3, 6]]);

    assert_eq!(format!("{row_major:?}"), "[[1, 2, 3], [4, 5, 6]]");
    assert_eq!(format!("{column_major:?}"), "[[1, 4], [2, 5], [3, 6]]");
    assert_eq!(format!("{row_major:#?}"), "[\n    [1, 2, 3],\n    [4, 5, 6],\n]");
    assert_eq!(format!("{column_major:#?}"), "[\n    [1, 4],\n    [2, 5],\n    [3, 6],\n]");
}
