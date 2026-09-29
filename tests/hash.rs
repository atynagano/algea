//! Tests for vector, mask, and matrix hashing.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use algea::{Mask, Vector, column_major, row_major};

fn hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn vector_hash_uses_logical_lanes() {
    let lanes = [1, -2, 3];
    let vector = Vector::<i32, 3>::from(lanes);
    let equal = Vector::<i32, 3>::from_array(lanes);

    assert_eq!(hash(&vector), hash(&equal));
    assert_eq!(hash(&vector), hash(&lanes));
}

#[test]
fn mask_hash_uses_logical_boolean_lanes() {
    let lanes = [true, false, true];
    let mask = Mask::<i64, 3>::from(lanes);
    let equal = Mask::<i64, 3>::from(lanes);

    assert_eq!(hash(&mask), hash(&equal));
    assert_eq!(hash(&mask), hash(&lanes));
}

#[test]
fn row_major_matrix_hash_uses_logical_rows() {
    let rows = [[1, 2, 3], [4, 5, 6]];
    let matrix = row_major::Matrix::<i32, 2, 3>::from_rows(rows);
    let equal = row_major::Matrix::from_row_vecs(matrix.to_row_vecs());

    assert_eq!(hash(&matrix), hash(&equal));
    assert_eq!(hash(&matrix), hash(&rows));
}

#[test]
fn column_major_matrix_hash_uses_logical_columns() {
    let columns = [[1_i64, 4], [2, 5], [3, 6]];
    let matrix = column_major::Matrix::<i64, 2, 3>::from_columns(columns);
    let equal = column_major::Matrix::from_column_vecs(matrix.to_column_vecs());

    assert_eq!(hash(&matrix), hash(&equal));
    assert_eq!(hash(&matrix), hash(&columns));
}
