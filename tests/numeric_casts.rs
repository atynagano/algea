//! Tests for numeric vector and matrix casts.

use algea::{Vector, column_major, row_major};

macro_rules! assert_vector_cast {
    ($src:ty => $dst:ty, $d:literal, $values:expr) => {{
        let source: [$src; $d] = $values;
        let expected: [$dst; $d] = source.map(|value| value as $dst);
        let actual: [$dst; $d] = Vector::<$src, $d>::from(source).cast::<$dst>().into();
        assert_eq!(actual, expected, "{} -> {}, D={}", stringify!($src), stringify!($dst), $d);
    }};
}

macro_rules! assert_all_vector_casts {
    ($d:literal, $floats:expr, $signed:expr, $unsigned:expr) => {{
        assert_vector_cast!(f32 => f32, $d, $floats);
        assert_vector_cast!(f32 => i32, $d, $floats);
        assert_vector_cast!(f32 => u32, $d, $floats);
        assert_vector_cast!(i32 => f32, $d, $signed);
        assert_vector_cast!(i32 => i32, $d, $signed);
        assert_vector_cast!(i32 => u32, $d, $signed);
        assert_vector_cast!(u32 => f32, $d, $unsigned);
        assert_vector_cast!(u32 => i32, $d, $unsigned);
        assert_vector_cast!(u32 => u32, $d, $unsigned);
    }};
}

#[test]
fn vector_casts_match_scalar_as_for_all_types_and_dimensions() {
    assert_all_vector_casts!(1, [-1.75], [i32::MIN], [u32::MAX]);
    assert_all_vector_casts!(2, [-0.0, 1.75], [i32::MIN, i32::MAX], [0, u32::MAX]);
    assert_all_vector_casts!(3, [-12.75, 0.0, 16_777_217.0], [-1, 0, 16_777_217], [
        0,
        16_777_217,
        u32::MAX
    ]);
    assert_all_vector_casts!(
        4,
        [-1.5, 0.0, 1.5, 4_294_967_296.0],
        [i32::MIN, -16_777_217, 16_777_217, i32::MAX],
        [0, 1, 16_777_217, u32::MAX]
    );
}

#[test]
fn float_to_integer_casts_match_scalar_as_at_special_values_and_boundaries() {
    let inputs = [
        [f32::NAN, f32::NEG_INFINITY, -1.9, -0.0],
        [0.0, 1.9, 2_147_483_648.0, f32::INFINITY],
        [f32::MIN, f32::MAX, 4_294_967_296.0, -2_147_483_648.0],
    ];

    for input in inputs {
        assert_vector_cast!(f32 => i32, 4, input);
        assert_vector_cast!(f32 => u32, 4, input);
    }
}

#[test]
fn identity_float_cast_preserves_bits() {
    let source = [f32::from_bits(0x7fc1_2345), f32::from_bits(0xffc5_4321), -0.0, f32::INFINITY];
    let actual: [f32; 4] = Vector::<f32, 4>::from(source).cast::<f32>().into();

    assert_eq!(actual.map(f32::to_bits), source.map(f32::to_bits));
}

macro_rules! assert_row_major_cast {
    ($src:ty => $dst:ty, $r:literal, $c:literal, $values:expr) => {{
        let source: [[$src; $c]; $r] = $values;
        let expected: [[$dst; $c]; $r] = source.map(|row| row.map(|value| value as $dst));
        let actual: [[$dst; $c]; $r] =
            row_major::Matrix::<$src, $r, $c>::from(source).cast::<$dst>().into();
        assert_eq!(
            actual,
            expected,
            "row-major {} -> {}, {}x{}",
            stringify!($src),
            stringify!($dst),
            $r,
            $c
        );
    }};
}

macro_rules! assert_column_major_cast {
    ($src:ty => $dst:ty, $r:literal, $c:literal, $values:expr) => {{
        let source: [[$src; $r]; $c] = $values;
        let expected: [[$dst; $r]; $c] = source.map(|column| column.map(|value| value as $dst));
        let actual: [[$dst; $r]; $c] =
            column_major::Matrix::<$src, $r, $c>::from(source).cast::<$dst>().into();
        assert_eq!(
            actual,
            expected,
            "column-major {} -> {}, {}x{}",
            stringify!($src),
            stringify!($dst),
            $r,
            $c
        );
    }};
}

macro_rules! assert_all_matrix_casts {
    ($r:literal, $c:literal) => {{
        let float_rows: [[f32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column) as f32 - 3.5));
        let signed_rows: [[i32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column) as i32 - 5));
        let unsigned_rows: [[u32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column + 1) as u32));

        assert_row_major_cast!(f32 => f32, $r, $c, float_rows);
        assert_row_major_cast!(f32 => i32, $r, $c, float_rows);
        assert_row_major_cast!(f32 => u32, $r, $c, float_rows);
        assert_row_major_cast!(i32 => f32, $r, $c, signed_rows);
        assert_row_major_cast!(i32 => i32, $r, $c, signed_rows);
        assert_row_major_cast!(i32 => u32, $r, $c, signed_rows);
        assert_row_major_cast!(u32 => f32, $r, $c, unsigned_rows);
        assert_row_major_cast!(u32 => i32, $r, $c, unsigned_rows);
        assert_row_major_cast!(u32 => u32, $r, $c, unsigned_rows);

        let float_columns: [[f32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column) as f32 - 3.5));
        let signed_columns: [[i32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column) as i32 - 5));
        let unsigned_columns: [[u32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column + 1) as u32));

        assert_column_major_cast!(f32 => f32, $r, $c, float_columns);
        assert_column_major_cast!(f32 => i32, $r, $c, float_columns);
        assert_column_major_cast!(f32 => u32, $r, $c, float_columns);
        assert_column_major_cast!(i32 => f32, $r, $c, signed_columns);
        assert_column_major_cast!(i32 => i32, $r, $c, signed_columns);
        assert_column_major_cast!(i32 => u32, $r, $c, signed_columns);
        assert_column_major_cast!(u32 => f32, $r, $c, unsigned_columns);
        assert_column_major_cast!(u32 => i32, $r, $c, unsigned_columns);
        assert_column_major_cast!(u32 => u32, $r, $c, unsigned_columns);
    }};
}

#[test]
fn matrix_casts_match_scalar_as_for_all_types_shapes_and_layouts() {
    assert_all_matrix_casts!(1, 1);
    assert_all_matrix_casts!(1, 2);
    assert_all_matrix_casts!(1, 3);
    assert_all_matrix_casts!(1, 4);
    assert_all_matrix_casts!(2, 1);
    assert_all_matrix_casts!(2, 2);
    assert_all_matrix_casts!(2, 3);
    assert_all_matrix_casts!(2, 4);
    assert_all_matrix_casts!(3, 1);
    assert_all_matrix_casts!(3, 2);
    assert_all_matrix_casts!(3, 3);
    assert_all_matrix_casts!(3, 4);
    assert_all_matrix_casts!(4, 1);
    assert_all_matrix_casts!(4, 2);
    assert_all_matrix_casts!(4, 3);
    assert_all_matrix_casts!(4, 4);
}
