//! Tests for constant vector and matrix constructors.

use algea::{Vector, row_major::Matrix};

macro_rules! check_matrix_from_rows {
    ($name:ident, $t:ty, $r:literal, $c:literal, $rows:expr) => {
        #[test]
        fn $name() {
            const ROWS: [[$t; $c]; $r] = $rows;
            const MATRIX: Matrix<$t, $r, $c> = Matrix::<$t, $r, $c>::from_rows(ROWS);
            let actual: [[$t; $c]; $r] = MATRIX.into();
            assert_eq!(actual, ROWS);
        }
    };
}

macro_rules! check_matrix_zero_one {
    ($name:ident, $r:literal, $c:literal) => {
        #[test]
        fn $name() {
            assert_eq!(<[[f32; $c]; $r]>::from(Matrix::<f32, $r, $c>::ZERO), [[0.0; $c]; $r]);
            assert_eq!(<[[f32; $c]; $r]>::from(Matrix::<f32, $r, $c>::ONE), [[1.0; $c]; $r]);
            assert_eq!(<[[f64; $c]; $r]>::from(Matrix::<f64, $r, $c>::ZERO), [[0.0; $c]; $r]);
            assert_eq!(<[[f64; $c]; $r]>::from(Matrix::<f64, $r, $c>::ONE), [[1.0; $c]; $r]);
            assert_eq!(<[[i64; $c]; $r]>::from(Matrix::<i64, $r, $c>::ZERO), [[0; $c]; $r]);
            assert_eq!(<[[i64; $c]; $r]>::from(Matrix::<i64, $r, $c>::ONE), [[1; $c]; $r]);
            assert_eq!(<[[u64; $c]; $r]>::from(Matrix::<u64, $r, $c>::ZERO), [[0; $c]; $r]);
            assert_eq!(<[[u64; $c]; $r]>::from(Matrix::<u64, $r, $c>::ONE), [[1; $c]; $r]);
        }
    };
}

check_matrix_zero_one!(zero_one_1x1, 1, 1);
check_matrix_zero_one!(zero_one_1x2, 1, 2);
check_matrix_zero_one!(zero_one_1x3, 1, 3);
check_matrix_zero_one!(zero_one_1x4, 1, 4);
check_matrix_zero_one!(zero_one_2x1, 2, 1);
check_matrix_zero_one!(zero_one_2x2, 2, 2);
check_matrix_zero_one!(zero_one_2x3, 2, 3);
check_matrix_zero_one!(zero_one_2x4, 2, 4);
check_matrix_zero_one!(zero_one_3x1, 3, 1);
check_matrix_zero_one!(zero_one_3x2, 3, 2);
check_matrix_zero_one!(zero_one_3x3, 3, 3);
check_matrix_zero_one!(zero_one_3x4, 3, 4);
check_matrix_zero_one!(zero_one_4x1, 4, 1);
check_matrix_zero_one!(zero_one_4x2, 4, 2);
check_matrix_zero_one!(zero_one_4x3, 4, 3);
check_matrix_zero_one!(zero_one_4x4, 4, 4);

check_matrix_from_rows!(f32_1x1, f32, 1, 1, [[1.0]]);
check_matrix_from_rows!(f32_1x2, f32, 1, 2, [[1.0, 2.0]]);
check_matrix_from_rows!(f32_1x3, f32, 1, 3, [[1.0, 2.0, 3.0]]);
check_matrix_from_rows!(f32_1x4, f32, 1, 4, [[1.0, 2.0, 3.0, 4.0]]);
check_matrix_from_rows!(f32_2x1, f32, 2, 1, [[1.0], [2.0]]);
check_matrix_from_rows!(f32_2x2, f32, 2, 2, [[1.0, 2.0], [3.0, 4.0]]);
check_matrix_from_rows!(f32_2x3, f32, 2, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
check_matrix_from_rows!(f32_2x4, f32, 2, 4, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
check_matrix_from_rows!(f32_3x1, f32, 3, 1, [[1.0], [2.0], [3.0]]);
check_matrix_from_rows!(f32_3x2, f32, 3, 2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
check_matrix_from_rows!(f32_3x3, f32, 3, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
check_matrix_from_rows!(f32_3x4, f32, 3, 4, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [
    9.0, 10.0, 11.0, 12.0
]]);
check_matrix_from_rows!(f32_4x1, f32, 4, 1, [[1.0], [2.0], [3.0], [4.0]]);
check_matrix_from_rows!(f32_4x2, f32, 4, 2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
check_matrix_from_rows!(f32_4x3, f32, 4, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [
    10.0, 11.0, 12.0
]]);
check_matrix_from_rows!(f32_4x4, f32, 4, 4, [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
]);

#[test]
fn special_from_rows_constructors_preserve_rows() {
    const M2: Matrix<f32, 2, 2> = Matrix::<f32, 2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    const M43: Matrix<f32, 4, 3> =
        Matrix::<f32, 4, 3>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [
            10.0, 11.0, 12.0,
        ]]);
    const M4: Matrix<f32, 4, 4> = Matrix::<f32, 4, 4>::from_rows([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);

    assert_eq!(<[[_; 2]; 2]>::from(M2), [[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(<[[_; 3]; 4]>::from(M43), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [
        10.0, 11.0, 12.0
    ]]);
    assert_eq!(<[[_; 4]; 4]>::from(M4), [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
}

#[test]
fn identity_constants_preserve_public_coordinates() {
    assert_eq!(<[[f32; 2]; 2]>::from(Matrix::<f32, 2, 2>::IDENTITY), [[1.0, 0.0], [0.0, 1.0]]);
    assert_eq!(<[[f32; 3]; 3]>::from(Matrix::<f32, 3, 3>::IDENTITY), [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]);
    assert_eq!(<[[f32; 4]; 4]>::from(Matrix::<f32, 4, 4>::IDENTITY), [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
}

#[test]
fn integer_and_non_square_constants_preserve_public_coordinates() {
    const M43: Matrix<i32, 4, 3> =
        Matrix::<i32, 4, 3>::from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

    assert_eq!(<[[i32; 3]; 4]>::from(M43), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    assert_eq!(<[[i32; 3]; 2]>::from(Matrix::<i32, 2, 3>::ZERO), [[0; 3]; 2]);
    assert_eq!(<[[i32; 3]; 2]>::from(Matrix::<i32, 2, 3>::ONE), [[1; 3]; 2]);
    assert_eq!(<[[i32; 3]; 3]>::from(Matrix::<i32, 3, 3>::IDENTITY), [[1, 0, 0], [0, 1, 0], [
        0, 0, 1
    ],]);
    assert_eq!(<[i32; 4]>::from(Vector::<i32, 4>::NEG_W), [0, 0, 0, -1]);
}

#[test]
fn vector_const_constructors_and_directions_preserve_lane_order() {
    const V1: Vector<f32, 1> = Vector::<f32, 1>::from_array([1.0]);
    const V2: Vector<f32, 2> = Vector::<f32, 2>::from_array([1.0, 2.0]);
    const V3: Vector<f32, 3> = Vector::<f32, 3>::from_array([1.0, 2.0, 3.0]);
    const V4: Vector<f32, 4> = Vector::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);

    assert_eq!(<[f32; 1]>::from(V1), [1.0]);
    assert_eq!(<[f32; 2]>::from(V2), [1.0, 2.0]);
    assert_eq!(<[f32; 3]>::from(V3), [1.0, 2.0, 3.0]);
    assert_eq!(<[f32; 4]>::from(V4), [1.0, 2.0, 3.0, 4.0]);

    assert_eq!(<[f32; 2]>::from(Vector::<f32, 2>::POS_X), [1.0, 0.0]);
    assert_eq!(<[f32; 2]>::from(Vector::<f32, 2>::NEG_Y), [0.0, -1.0]);
    assert_eq!(<[f32; 3]>::from(Vector::<f32, 3>::POS_Z), [0.0, 0.0, 1.0]);
    assert_eq!(<[f32; 3]>::from(Vector::<f32, 3>::NEG_X), [-1.0, 0.0, 0.0]);
    assert_eq!(<[f32; 4]>::from(Vector::<f32, 4>::POS_W), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(<[f32; 4]>::from(Vector::<f32, 4>::NEG_Z), [0.0, 0.0, -1.0, 0.0]);
}
