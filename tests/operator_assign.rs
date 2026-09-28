//! Tests for assignment operators.

use algea::{Vector, column_major, row_major};

macro_rules! integer_assignment_tests {
    ($module:ident, $t:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn vector_assignment_operators_update_each_active_lane() {
                let mut vector = Vector::<$t, 3>::from([24, -36, 48]);

                vector += Vector::from([1, 2, 3]);
                assert_eq!(vector.to_array(), [25, -34, 51]);

                vector -= Vector::from([5, -6, 1]);
                assert_eq!(vector.to_array(), [20, -28, 50]);

                vector *= Vector::from([2, -1, 3]);
                assert_eq!(vector.to_array(), [40, 28, 150]);

                vector /= Vector::from([4, 7, 5]);
                assert_eq!(vector.to_array(), [10, 4, 30]);

                vector += 2;
                vector -= 1;
                vector *= 3;
                vector /= 3;
                assert_eq!(vector.to_array(), [11, 5, 31]);
            }

            #[test]
            fn row_major_assignment_operators_preserve_logical_rows() {
                let mut matrix =
                    row_major::Matrix::<$t, 2, 3>::from_rows([[24, -36, 48], [60, -72, 84]]);

                matrix += row_major::Matrix::<$t, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);
                assert_eq!(matrix.to_rows(), [[25, -34, 51], [64, -67, 90]]);

                matrix -= row_major::Matrix::<$t, 2, 3>::from_rows([[5, -6, 1], [4, 3, 2]]);
                assert_eq!(matrix.to_rows(), [[20, -28, 50], [60, -70, 88]]);

                matrix += 2;
                matrix -= 1;
                matrix *= 3;
                matrix /= 3;
                assert_eq!(matrix.to_rows(), [[21, -27, 51], [61, -69, 89]]);
            }

            #[test]
            fn column_major_assignment_operators_preserve_logical_columns() {
                let mut matrix =
                    column_major::Matrix::<$t, 2, 3>::from_columns([[24, 60], [-36, -72], [
                        48, 84,
                    ]]);

                matrix += column_major::Matrix::<$t, 2, 3>::from_columns([[1, 4], [2, 5], [3, 6]]);
                assert_eq!(matrix.to_columns(), [[25, 64], [-34, -67], [51, 90]]);

                matrix -= column_major::Matrix::<$t, 2, 3>::from_columns([[5, 4], [-6, 3], [1, 2]]);
                assert_eq!(matrix.to_columns(), [[20, 60], [-28, -70], [50, 88]]);

                matrix += 2;
                matrix -= 1;
                matrix *= 3;
                matrix /= 3;
                assert_eq!(matrix.to_columns(), [[21, 61], [-27, -69], [51, 89]]);
            }
        }
    };
}

integer_assignment_tests!(i64_assignment, i64);

#[test]
fn vector_assignment_operators_update_each_active_lane() {
    let mut vector = Vector::<i32, 3>::from([24, -36, 48]);

    vector += Vector::from([1, 2, 3]);
    assert_eq!(vector.to_array(), [25, -34, 51]);

    vector -= Vector::from([5, -6, 1]);
    assert_eq!(vector.to_array(), [20, -28, 50]);

    vector *= Vector::from([2, -1, 3]);
    assert_eq!(vector.to_array(), [40, 28, 150]);

    vector /= Vector::from([4, 7, 5]);
    assert_eq!(vector.to_array(), [10, 4, 30]);

    vector += 2;
    vector -= 1;
    vector *= 3;
    vector /= 3;
    assert_eq!(vector.to_array(), [11, 5, 31]);
}

#[test]
fn row_major_matrix_assignment_operators_preserve_logical_rows() {
    let mut matrix = row_major::Matrix::<i32, 2, 3>::from_rows([[24, -36, 48], [60, -72, 84]]);

    matrix += row_major::Matrix::<i32, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(matrix.to_rows(), [[25, -34, 51], [64, -67, 90]]);

    matrix -= row_major::Matrix::<i32, 2, 3>::from_rows([[5, -6, 1], [4, 3, 2]]);
    assert_eq!(matrix.to_rows(), [[20, -28, 50], [60, -70, 88]]);

    matrix += 2;
    matrix -= 1;
    matrix *= 3;
    matrix /= 3;
    assert_eq!(matrix.to_rows(), [[21, -27, 51], [61, -69, 89]]);
}

#[test]
fn column_major_matrix_assignment_operators_preserve_logical_columns() {
    let mut matrix =
        column_major::Matrix::<i32, 2, 3>::from_columns([[24, 60], [-36, -72], [48, 84]]);

    matrix += column_major::Matrix::<i32, 2, 3>::from_columns([[1, 4], [2, 5], [3, 6]]);
    assert_eq!(matrix.to_columns(), [[25, 64], [-34, -67], [51, 90]]);

    matrix -= column_major::Matrix::<i32, 2, 3>::from_columns([[5, 4], [-6, 3], [1, 2]]);
    assert_eq!(matrix.to_columns(), [[20, 60], [-28, -70], [50, 88]]);

    matrix += 2;
    matrix -= 1;
    matrix *= 3;
    matrix /= 3;
    assert_eq!(matrix.to_columns(), [[21, 61], [-27, -69], [51, 89]]);
}

macro_rules! vector_matrix_mul_assign_test {
    ($name:ident, $t:ty, $n:literal, $vector:expr, $matrix:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let mut vector = Vector::<$t, $n>::from($vector);
            vector *= row_major::Matrix::<$t, $n, $n>::from_rows($matrix);
            assert_eq!(vector.to_array(), $expected);
        }
    };
}

macro_rules! all_vector_matrix_mul_assign {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            vector_matrix_mul_assign_test!([<$prefix _1x1>], $t, 1, [2.0], [[3.0]], [6.0]);
            vector_matrix_mul_assign_test!(
                [<$prefix _2x2>], $t, 2,
                [2.0, 3.0],
                [[4.0, 5.0], [6.0, 7.0]],
                [26.0, 31.0]
            );
            vector_matrix_mul_assign_test!(
                [<$prefix _3x3>], $t, 3,
                [2.0, 3.0, 5.0],
                [[7.0, 11.0, 13.0], [17.0, 19.0, 23.0], [29.0, 31.0, 37.0]],
                [210.0, 234.0, 280.0]
            );
            vector_matrix_mul_assign_test!(
                [<$prefix _4x4>], $t, 4,
                [2.0, 3.0, 5.0, 7.0],
                [
                    [11.0, 13.0, 17.0, 19.0],
                    [23.0, 29.0, 31.0, 37.0],
                    [41.0, 43.0, 47.0, 53.0],
                    [59.0, 61.0, 67.0, 71.0],
                ],
                [709.0, 755.0, 831.0, 911.0]
            );
        }
    };
}

all_vector_matrix_mul_assign!(f32, vector_matrix_mul_assign);
all_vector_matrix_mul_assign!(f64, f64_vector_matrix_mul_assign);

macro_rules! matrix_mul_assign_test {
    ($name:ident, $module:ident, $constructor:ident, $extractor:ident, $lhs:expr, $rhs:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let mut matrix = $module::Matrix::<f32, 2, 3>::$constructor($lhs);
            matrix *= $module::Matrix::<f32, 3, 3>::$constructor($rhs);
            assert_eq!(matrix.$extractor(), $expected);
        }
    };
}

matrix_mul_assign_test!(
    row_major_matrix_mul_assign,
    row_major,
    from_rows,
    to_rows,
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
    [[66.0, 72.0, 78.0], [156.0, 171.0, 186.0]]
);
matrix_mul_assign_test!(
    column_major_matrix_mul_assign,
    column_major,
    from_columns,
    to_columns,
    [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    [[7.0, 10.0, 13.0], [8.0, 11.0, 14.0], [9.0, 12.0, 15.0]],
    [[66.0, 156.0], [72.0, 171.0], [78.0, 186.0]]
);
