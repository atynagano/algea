//! Tests for matrix indexing.

use algea::row_major::Matrix;

macro_rules! matrix_index_tests {
    ($module:ident, $r:literal, $c:literal, $values:expr) => {
        mod $module {
            use super::*;

            macro_rules! per_type {
                ($sub:ident, $t:ty) => {
                    mod $sub {
                        use super::*;

                        // The shared value lists are written as integers, so each element type
                        // converts them rather than repeating them.
                        fn rows() -> [[$t; $c]; $r] {
                            let source: [[i32; $c]; $r] = $values;
                            core::array::from_fn(|i| core::array::from_fn(|j| source[i][j] as $t))
                        }

                        #[test]
                        fn indexes_match_source_rows() {
                            let rows = rows();
                            let matrix = Matrix::<$t, $r, $c>::from_rows(rows);

                            for i in 0..$r {
                                for j in 0..$c {
                                    assert_eq!(matrix[(i, j)], rows[i][j]);
                                }
                            }
                        }

                        #[test]
                        fn array_round_trip_preserves_source_rows() {
                            let rows = rows();
                            let matrix = Matrix::<$t, $r, $c>::from_rows(rows);
                            let actual: [[$t; $c]; $r] = matrix.into();

                            assert_eq!(actual, rows);
                        }
                    }
                };
            }

            per_type!(i32_elements, i32);
            per_type!(i64_elements, i64);
            per_type!(u64_elements, u64);
            per_type!(f64_elements, f64);
        }
    };
}

matrix_index_tests!(dim_1x1, 1, 1, [[1]]);
matrix_index_tests!(dim_1x2, 1, 2, [[1, 2]]);
matrix_index_tests!(dim_1x3, 1, 3, [[1, 2, 3]]);
matrix_index_tests!(dim_1x4, 1, 4, [[1, 2, 3, 4]]);
matrix_index_tests!(dim_2x1, 2, 1, [[1], [2]]);
matrix_index_tests!(dim_2x2, 2, 2, [[1, 2], [3, 4]]);
matrix_index_tests!(dim_2x3, 2, 3, [[1, 2, 3], [4, 5, 6]]);
matrix_index_tests!(dim_2x4, 2, 4, [[1, 2, 3, 4], [5, 6, 7, 8]]);
matrix_index_tests!(dim_3x1, 3, 1, [[1], [2], [3]]);
matrix_index_tests!(dim_3x2, 3, 2, [[1, 2], [3, 4], [5, 6]]);
matrix_index_tests!(dim_3x3, 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
matrix_index_tests!(dim_3x4, 3, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
matrix_index_tests!(dim_4x1, 4, 1, [[1], [2], [3], [4]]);
matrix_index_tests!(dim_4x2, 4, 2, [[1, 2], [3, 4], [5, 6], [7, 8]]);
matrix_index_tests!(dim_4x3, 4, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
matrix_index_tests!(dim_4x4, 4, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
