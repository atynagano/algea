/// The body is a macro so both floating-point widths can be instantiated from it. The including
/// file supplies `matrix_from_rows!`/`matrix_to_rows!` for its own storage order.
macro_rules! square_kernel_tests {
    ($module:ident, $t:ty, $tolerance:expr) => {
        mod $module {
            use super::*;

            macro_rules! transpose_case {
                ($name:ident, $r:literal, $c:literal) => {
                    #[test]
                    fn $name() {
                        let source: [[$t; $c]; $r] = core::array::from_fn(|i| {
                            core::array::from_fn(|j| (1 + i * $c + j) as $t)
                        });
                        let actual: [[$t; $r]; $c] =
                            matrix_to_rows!(matrix_from_rows!(source).transpose());
                        let expected =
                            core::array::from_fn(|j| core::array::from_fn(|i| source[i][j]));
                        assert_eq!(actual, expected);
                    }
                };
            }

            #[rustfmt::skip]
            mod transpose {
                use super::*;
                transpose_case!(transpose_1x1, 1, 1); transpose_case!(transpose_1x2, 1, 2);
                transpose_case!(transpose_1x3, 1, 3); transpose_case!(transpose_1x4, 1, 4);
                transpose_case!(transpose_2x1, 2, 1); transpose_case!(transpose_2x2, 2, 2);
                transpose_case!(transpose_2x3, 2, 3); transpose_case!(transpose_2x4, 2, 4);
                transpose_case!(transpose_3x1, 3, 1); transpose_case!(transpose_3x2, 3, 2);
                transpose_case!(transpose_3x3, 3, 3); transpose_case!(transpose_3x4, 3, 4);
                transpose_case!(transpose_4x1, 4, 1); transpose_case!(transpose_4x2, 4, 2);
                transpose_case!(transpose_4x3, 4, 3); transpose_case!(transpose_4x4, 4, 4);
            }

            #[rustfmt::skip]
            fn reference_inverse<const D: usize>(source: [[$t; D]; D]) -> [[$t; D]; D] {
                let mut a = source;
                let mut inverse =
                    core::array::from_fn(|i| core::array::from_fn(|j| (i == j) as u8 as $t));
                for pivot in 0..D {
                    let best = (pivot..D)
                        .max_by(|&lhs, &rhs| a[lhs][pivot].abs().total_cmp(&a[rhs][pivot].abs()))
                        .unwrap();
                    assert!(a[best][pivot].abs() > 1.0e-6);
                    a.swap(pivot, best);
                    inverse.swap(pivot, best);
                    let scale = a[pivot][pivot];
                    for j in 0..D { a[pivot][j] /= scale; inverse[pivot][j] /= scale; }
                    for i in 0..D {
                        if i == pivot { continue; }
                        let factor = a[i][pivot];
                        for j in 0..D {
                            a[i][j] -= factor * a[pivot][j];
                            inverse[i][j] -= factor * inverse[pivot][j];
                        }
                    }
                }
                inverse
            }

            fn assert_matrix_close<const D: usize>(
                actual: [[$t; D]; D],
                expected: [[$t; D]; D],
            ) {
                for i in 0..D {
                    for j in 0..D {
                        let tolerance = $tolerance * expected[i][j].abs().max(1.0);
                        assert!(
                            (actual[i][j] - expected[i][j]).abs() <= tolerance,
                            "({i}, {j}): actual={}, expected={}",
                            actual[i][j],
                            expected[i][j]
                        );
                    }
                }
            }

            fn assert_scalar_close(actual: $t, expected: $t) {
                let tolerance = $tolerance * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "actual={actual}, expected={expected}, tolerance={tolerance}"
                );
            }

            #[rustfmt::skip]
            #[test]
            fn diagonal_preserves_logical_coordinates() {
                let m2 = matrix_from_rows!([[2.0, 3.0], [5.0, 7.0]]);
                let m3 = matrix_from_rows!([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0], [17.0, 19.0, 23.0]]);
                let m4 = matrix_from_rows!([[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 17.0, 19.0], [23.0, 29.0, 31.0, 37.0], [41.0, 43.0, 47.0, 53.0]]);
                assert_eq!(<[$t; 2]>::from(m2.diagonal()), [2.0, 7.0]);
                assert_eq!(<[$t; 3]>::from(m3.diagonal()), [2.0, 11.0, 23.0]);
                assert_eq!(<[$t; 4]>::from(m4.diagonal()), [2.0, 13.0, 31.0, 53.0]);
            }

            #[rustfmt::skip]
            #[test]
            fn all_square_determinants_preserve_logical_semantics() {
                assert_eq!(matrix_from_rows!([[4.0]]).determinant(), 4.0);
                assert_eq!(matrix_from_rows!([[2.0, 3.0], [5.0, 7.0]]).determinant(), -1.0);
                assert_eq!(matrix_from_rows!([[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]]).determinant(), 10.0);
                assert_eq!(matrix_from_rows!([[5.0, 7.0, 9.0, 10.0], [2.0, 3.0, 3.0, 8.0], [8.0, 10.0, 2.0, 3.0], [3.0, 3.0, 4.0, 8.0]]).determinant(), -361.0);
            }

            #[rustfmt::skip]
            #[test]
            fn all_square_inverses_match_scalar_reference() {
                macro_rules! check { ($source:expr) => {{
                    let source = $source;
                    assert_matrix_close(matrix_to_rows!(matrix_from_rows!(source).inverse()), reference_inverse(source));
                }}; }
                check!([[4.0]]);
                check!([[4.0, 7.0], [2.0, 6.0]]);
                check!([[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]]);
                check!([[5.0, 7.0, 9.0, 10.0], [2.0, 3.0, 3.0, 8.0], [8.0, 10.0, 2.0, 3.0], [3.0, 3.0, 4.0, 8.0]]);
            }

            #[rustfmt::skip]
            #[test]
            fn all_square_inverses_and_determinants_match_nalgebra() {
                macro_rules! compare { ($d:literal, $source:expr) => {{
                    let source: [[$t; $d]; $d] = $source;
                    let flat: Vec<$t> = source.iter().flatten().copied().collect();
                    let reference = nalgebra::SMatrix::<$t, $d, $d>::from_row_slice(&flat);
                    let matrix: Matrix<$t, $d, $d> = matrix_from_rows!(source);
                    assert_scalar_close(matrix.determinant(), reference.determinant());
                    let inverse = reference.try_inverse().expect("test matrix must be invertible");
                    let expected = core::array::from_fn(|i| core::array::from_fn(|j| inverse[(i, j)]));
                    assert_matrix_close(matrix_to_rows!(matrix.inverse()), expected);
                }}; }
                compare!(1, [[4.0]]);
                compare!(2, [[4.0, 7.0], [2.0, 6.0]]);
                compare!(3, [[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]]);
                compare!(4, [[5.0, 7.0, 9.0, 10.0], [2.0, 3.0, 3.0, 8.0], [8.0, 10.0, 2.0, 3.0], [3.0, 3.0, 4.0, 8.0]]);
            }
        }
    };
}

// `f64` carries about 29 more bits of mantissa, so its inverses land far closer to the reference.
square_kernel_tests!(f32_square_kernels, f32, 2.0e-4);
square_kernel_tests!(f64_square_kernels, f64, 2.0e-12);
