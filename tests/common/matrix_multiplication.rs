/// The body is a macro so both floating-point widths can be instantiated from it. The including
/// file supplies `matrix_from_rows!`/`matrix_to_rows!` for its own storage order, and the
/// `VectorProduct`/`OuterProductAdapter` adapters for each element type.
macro_rules! matrix_multiplication_tests {
    ($module:ident, $t:ty) => {
        mod $module {
            use super::*;

            fn check<const R: usize, const K: usize, const C: usize>(failures: &mut Vec<String>)
            where
                $t: MatrixProduct<R, K, C>,
            {
                let lhs: [[$t; K]; R] = core::array::from_fn(|i| {
                    core::array::from_fn(|k| (1 + i * K + k) as $t * 0.25)
                });
                let rhs: [[$t; C]; K] = core::array::from_fn(|k| {
                    core::array::from_fn(|j| {
                        (2 + k * C + j) as $t * if (k + j) % 2 == 0 { 0.5 } else { -0.5 }
                    })
                });
                let expected: [[$t; C]; R] = core::array::from_fn(|i| {
                    core::array::from_fn(|j| (0..K).map(|k| lhs[i][k] * rhs[k][j]).sum())
                });
                let actual: [[$t; C]; R] =
                    matrix_to_rows!(matrix_from_rows!(lhs) * matrix_from_rows!(rhs));

                for i in 0..R {
                    for j in 0..C {
                        let tolerance = 1.0e-5 * expected[i][j].abs().max(1.0);
                        if (actual[i][j] - expected[i][j]).abs() > tolerance {
                            failures.push(format!(
                                "{R}x{K}x{C} ({i}, {j}): actual={}, expected={}",
                                actual[i][j], expected[i][j]
                            ));
                        }
                    }
                }
            }

            fn check_vector_matrix<const R: usize, const C: usize>(failures: &mut Vec<String>)
            where
                $t: VectorProduct<R, C>,
            {
                let lhs: [$t; R] = core::array::from_fn(|i| (i + 1) as $t * 0.25);
                let rhs: [[$t; C]; R] = core::array::from_fn(|i| {
                    core::array::from_fn(|j| {
                        (2 + i * C + j) as $t * if (i + j) % 2 == 0 { 0.5 } else { -0.5 }
                    })
                });
                let expected: [$t; C] =
                    core::array::from_fn(|j| (0..R).map(|i| lhs[i] * rhs[i][j]).sum());
                let actual = <$t as VectorProduct<R, C>>::product(lhs, rhs);

                for j in 0..C {
                    let tolerance = 1.0e-5 * expected[j].abs().max(1.0);
                    if (actual[j] - expected[j]).abs() > tolerance {
                        failures.push(format!(
                            "vector product <{R}, {C}> [{j}]: actual={}, expected={}",
                            actual[j], expected[j]
                        ));
                    }
                }
            }

            fn check_outer_product<const R: usize, const C: usize>(failures: &mut Vec<String>)
            where
                $t: OuterProduct<R, C>,
            {
                let lhs: [$t; R] = core::array::from_fn(|i| (i + 1) as $t * 0.25);
                let rhs: [$t; C] =
                    core::array::from_fn(|j| (j + 2) as $t * if j % 2 == 0 { 0.5 } else { -0.5 });
                let expected: [[$t; C]; R] =
                    core::array::from_fn(|i| core::array::from_fn(|j| lhs[i] * rhs[j]));
                let actual = <$t as OuterProductAdapter<R, C>>::product(lhs, rhs);

                for i in 0..R {
                    for j in 0..C {
                        let tolerance = 1.0e-5 * expected[i][j].abs().max(1.0);
                        if (actual[i][j] - expected[i][j]).abs() > tolerance {
                            failures.push(format!(
                                "outer product <{R}, {C}> [{i}][{j}]: actual={}, expected={}",
                                actual[i][j], expected[i][j]
                            ));
                        }
                    }
                }
            }

            macro_rules! check_all_c {
                ($check:ident, $failures:ident, $r:literal) => {
                    $check::<$r, 1>(&mut $failures);
                    $check::<$r, 2>(&mut $failures);
                    $check::<$r, 3>(&mut $failures);
                    $check::<$r, 4>(&mut $failures);
                };
            }

            macro_rules! check_c {
                ($failures:ident, $r:literal, $k:literal) => {
                    check::<$r, $k, 1>(&mut $failures);
                    check::<$r, $k, 2>(&mut $failures);
                    check::<$r, $k, 3>(&mut $failures);
                    check::<$r, $k, 4>(&mut $failures);
                };
            }

            #[rustfmt::skip]
            #[test]
            fn all_64_matrix_multiplication_shapes_match_scalar_reference() {
                let mut failures = Vec::new();
                check_c!(failures, 1, 1); check_c!(failures, 1, 2); check_c!(failures, 1, 3); check_c!(failures, 1, 4);
                check_c!(failures, 2, 1); check_c!(failures, 2, 2); check_c!(failures, 2, 3); check_c!(failures, 2, 4);
                check_c!(failures, 3, 1); check_c!(failures, 3, 2); check_c!(failures, 3, 3); check_c!(failures, 3, 4);
                check_c!(failures, 4, 1); check_c!(failures, 4, 2); check_c!(failures, 4, 3); check_c!(failures, 4, 4);
                assert!(failures.is_empty(), "{}", failures.join("\n"));
            }

            #[rustfmt::skip]
            #[test]
            fn all_16_vector_product_shapes_match_scalar_reference() {
                let mut failures = Vec::new();
                check_all_c!(check_vector_matrix, failures, 1); check_all_c!(check_vector_matrix, failures, 2);
                check_all_c!(check_vector_matrix, failures, 3); check_all_c!(check_vector_matrix, failures, 4);
                assert!(failures.is_empty(), "{}", failures.join("\n"));
            }

            #[rustfmt::skip]
            #[test]
            fn all_16_outer_product_shapes_match_scalar_reference() {
                let mut failures = Vec::new();
                check_all_c!(check_outer_product, failures, 1); check_all_c!(check_outer_product, failures, 2);
                check_all_c!(check_outer_product, failures, 3); check_all_c!(check_outer_product, failures, 4);
                assert!(failures.is_empty(), "{}", failures.join("\n"));
            }
        }
    };
}

matrix_multiplication_tests!(f32_products, f32);
matrix_multiplication_tests!(f64_products, f64);
