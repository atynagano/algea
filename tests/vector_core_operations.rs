//! Tests for core floating-point vector operations.

use algea::Vector;

macro_rules! vector_core_tests {
    ($module:ident, $t:ty, $epsilon:expr) => {
        mod $module {
            use super::*;

            /// The reductions accumulate, so an exact comparison would be over-specified; anything
            /// bit-exact is caught by the identical-bits arm.
            fn assert_float_eq(actual: $t, expected: $t) {
                assert!(
                    actual.to_bits() == expected.to_bits() || (actual - expected).abs() <= $epsilon,
                    "actual: {actual:?}, expected: {expected:?}",
                );
            }

            fn assert_array_eq<const D: usize>(actual: [$t; D], expected: [$t; D]) {
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert_float_eq(actual, expected);
                }
            }

            fn dot<const D: usize>(a: [$t; D], b: [$t; D]) -> $t {
                a.into_iter().zip(b).map(|(a, b)| a * b).sum()
            }

            fn norm<const D: usize>(a: [$t; D]) -> $t { dot(a, a).sqrt() }

            fn distance<const D: usize>(a: [$t; D], b: [$t; D]) -> $t {
                let mut sum = 0.0;
                for i in 0..D {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                sum.sqrt()
            }

            fn map_with<const D: usize>(
                a: [$t; D],
                b: [$t; D],
                f: impl Fn($t, $t) -> $t,
            ) -> [$t; D] {
                core::array::from_fn(|i| f(a[i], b[i]))
            }

            macro_rules! dimension_tests {
                ($sub:ident, $d:literal, $a:expr, $b:expr, $scalar:expr) => {
                    mod $sub {
                        use super::*;

                        #[test]
                        fn vector_vector_ops() {
                            let aa: [$t; $d] = $a;
                            let bb: [$t; $d] = $b;
                            let a = Vector::<$t, $d>::from(aa);
                            let b = Vector::<$t, $d>::from(bb);

                            assert_array_eq(
                                Into::<[$t; $d]>::into(a + b),
                                map_with(aa, bb, |a, b| a + b),
                            );
                            assert_array_eq(
                                Into::<[$t; $d]>::into(a - b),
                                map_with(aa, bb, |a, b| a - b),
                            );
                            assert_array_eq(
                                Into::<[$t; $d]>::into(a * b),
                                map_with(aa, bb, |a, b| a * b),
                            );
                            assert_array_eq(
                                Into::<[$t; $d]>::into(a / b),
                                map_with(aa, bb, |a, b| a / b),
                            );
                        }

                        #[test]
                        fn vector_scalar_ops() {
                            let aa: [$t; $d] = $a;
                            let a = Vector::<$t, $d>::from(aa);
                            let s: $t = $scalar;

                            assert_array_eq(Into::<[$t; $d]>::into(a + s), aa.map(|a| a + s));
                            assert_array_eq(Into::<[$t; $d]>::into(a - s), aa.map(|a| a - s));
                            assert_array_eq(Into::<[$t; $d]>::into(a * s), aa.map(|a| a * s));
                            assert_array_eq(Into::<[$t; $d]>::into(a / s), aa.map(|a| a / s));

                            assert_array_eq(Into::<[$t; $d]>::into(s + a), aa.map(|a| s + a));
                            assert_array_eq(Into::<[$t; $d]>::into(s - a), aa.map(|a| s - a));
                            assert_array_eq(Into::<[$t; $d]>::into(s * a), aa.map(|a| s * a));
                            assert_array_eq(Into::<[$t; $d]>::into(s / a), aa.map(|a| s / a));
                        }

                        #[test]
                        fn dot_norm_distance_and_normalize() {
                            let aa: [$t; $d] = $a;
                            let bb: [$t; $d] = $b;
                            let a = Vector::<$t, $d>::from(aa);
                            let b = Vector::<$t, $d>::from(bb);

                            assert_float_eq(a.dot(b), dot(aa, bb));
                            assert_float_eq(a.norm_squared(), dot(aa, aa));
                            assert_float_eq(a.norm(), norm(aa));
                            assert_float_eq(a.distance_squared(b), {
                                let d = map_with(aa, bb, |a, b| a - b);
                                dot(d, d)
                            });
                            assert_float_eq(a.distance(b), distance(aa, bb));
                            assert_array_eq(Into::<[$t; $d]>::into(a.normalize()), {
                                let n = norm(aa);
                                aa.map(|a| a / n)
                            });
                        }
                    }
                };
            }

            dimension_tests!(dim1, 1, [2.0], [4.0], 0.5);
            dimension_tests!(dim2, 2, [2.0, -3.5], [4.0, 0.25], 0.5);
            dimension_tests!(dim3, 3, [2.0, -3.5, 8.0], [4.0, 0.25, -2.0], 0.5);
            dimension_tests!(dim4, 4, [2.0, -3.5, 8.0, 0.125], [4.0, 0.25, -2.0, -0.5], 0.5);
        }
    };
}

vector_core_tests!(f32_core, f32, 1e-6);
vector_core_tests!(f64_core, f64, 1e-12);
