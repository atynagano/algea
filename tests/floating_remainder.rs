//! Tests for floating-point vector remainders.

use algea::{Element, Vector};

macro_rules! remainder_tests {
    ($module:ident, $t:ty) => {
        mod $module {
            use super::*;

            /// A remainder is exact, so the comparison is on bits; that is also what catches a lost
            /// zero sign. NaN is exempt.
            fn assert_float_eq(actual: $t, expected: $t) {
                if expected.is_nan() {
                    assert!(actual.is_nan(), "expected NaN, got {actual:?}");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "actual = {actual:?}, expected = {expected:?}"
                    );
                }
            }

            fn assert_array_eq<const D: usize>(actual: [$t; D], expected: [$t; D]) {
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert_float_eq(actual, expected);
                }
            }

            fn map<const D: usize>(values: [$t; D], f: impl Fn($t) -> $t) -> [$t; D] {
                values.map(f)
            }

            fn map2<const D: usize>(lhs: [$t; D], rhs: [$t; D]) -> [$t; D] {
                core::array::from_fn(|i| lhs[i] % rhs[i])
            }

            fn assert_remainder_operators<const D: usize>(lhs: [$t; D], rhs: [$t; D], scalar: $t)
            where
                $t: Element<D>,
            {
                let lhs_vector = Vector::<$t, D>::from(lhs);
                let rhs_vector = Vector::<$t, D>::from(rhs);

                assert_array_eq((lhs_vector % rhs_vector).to_array(), map2(lhs, rhs));
                assert_array_eq((lhs_vector % scalar).to_array(), map(lhs, |v| v % scalar));
                assert_array_eq((scalar % rhs_vector).to_array(), map(rhs, |v| scalar % v));

                let mut assigned = lhs_vector;
                assigned %= rhs_vector;
                assert_array_eq(assigned.to_array(), map2(lhs, rhs));

                assigned = lhs_vector;
                assigned %= scalar;
                assert_array_eq(assigned.to_array(), map(lhs, |v| v % scalar));
            }

            #[test]
            fn operators_match_scalar_for_all_dimensions() {
                assert_remainder_operators([5.5], [2.0], -2.0);
                assert_remainder_operators([-5.5, 5.5], [2.0, -2.0], -2.0);
                assert_remainder_operators([-5.5, 5.5, 1.0], [2.0, -2.0, <$t>::from_bits(1)], -2.0);
                assert_remainder_operators(
                    [-5.5, 5.5, 16_777_216.0, -16_777_216.0],
                    [2.0, -2.0, 3.0, -3.0],
                    -2.0,
                );
            }

            #[test]
            fn matches_scalar_for_special_values() {
                let lhs = [5.0, <$t>::INFINITY, 1.0, <$t>::NAN];
                let rhs = [<$t>::INFINITY, 2.0, <$t>::from_bits(1), 3.0];
                assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));

                // A remainder keeps the dividend's sign, including the sign of a zero.
                let lhs = [-0.0, 0.0, -0.0, 0.0];
                let rhs = [-1.0, -1.0, 1.0, 1.0];
                assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));

                let lhs = [1.0, -1.0, 0.0, -0.0];
                let rhs = [0.0, -0.0, 0.0, -0.0];
                assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));
            }
        }
    };
}

remainder_tests!(f32_remainder, f32);
remainder_tests!(f64_remainder, f64);
