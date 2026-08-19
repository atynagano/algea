//! Tests for floating-point vector rounding.

use algea::Vector;

/// The rounding operations are bit-exact, so the comparison is on bits. NaN is exempt: only that a
/// NaN input yields some NaN is checked.
macro_rules! rounding_tests {
    ($module:ident, $t:ty, [$($value:expr),+ $(,)?]) => {
        mod $module {
            use super::*;

            fn assert_same_float(actual: $t, expected: $t, input: $t, operation: &str) {
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "{operation}({input:?}): actual={actual:?}, expected NaN",
                    );
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{operation}({input:?}): actual={actual:?}, expected={expected:?}",
                    );
                }
            }

            fn boundary_values() -> Vec<$t> { vec![$($value),+] }

            /// Every boundary value in every lane: the window rotates, so a lane-dependent bug
            /// cannot hide, and no value is dropped the way a non-multiple-of-four chunking would
            /// drop the tail.
            fn assert_operation_matches_scalar(
                operation: &str,
                scalar: fn($t) -> $t,
                vector: fn(Vector<$t, 4>) -> Vector<$t, 4>,
            ) {
                let values = boundary_values();
                for offset in 0..values.len() {
                    let lanes: [$t; 4] =
                        core::array::from_fn(|i| values[(offset + i) % values.len()]);
                    let actual = vector(Vector::from(lanes)).to_array();
                    for (input, actual) in lanes.into_iter().zip(actual) {
                        assert_same_float(actual, scalar(input), input, operation);
                    }
                }
            }

            #[test]
            fn round_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar("round", <$t>::round, Vector::<$t, 4>::round);
            }

            #[test]
            fn round_ties_even_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar(
                    "round_ties_even",
                    <$t>::round_ties_even,
                    Vector::<$t, 4>::round_ties_even,
                );
            }

            #[test]
            fn floor_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar("floor", <$t>::floor, Vector::<$t, 4>::floor);
            }

            #[test]
            fn ceil_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar("ceil", <$t>::ceil, Vector::<$t, 4>::ceil);
            }

            #[test]
            fn trunc_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar("trunc", <$t>::trunc, Vector::<$t, 4>::trunc);
            }

            #[test]
            fn fract_matches_scalar_at_boundaries_and_special_values() {
                assert_operation_matches_scalar("fract", <$t>::fract, Vector::<$t, 4>::fract);
            }

            fn assert_dimension<const D: usize>(input: [$t; D])
            where
                $t: algea::FloatElement<D>,
            {
                let rounded = Vector::<$t, D>::from(input).round().to_array();
                let ties_even = Vector::<$t, D>::from(input).round_ties_even().to_array();
                let floor = Vector::<$t, D>::from(input).floor().to_array();
                let ceil = Vector::<$t, D>::from(input).ceil().to_array();
                let trunc = Vector::<$t, D>::from(input).trunc().to_array();
                let fract = Vector::<$t, D>::from(input).fract().to_array();

                for i in 0..D {
                    assert_same_float(rounded[i], input[i].round(), input[i], "round");
                    assert_same_float(
                        ties_even[i],
                        input[i].round_ties_even(),
                        input[i],
                        "round_ties_even",
                    );
                    assert_same_float(floor[i], input[i].floor(), input[i], "floor");
                    assert_same_float(ceil[i], input[i].ceil(), input[i], "ceil");
                    assert_same_float(trunc[i], input[i].trunc(), input[i], "trunc");
                    assert_same_float(fract[i], input[i].fract(), input[i], "fract");
                }
            }

            #[test]
            fn rounding_matches_scalar_for_all_vector_dimensions() {
                assert_dimension([-0.5]);
                assert_dimension([-0.5, 0.5]);
                assert_dimension([-1.5, 1.5, 2.5]);
                assert_dimension([-0.0, -2.5, 3.5, <$t>::INFINITY]);
            }
        }
    };
}

// The two lists mirror each other: signed zeros, the smallest subnormals, the values one bit either
// side of 0.5 and 1.5, the ties at 2.5, the first magnitude where the mantissa can no longer hold a
// half (2^23 for `f32` and 2^52 for `f64`), the extremes, the infinities, and both quiet and
// signalling NaN patterns.
rounding_tests!(f32_rounding, f32, [
    0.0,
    -0.0,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    f32::from_bits(0x3eff_fffe),
    f32::from_bits(0x3eff_ffff),
    0.5,
    f32::from_bits(0x3f00_0001),
    f32::from_bits(0xbeff_fffe),
    f32::from_bits(0xbeff_ffff),
    -0.5,
    f32::from_bits(0xbf00_0001),
    1.0,
    -1.0,
    f32::from_bits(1.5_f32.to_bits() - 1),
    1.5,
    f32::from_bits(1.5_f32.to_bits() + 1),
    f32::from_bits((-1.5_f32).to_bits() - 1),
    -1.5,
    f32::from_bits((-1.5_f32).to_bits() + 1),
    2.5,
    -2.5,
    8_388_607.0,
    8_388_607.5,
    8_388_608.0,
    8_388_609.0,
    -8_388_607.0,
    -8_388_607.5,
    -8_388_608.0,
    -8_388_609.0,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0xffc1_2345),
    f32::from_bits(0x7f81_2345),
    f32::from_bits(0xff81_2345),
]);

rounding_tests!(f64_rounding, f64, [
    0.0,
    -0.0,
    f64::from_bits(1),
    f64::from_bits(0x8000_0000_0000_0001),
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    f64::from_bits(0x3fdf_ffff_ffff_fffe),
    f64::from_bits(0x3fdf_ffff_ffff_ffff),
    0.5,
    f64::from_bits(0x3fe0_0000_0000_0001),
    f64::from_bits(0xbfdf_ffff_ffff_fffe),
    f64::from_bits(0xbfdf_ffff_ffff_ffff),
    -0.5,
    f64::from_bits(0xbfe0_0000_0000_0001),
    1.0,
    -1.0,
    f64::from_bits(1.5_f64.to_bits() - 1),
    1.5,
    f64::from_bits(1.5_f64.to_bits() + 1),
    f64::from_bits((-1.5_f64).to_bits() - 1),
    -1.5,
    f64::from_bits((-1.5_f64).to_bits() + 1),
    2.5,
    -2.5,
    4_503_599_627_370_495.0,
    4_503_599_627_370_495.5,
    4_503_599_627_370_496.0,
    4_503_599_627_370_497.0,
    -4_503_599_627_370_495.0,
    -4_503_599_627_370_495.5,
    -4_503_599_627_370_496.0,
    -4_503_599_627_370_497.0,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::from_bits(0xfff8_1234_5678_9abc),
    f64::from_bits(0x7ff0_1234_5678_9abc),
    f64::from_bits(0xfff0_1234_5678_9abc),
]);
