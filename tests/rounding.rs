//! Tests for floating-point vector rounding.

use algea::Vector;

fn assert_same_float(actual: f32, expected: f32, input: f32, operation: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{operation}({input:?}): actual={actual:?}, expected NaN",);
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{operation}({input:?}): actual={actual:?}, expected={expected:?}",
        );
    }
}

fn boundary_values() -> Vec<f32> {
    vec![
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
    ]
}

#[test]
fn round_matches_scalar_at_boundaries_and_special_values() {
    for lanes in boundary_values().chunks_exact(4) {
        let lanes: [f32; 4] = lanes.try_into().unwrap();
        let actual = Vector::<f32, 4>::from(lanes).round().to_array();
        for (input, actual) in lanes.into_iter().zip(actual) {
            assert_same_float(actual, input.round(), input, "round");
        }
    }
}

#[test]
fn round_ties_even_matches_scalar_at_boundaries_and_special_values() {
    for lanes in boundary_values().chunks_exact(4) {
        let lanes: [f32; 4] = lanes.try_into().unwrap();
        let actual = Vector::<f32, 4>::from(lanes).round_ties_even().to_array();
        for (input, actual) in lanes.into_iter().zip(actual) {
            assert_same_float(actual, input.round_ties_even(), input, "round_ties_even");
        }
    }
}

fn assert_f32x4_operation_matches_scalar(
    operation: &str,
    scalar: fn(f32) -> f32,
    vector: fn(Vector<f32, 4>) -> Vector<f32, 4>,
) {
    for lanes in boundary_values().chunks_exact(4) {
        let lanes: [f32; 4] = lanes.try_into().unwrap();
        let actual = vector(Vector::from(lanes)).to_array();
        for (input, actual) in lanes.into_iter().zip(actual) {
            assert_same_float(actual, scalar(input), input, operation);
        }
    }
}

#[test]
fn floor_matches_scalar_at_boundaries_and_special_values() {
    assert_f32x4_operation_matches_scalar("floor", f32::floor, Vector::<f32, 4>::floor);
}

#[test]
fn ceil_matches_scalar_at_boundaries_and_special_values() {
    assert_f32x4_operation_matches_scalar("ceil", f32::ceil, Vector::<f32, 4>::ceil);
}

#[test]
fn trunc_matches_scalar_at_boundaries_and_special_values() {
    assert_f32x4_operation_matches_scalar("trunc", f32::trunc, Vector::<f32, 4>::trunc);
}

#[test]
fn fract_matches_scalar_at_boundaries_and_special_values() {
    assert_f32x4_operation_matches_scalar("fract", f32::fract, Vector::<f32, 4>::fract);
}

#[test]
fn rounding_matches_scalar_for_all_vector_dimensions() {
    let input1 = [-0.5];
    let input2 = [-0.5, 0.5];
    let input3 = [-1.5, 1.5, 2.5];
    let input4 = [-0.0, -2.5, 3.5, f32::INFINITY];

    assert_dimension(input1);
    assert_dimension(input2);
    assert_dimension(input3);
    assert_dimension(input4);
}

fn assert_dimension<const D: usize>(input: [f32; D])
where
    f32: algea::FloatElement<D>,
{
    let rounded = Vector::<f32, D>::from(input).round().to_array();
    let ties_even = Vector::<f32, D>::from(input).round_ties_even().to_array();
    let floor = Vector::<f32, D>::from(input).floor().to_array();
    let ceil = Vector::<f32, D>::from(input).ceil().to_array();
    let trunc = Vector::<f32, D>::from(input).trunc().to_array();
    let fract = Vector::<f32, D>::from(input).fract().to_array();

    for i in 0..D {
        assert_same_float(rounded[i], input[i].round(), input[i], "round");
        assert_same_float(ties_even[i], input[i].round_ties_even(), input[i], "round_ties_even");
        assert_same_float(floor[i], input[i].floor(), input[i], "floor");
        assert_same_float(ceil[i], input[i].ceil(), input[i], "ceil");
        assert_same_float(trunc[i], input[i].trunc(), input[i], "trunc");
        assert_same_float(fract[i], input[i].fract(), input[i], "fract");
    }
}
