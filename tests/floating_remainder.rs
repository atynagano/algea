//! Tests for floating-point vector remainders.

use algea::{Element, Vector};

fn assert_f32_eq(actual: f32, expected: f32) {
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

fn assert_array_eq<const D: usize>(actual: [f32; D], expected: [f32; D]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_f32_eq(actual, expected);
    }
}

fn map<const D: usize>(values: [f32; D], f: impl Fn(f32) -> f32) -> [f32; D] { values.map(f) }

fn map2<const D: usize>(lhs: [f32; D], rhs: [f32; D]) -> [f32; D] {
    core::array::from_fn(|i| lhs[i] % rhs[i])
}

fn assert_remainder_operators<const D: usize>(lhs: [f32; D], rhs: [f32; D], scalar: f32)
where
    f32: Element<D>,
{
    let lhs_vector = Vector::<f32, D>::from(lhs);
    let rhs_vector = Vector::<f32, D>::from(rhs);

    assert_array_eq((lhs_vector % rhs_vector).to_array(), map2(lhs, rhs));
    assert_array_eq((lhs_vector % scalar).to_array(), map(lhs, |value| value % scalar));
    assert_array_eq((scalar % rhs_vector).to_array(), map(rhs, |value| scalar % value));

    let mut assigned = lhs_vector;
    assigned %= rhs_vector;
    assert_array_eq(assigned.to_array(), map2(lhs, rhs));

    assigned = lhs_vector;
    assigned %= scalar;
    assert_array_eq(assigned.to_array(), map(lhs, |value| value % scalar));
}

#[test]
fn floating_remainder_operators_match_scalar_for_all_dimensions() {
    assert_remainder_operators([5.5], [2.0], -2.0);
    assert_remainder_operators([-5.5, 5.5], [2.0, -2.0], -2.0);
    assert_remainder_operators([-5.5, 5.5, 1.0], [2.0, -2.0, f32::from_bits(1)], -2.0);
    assert_remainder_operators(
        [-5.5, 5.5, 16_777_216.0, -16_777_216.0],
        [2.0, -2.0, 3.0, -3.0],
        -2.0,
    );
}

#[test]
fn floating_remainder_matches_scalar_for_special_values() {
    let lhs = [5.0, f32::INFINITY, 1.0, f32::NAN];
    let rhs = [f32::INFINITY, 2.0, f32::from_bits(1), 3.0];
    assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));

    let lhs = [-0.0, 0.0, -0.0, 0.0];
    let rhs = [-1.0, -1.0, 1.0, 1.0];
    assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));

    let lhs = [1.0, -1.0, 0.0, -0.0];
    let rhs = [0.0, -0.0, 0.0, -0.0];
    assert_array_eq((Vector::from(lhs) % Vector::from(rhs)).to_array(), map2(lhs, rhs));
}
