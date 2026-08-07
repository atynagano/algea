//! Tests for core floating-point vector operations.

use algea::Vector;

fn assert_f32_eq(actual: f32, expected: f32) {
    assert!(
        actual.to_bits() == expected.to_bits() || (actual - expected).abs() <= 1e-6,
        "actual: {actual:?}, expected: {expected:?}",
    );
}

fn assert_array_eq<const D: usize>(actual: [f32; D], expected: [f32; D]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_f32_eq(actual, expected);
    }
}

fn dot<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

fn norm<const D: usize>(a: [f32; D]) -> f32 { dot(a, a).sqrt() }

fn distance<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    let mut sum = 0.0;
    for i in 0..D {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

macro_rules! vector_core_tests {
    ($module:ident, $d:literal, $a:expr, $b:expr, $scalar:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn vector_vector_ops() {
                let a = Vector::<f32, $d>::from($a);
                let b = Vector::<f32, $d>::from($b);

                assert_array_eq(Into::<[f32; $d]>::into(a + b), $a.map_with($b, |a, b| a + b));
                assert_array_eq(Into::<[f32; $d]>::into(a - b), $a.map_with($b, |a, b| a - b));
                assert_array_eq(Into::<[f32; $d]>::into(a * b), $a.map_with($b, |a, b| a * b));
                assert_array_eq(Into::<[f32; $d]>::into(a / b), $a.map_with($b, |a, b| a / b));
            }

            #[test]
            fn vector_scalar_ops() {
                let a = Vector::<f32, $d>::from($a);
                let s = $scalar;

                assert_array_eq(Into::<[f32; $d]>::into(a + s), $a.map(|a| a + s));
                assert_array_eq(Into::<[f32; $d]>::into(a - s), $a.map(|a| a - s));
                assert_array_eq(Into::<[f32; $d]>::into(a * s), $a.map(|a| a * s));
                assert_array_eq(Into::<[f32; $d]>::into(a / s), $a.map(|a| a / s));

                assert_array_eq(Into::<[f32; $d]>::into(s + a), $a.map(|a| s + a));
                assert_array_eq(Into::<[f32; $d]>::into(s - a), $a.map(|a| s - a));
                assert_array_eq(Into::<[f32; $d]>::into(s * a), $a.map(|a| s * a));
                assert_array_eq(Into::<[f32; $d]>::into(s / a), $a.map(|a| s / a));
            }

            #[test]
            fn dot_norm_distance_and_normalize() {
                let a = Vector::<f32, $d>::from($a);
                let b = Vector::<f32, $d>::from($b);

                assert_f32_eq(a.dot(b), dot($a, $b));
                assert_f32_eq(a.norm_squared(), dot($a, $a));
                assert_f32_eq(a.norm(), norm($a));
                assert_f32_eq(a.distance_squared(b), {
                    let d = $a.map_with($b, |a, b| a - b);
                    dot(d, d)
                });
                assert_f32_eq(a.distance(b), distance($a, $b));
                assert_array_eq(Into::<[f32; $d]>::into(a.normalize()), {
                    let n = norm($a);
                    $a.map(|a| a / n)
                });
            }
        }
    };
}

trait MapWith<const D: usize> {
    fn map_with(self, rhs: Self, f: impl Fn(f32, f32) -> f32) -> Self;
}

impl<const D: usize> MapWith<D> for [f32; D] {
    fn map_with(self, rhs: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        core::array::from_fn(|i| f(self[i], rhs[i]))
    }
}

vector_core_tests!(dim1, 1, [2.0], [4.0], 0.5);
vector_core_tests!(dim2, 2, [2.0, -3.5], [4.0, 0.25], 0.5);
vector_core_tests!(dim3, 3, [2.0, -3.5, 8.0], [4.0, 0.25, -2.0], 0.5);
vector_core_tests!(dim4, 4, [2.0, -3.5, 8.0, 0.125], [4.0, 0.25, -2.0, -0.5], 0.5);
