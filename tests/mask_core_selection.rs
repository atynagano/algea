//! Tests for core mask selection operations.

use algea::{Mask, MaskElement, Select, Vector};

fn mask_bits<M, const D: usize>(mask: Mask<M, D>) -> u64
where
    M: MaskElement<D>,
    i32: algea::Element<D>,
{
    let lanes: [i32; D] = mask.select(Vector::from([1; D]), Vector::from([0; D])).into();
    lanes
        .into_iter()
        .enumerate()
        .fold(0, |bits, (lane, value)| bits | (u64::from(value != 0) << lane))
}

macro_rules! assert_mask {
    ($actual:expr, $expected_bitmask:expr, $expected_all:expr, $expected_any:expr $(,)?) => {
        let actual = $actual;
        assert_eq!(mask_bits(actual), $expected_bitmask);
        assert_eq!(actual.all(), $expected_all);
        assert_eq!(actual.any(), $expected_any);
    };
}

macro_rules! assert_array_eq {
    ($t:ty, $d:literal, $actual:expr, $expected:expr) => {
        let actual: [$t; $d] = $actual.into();
        assert_eq!(actual, $expected);
    };
}

macro_rules! mask_core_tests {
    ($module:ident, $t:ty, $d:literal, $a:expr, $same:expr, $partial:expr, $true_values:expr, $false_values:expr, $selected:expr, $partial_bitmask:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn each_eq_mask_queries() {
                let a = Vector::<$t, $d>::from($a);
                let partial_all = $partial_bitmask == (1 << $d) - 1;

                assert_mask!(a.each_eq(Vector::<$t, $d>::from($same)), (1 << $d) - 1, true, true);
                assert_mask!(
                    a.each_eq(Vector::<$t, $d>::from($partial)),
                    $partial_bitmask,
                    partial_all,
                    true,
                );
                assert_mask!(
                    a.each_eq(Vector::<$t, $d>::from($a.map(|x| x + 100.0))),
                    0,
                    false,
                    false,
                );
            }

            #[test]
            fn mask_selects_vector_lanes() {
                let mask = Vector::<$t, $d>::from($a).each_eq(Vector::<$t, $d>::from($partial));
                let selected = mask.select(
                    Vector::<$t, $d>::from($true_values),
                    Vector::<$t, $d>::from($false_values),
                );

                assert_array_eq!($t, $d, selected, $selected);
            }

            #[test]
            fn mask_selects_mask_lanes() {
                let values = Vector::<$t, $d>::from($a);
                let selector = values.each_eq(Vector::<$t, $d>::from($partial));
                let all_true = values.each_eq(values);
                let all_false = values.each_ne(values);

                assert_mask!(
                    selector.select(all_true, all_false),
                    $partial_bitmask,
                    $partial_bitmask == (1 << $d) - 1,
                    $partial_bitmask != 0,
                );
            }

            #[test]
            fn mask_not_inverts_active_lanes() {
                let selector = Vector::<$t, $d>::from($a).each_eq(Vector::<$t, $d>::from($partial));
                let expected = ((1_u64 << $d) - 1) ^ $partial_bitmask;
                assert_eq!(mask_bits(!selector), expected);
            }
        }
    };
}

mask_core_tests!(f32_dim1, f32, 1, [1.0], [1.0], [1.0], [10.0], [20.0], [10.0], 0b1);
mask_core_tests!(
    f32_dim2,
    f32,
    2,
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 9.0],
    [10.0, 20.0],
    [30.0, 40.0],
    [10.0, 40.0],
    0b01
);
mask_core_tests!(
    f32_dim3,
    f32,
    3,
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
    [1.0, 9.0, 3.0],
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0],
    [10.0, 50.0, 30.0],
    0b101
);
mask_core_tests!(
    f32_dim4,
    f32,
    4,
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [0.0, 2.0, 0.0, 4.0],
    [10.0, 20.0, 30.0, 40.0],
    [50.0, 60.0, 70.0, 80.0],
    [50.0, 20.0, 70.0, 40.0],
    0b1010
);

mask_core_tests!(f64_dim1, f64, 1, [1.0], [1.0], [1.0], [10.0], [20.0], [10.0], 0b1);
mask_core_tests!(
    f64_dim2,
    f64,
    2,
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 9.0],
    [10.0, 20.0],
    [30.0, 40.0],
    [10.0, 40.0],
    0b01
);
mask_core_tests!(
    f64_dim3,
    f64,
    3,
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
    [1.0, 9.0, 3.0],
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0],
    [10.0, 50.0, 30.0],
    0b101
);
mask_core_tests!(
    f64_dim4,
    f64,
    4,
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [0.0, 2.0, 0.0, 4.0],
    [10.0, 20.0, 30.0, 40.0],
    [50.0, 60.0, 70.0, 80.0],
    [50.0, 20.0, 70.0, 40.0],
    0b1010
);
