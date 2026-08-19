//! Tests for lane comparisons, masks, and selection.

use algea::{EachOrd, Element, Mask, MaskElement, Select, Vector};

#[cfg(not(target_arch = "wasm32"))]
fn panic_message(f: impl FnOnce()) -> String {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .expect_err("operation did not panic");
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        panic!("panic payload was not a string")
    }
}

fn expected_mask<T, const D: usize>(a: [T; D], b: [T; D], f: impl Fn(T, T) -> bool) -> u64
where
    T: Copy,
{
    (0..D).fold(0, |mask, i| mask | (u64::from(f(a[i], b[i])) << i))
}

fn mask_bits<M, const D: usize>(mask: Mask<M, D>) -> u64
where
    M: MaskElement<D>,
    i32: Element<D>,
{
    let lanes: [i32; D] = mask.select(Vector::from([1; D]), Vector::from([0; D])).into();
    lanes
        .into_iter()
        .enumerate()
        .fold(0, |bits, (lane, value)| bits | (u64::from(value != 0) << lane))
}

macro_rules! comparison_tests {
    ($name:ident, $t:ty, $d:literal, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            let a: [$t; $d] = $a;
            let b: [$t; $d] = $b;
            let va = Vector::<$t, $d>::from(a);
            let vb = Vector::<$t, $d>::from(b);

            assert_eq!(mask_bits(va.each_eq(vb)), expected_mask(a, b, |x, y| x == y));
            assert_eq!(mask_bits(va.each_ne(vb)), expected_mask(a, b, |x, y| x != y));
            assert_eq!(mask_bits(va.each_lt(vb)), expected_mask(a, b, |x, y| x < y));
            assert_eq!(mask_bits(va.each_le(vb)), expected_mask(a, b, |x, y| x <= y));
            assert_eq!(mask_bits(va.each_gt(vb)), expected_mask(a, b, |x, y| x > y));
            assert_eq!(mask_bits(va.each_ge(vb)), expected_mask(a, b, |x, y| x >= y));

            let active = (1_u64 << $d) - 1;
            for mask in [
                va.each_eq(vb),
                va.each_ne(vb),
                va.each_lt(vb),
                va.each_le(vb),
                va.each_gt(vb),
                va.each_ge(vb),
            ] {
                let bits = mask_bits(mask);
                assert_eq!(bits & !active, 0, "padding bits must not be observable");
                assert_eq!(mask.any(), bits != 0);
                assert_eq!(mask.all(), bits == active);
            }
        }
    };
}

macro_rules! comparisons_all_dimensions {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            comparison_tests!([<$prefix _comparison_1>], $t, 1, [2 as $t], [3 as $t]);
            comparison_tests!([<$prefix _comparison_2>], $t, 2, [2 as $t, 4 as $t], [3 as $t, 4 as $t]);
            comparison_tests!([<$prefix _comparison_3>], $t, 3, [2 as $t, 4 as $t, 8 as $t], [3 as $t, 4 as $t, 7 as $t]);
            comparison_tests!([<$prefix _comparison_4>], $t, 4, [2 as $t, 4 as $t, 8 as $t, 1 as $t], [3 as $t, 4 as $t, 7 as $t, 1 as $t]);
        }
    };
}

comparisons_all_dimensions!(f32, f32);
comparisons_all_dimensions!(i32, i32);
comparisons_all_dimensions!(u32, u32);
comparisons_all_dimensions!(f64, f64);
comparisons_all_dimensions!(i64, i64);
comparisons_all_dimensions!(u64, u64);

macro_rules! integer_min_max_clamp {
    ($name:ident, $t:ty, $d:literal, $values:expr) => {
        #[test]
        fn $name() {
            let values: [$t; $d] = $values;
            let other: [$t; $d] =
                core::array::from_fn(|i| if i % 2 == 0 { <$t>::MAX } else { <$t>::MIN });
            let min: [$t; $d] = core::array::from_fn(|i| (i + 2) as $t);
            let max: [$t; $d] = core::array::from_fn(|i| (i + 6) as $t);
            let vector = Vector::<$t, $d>::from(values);

            assert_eq!(
                <[$t; $d]>::from(vector.each_max(Vector::from(other))),
                core::array::from_fn(|i| values[i].max(other[i]))
            );
            assert_eq!(
                <[$t; $d]>::from(vector.each_min(Vector::from(other))),
                core::array::from_fn(|i| values[i].min(other[i]))
            );
            assert_eq!(
                <[$t; $d]>::from(vector.each_clamp(Vector::from(min), Vector::from(max))),
                core::array::from_fn(|i| values[i].clamp(min[i], max[i])),
            );
        }
    };
}

macro_rules! integer_min_max_all_dimensions {
    ($t:ty, $prefix:ident, $min:expr) => {
        paste::paste! {
            integer_min_max_clamp!([<$prefix _min_max_1>], $t, 1, [<$t>::MAX]);
            integer_min_max_clamp!([<$prefix _min_max_2>], $t, 2, [<$t>::MIN, <$t>::MAX]);
            integer_min_max_clamp!([<$prefix _min_max_3>], $t, 3, [$min, 4 as $t, <$t>::MAX]);
            integer_min_max_clamp!([<$prefix _min_max_4>], $t, 4, [$min, 4 as $t, 8 as $t, <$t>::MAX]);
        }
    };
}

integer_min_max_all_dimensions!(i32, i32, i32::MIN);
integer_min_max_all_dimensions!(u32, u32, u32::MIN);
integer_min_max_all_dimensions!(f32, f32, f32::MIN);
integer_min_max_all_dimensions!(i64, i64, i64::MIN);
integer_min_max_all_dimensions!(u64, u64, u64::MIN);
integer_min_max_all_dimensions!(f64, f64, f64::MIN);

macro_rules! select_tests {
    ($name:ident, $t:ty, $d:literal) => {
        #[test]
        fn $name() {
            let a: [$t; $d] = core::array::from_fn(|i| (i + 1) as $t);
            let b: [$t; $d] = core::array::from_fn(|i| (i + 11) as $t);
            let selector_bits = 0b1010_0101_u64 & ((1 << $d) - 1);
            let expected: [$t; $d] =
                core::array::from_fn(|i| if selector_bits & (1 << i) != 0 { a[i] } else { b[i] });

            let selector = Vector::<$t, $d>::from(core::array::from_fn(|i| {
                if selector_bits & (1 << i) != 0 { 1 as $t } else { 0 as $t }
            }))
            .each_ne(Vector::ZERO);
            assert_eq!(
                <[$t; $d]>::from(selector.select(Vector::from(a), Vector::from(b))),
                expected
            );

            assert_eq!(mask_bits(selector), selector_bits);
            assert_eq!(mask_bits(!selector), ((1_u64 << $d) - 1) ^ selector_bits);
        }
    };
}

macro_rules! select_all_dimensions {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            select_tests!([<$prefix _select_1>], $t, 1);
            select_tests!([<$prefix _select_2>], $t, 2);
            select_tests!([<$prefix _select_3>], $t, 3);
            select_tests!([<$prefix _select_4>], $t, 4);
        }
    };
}

select_all_dimensions!(f32, f32);
select_all_dimensions!(i32, i32);
select_all_dimensions!(u32, u32);
select_all_dimensions!(f64, f64);
select_all_dimensions!(i64, i64);
select_all_dimensions!(u64, u64);

macro_rules! float_special_value_comparisons {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let a = Vector::<$t, 3>::from([<$t>::NAN, -0.0, 1.0]);
            let b = Vector::<$t, 3>::from([0.0, 0.0, <$t>::NAN]);

            assert_eq!(mask_bits(a.each_eq(b)), 0b010);
            assert_eq!(mask_bits(a.each_ne(b)), 0b101);
            assert_eq!(mask_bits(a.each_lt(b)), 0);
            assert_eq!(mask_bits(a.each_le(b)), 0b010);
            assert_eq!(mask_bits(a.each_gt(b)), 0);
            assert_eq!(mask_bits(a.each_ge(b)), 0b010);
        }
    };
}

float_special_value_comparisons!(f32_comparisons_match_scalar_for_nan_signed_zero_and_padding, f32);
float_special_value_comparisons!(f64_comparisons_match_scalar_for_nan_signed_zero_and_padding, f64);

macro_rules! float_min_max_clamp_special_values {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let a = Vector::<$t, 4>::from([<$t>::NAN, -0.0, <$t>::NEG_INFINITY, <$t>::INFINITY]);
            let b = Vector::<$t, 4>::from([3.0, 0.0, <$t>::INFINITY, <$t>::NEG_INFINITY]);
            let max: [$t; 4] = a.each_max(b).into();
            let min: [$t; 4] = a.each_min(b).into();
            assert_eq!(max[0], 3.0);
            assert_eq!(min[0], 3.0);
            assert_eq!(max[1], 0.0);
            assert_eq!(min[1], 0.0);
            assert_eq!(max[2], <$t>::INFINITY);
            assert_eq!(min[2], <$t>::NEG_INFINITY);
            assert_eq!(max[3], <$t>::INFINITY);
            assert_eq!(min[3], <$t>::NEG_INFINITY);

            let values =
                Vector::<$t, 4>::from([<$t>::NAN, -0.0, <$t>::NEG_INFINITY, <$t>::INFINITY]);
            let clamped: [$t; 4] =
                values.each_clamp(Vector::from([-1.0; 4]), Vector::from([1.0; 4])).into();
            assert!(clamped[0].is_nan());
            assert_eq!(clamped[1].to_bits(), (-0.0 as $t).to_bits());
            assert_eq!(clamped[2], -1.0);
            assert_eq!(clamped[3], 1.0);
        }
    };
}

float_min_max_clamp_special_values!(f32_min_max_clamp_cover_special_values, f32);
float_min_max_clamp_special_values!(f64_min_max_clamp_cover_special_values, f64);

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn each_clamp_rejects_invalid_active_lane_bounds() {
    assert_eq!(
        panic_message(|| {
            Vector::<i32, 3>::ZERO.each_clamp(Vector::from([0, 2, 0]), Vector::from([1, 1, 1]));
        }),
        "each element in `min` must be less than or equal to the corresponding element in `max`. \
         min = [0, 2, 0], max = [1, 1, 1]",
    );
    assert!(
        std::panic::catch_unwind(|| {
            Vector::<u32, 3>::ZERO.each_clamp(Vector::from([0, 2, 0]), Vector::from([1, 1, 1]))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            Vector::<f32, 3>::ZERO
                .each_clamp(Vector::from([0.0, f32::NAN, 0.0]), Vector::from([1.0, 1.0, 1.0]))
        })
        .is_err()
    );
    assert_eq!(
        panic_message(|| {
            Vector::<i64, 3>::ZERO.each_clamp(Vector::from([0, 2, 0]), Vector::from([1, 1, 1]));
        }),
        "each element in `min` must be less than or equal to the corresponding element in `max`. \
         min = [0, 2, 0], max = [1, 1, 1]",
    );
    assert!(
        std::panic::catch_unwind(|| {
            Vector::<u64, 3>::ZERO.each_clamp(Vector::from([0, 2, 0]), Vector::from([1, 1, 1]))
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            Vector::<f64, 3>::ZERO
                .each_clamp(Vector::from([0.0, f64::NAN, 0.0]), Vector::from([1.0, 1.0, 1.0]))
        })
        .is_err()
    );
}
