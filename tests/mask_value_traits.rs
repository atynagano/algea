//! Tests for mask value traits and lane-wise ordering operations.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use algea::{EachOrd, Mask, support::MaskElement};

fn hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn require_eq<T: Eq>() {}

#[allow(clippy::bool_comparison)]
fn exercise_mask_value_traits<T: MaskElement<D>, const D: usize>() {
    require_eq::<Mask<T, D>>();

    let lhs_lanes = core::array::from_fn(|i| i % 2 == 0);
    let rhs_lanes = core::array::from_fn(|i| i % 3 != 0);
    let lhs = Mask::<T, D>::from(lhs_lanes);
    let same_lhs = Mask::<T, D>::from(lhs_lanes);
    let rhs = Mask::<T, D>::from(rhs_lanes);

    assert_eq!(lhs, same_lhs);
    assert_ne!(lhs, rhs);
    assert_eq!(hash(&lhs), hash(&same_lhs));

    assert_eq!(lhs.each_eq(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] == rhs_lanes[i]));
    assert_eq!(lhs.each_ne(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] != rhs_lanes[i]));
    assert_eq!(lhs.each_lt(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] < rhs_lanes[i]));
    assert_eq!(lhs.each_le(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] <= rhs_lanes[i]));
    assert_eq!(lhs.each_gt(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] > rhs_lanes[i]));
    assert_eq!(lhs.each_ge(rhs).to_array(), core::array::from_fn(|i| lhs_lanes[i] >= rhs_lanes[i]));
    assert_eq!(
        lhs.each_max(rhs).to_array(),
        core::array::from_fn(|i| lhs_lanes[i].max(rhs_lanes[i]))
    );
    assert_eq!(
        lhs.each_min(rhs).to_array(),
        core::array::from_fn(|i| lhs_lanes[i].min(rhs_lanes[i]))
    );

    let min_lanes = core::array::from_fn(|i| i % 4 == 3);
    let max_lanes = core::array::from_fn(|i| i % 4 != 0);
    assert_eq!(
        lhs.each_clamp(Mask::from(min_lanes), Mask::from(max_lanes)).to_array(),
        core::array::from_fn(|i| lhs_lanes[i].clamp(min_lanes[i], max_lanes[i]))
    );
}

#[test]
fn value_traits_cover_every_mask_shape() {
    exercise_mask_value_traits::<i32, 1>();
    exercise_mask_value_traits::<i32, 2>();
    exercise_mask_value_traits::<i32, 3>();
    exercise_mask_value_traits::<i32, 4>();
    exercise_mask_value_traits::<i64, 1>();
    exercise_mask_value_traits::<i64, 2>();
    exercise_mask_value_traits::<i64, 3>();
    exercise_mask_value_traits::<i64, 4>();
}

#[test]
#[should_panic(expected = "each lane in `min` must be less than or equal")]
fn each_clamp_rejects_an_invalid_mask_range() {
    Mask::<i32, 3>::splat(false)
        .each_clamp(Mask::from([false, true, false]), Mask::from([true, false, true]));
}
