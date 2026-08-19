use super::mask::i32::*;
use crate::{
    private,
    simd::utils::{compute_i32x2, f32x2, i32x2, u32x2},
    utils::{ArithPrimitive, MaskPrimitive, MaskStorage, Store},
};
use wide::{f32x4, i32x4, u32x4};

const T: i32 = -1;
const F: i32 = 0;

/// Builds the two-lane storage that `compute_i32x2` resolves to on the current target: a
/// genuinely two-lane register on aarch64/NEON, or an `i32x4` with the upper two lanes unused
/// everywhere else (matching how `mask::from_array_2`/`to_array_2` treat those lanes).
fn compact2(values: [i32; 2]) -> compute_i32x2 {
    cfg_select! {
        all(target_feature = "neon", target_arch = "aarch64") => i32x2::new(values),
        _ => i32x4::new([values[0], values[1], 0, 0]),
    }
}

fn canonical<T: MaskPrimitive>(value: T) -> MaskStorage<T> {
    assert!(value.is_valid());
    // SAFETY: `is_valid` verified that every physical lane is either 0 or -1.
    unsafe { MaskStorage::new_unchecked(value) }
}

fn assert_canonical_lanes(mask: MaskStorage<i32x4>, expected: [i32; 4]) {
    let storage = mask.into_inner();
    assert!(storage.is_valid());
    assert_eq!(storage.to_array(), expected);
}

#[test]
fn compact_two_lane_storage_has_expected_layout() {
    assert_eq!(core::mem::size_of::<f32x2>(), 8);
    assert_eq!(core::mem::align_of::<f32x2>(), 8);
    assert_eq!(core::mem::size_of::<i32x2>(), 8);
    assert_eq!(core::mem::align_of::<i32x2>(), 8);
    assert_eq!(core::mem::size_of::<u32x2>(), 8);
    assert_eq!(core::mem::align_of::<u32x2>(), 8);
}

macro_rules! assert_mask_query {
    ($m:tt, $n:tt, $all_true:expr, $all_false:expr, $mixed:expr) => {{
        assert!(<i32 as private::SealedElement<$m, $n>>::all(canonical($all_true)));
        assert!(<i32 as private::SealedElement<$m, $n>>::any(canonical($all_true)));
        assert!(!<i32 as private::SealedElement<$m, $n>>::all(canonical($all_false)));
        assert!(!<i32 as private::SealedElement<$m, $n>>::any(canonical($all_false)));
        assert!(!<i32 as private::SealedElement<$m, $n>>::all(canonical($mixed)));
        assert!(<i32 as private::SealedElement<$m, $n>>::any(canonical($mixed)));
    }};
}

#[test]
fn all_and_any_cover_all_16_storage_shapes() {
    assert!(<i32 as private::SealedElement<1, 1>>::all(canonical(T)));
    assert!(<i32 as private::SealedElement<1, 1>>::any(canonical(T)));
    assert!(!<i32 as private::SealedElement<1, 1>>::all(canonical(F)));
    assert!(!<i32 as private::SealedElement<1, 1>>::any(canonical(F)));
    assert_mask_query!(
        2,
        1,
        compact2([T, T]).store(),
        compact2([F, F]).store(),
        compact2([T, F]).store()
    );
    assert_mask_query!(
        3,
        1,
        i32x4::new([T, T, T, F]),
        i32x4::new([F, F, F, T]),
        i32x4::new([F, T, F, F])
    );
    assert_mask_query!(4, 1, i32x4::splat(T), i32x4::splat(F), i32x4::new([F, F, T, F]));

    assert_mask_query!(
        1,
        2,
        compact2([T, T]).store(),
        compact2([F, F]).store(),
        compact2([F, T]).store()
    );
    assert_mask_query!(2, 2, i32x4::splat(T), i32x4::splat(F), i32x4::new([T, F, F, F]));
    assert_mask_query!(3, 2, [i32x4::new([T, T, T, F]); 2], [i32x4::new([F, F, F, T]); 2], [
        i32x4::new([F, T, F, F]),
        i32x4::splat(F)
    ]);
    assert_mask_query!(4, 2, [i32x4::splat(T); 2], [i32x4::splat(F); 2], [
        i32x4::splat(F),
        i32x4::new([F, F, F, T])
    ]);

    assert_mask_query!(
        1,
        3,
        i32x4::new([T, T, T, F]),
        i32x4::new([F, F, F, T]),
        i32x4::new([T, F, F, F])
    );
    assert_mask_query!(
        2,
        3,
        [i32x4::splat(T), i32x4::new([T, T, F, F])],
        [i32x4::splat(F), i32x4::new([F, F, T, T])],
        [i32x4::splat(F), i32x4::new([F, T, F, F])]
    );
    assert_mask_query!(3, 3, [i32x4::new([T, T, T, F]); 3], [i32x4::new([F, F, F, T]); 3], [
        i32x4::new([F, F, T, F]),
        i32x4::splat(F),
        i32x4::splat(F)
    ]);
    assert_mask_query!(4, 3, [i32x4::splat(T); 3], [i32x4::splat(F); 3], [
        i32x4::splat(F),
        i32x4::new([T, F, F, F]),
        i32x4::splat(F)
    ]);

    assert_mask_query!(1, 4, i32x4::splat(T), i32x4::splat(F), i32x4::new([F, T, F, F]));
    assert_mask_query!(2, 4, [i32x4::splat(T); 2], [i32x4::splat(F); 2], [
        i32x4::new([F, F, T, F]),
        i32x4::splat(F)
    ]);
    assert_mask_query!(3, 4, [i32x4::new([T, T, T, F]); 4], [i32x4::new([F, F, F, T]); 4], [
        i32x4::splat(F),
        i32x4::splat(F),
        i32x4::new([F, T, F, F]),
        i32x4::splat(F),
    ]);
    assert_mask_query!(4, 4, [i32x4::splat(T); 4], [i32x4::splat(F); 4], [
        i32x4::splat(F),
        i32x4::splat(F),
        i32x4::splat(F),
        i32x4::new([F, F, F, T]),
    ]);
}

macro_rules! assert_array_round_trip {
    ($from:ident, $to:ident, $value:expr) => {{
        let value = $value;
        assert_eq!($to($from(value)), value);
    }};
}

#[test]
fn mask_array_conversion_covers_all_16_storage_shapes() {
    assert_array_round_trip!(from_array_1x1, to_array_1x1, [[true]]);
    assert_array_round_trip!(from_array_2x1, to_array_2x1, [[true, false]]);
    assert_array_round_trip!(from_array_3x1, to_array_3x1, [[true, false, true]]);
    assert_array_round_trip!(from_array_4x1, to_array_4x1, [[true, false, true, false]]);

    assert_array_round_trip!(from_array_1x2, to_array_1x2, [[true], [false]]);
    assert_array_round_trip!(from_array_2x2, to_array_2x2, [[true, false], [false, true]]);
    assert_array_round_trip!(from_array_3x2, to_array_3x2, [[true, false, true], [
        false, true, false
    ]]);
    assert_array_round_trip!(from_array_4x2, to_array_4x2, [[true, false, true, false], [
        false, true, false, true
    ]]);

    assert_array_round_trip!(from_array_1x3, to_array_1x3, [[true], [false], [true]]);
    assert_array_round_trip!(from_array_2x3, to_array_2x3, [[true, false], [false, true], [
        true, true
    ]]);
    assert_array_round_trip!(from_array_3x3, to_array_3x3, [
        [true, false, true],
        [false, true, false],
        [true, true, false]
    ]);
    assert_array_round_trip!(from_array_4x3, to_array_4x3, [
        [true, false, true, false],
        [false, true, false, true],
        [true, true, false, false],
    ]);

    assert_array_round_trip!(from_array_1x4, to_array_1x4, [[true], [false], [true], [true]]);
    assert_array_round_trip!(from_array_2x4, to_array_2x4, [
        [true, false],
        [false, true],
        [true, true],
        [false, false]
    ]);
    assert_array_round_trip!(from_array_3x4, to_array_3x4, [
        [true, false, true],
        [false, true, false],
        [true, true, false],
        [false, false, true],
    ]);
    assert_array_round_trip!(from_array_4x4, to_array_4x4, [
        [true, false, true, false],
        [false, true, false, true],
        [true, true, false, false],
        [false, false, true, true],
    ]);
}

#[test]
fn mask_storage_validation_not_and_pack_unpack_preserve_canonical_lanes() {
    assert!(F.is_valid());
    assert!(T.is_valid());
    assert!(!1_i32.is_valid());
    assert!(!(-2_i32).is_valid());

    assert!(i32x4::new([T, F, T, F]).is_valid());
    assert!(!i32x4::new([T, F, 1, F]).is_valid());
    assert!([i32x4::new([T, F, T, F]), i32x4::splat(F)].is_valid());
    assert!(![i32x4::new([T, F, T, F]), i32x4::new([F, -2, F, F])].is_valid());

    let first = canonical(i32x4::new([T, F, T, F]));
    let second = canonical(i32x4::new([F, T, F, T]));
    assert_canonical_lanes(!first, [F, T, F, T]);

    let packed: MaskStorage<[i32x4; 2]> = [first, second].into();
    assert!(packed.into_inner().is_valid());
    let unpacked = packed.unpack();
    assert_canonical_lanes(unpacked[0], [T, F, T, F]);
    assert_canonical_lanes(unpacked[1], [F, T, F, T]);
}

#[test]
fn f32x4_comparisons_produce_canonical_lanes() {
    let a = f32x4::new([1.0, 2.0, f32::NAN, -0.0]);
    let b = f32x4::new([1.0, 3.0, f32::INFINITY, 0.0]);

    assert_canonical_lanes(<f32x4 as ArithPrimitive>::eq_(a, b), [T, F, F, T]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::ne_(a, b), [F, T, T, F]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::lt_(a, b), [F, T, F, F]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::le_(a, b), [T, T, F, T]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::gt_(a, b), [F, F, F, F]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::ge_(a, b), [T, F, F, T]);
    assert_canonical_lanes(<f32x4 as ArithPrimitive>::is_nan_(a), [F, F, T, F]);
}

#[test]
fn i32x4_comparisons_produce_canonical_lanes() {
    let a = i32x4::new([-2, 5, 7, i32::MAX]);
    let b = i32x4::new([-1, 5, 3, i32::MIN]);

    assert_canonical_lanes(<i32x4 as ArithPrimitive>::eq_(a, b), [F, T, F, F]);
    assert_canonical_lanes(<i32x4 as ArithPrimitive>::ne_(a, b), [T, F, T, T]);
    assert_canonical_lanes(<i32x4 as ArithPrimitive>::lt_(a, b), [T, F, F, F]);
    assert_canonical_lanes(<i32x4 as ArithPrimitive>::le_(a, b), [T, T, F, F]);
    assert_canonical_lanes(<i32x4 as ArithPrimitive>::gt_(a, b), [F, F, T, T]);
    assert_canonical_lanes(<i32x4 as ArithPrimitive>::ge_(a, b), [F, T, T, T]);
}

#[test]
fn u32x4_comparisons_produce_canonical_lanes() {
    let a = u32x4::new([0, 5, u32::MAX, 1]);
    let b = u32x4::new([1, 5, 0, 2]);

    assert_canonical_lanes(<u32x4 as ArithPrimitive>::eq_(a, b), [F, T, F, F]);
    assert_canonical_lanes(<u32x4 as ArithPrimitive>::ne_(a, b), [T, F, T, T]);
    assert_canonical_lanes(<u32x4 as ArithPrimitive>::lt_(a, b), [T, F, F, T]);
    assert_canonical_lanes(<u32x4 as ArithPrimitive>::le_(a, b), [T, T, F, T]);
    assert_canonical_lanes(<u32x4 as ArithPrimitive>::gt_(a, b), [F, F, T, F]);
    assert_canonical_lanes(<u32x4 as ArithPrimitive>::ge_(a, b), [F, T, T, F]);
}
