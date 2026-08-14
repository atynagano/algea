use crate::{
    private,
    utils::{Load, MaskStorage, Store},
};
#[allow(unused_imports)]
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

pub(crate) trait Simd2Ext {
    type Vector4;
    fn widen(self) -> Self::Vector4;
}
pub(crate) trait Simd4Ext {
    type Vector2;
    fn xy(self) -> Self::Vector2;
}
pub(crate) trait ComputeVector: Copy {
    type Vector2: ComputeVector2;
    type Vector4: ComputeVector4;
}
macro_rules! impl_compute_vector {
    ([$($type:ty),+]: [$vector2:ty, $vector4:ty]) => {
        $(impl ComputeVector for $type {
            type Vector2 = $vector2;
            type Vector4 = $vector4;
        })+
    };
}
impl_compute_vector!([f32x2, f32x4]: [compute_f32x2, f32x4]);
impl_compute_vector!([i32x2, i32x4]: [compute_i32x2, i32x4]);
impl_compute_vector!([u32x2, u32x4]: [compute_u32x2, u32x4]);
#[cfg(target_feature = "sse2")]
impl_compute_vector!([f64x2, f64x4]: [f64x2, f64x4]);
#[cfg(target_feature = "sse2")]
impl_compute_vector!([i64x2, i64x4]: [i64x2, i64x4]);
#[cfg(target_feature = "sse2")]
impl_compute_vector!([u64x2, u64x4]: [u64x2, u64x4]);

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) mod swizzle_impl {
    use super::{
        _64bit_types::{f32x2, i32x2, u32x2},
        ComputeVector,
    };
    use core::arch::aarch64::*;
    use wide::{f32x4, i32x4, u32x4};

    pub(crate) trait SwizzleBase: ComputeVector {
        fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> Self::Vector2;
        fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
            a: Self,
        ) -> Self::Vector4;
        fn swizzle_concat2<const I0: usize, const I1: usize>(a: Self, b: Self) -> Self::Vector2;
        fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
            a: Self,
            b: Self,
        ) -> Self::Vector4;
    }

    pub(crate) trait ComputeVector4:
        SwizzleBase<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
    {
    }
    pub(crate) trait ComputeVector2:
        SwizzleBase<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
    {
    }
    impl<T> ComputeVector4 for T where
        T: SwizzleBase<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
    {
    }
    impl<T> ComputeVector2 for T where
        T: SwizzleBase<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
    {
    }

    // --- Shared byte-level core, used by every element type (f32/i32/u32) and both register
    // widths (2-lane/64-bit, 4-lane/128-bit) below ---
    //
    // Every lane here, regardless of whether the caller interprets it as `f32`, `i32`, or `u32`,
    // is 4 bytes wide, so a lane index `I` always selects bytes `[I*4, I*4+4)`. `f32x4`/`i32x4`/
    // `u32x4` (and `f32x2`/`i32x2`/`u32x2`) therefore share this single implementation instead of
    // each hand-rolling their own NEON shuffle.
    //
    // Verified in dev/neon-tbl-vs-simd-swizzle.md: passing a compile-time-constant index table to
    // `vqtbl1q_u8`/`vqtbl2q_u8` lets LLVM pick the same specialized instruction (`zip1`, `ext` +
    // `trn1`, etc.) it would for `core::simd::simd_swizzle!`, rather than emitting a literal table
    // lookup — confirmed exhaustively for the 2-input, 4-lane case (4096 patterns: 98.6% identical
    // instruction count, no systematic disadvantage in the remaining 1.4%). The 2-lane (64-bit)
    // path below reuses the same 128-bit primitives by zero-widening first, which has not been
    // separately codegen-checked.
    #[inline(always)]
    fn byte_table<const I0: usize, const I1: usize, const I2: usize, const I3: usize>() -> [u8; 16]
    {
        [
            (I0 * 4) as u8,
            (I0 * 4 + 1) as u8,
            (I0 * 4 + 2) as u8,
            (I0 * 4 + 3) as u8,
            (I1 * 4) as u8,
            (I1 * 4 + 1) as u8,
            (I1 * 4 + 2) as u8,
            (I1 * 4 + 3) as u8,
            (I2 * 4) as u8,
            (I2 * 4 + 1) as u8,
            (I2 * 4 + 2) as u8,
            (I2 * 4 + 3) as u8,
            (I3 * 4) as u8,
            (I3 * 4 + 1) as u8,
            (I3 * 4 + 2) as u8,
            (I3 * 4 + 3) as u8,
        ]
    }

    #[inline(always)]
    fn tbl1_16<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: uint8x16_t,
    ) -> uint8x16_t {
        let idx = byte_table::<I0, I1, I2, I3>();
        // SAFETY: `idx` is a fully initialized 16-byte array; NEON permits unaligned loads, and
        // NEON itself is guaranteed available under this module's `target_feature = "neon"` gate.
        unsafe { vqtbl1q_u8(a, vld1q_u8(idx.as_ptr())) }
    }
    #[inline(always)]
    fn tbl2_16<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: uint8x16_t,
        b: uint8x16_t,
    ) -> uint8x16_t {
        let idx = byte_table::<I0, I1, I2, I3>();
        // SAFETY: see `tbl1_16`.
        unsafe { vqtbl2q_u8(uint8x16x2_t(a, b), vld1q_u8(idx.as_ptr())) }
    }

    // `$t`, and its `Vector2`/`Vector4` associated types, are all NEON vector-register wrapper
    // types (`wide::{f32x4,i32x4,u32x4}`, each a `#[repr(C)]` newtype around a 128-bit NEON
    // register) with the same size and bit validity as `uint8x16_t`/`uint8x8_t`, so transmuting
    // between them is sound.
    macro_rules! impl_swizzle_base_4lane {
        ($t:ty) => {
            impl SwizzleBase for $t {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> Self::Vector2 {
                    // SAFETY: see the comment on `impl_swizzle_base_4lane!`.
                    unsafe {
                        let bytes: uint8x16_t = core::mem::transmute(a);
                        let result = vget_low_u8(tbl1_16::<I0, I1, 0, 0>(bytes));
                        core::mem::transmute(result)
                    }
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
                    a: Self,
                ) -> Self::Vector4 {
                    // SAFETY: see the comment on `impl_swizzle_base_4lane!`.
                    unsafe {
                        let bytes: uint8x16_t = core::mem::transmute(a);
                        core::mem::transmute(tbl1_16::<I0, I1, I2, I3>(bytes))
                    }
                }
                #[inline(always)]
                fn swizzle_concat2<const I0: usize, const I1: usize>(
                    a: Self,
                    b: Self,
                ) -> Self::Vector2 {
                    // SAFETY: see the comment on `impl_swizzle_base_4lane!`.
                    unsafe {
                        let a_bytes: uint8x16_t = core::mem::transmute(a);
                        let b_bytes: uint8x16_t = core::mem::transmute(b);
                        let result = vget_low_u8(tbl2_16::<I0, I1, 0, 0>(a_bytes, b_bytes));
                        core::mem::transmute(result)
                    }
                }
                #[inline(always)]
                fn swizzle_concat4<
                    const I0: usize,
                    const I1: usize,
                    const I2: usize,
                    const I3: usize,
                >(
                    a: Self,
                    b: Self,
                ) -> Self::Vector4 {
                    // SAFETY: see the comment on `impl_swizzle_base_4lane!`.
                    unsafe {
                        let a_bytes: uint8x16_t = core::mem::transmute(a);
                        let b_bytes: uint8x16_t = core::mem::transmute(b);
                        core::mem::transmute(tbl2_16::<I0, I1, I2, I3>(a_bytes, b_bytes))
                    }
                }
            }
        };
    }
    impl_swizzle_base_4lane!(f32x4);
    impl_swizzle_base_4lane!(i32x4);
    impl_swizzle_base_4lane!(u32x4);

    // 2-lane (64-bit) self: widen to a 128-bit register (the upper 8 bytes are unused padding,
    // never selected by a valid index) and reuse the exact same `tbl1_16`/`tbl2_16` primitives
    // as the 4-lane case above. This keeps the whole implementation on one, already-checked code
    // path, at the cost of not being separately verified for codegen quality (see the comment on
    // `tbl1_16`/`tbl2_16`).
    macro_rules! impl_swizzle_base_2lane {
        ($t:ty) => {
            impl SwizzleBase for $t {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> Self::Vector2 {
                    // SAFETY: `$t` is a `#[repr(transparent)]` newtype around a 64-bit NEON
                    // register, the same size as `uint8x8_t`, so transmuting is sound.
                    unsafe {
                        let low: uint8x8_t = core::mem::transmute(a);
                        let bytes = vcombine_u8(low, vdup_n_u8(0));
                        core::mem::transmute(vget_low_u8(tbl1_16::<I0, I1, 0, 0>(bytes)))
                    }
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
                    a: Self,
                ) -> Self::Vector4 {
                    // SAFETY: see `swizzle2` above.
                    unsafe {
                        let low: uint8x8_t = core::mem::transmute(a);
                        let bytes = vcombine_u8(low, vdup_n_u8(0));
                        core::mem::transmute(tbl1_16::<I0, I1, I2, I3>(bytes))
                    }
                }
                #[inline(always)]
                fn swizzle_concat2<const I0: usize, const I1: usize>(
                    a: Self,
                    b: Self,
                ) -> Self::Vector2 {
                    // SAFETY: see `swizzle2` above.
                    unsafe {
                        let a_low: uint8x8_t = core::mem::transmute(a);
                        let b_low: uint8x8_t = core::mem::transmute(b);
                        let a_bytes = vcombine_u8(a_low, vdup_n_u8(0));
                        let b_bytes = vcombine_u8(b_low, vdup_n_u8(0));
                        let result = vget_low_u8(tbl2_16::<I0, I1, 0, 0>(a_bytes, b_bytes));
                        core::mem::transmute(result)
                    }
                }
                #[inline(always)]
                fn swizzle_concat4<
                    const I0: usize,
                    const I1: usize,
                    const I2: usize,
                    const I3: usize,
                >(
                    a: Self,
                    b: Self,
                ) -> Self::Vector4 {
                    // SAFETY: see `swizzle2` above.
                    unsafe {
                        let a_low: uint8x8_t = core::mem::transmute(a);
                        let b_low: uint8x8_t = core::mem::transmute(b);
                        let a_bytes = vcombine_u8(a_low, vdup_n_u8(0));
                        let b_bytes = vcombine_u8(b_low, vdup_n_u8(0));
                        core::mem::transmute(tbl2_16::<I0, I1, I2, I3>(a_bytes, b_bytes))
                    }
                }
            }
        };
    }
    impl_swizzle_base_2lane!(f32x2);
    impl_swizzle_base_2lane!(i32x2);
    impl_swizzle_base_2lane!(u32x2);

    macro_rules! swizzle {
        ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i0, $i1])
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i2, $i2])
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle4::<
                { $crate::simd::utils::validate_lane4!($i0) },
                { $crate::simd::utils::validate_lane4!($i1) },
                { $crate::simd::utils::validate_lane4!($i2) },
                { $crate::simd::utils::validate_lane4!($i3) },
            >($a)
        };
        ($a:expr, [$i0:tt, $i1:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle2::<
                { $crate::simd::utils::validate_lane4!($i0) },
                { $crate::simd::utils::validate_lane4!($i1) },
            >($a)
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i2, _])
        };

        ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i0, $i1])
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i2, $i2])
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle_concat4::<
                { $crate::simd::utils::validate_lane8!($i0) },
                { $crate::simd::utils::validate_lane8!($i1) },
                { $crate::simd::utils::validate_lane8!($i2) },
                { $crate::simd::utils::validate_lane8!($i3) },
            >($a, $b)
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle_concat2::<
                { $crate::simd::utils::validate_lane8!($i0) },
                { $crate::simd::utils::validate_lane8!($i1) },
            >($a, $b)
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i2, _])
        };
    }

    pub(crate) use swizzle;

    #[cfg(test)]
    mod tests {
        use super::{SwizzleBase, u32x2, u32x4};

        #[test]
        fn swizzle4_selects_lanes_from_one_input() {
            let a = u32x4::new([10, 11, 12, 13]);

            let actual = SwizzleBase::swizzle4::<3, 1, 0, 2>(a);

            assert_eq!(actual.to_array(), [13, 11, 10, 12]);
        }

        #[test]
        fn swizzle_concat4_selects_lanes_from_both_inputs() {
            let a = u32x4::new([10, 11, 12, 13]);
            let b = u32x4::new([20, 21, 22, 23]);

            let actual = SwizzleBase::swizzle_concat4::<7, 0, 5, 2>(a, b);

            assert_eq!(actual.to_array(), [23, 10, 21, 12]);
        }

        #[test]
        fn swizzle2_selects_lanes_from_one_input() {
            let a = u32x2::new([10, 11]);

            let actual = SwizzleBase::swizzle2::<1, 0>(a);

            assert_eq!(actual.to_array(), [11, 10]);
        }

        #[test]
        fn swizzle_concat2_selects_lanes_from_both_inputs() {
            // Indices 0/1 select `a`'s two lanes and 4/5 select `b`'s, matching the same
            // "`a` widened into the low half, `b` into the high half" convention as
            // `Simd2Ext::widen`; 2/3/6/7 would read each operand's zero-padding.
            let a = u32x2::new([10, 11]);
            let b = u32x2::new([20, 21]);

            let actual = SwizzleBase::swizzle_concat2::<5, 0>(a, b);

            assert_eq!(actual.to_array(), [21, 10]);
        }
    }
}

#[cfg(target_feature = "simd128")]
pub(crate) mod swizzle_impl {
    use super::ComputeVector;
    use wide::{f32x4, i32x4, u32x4};

    pub(crate) trait SwizzleBase: ComputeVector {
        fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> Self::Vector2;
        fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
            a: Self,
        ) -> Self::Vector4;
        fn swizzle_concat2<const I0: usize, const I1: usize>(a: Self, b: Self) -> Self::Vector2;
        fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
            a: Self,
            b: Self,
        ) -> Self::Vector4;
    }

    pub(crate) trait ComputeVector4:
        SwizzleBase<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
    {
    }
    pub(crate) trait ComputeVector2:
        SwizzleBase<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
    {
    }
    impl<T> ComputeVector4 for T where
        T: SwizzleBase<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
    {
    }
    impl<T> ComputeVector2 for T where
        T: SwizzleBase<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
    {
    }

    /// The single general two-input shuffle WebAssembly provides, so no per-pattern instruction
    /// selection is needed. Every lane is four bytes wide whether the caller reads it as `f32`,
    /// `i32`, or `u32`, so one `u32x4_shuffle` serves all three element types.
    #[inline(always)]
    fn shuffle<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: f32x4,
        b: f32x4,
    ) -> f32x4 {
        use core::arch::wasm32::{u32x4_shuffle, v128_load, v128_store};

        let a = a.to_array();
        let b = b.to_array();
        // SAFETY: Both arrays contain exactly 16 initialized bytes.
        // WebAssembly's v128 loads and stores permit unaligned addresses.
        unsafe {
            let a = v128_load(a.as_ptr().cast());
            let b = v128_load(b.as_ptr().cast());
            let shuffled = u32x4_shuffle::<I0, I1, I2, I3>(a, b);
            let mut result = [0.; 4];
            v128_store(result.as_mut_ptr().cast(), shuffled);
            f32x4::new(result)
        }
    }

    // The two-lane compute type is the four-lane type itself here, so a two-lane result is the
    // four-lane shuffle with the requested pair repeated into the padding lanes.
    macro_rules! impl_swizzle_base {
        ($type:ty; |$value:ident| $to_f32x4:expr; |$bits:ident| $from_f32x4:expr) => {
            impl SwizzleBase for $type {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> Self::Vector2 {
                    Self::swizzle4::<I0, I1, I0, I1>(a)
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
                    a: Self,
                ) -> Self::Vector4 {
                    Self::swizzle_concat4::<I0, I1, I2, I3>(a, a)
                }
                #[inline(always)]
                fn swizzle_concat2<const I0: usize, const I1: usize>(
                    a: Self,
                    b: Self,
                ) -> Self::Vector2 {
                    Self::swizzle_concat4::<I0, I1, I0, I1>(a, b)
                }
                #[inline(always)]
                fn swizzle_concat4<
                    const I0: usize,
                    const I1: usize,
                    const I2: usize,
                    const I3: usize,
                >(
                    a: Self,
                    b: Self,
                ) -> Self::Vector4 {
                    let into = |$value: Self| $to_f32x4;
                    let from = |$bits: f32x4| $from_f32x4;
                    from(shuffle::<I0, I1, I2, I3>(into(a), into(b)))
                }
            }
        };
    }
    impl_swizzle_base!(f32x4; |value| value; |bits| bits);
    impl_swizzle_base!(i32x4;
        |value| f32x4::from_bits(value.cast_unsigned());
        |bits| bits.to_bits().cast_signed()
    );
    impl_swizzle_base!(u32x4; |value| f32x4::from_bits(value); |bits| bits.to_bits());

    macro_rules! swizzle {
        ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i0, $i1])
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i2, $i2])
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle4::<
                { $crate::simd::utils::validate_lane4!($i0) },
                { $crate::simd::utils::validate_lane4!($i1) },
                { $crate::simd::utils::validate_lane4!($i2) },
                { $crate::simd::utils::validate_lane4!($i3) },
            >($a)
        };
        ($a:expr, [$i0:tt, $i1:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle2::<
                { $crate::simd::utils::validate_lane4!($i0) },
                { $crate::simd::utils::validate_lane4!($i1) },
            >($a)
        };
        ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
            $crate::simd::utils::swizzle!($a, [$i0, $i1, $i2, _])
        };

        ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i0, $i1])
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i2, $i2])
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle_concat4::<
                { $crate::simd::utils::validate_lane8!($i0) },
                { $crate::simd::utils::validate_lane8!($i1) },
                { $crate::simd::utils::validate_lane8!($i2) },
                { $crate::simd::utils::validate_lane8!($i3) },
            >($a, $b)
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt]) => {
            $crate::simd::utils::swizzle_impl::SwizzleBase::swizzle_concat2::<
                { $crate::simd::utils::validate_lane8!($i0) },
                { $crate::simd::utils::validate_lane8!($i1) },
            >($a, $b)
        };
        ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
            $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i2, _])
        };
    }

    pub(crate) use swizzle;
}

// On x86 these come from the blanket impls in `simd/swizzle_x86.rs`; only the targets without a
// dedicated swizzle module of their own need the identity impls spelled out here.
#[cfg(not(any(all(target_feature = "neon", target_arch = "aarch64"), target_feature = "sse2")))]
mod aliased_2lane {
    use super::{Simd2Ext, Simd4Ext};
    use crate::utils::MaskStorage;
    use wide::{f32x4, i32x4, u32x4};

    impl Simd4Ext for f32x4 {
        type Vector2 = Self;
        #[inline(always)]
        fn xy(self) -> Self::Vector2 { self }
    }
    impl Simd4Ext for i32x4 {
        type Vector2 = Self;
        #[inline(always)]
        fn xy(self) -> Self::Vector2 { self }
    }
    impl Simd4Ext for u32x4 {
        type Vector2 = Self;
        #[inline(always)]
        fn xy(self) -> Self::Vector2 { self }
    }
    impl Simd4Ext for MaskStorage<i32x4> {
        type Vector2 = Self;
        #[inline(always)]
        fn xy(self) -> Self::Vector2 { self }
    }
    impl Simd2Ext for f32x4 {
        type Vector4 = Self;
        #[inline(always)]
        fn widen(self) -> Self::Vector4 { self }
    }
    impl Simd2Ext for i32x4 {
        type Vector4 = Self;
        #[inline(always)]
        fn widen(self) -> Self::Vector4 { self }
    }
    impl Simd2Ext for u32x4 {
        type Vector4 = Self;
        #[inline(always)]
        fn widen(self) -> Self::Vector4 { self }
    }
    impl Simd2Ext for MaskStorage<i32x4> {
        type Vector4 = Self;
        #[inline(always)]
        fn widen(self) -> Self::Vector4 { self }
    }
}

#[rustfmt::skip]
#[allow(unused_macros)]
macro_rules! validate_lane4 {
    (0) => { 0 };
    (1) => { 1 };
    (2) => { 2 };
    (3) => { 3 };
}
#[rustfmt::skip]
#[allow(unused_macros)]
macro_rules! validate_lane8 {
    (0) => { 0 };
    (1) => { 1 };
    (2) => { 2 };
    (3) => { 3 };
    (4) => { 4 };
    (5) => { 5 };
    (6) => { 6 };
    (7) => { 7 };
}

macro_rules! sign {
    ($vector:expr, [+, +, +, -]) => {
        $vector ^ f32x4::new([0., 0., 0., -0.])
    };
    ($vector:expr, [+, +, -, +]) => {
        $vector ^ f32x4::new([0., 0., -0., 0.])
    };
    ($vector:expr, [+, +, -, -]) => {
        $vector ^ f32x4::new([0., 0., -0., -0.])
    };
    ($vector:expr, [+, -, +, +]) => {
        $vector ^ f32x4::new([0., -0., 0., 0.])
    };
    ($vector:expr, [+, -, +, -]) => {
        $vector ^ f32x4::new([0., -0., 0., -0.])
    };
    ($vector:expr, [+, -, -, +]) => {
        $vector ^ f32x4::new([0., -0., -0., 0.])
    };
    ($vector:expr, [+, -, -, -]) => {
        $vector ^ f32x4::new([0., -0., -0., -0.])
    };
    ($vector:expr, [-, +, +, +]) => {
        $vector ^ f32x4::new([-0., 0., 0., 0.])
    };
    ($vector:expr, [-, +, +, -]) => {
        $vector ^ f32x4::new([-0., 0., 0., -0.])
    };
    ($vector:expr, [-, +, -, +]) => {
        $vector ^ f32x4::new([-0., 0., -0., 0.])
    };
    ($vector:expr, [-, +, -, -]) => {
        $vector ^ f32x4::new([-0., 0., -0., -0.])
    };
    ($vector:expr, [-, -, +, +]) => {
        $vector ^ f32x4::new([-0., -0., 0., 0.])
    };
    ($vector:expr, [-, -, +, -]) => {
        $vector ^ f32x4::new([-0., -0., 0., -0.])
    };
    ($vector:expr, [-, -, -, +]) => {
        $vector ^ f32x4::new([-0., -0., -0., 0.])
    };
}

#[cfg(target_feature = "sse2")]
pub(crate) use super::swizzle_x86::{ComputeVector2, ComputeVector4, swizzle4 as swizzle};
#[cfg(not(target_feature = "sse2"))]
pub(crate) use swizzle_impl::{ComputeVector2, ComputeVector4, swizzle};
#[allow(unused_imports)]
pub(crate) use {sign, validate_lane4, validate_lane8};

// A future `std::simd::Simd` backend must preserve the current eight-byte two-lane storage layout.
// `std::simd` can represent LLVM `<2 x float>` directly, whereas `[f32; 2]` remains an aggregate;
// stable Rust currently offers no equally optimizable portable representation with this layout.
#[cfg(not(all(target_feature = "neon", target_arch = "aarch64")))]
mod _64bit_types {
    use crate::utils::{Load, MaskStorage, Store};
    use wide::{f32x4, i32x4, u32x4};

    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct f32x2([f32; 2]);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct i32x2([i32; 2]);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct u32x2([u32; 2]);

    impl f32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [f32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [f32; 2] { self.0 }
        #[inline(always)]
        pub(crate) fn from_bits(bits: u32x2) -> Self {
            Self::new(bits.to_array().map(f32::from_bits))
        }
        #[inline(always)]
        pub(crate) fn to_bits(self) -> u32x2 { u32x2::new(self.to_array().map(f32::to_bits)) }
    }
    impl i32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [i32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [i32; 2] { self.0 }
    }
    impl u32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [u32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [u32; 2] { self.0 }
    }

    // SAFETY: the C layout contains exactly two integer lanes. Their total
    // size is eight bytes, equal to the explicit alignment, so there is no
    // padding, and every bit pattern is valid for the fields.
    const _: () = {
        // Confirm that the two-lane storage types contain no padding bytes.
        assert!(size_of::<f32x2>() == size_of::<[f32; 2]>());
        assert!(size_of::<i32x2>() == size_of::<[i32; 2]>());
        assert!(size_of::<u32x2>() == size_of::<[u32; 2]>());
    };
    unsafe impl wide::bytemuck::Zeroable for i32x2 {}
    unsafe impl wide::bytemuck::Zeroable for u32x2 {}
    unsafe impl wide::bytemuck::Pod for i32x2 {}
    unsafe impl wide::bytemuck::Pod for u32x2 {}

    unsafe impl crate::utils::MaskPrimitive for i32x2 {
        fn is_valid(self) -> bool {
            self.to_array().into_iter().all(crate::utils::MaskPrimitive::is_valid)
        }
        fn not(self) -> Self { unimplemented!() }
        fn bitand(self, _rhs: Self) -> Self { unimplemented!() }
        fn bitor(self, _rhs: Self) -> Self { unimplemented!() }
        fn bitxor(self, _rhs: Self) -> Self { unimplemented!() }
        fn select(self, _true_values: Self, _false_values: Self) -> Self { unimplemented!() }
        fn any<const N: usize>(self) -> bool { unimplemented!() }
        fn all<const N: usize>(self) -> bool { unimplemented!() }
    }
    impl crate::utils::ArithPrimitive for f32x2 {
        type Scalar = f32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0., 0.]);
        const ONE_: Self = Self::new([1., 1.]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }
    impl crate::utils::ArithPrimitive for i32x2 {
        type Scalar = i32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }
    impl crate::utils::ArithPrimitive for u32x2 {
        type Scalar = u32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }

    impl Load for f32x2 {
        type Output = f32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { f32x4::new([self.0[0], self.0[1], 0.0, 0.0]) }
    }
    impl Load for i32x2 {
        type Output = i32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { i32x4::new([self.0[0], self.0[1], 0, 0]) }
    }
    impl Load for u32x2 {
        type Output = u32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { u32x4::new([self.0[0], self.0[1], 0, 0]) }
    }
    impl Store<f32x2> for f32x4 {
        #[inline(always)]
        fn store(self) -> f32x2 {
            let [a, b, ..] = self.to_array();
            f32x2([a, b])
        }
    }
    impl Store<i32x2> for i32x4 {
        #[inline(always)]
        fn store(self) -> i32x2 {
            let [a, b, ..] = self.to_array();
            i32x2([a, b])
        }
    }
    impl Store<u32x2> for u32x4 {
        #[inline(always)]
        fn store(self) -> u32x2 {
            let [a, b, ..] = self.to_array();
            u32x2([a, b])
        }
    }
    impl Load for MaskStorage<i32x4> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl<const N: usize> Load for MaskStorage<[i32x4; N]> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl Load for MaskStorage<i32x2> {
        type Output = MaskStorage<i32x4>;
        #[inline(always)]
        fn load(self) -> Self::Output {
            let inner = self.into_inner().load();
            unsafe {
                // SAFETY: padding lanes are zeroed, so this is safe
                MaskStorage::new_unchecked(inner)
            }
        }
    }
    impl Store<MaskStorage<i32x2>> for MaskStorage<i32x4> {
        #[inline(always)]
        fn store(self) -> MaskStorage<i32x2> {
            let inner = self.into_inner().store();
            unsafe {
                // SAFETY: just dropping the padding lanes, so this is safe
                MaskStorage::new_unchecked(inner)
            }
        }
    }

    macro_rules! impl_load {
        ($($t:ty),*) => {
            $(
                impl Load for $t {
                    type Output = $t;
                    #[inline(always)]
                    fn load(self) -> Self::Output { self }
                }
            )*
        };
    }
    impl_load!(f32, i32, u32);
    impl_load!(f32x4, i32x4, u32x4);
    impl<T: Load<Output = T>, const N: usize> Load for [T; N] {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl Load for MaskStorage<i32> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }

    pub(crate) use wide::{f32x4 as compute_f32x2, i32x4 as compute_i32x2, u32x4 as compute_u32x2};
}

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
mod _64bit_types {
    use crate::utils::{ArithPrimitive, MaskPrimitive, MaskStorage};
    use core::arch::aarch64::*;

    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct f32x2(pub(crate) float32x2_t);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct i32x2(pub(crate) int32x2_t);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct u32x2(pub(crate) uint32x2_t);

    impl From<float32x2_t> for f32x2 {
        #[inline(always)]
        fn from(value: float32x2_t) -> Self { Self(value) }
    }
    impl From<int32x2_t> for i32x2 {
        #[inline(always)]
        fn from(value: int32x2_t) -> Self { Self(value) }
    }
    impl From<uint32x2_t> for u32x2 {
        #[inline(always)]
        fn from(value: uint32x2_t) -> Self { Self(value) }
    }
    impl From<f32x2> for float32x2_t {
        #[inline(always)]
        fn from(value: f32x2) -> Self { value.0 }
    }
    impl From<i32x2> for int32x2_t {
        #[inline(always)]
        fn from(value: i32x2) -> Self { value.0 }
    }
    impl From<u32x2> for uint32x2_t {
        #[inline(always)]
        fn from(value: u32x2) -> Self { value.0 }
    }

    impl f32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [f32; 2]) -> Self {
            // SAFETY: `float32x2_t` is guaranteed to be 8 bytes, the same size as `[f32; 2]`.
            unsafe { core::mem::transmute(a) }
        }
        #[inline(always)]
        pub(crate) const fn splat(a: f32) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [f32; 2] {
            let mut arr = [0.; 2];
            unsafe { vst1_f32(arr.as_mut_ptr(), self.0) };
            arr
        }
        #[inline(always)]
        pub(crate) fn from_bits(bits: u32x2) -> Self {
            unsafe { Self(vreinterpret_f32_u32(bits.0)) }
        }
        #[inline(always)]
        pub(crate) fn to_bits(self) -> u32x2 { unsafe { u32x2(vreinterpret_u32_f32(self.0)) } }

        #[inline(always)]
        pub(crate) fn sqrt(self) -> Self { unsafe { Self(vsqrt_f32(self.0)) } }
        #[inline(always)]
        pub(crate) fn floor(self) -> Self { unsafe { Self(vrndm_f32(self.0)) } }
        #[inline(always)]
        pub(crate) fn ceil(self) -> Self { unsafe { Self(vrndp_f32(self.0)) } }
        #[inline(always)]
        pub(crate) fn round(self) -> Self { unsafe { Self(vrnda_f32(self.0)) } }
        #[inline(always)]
        pub(crate) fn trunc(self) -> Self { unsafe { Self(vrnd_f32(self.0)) } }
        #[inline(always)]
        pub(crate) fn fract(self) -> Self { unsafe { Self(vsub_f32(self.0, vrnd_f32(self.0))) } }
    }
    impl i32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [i32; 2]) -> Self {
            // SAFETY: `int32x2_t` is guaranteed to be 8 bytes, the same size as `[i32; 2]`.
            unsafe { core::mem::transmute(a) }
        }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [i32; 2] {
            let mut arr = [0; 2];
            unsafe { vst1_s32(arr.as_mut_ptr(), self.0) };
            arr
        }
        #[inline(always)]
        pub(crate) fn to_bitmask(self) -> u32 {
            unsafe {
                // Set every bit of a lane to 1 if that lane is negative (canonical mask
                // lanes are always 0 or -1, so this is equivalent to "lane is true").
                let masked = vclt_s32(self.0, vdup_n_s32(0));
                // SAFETY: `uint32x2_t` is guaranteed to be 8 bytes, the same size as `[u32; 2]`.
                let select_bit: uint32x2_t = core::mem::transmute([1u32, 2]);
                let bits = vand_u32(masked, select_bit);
                vaddv_u32(bits)
            }
        }
    }
    impl u32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [u32; 2]) -> Self {
            // SAFETY: `uint32x2_t` is guaranteed to be 8 bytes, the same size as `[u32; 2]`.
            unsafe { core::mem::transmute(a) }
        }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [u32; 2] {
            let mut arr = [0; 2];
            unsafe { vst1_u32(arr.as_mut_ptr(), self.0) };
            arr
        }
    }

    const _: () = {
        assert!(size_of::<f32x2>() == size_of::<[f32; 2]>());
        assert!(size_of::<i32x2>() == size_of::<[i32; 2]>());
        assert!(size_of::<u32x2>() == size_of::<[u32; 2]>());
    };
    unsafe impl wide::bytemuck::Zeroable for i32x2 {}
    unsafe impl wide::bytemuck::Zeroable for u32x2 {}
    unsafe impl wide::bytemuck::Pod for i32x2 {}
    unsafe impl wide::bytemuck::Pod for u32x2 {}

    // SAFETY: validation and `not` operate lane-wise. With a canonical selector,
    // `select` copies each complete physical lane from one of the canonical inputs.
    unsafe impl MaskPrimitive for i32x2 {
        fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
        #[inline(always)]
        fn not(self) -> Self { unsafe { Self(vmvn_s32(self.0)) } }
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self { unsafe { Self(vand_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self { unsafe { Self(vorr_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self { unsafe { Self(veor_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn select(self, true_values: Self, false_values: Self) -> Self {
            unsafe { Self(vbsl_s32(vreinterpret_u32_s32(self.0), true_values.0, false_values.0)) }
        }
        #[inline(always)]
        fn any<const N: usize>(self) -> bool {
            assert_eq!(N, 2);
            unsafe { vminv_s32(self.into()) < 0 }
        }
        #[inline(always)]
        fn all<const N: usize>(self) -> bool {
            assert_eq!(N, 2);
            unsafe { vmaxv_s32(self.into()) < 0 }
        }
    }

    impl ArithPrimitive for f32x2 {
        type Scalar = f32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0., 0.]);
        const ONE_: Self = Self::new([1., 1.]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { unsafe { Self(vdup_n_f32(a)) } }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] {
            unsafe { core::slice::from_raw_parts(self as *const _ as *const f32, 2) }
        }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] {
            unsafe { core::slice::from_raw_parts_mut(self as *mut _ as *mut f32, 2) }
        }
        #[inline(always)]
        fn cast_from_f32_(a: Self::F32) -> Self { a }
        #[inline(always)]
        fn cast_from_i32_(a: Self::I32) -> Self { unsafe { Self(vcvt_f32_s32(a.0)) } }
        #[inline(always)]
        fn cast_from_u32_(a: Self::U32) -> Self { unsafe { Self(vcvt_f32_u32(a.0)) } }
        #[inline(always)]
        fn max_(self, other: Self) -> Self { unsafe { Self(vmaxnm_f32(self.0, other.0)) } }
        #[inline(always)]
        fn min_(self, other: Self) -> Self { unsafe { Self(vminnm_f32(self.0, other.0)) } }
        #[inline(always)]
        fn clamp_noexcept_(mut self, min: Self, max: Self) -> Self {
            self = Self::select_(self.lt_(min), min, self);
            self = Self::select_(self.gt_(max), max, self);
            self
        }
        #[inline(always)]
        fn add_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vadd_f32(self.0, rhs.0)) } }
        #[inline(always)]
        fn sub_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vsub_f32(self.0, rhs.0)) } }
        #[inline(always)]
        fn mul_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vmul_f32(self.0, rhs.0)) } }
        #[inline(always)]
        fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_f32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vceq_f32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn ne_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_f32` produces an all-zero or all-one bit pattern in every lane,
            // `vmvn_u32` complements it (still all-zero or all-one), and
            // `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vmvn_u32(vceq_f32(
                    self.0, other.0,
                )))))
            }
        }
        #[inline(always)]
        fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcgt_f32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcgt_f32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vclt_f32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vclt_f32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn ge_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcge_f32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcge_f32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn le_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcle_f32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcle_f32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
            unsafe {
                Self(vbsl_f32(
                    vreinterpret_u32_s32(mask.into_inner().0),
                    true_values.0,
                    false_values.0,
                ))
            }
        }
        #[inline(always)]
        fn neg_noexcept_(self) -> Self { unsafe { Self(vneg_f32(self.0)) } }
        #[inline(always)]
        fn abs_noexcept_(self) -> Self { unsafe { Self(vabs_f32(self.0)) } }
        #[inline(always)]
        fn round_ties_even_(self) -> Self { unsafe { Self(vrndx_f32(self.0)) } }
        #[inline(always)]
        fn is_nan_(self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_f32(self, self)` is all-zero exactly where `self` is NaN (and
            // all-one elsewhere); `vmvn_u32` complements it to all-one where NaN, and
            // `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vmvn_u32(vceq_f32(
                    self.0, self.0,
                )))))
            }
        }
        #[inline(always)]
        fn mul_add_(a: Self, b: Self, c: Self) -> Self { unsafe { Self(vfma_f32(c.0, a.0, b.0)) } }
        #[inline(always)]
        fn mul_sub_(a: Self, b: Self, c: Self) -> Self {
            unsafe { Self(vfma_f32(vneg_f32(c.0), a.0, b.0)) }
        }
        #[inline(always)]
        fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self {
            unsafe { Self(vfms_f32(c.0, a.0, b.0)) }
        }
    }

    impl ArithPrimitive for i32x2 {
        type Scalar = i32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { unsafe { Self(vdup_n_s32(a)) } }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] {
            unsafe { core::slice::from_raw_parts(self as *const _ as *const i32, 2) }
        }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] {
            unsafe { core::slice::from_raw_parts_mut(self as *mut _ as *mut i32, 2) }
        }
        #[inline(always)]
        fn cast_from_f32_(a: Self::F32) -> Self { unsafe { Self(vcvt_s32_f32(a.0)) } }
        #[inline(always)]
        fn cast_from_i32_(a: Self::I32) -> Self { a }
        #[inline(always)]
        fn cast_from_u32_(a: Self::U32) -> Self { unsafe { Self(vreinterpret_s32_u32(a.0)) } }
        #[inline(always)]
        fn max_(self, other: Self) -> Self { unsafe { Self(vmax_s32(self.0, other.0)) } }
        #[inline(always)]
        fn min_(self, other: Self) -> Self { unsafe { Self(vmin_s32(self.0, other.0)) } }
        #[inline(always)]
        fn add_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vadd_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn sub_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vsub_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn mul_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vmul_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_s32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vceq_s32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcgt_s32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcgt_s32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vclt_s32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vclt_s32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn ge_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcge_s32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcge_s32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn le_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcle_s32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcle_s32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
            unsafe {
                Self(vbsl_s32(
                    vreinterpret_u32_s32(mask.into_inner().0),
                    true_values.0,
                    false_values.0,
                ))
            }
        }
        #[inline(always)]
        fn neg_noexcept_(self) -> Self { unsafe { Self(vneg_s32(self.0)) } }
        #[inline(always)]
        fn abs_noexcept_(self) -> Self { unsafe { Self(vabs_s32(self.0)) } }
        #[inline(always)]
        fn shl_noexcept_(self, rhs: Self) -> Self {
            // `SSHL`/`USHL` treat a shift magnitude >= the 32-bit lane width as a special
            // case (zero, rather than wrapping), so the shift amount is masked to `0..=31`
            // first to match `wrapping_shl`/`wrapping_shr`'s modulo-width semantics. Left
            // shift uses the unsigned intrinsic, matching `wide::i32x4`'s NEON shift and the
            // instruction LLVM itself picks for `Simd<i32, N> << Simd<i32, N>>`.
            unsafe {
                let masked = vand_s32(rhs.0, vdup_n_s32(31));
                Self(vreinterpret_s32_u32(vshl_u32(vreinterpret_u32_s32(self.0), masked)))
            }
        }
        #[inline(always)]
        fn shr_noexcept_(self, rhs: Self) -> Self {
            unsafe { Self(vshl_s32(self.0, vneg_s32(vand_s32(rhs.0, vdup_n_s32(31))))) }
        }
        #[inline(always)]
        fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self {
            unsafe {
                Self(vreinterpret_s32_u32(vshl_u32(
                    vreinterpret_u32_s32(self.0),
                    vdup_n_s32(rhs & 31),
                )))
            }
        }
        #[inline(always)]
        fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self {
            unsafe { Self(vshl_s32(self.0, vdup_n_s32(-(rhs & 31)))) }
        }
    }

    impl ArithPrimitive for u32x2 {
        type Scalar = u32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { unsafe { Self(vdup_n_u32(a)) } }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] {
            unsafe { core::slice::from_raw_parts(self as *const _ as *const u32, 2) }
        }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] {
            unsafe { core::slice::from_raw_parts_mut(self as *mut _ as *mut u32, 2) }
        }
        #[inline(always)]
        fn cast_from_f32_(a: Self::F32) -> Self { unsafe { Self(vcvt_u32_f32(a.0)) } }
        #[inline(always)]
        fn cast_from_i32_(a: Self::I32) -> Self { unsafe { Self(vreinterpret_u32_s32(a.0)) } }
        #[inline(always)]
        fn cast_from_u32_(a: Self::U32) -> Self { a }
        #[inline(always)]
        fn max_(self, other: Self) -> Self { unsafe { Self(vmax_u32(self.0, other.0)) } }
        #[inline(always)]
        fn min_(self, other: Self) -> Self { unsafe { Self(vmin_u32(self.0, other.0)) } }
        #[inline(always)]
        fn add_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vadd_u32(self.0, rhs.0)) } }
        #[inline(always)]
        fn sub_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vsub_u32(self.0, rhs.0)) } }
        #[inline(always)]
        fn mul_noexcept_(self, rhs: Self) -> Self { unsafe { Self(vmul_u32(self.0, rhs.0)) } }
        #[inline(always)]
        fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_u32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vceq_u32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcgt_u32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcgt_u32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vclt_u32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vclt_u32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn ge_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcge_u32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcge_u32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn le_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vcle_u32` produces an all-zero or all-one bit pattern in every lane,
            // and `vreinterpret_s32_u32` preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vcle_u32(self.0, other.0))))
            }
        }
        #[inline(always)]
        fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
            unsafe {
                Self(vbsl_u32(
                    vreinterpret_u32_s32(mask.into_inner().0),
                    true_values.0,
                    false_values.0,
                ))
            }
        }
        #[inline(always)]
        fn shl_noexcept_(self, rhs: Self) -> Self {
            // See the `i32x2` shift impls above: the shift amount is masked to `0..=31` so
            // an out-of-range magnitude doesn't hit `USHL`'s zero-fill special case.
            unsafe { Self(vshl_u32(self.0, vand_s32(vreinterpret_s32_u32(rhs.0), vdup_n_s32(31)))) }
        }
        #[inline(always)]
        fn shr_noexcept_(self, rhs: Self) -> Self {
            unsafe {
                Self(vshl_u32(
                    self.0,
                    vneg_s32(vand_s32(vreinterpret_s32_u32(rhs.0), vdup_n_s32(31))),
                ))
            }
        }
        #[inline(always)]
        fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self {
            unsafe { Self(vshl_u32(self.0, vdup_n_s32((rhs & 31) as i32))) }
        }
        #[inline(always)]
        fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self {
            unsafe { Self(vshl_u32(self.0, vdup_n_s32(-((rhs & 31) as i32)))) }
        }
    }

    crate::utils::impl_default_load!();

    macro_rules! impl_ops {
        (impl $trait:ident for [$($ty:ty),+] { $f_trait:ident => $f:ident }) => {
            $(impl core::ops::$trait for $ty {
                type Output = Self;
                fn $f_trait(self, rhs: Self) -> Self::Output { crate::utils::ArithPrimitive::$f(self, rhs) }
            })+
        };
    }
    impl_ops!(impl Add for [f32x2, i32x2, u32x2] { add => add_noexcept_ });
    impl_ops!(impl Sub for [f32x2, i32x2, u32x2] { sub => sub_noexcept_ });
    impl_ops!(impl Mul for [f32x2, i32x2, u32x2] { mul => mul_noexcept_ });
    impl core::ops::Div for f32x2 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self::Output { unsafe { Self(vdiv_f32(self.0, rhs.0)) } }
    }
    impl core::ops::BitAnd for i32x2 {
        type Output = Self;
        fn bitand(self, rhs: Self) -> Self::Output { unsafe { Self(vand_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitAnd for u32x2 {
        type Output = Self;
        fn bitand(self, rhs: Self) -> Self::Output { unsafe { Self(vand_u32(self.0, rhs.0)) } }
    }
    impl core::ops::BitOr for i32x2 {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output { unsafe { Self(vorr_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitOr for u32x2 {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output { unsafe { Self(vorr_u32(self.0, rhs.0)) } }
    }
    impl core::ops::BitXor for i32x2 {
        type Output = Self;
        fn bitxor(self, rhs: Self) -> Self::Output { unsafe { Self(veor_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitXor for u32x2 {
        type Output = Self;
        fn bitxor(self, rhs: Self) -> Self::Output { unsafe { Self(veor_u32(self.0, rhs.0)) } }
    }
    impl core::ops::Not for i32x2 {
        type Output = Self;
        fn not(self) -> Self::Output { unsafe { Self(vmvn_s32(self.0)) } }
    }
    impl core::ops::Not for u32x2 {
        type Output = Self;
        fn not(self) -> Self::Output { unsafe { Self(vmvn_u32(self.0)) } }
    }
    impl MaskStorage<i32x2> {
        #[inline(always)]
        pub(crate) fn unpack(self) -> Self { self }
    }

    impl super::Simd2Ext for f32x2 {
        type Vector4 = wide::f32x4;

        fn widen(self) -> Self::Vector4 {
            // SAFETY: `wide::f32x4` is a `#[repr(transparent)]`-equivalent wrapper around a
            // single `float32x4_t` NEON register on this target, so reinterpreting one as the
            // other is valid.
            unsafe { vcombine_f32(self.0, vdup_n_f32(0.)).into() }
        }
    }
    impl super::Simd2Ext for i32x2 {
        type Vector4 = wide::i32x4;

        fn widen(self) -> Self::Vector4 {
            // SAFETY: see `f32x2::widen`; `wide::i32x4` wraps a single `int32x4_t`.
            unsafe { vcombine_s32(self.0, vdup_n_s32(0)).into() }
        }
    }
    impl super::Simd2Ext for u32x2 {
        type Vector4 = wide::u32x4;

        fn widen(self) -> Self::Vector4 {
            // SAFETY: see `f32x2::widen`; `wide::u32x4` wraps a single `uint32x4_t`.
            unsafe { vcombine_u32(self.0, vdup_n_u32(0)).into() }
        }
    }
    impl super::Simd2Ext for MaskStorage<i32x2> {
        type Vector4 = MaskStorage<wide::i32x4>;

        fn widen(self) -> Self::Vector4 {
            // SAFETY: `i32x2::widen` zero-extends the two canonical mask lanes (each `0` or
            // `-1`) into a 4-lane vector; the padding lanes are `0`, itself a valid canonical
            // "false" mask value, so the result is a valid canonical mask.
            unsafe { MaskStorage::new_unchecked(self.into_inner().widen()) }
        }
    }
    impl super::Simd4Ext for wide::f32x4 {
        type Vector2 = f32x2;

        fn xy(self) -> Self::Vector2 {
            // SAFETY: see `f32x2::widen`; `wide::f32x4` wraps a single `float32x4_t`.
            unsafe { f32x2(vget_low_f32(self.into())) }
        }
    }
    impl super::Simd4Ext for MaskStorage<wide::i32x4> {
        type Vector2 = MaskStorage<i32x2>;

        fn xy(self) -> Self::Vector2 {
            // SAFETY: `wide::i32x4` wraps a single `int32x4_t` (see `f32x2::widen`), and the
            // low two lanes of an already-canonical mask are themselves canonical (`0` or
            // `-1`), so the narrowed value is a valid canonical mask.
            unsafe { MaskStorage::new_unchecked(self.into_inner().xy()) }
        }
    }
    impl super::Simd4Ext for wide::i32x4 {
        type Vector2 = i32x2;
        fn xy(self) -> Self::Vector2 { unsafe { i32x2(vget_low_s32(self.into())) } }
    }

    pub(crate) use f32x2 as compute_f32x2;
    pub(crate) use i32x2 as compute_i32x2;
    pub(crate) use u32x2 as compute_u32x2;
}

#[allow(unused_imports)]
pub(crate) use _64bit_types::{compute_f32x2, compute_i32x2, compute_u32x2, f32x2, i32x2, u32x2};

impl<const N: usize> Store<MaskStorage<[wide::i32x4; N]>> for [MaskStorage<wide::i32x4>; N] {
    #[inline(always)]
    fn store(self) -> MaskStorage<[wide::i32x4; N]> { self.into() }
}

macro_rules! impl_from {
    ($($tx2:ty:$t:ty),*) => {
        $(
            impl From<[$t; 2]> for $tx2 {
                #[inline]
                fn from(value: [$t; 2]) -> Self { Self::new(value) }
            }
            impl From<$tx2> for [$t; 2] {
                #[inline]
                fn from(value: $tx2) -> Self { value.to_array() }
            }
        )*
    };
}
impl_from!(f32x2:f32, i32x2:i32, u32x2:u32);

// FINDING (kept for future reference — do not "fix" this without re-reading it):
//
// A source of shape `SealedElement<M, 1>` stores exactly `M` meaningful lanes; reading index `M`
// or beyond means reading a padding lane with no defined value. So the constraint we actually
// want is "`IndicesN<I0, ...>` must not contain a value >= M" for whichever `M` a given
// `SwizzleDispatch<T, M, N>` impl targets — but that constraint cannot be derived from (or
// even expressed compatibly with) `Dimension<D>: __internal::AtLeast<b>`, the constraint that
// picks which `Vector<T, D>` swizzle accessors exist (`src/swizzle.rs`). Those accessors are
// generic over `D` — one function body serves every `D` satisfying `AtLeast<b>` — and calls
// `SealedElement::swizzle2`/`3`/`4` with `M == D` still generic at that point. Both narrowing
// `SwizzleDispatchAny<N>`'s bound to be `M`-specific and simply not implementing out-of-range
// `SwizzleDispatch<T, M, N>` impls were tried and fail to compile for that reason: a
// `D`-generic call site cannot resolve a bound that depends on a concrete `M`, and
// `SwizzleDispatchAny<N>` must keep bundling *all* `M` in {2, 3, 4} unconditionally so the
// `D`-generic accessors type-check regardless of which concrete `M` eventually gets used.
//
// A real fix would mean generating dimension-*concrete* swizzle accessors instead (dropping the
// `AtLeast`-based, `D`-generic accessor design in `src/swizzle.rs`/`build.rs`), which is a larger,
// out-of-scope redesign. So instead, every index combination is implemented (to satisfy
// `SwizzleDispatchAny<N>`), but out-of-range combinations get `unimplemented!()` bodies below
// instead of performing a swizzle. `build.rs`'s `AtLeast`/`required_index` machinery guarantees
// these bodies are never reached from public API: an out-of-range index for a given `M` is never
// generated as a call against that `M`.
// The actual swizzle, for an index combination already known to stay in range for `$m`.
macro_rules! impl_swizzle_dispatch_valid {
    ($t:ty, $m:tt, $n:tt, Indices2[$i0:tt, $i1:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices2<$i0, $i1> {
            #[inline(always)]
            fn dispatch(
                v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                swizzle!(v.load(), [$i0, $i1]).store()
            }
        }
    };
    ($t:ty, $m:tt, $n:tt, Indices3[$i0:tt, $i1:tt, $i2:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices3<$i0, $i1, $i2> {
            #[inline(always)]
            fn dispatch(
                v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                swizzle!(v.load(), [$i0, $i1, $i2]).store()
            }
        }
    };
    ($t:ty, $m:tt, $n:tt, Indices4[$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices4<$i0, $i1, $i2, $i3> {
            #[inline(always)]
            fn dispatch(
                v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                swizzle!(v.load(), [$i0, $i1, $i2, $i3]).store()
            }
        }
    };
}

// Walks the index list looking for one that reads a padding lane for `$m` (see the FINDING
// comment above): index 2 or 3 for `$m == 2`, index 3 for `$m == 3`. `$m == 4` has no padding
// lane, so no arm ever matches it and the scan always reaches the end. `[$($orig),*]` is carried
// through unchanged so the final dispatch (real or unimplemented) still has the full index list.
macro_rules! impl_swizzle_dispatch_one {
    ($t:ty, $m:tt, $n:tt, $kind:ident[$($i:tt),*]) => {
        impl_swizzle_dispatch_one!(@scan $t, $m, $n, $kind[$($i),*]; [$($i),*]);
    };
    (@scan $t:ty, 2, $n:tt, $kind:ident[$($orig:tt),*]; [2 $(, $rest:tt)*]) => {
        impl_swizzle_dispatch_unimplemented!($t, 2, $n, $kind[$($orig),*]);
    };
    (@scan $t:ty, 2, $n:tt, $kind:ident[$($orig:tt),*]; [3 $(, $rest:tt)*]) => {
        impl_swizzle_dispatch_unimplemented!($t, 2, $n, $kind[$($orig),*]);
    };
    (@scan $t:ty, 3, $n:tt, $kind:ident[$($orig:tt),*]; [3 $(, $rest:tt)*]) => {
        impl_swizzle_dispatch_unimplemented!($t, 3, $n, $kind[$($orig),*]);
    };
    (@scan $t:ty, $m:tt, $n:tt, $kind:ident[$($orig:tt),*]; [$i:tt $(, $rest:tt)*]) => {
        impl_swizzle_dispatch_one!(@scan $t, $m, $n, $kind[$($orig),*]; [$($rest),*]);
    };
    (@scan $t:ty, $m:tt, $n:tt, $kind:ident[$($orig:tt),*]; []) => {
        impl_swizzle_dispatch_valid!($t, $m, $n, $kind[$($orig),*]);
    };
}

// Emits a `dispatch` that panics: `$i0`/etc. select a lane that is padding (undefined) for a
// source of shape `M`. Unreachable through the public API (see the FINDING comment above), so no
// message is attached.
macro_rules! impl_swizzle_dispatch_unimplemented {
    ($t:ty, $m:tt, $n:tt, Indices2[$i0:tt, $i1:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices2<$i0, $i1> {
            #[inline(always)]
            fn dispatch(
                _v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                unimplemented!()
            }
        }
    };
    ($t:ty, $m:tt, $n:tt, Indices3[$i0:tt, $i1:tt, $i2:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices3<$i0, $i1, $i2> {
            #[inline(always)]
            fn dispatch(
                _v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                unimplemented!()
            }
        }
    };
    ($t:ty, $m:tt, $n:tt, Indices4[$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        impl private::SwizzleDispatch<$t, $m, $n> for private::Indices4<$i0, $i1, $i2, $i3> {
            #[inline(always)]
            fn dispatch(
                _v: <$t as private::SealedElement<$m, 1>>::Storage,
            ) -> <$t as private::SealedElement<$n, 1>>::Storage {
                unimplemented!()
            }
        }
    };
}

// `f32`, `i32`, and `u32` share the exact same dispatch body, so every index combination is
// implemented for all three element types through this one macro.
macro_rules! impl_swizzle_dispatch {
    ($m:tt, $n:tt, $kind:ident[$($i:tt),*]) => {
        impl_swizzle_dispatch_one!(f32, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(i32, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(u32, $m, $n, $kind[$($i),*]);
    };
}

macro_rules! impl_swizzle2_for_i0 {
    ($i0:tt; $($i1:tt),*) => {$(
        impl_swizzle_dispatch!(2, 2, Indices2[$i0, $i1]);
        impl_swizzle_dispatch!(3, 2, Indices2[$i0, $i1]);
        impl_swizzle_dispatch!(4, 2, Indices2[$i0, $i1]);
    )*};
}
macro_rules! impl_swizzle3_for_i0_i1 {
    ($i0:tt, $i1:tt; $($i2:tt),*) => {$(
        impl_swizzle_dispatch!(2, 3, Indices3[$i0, $i1, $i2]);
        impl_swizzle_dispatch!(3, 3, Indices3[$i0, $i1, $i2]);
        impl_swizzle_dispatch!(4, 3, Indices3[$i0, $i1, $i2]);
    )*};
}
macro_rules! impl_swizzle3_for_i0 {
    ($i0:tt; $($i1:tt),*) => {
        $(impl_swizzle3_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0_i1_i2 {
    ($i0:tt, $i1:tt, $i2:tt; $($i3:tt),*) => {$(
        impl_swizzle_dispatch!(2, 4, Indices4[$i0, $i1, $i2, $i3]);
        impl_swizzle_dispatch!(3, 4, Indices4[$i0, $i1, $i2, $i3]);
        impl_swizzle_dispatch!(4, 4, Indices4[$i0, $i1, $i2, $i3]);
    )*};
}
macro_rules! impl_swizzle4_for_i0_i1 {
    ($i0:tt, $i1:tt; $($i2:tt),*) => {
        $(impl_swizzle4_for_i0_i1_i2!($i0, $i1, $i2; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0 {
    ($i0:tt; $($i1:tt),*) => {
        $(impl_swizzle4_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}

impl_swizzle2_for_i0!(0; 0, 1, 2, 3);
impl_swizzle2_for_i0!(1; 0, 1, 2, 3);
impl_swizzle2_for_i0!(2; 0, 1, 2, 3);
impl_swizzle2_for_i0!(3; 0, 1, 2, 3);
impl_swizzle3_for_i0!(0; 0, 1, 2, 3);
impl_swizzle3_for_i0!(1; 0, 1, 2, 3);
impl_swizzle3_for_i0!(2; 0, 1, 2, 3);
impl_swizzle3_for_i0!(3; 0, 1, 2, 3);
impl_swizzle4_for_i0!(0; 0, 1, 2, 3);
impl_swizzle4_for_i0!(1; 0, 1, 2, 3);
impl_swizzle4_for_i0!(2; 0, 1, 2, 3);
impl_swizzle4_for_i0!(3; 0, 1, 2, 3);
