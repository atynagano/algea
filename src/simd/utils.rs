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
impl_compute_vector!([f64x2, f64x4]: [f64x2, f64x4]);
impl_compute_vector!([i64x2, i64x4]: [i64x2, i64x4]);
impl_compute_vector!([u64x2, u64x4]: [u64x2, u64x4]);

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) use super::swizzle_arm::{Swizzle, SwizzleConcat, swizzle4 as swizzle};
#[cfg(target_feature = "simd128")]
pub(crate) use super::swizzle_wasm::{Swizzle, SwizzleConcat, swizzle4 as swizzle};
#[cfg(target_feature = "sse2")]
pub(crate) use super::swizzle_x86::{Swizzle, SwizzleConcat, swizzle4 as swizzle};

pub(crate) trait ComputeVector4:
    SwizzleConcat<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
{
}
pub(crate) trait ComputeVector2:
    Swizzle<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
{
}
impl<T> ComputeVector4 for T where
    T: SwizzleConcat<Vector4 = Self, Vector2: ComputeVector<Vector4 = Self>>
{
}
impl<T> ComputeVector2 for T where T: Swizzle<Vector2 = Self, Vector4: ComputeVector<Vector2 = Self>>
{}

impl<T: ComputeVector4> Simd4Ext for T {
    type Vector2 = <T as ComputeVector>::Vector2;
    #[inline(always)]
    fn xy(self) -> Self::Vector2 { <T as Swizzle>::__xy(self) }
}
impl<T: ComputeVector2> Simd2Ext for T {
    type Vector4 = <T as ComputeVector>::Vector4;
    #[inline(always)]
    fn widen(self) -> Self::Vector4 { <T as Swizzle>::__widen(self) }
}
// Mask storage is never swizzled, so it is given `Simd2Ext`/`Simd4Ext` directly rather than a
// `Swizzle` implementation it would never call just to reach the blanket impls above. Each one
// forwards to the storage vector it wraps, which those blanket impls do cover.
impl Simd4Ext for MaskStorage<i32x4> {
    type Vector2 = MaskStorage<compute_i32x2>;
    #[inline(always)]
    fn xy(self) -> Self::Vector2 {
        // SAFETY: narrowing keeps the low lanes of an already-canonical mask, each of which is
        // `0` or `-1` and so canonical on its own.
        unsafe { MaskStorage::new_unchecked(self.into_inner().xy()) }
    }
}
impl Simd2Ext for MaskStorage<compute_i32x2> {
    type Vector4 = MaskStorage<i32x4>;
    #[inline(always)]
    fn widen(self) -> Self::Vector4 {
        // SAFETY: widening zero-fills the padding lanes, and `0` is itself the canonical "false"
        // value, so the result still satisfies the invariant.
        unsafe { MaskStorage::new_unchecked(self.into_inner().widen()) }
    }
}
impl Simd4Ext for MaskStorage<i64x4> {
    type Vector2 = MaskStorage<i64x2>;
    #[inline(always)]
    fn xy(self) -> Self::Vector2 {
        // SAFETY: see `MaskStorage<i32x4>::xy`.
        unsafe { MaskStorage::new_unchecked(self.into_inner().xy()) }
    }
}
impl Simd2Ext for MaskStorage<i64x2> {
    type Vector4 = MaskStorage<i64x4>;
    #[inline(always)]
    fn widen(self) -> Self::Vector4 {
        // SAFETY: see `MaskStorage<compute_i32x2>::widen`.
        unsafe { MaskStorage::new_unchecked(self.into_inner().widen()) }
    }
}

/// Completes a four-lane index list whose last lane is `_`, then hands it to the backend macro
/// `$mac`. The operands are passed through as one group, so this serves both the one-operand and
/// the two-operand form.
///
/// Where the third lane is even, the padding lane takes the lane after it. The upper half of the
/// result is then one whole 64-bit half of a source, which every backend moves in a single
/// instruction; leaving the choice to the backend instead costs two lane inserts on targets
/// without a general two-input four-lane 32-bit shuffle. Repeating the third lane cannot beat
/// that, since a duplicated lane is a move of its own.
///
/// Unless the two given lanes are equal. Then the whole list may collapse into one instruction —
/// `[x, x, x, x]` is a broadcast, `[x, x, y, y]` an interleave of one operand with itself — and
/// naming a different lane would break that, so the third lane repeats instead.
#[rustfmt::skip]
macro_rules! complete_swizzle4 {
    ([$($mac:tt)*], ($($operands:tt)*), [0, 0, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [0, 0, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [1, 1, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [1, 1, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [2, 2, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [2, 2, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [3, 3, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [3, 3, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [4, 4, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [4, 4, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [5, 5, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [5, 5, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [6, 6, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [6, 6, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [7, 7, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [7, 7, $i2, $i2])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [$i0:tt, $i1:tt, 0, _]) => {
        $($mac)*!($($operands)*, [$i0, $i1, 0, 1])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [$i0:tt, $i1:tt, 2, _]) => {
        $($mac)*!($($operands)*, [$i0, $i1, 2, 3])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [$i0:tt, $i1:tt, 4, _]) => {
        $($mac)*!($($operands)*, [$i0, $i1, 4, 5])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [$i0:tt, $i1:tt, 6, _]) => {
        $($mac)*!($($operands)*, [$i0, $i1, 6, 7])
    };
    ([$($mac:tt)*], ($($operands:tt)*), [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $($mac)*!($($operands)*, [$i0, $i1, $i2, $i2])
    };
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

// NOTE: lets a caller write `__bitxor` without annotating whether the operand is `f32` or
// `f64`, which inference cannot pin down on its own here.
pub(crate) trait __BitXorSelf: core::ops::BitXor + Sized {
    fn __bitxor(lhs: Self, rhs: Self) -> Self::Output;
}
impl<T: core::ops::BitXor> __BitXorSelf for T {
    #[inline(always)]
    fn __bitxor(lhs: Self, rhs: Self) -> Self::Output { core::ops::BitXor::bitxor(lhs, rhs) }
}

macro_rules! sign {
    ($vector:expr, [+, +, +, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., 0., 0., -0.].into())
    };
    ($vector:expr, [+, +, -, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., 0., -0., 0.].into())
    };
    ($vector:expr, [+, +, -, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., 0., -0., -0.].into())
    };
    ($vector:expr, [+, -, +, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., -0., 0., 0.].into())
    };
    ($vector:expr, [+, -, +, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., -0., 0., -0.].into())
    };
    ($vector:expr, [+, -, -, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., -0., -0., 0.].into())
    };
    ($vector:expr, [+, -, -, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [0., -0., -0., -0.].into())
    };
    ($vector:expr, [-, +, +, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., 0., 0., 0.].into())
    };
    ($vector:expr, [-, +, +, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., 0., 0., -0.].into())
    };
    ($vector:expr, [-, +, -, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., 0., -0., 0.].into())
    };
    ($vector:expr, [-, +, -, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., 0., -0., -0.].into())
    };
    ($vector:expr, [-, -, +, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., -0., 0., 0.].into())
    };
    ($vector:expr, [-, -, +, -]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., -0., 0., -0.].into())
    };
    ($vector:expr, [-, -, -, +]) => {
        $crate::simd::utils::__BitXorSelf::__bitxor($vector, [-0., -0., -0., 0.].into())
    };
}

#[allow(unused_imports)]
pub(crate) use {complete_swizzle4, sign, validate_lane4, validate_lane8};

// A future `std::simd::Simd` backend must preserve the current eight-byte two-lane storage layout.
// `std::simd` can represent LLVM `<2 x float>` directly, whereas `[f32; 2]` remains an aggregate;
// stable Rust currently offers no equally optimizable portable representation with this layout.
#[cfg(not(all(target_feature = "neon", target_arch = "aarch64")))]
mod _64bit_types {
    use crate::{
        simd::kernels,
        utils::{Load, MaskPrimitive, MaskStorage, Store},
    };
    use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

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
        #[inline(always)]
        pub(crate) fn cast_unsigned(self) -> u32x2 { u32x2(self.0.map(i32::cast_unsigned)) }
    }
    impl u32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [u32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [u32; 2] { self.0 }
        #[inline(always)]
        pub(crate) fn cast_signed(self) -> i32x2 { i32x2(self.0.map(u32::cast_signed)) }
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

    // SAFETY: `is_valid` accepts a value only when every lane is canonical. The operations are
    // unreachable and hold vacuously: this type is a storage width alone, and `MaskLoad` below
    // widens it to `i32x4` before anything operates on it.
    unsafe impl MaskPrimitive for i32x2 {
        fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
        fn canonical_not(self) -> Self { unimplemented!() }
        fn canonical_bitand(self, _rhs: Self) -> Self { unimplemented!() }
        fn canonical_bitor(self, _rhs: Self) -> Self { unimplemented!() }
        fn canonical_bitxor(self, _rhs: Self) -> Self { unimplemented!() }
        fn canonical_select(self, _true_values: Self, _false_values: Self) -> Self {
            unimplemented!()
        }
        fn any<const N: usize>(self) -> bool { unimplemented!() }
        fn all<const N: usize>(self) -> bool { unimplemented!() }
    }
    // SAFETY: `__load` zeroes the two lanes it adds and copies the other two, and `__store` drops
    // those two again, so a canonical value maps to a canonical value in either direction.
    unsafe impl crate::utils::MaskLoad for i32x2 {
        type Primitive = i32x4;

        #[inline(always)]
        fn is_valid_storage(self) -> bool {
            self.to_array().into_iter().all(MaskPrimitive::is_valid)
        }
        #[inline(always)]
        fn __load(self) -> Self::Primitive { i32x4::new([self.0[0], self.0[1], 0, 0]) }
        #[inline(always)]
        fn __store(v: Self::Primitive) -> Self {
            let [a, b, ..] = v.to_array();
            i32x2([a, b])
        }
    }

    mod cast {
        pub(super) use super::kernels::cast::*;
        pub(crate) use core::convert::{
            identity as f32x2_from_f32,
            identity as i32x2_from_i32,
            identity as u32x2_from_u32,
        };
    }

    macro_rules! impl_arith_primitive {
        ($scalar:ty, $vec2:ty) => {
            impl ArithPrimitive for $vec2 {
                type Scalar = $scalar;
                type F32 = f32x2;
                type F64 = f64x2;
                type I32 = i32x2;
                type I64 = i64x2;
                type U32 = u32x2;
                type U64 = u64x2;
                type Mask = i32x2;
                const ZERO_: Self = Self::new([0 as _; 2]);
                const ONE_: Self = Self::new([1 as _; 2]);

                #[inline(always)]
                fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
                #[inline(always)]
                fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
                #[inline(always)]
                fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }

                #[inline(always)]
                fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self { paste::paste!(cast::[<$scalar x2_from_f32>] (a)) }
                #[inline(always)]
                fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self { paste::paste!(cast::[<$scalar x2_from_f64>] (a)) }
                #[inline(always)]
                fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self { paste::paste!(cast::[<$scalar x2_from_i32>] (a)) }
                #[inline(always)]
                fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self { paste::paste!(cast::[<$scalar x2_from_i64>] (a)) }
                #[inline(always)]
                fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self { paste::paste!(cast::[<$scalar x2_from_u32>] (a)) }
                #[inline(always)]
                fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self { paste::paste!(cast::[<$scalar x2_from_u64>] (a)) }

                #[inline(always)]
                fn max_(self, other: Self) -> Self { self.load().max_(other.load()).store() }
                #[inline(always)]
                fn min_(self, other: Self) -> Self { self.load().min_(other.load()).store() }
                #[inline(always)]
                fn clamp_noexcept_(self, min: Self, max: Self) -> Self { self.load().clamp_noexcept_(min.load(), max.load()).store() }
                #[inline(always)]
                fn add_noexcept_(self, rhs: Self) -> Self { self.load().add_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn sub_noexcept_(self, rhs: Self) -> Self { self.load().sub_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn mul_noexcept_(self, rhs: Self) -> Self { self.load().mul_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn div_(self, rhs: Self) -> Self { self.load().div_(rhs.load()).store() }

                #[inline(always)]
                fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().eq_(other.load())) }
                #[inline(always)]
                fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().ne_(other.load())) }
                #[inline(always)]
                fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().gt_(other.load())) }
                #[inline(always)]
                fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().lt_(other.load())) }
                #[inline(always)]
                fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().ge_(other.load())) }
                #[inline(always)]
                fn le_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().le_(other.load())) }
                #[inline(always)]
                fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
                    ArithPrimitive::select_(mask.load_mask(), true_values.load(), false_values.load()).store()
                }

                #[inline(always)]
                fn neg_noexcept_(self) -> Self { self.load().neg_noexcept_().store() }
                #[inline(always)]
                fn abs_noexcept_(self) -> Self { self.load().abs_noexcept_().store() }
                #[inline(always)]
                fn signum_(self) -> Self { self.load().signum_().store() }

                #[inline(always)]
                fn sqrt_(self) -> Self { self.load().sqrt_().store() }
                #[inline(always)]
                fn floor_(self) -> Self { self.load().floor_().store() }
                #[inline(always)]
                fn ceil_(self) -> Self { self.load().ceil_().store() }
                #[inline(always)]
                fn round_(self) -> Self { self.load().round_().store() }
                #[inline(always)]
                fn trunc_(self) -> Self { self.load().trunc_().store() }
                #[inline(always)]
                fn fract_(self) -> Self { self.load().fract_().store() }
                #[inline(always)]
                fn round_ties_even_(self) -> Self { self.load().round_ties_even_().store() }
                #[inline(always)]
                fn is_nan_(self) -> MaskStorage<Self::Mask> { MaskStorage::store_mask(self.load().is_nan_()) }
                #[inline(always)]
                fn mul_add_(a: Self, b: Self, c: Self) -> Self { ArithPrimitive::mul_add_(a.load(), b.load(), c.load()).store() }
                #[inline(always)]
                fn mul_sub_(a: Self, b: Self, c: Self) -> Self { ArithPrimitive::mul_sub_(a.load(), b.load(), c.load()).store() }
                #[inline(always)]
                fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self { ArithPrimitive::neg_mul_add_(a.load(), b.load(), c.load()).store() }

                #[inline(always)]
                fn bitand_(self, rhs: Self) -> Self { self.load().bitand_(rhs.load()).store() }
                #[inline(always)]
                fn bitor_(self, rhs: Self) -> Self { self.load().bitor_(rhs.load()).store() }
                #[inline(always)]
                fn bitxor_(self, rhs: Self) -> Self { self.load().bitxor_(rhs.load()).store() }
                #[inline(always)]
                fn not_(self) -> Self { self.load().not_().store() }
                #[inline(always)]
                fn shl_noexcept_(self, rhs: Self) -> Self { self.load().shl_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn shr_noexcept_(self, rhs: Self) -> Self { self.load().shr_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.load().shl_scalar_noexcept_(rhs.load()).store() }
                #[inline(always)]
                fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.load().shr_scalar_noexcept_(rhs.load()).store() }
            }
        };
    }

    impl_arith_primitive!(f32, f32x2);
    impl_arith_primitive!(i32, i32x2);
    impl_arith_primitive!(u32, u32x2);

    // TODO(module-naming): what follows has nothing to do with 64-bit types; rename the
    // module around it.
    macro_rules! impl_load {
        ($($t:ty),*) => {
            $(
                impl Load for $t {
                    type Output = Self;
                    #[inline(always)]
                    fn load(self) -> Self::Output { self }
                }
            )*
        };
    }
    impl_load!(f32, i32, u32, f32x4, i32x4, u32x4, MaskStorage<i32>, MaskStorage<i32x4>);
    impl_load!(f64, i64, u64, f64x4, i64x4, u64x4, MaskStorage<i64>, MaskStorage<i64x4>);
    impl_load!(f64x2, i64x2, u64x2, MaskStorage<i64x2>);

    impl Load for f32x2 {
        type Output = f32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { f32x4::new([self.0[0], self.0[1], 0., 0.]) }
    }
    impl Load for i32x2 {
        type Output = i32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { crate::utils::MaskLoad::__load(self) }
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
        fn store(self) -> i32x2 { crate::utils::MaskLoad::__store(self) }
    }
    impl Store<u32x2> for u32x4 {
        #[inline(always)]
        fn store(self) -> u32x2 {
            let [a, b, ..] = self.to_array();
            u32x2([a, b])
        }
    }
    impl<T: Load<Output = T>, const N: usize> Load for [T; N] {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }

    use crate::utils::ArithPrimitive;
    pub(crate) use wide::{f32x4 as compute_f32x2, i32x4 as compute_i32x2, u32x4 as compute_u32x2};
}

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
mod _64bit_types {
    use crate::{
        simd::kernels,
        utils::{ArithPrimitive, MaskPrimitive, MaskStorage},
    };
    use core::arch::aarch64::*;
    use wide::{f64x2, i64x2, u64x2};

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
        pub(crate) fn cast_unsigned(self) -> u32x2 {
            unsafe { u32x2(vreinterpret_u32_s32(self.0)) }
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
        #[inline(always)]
        pub(crate) fn cast_signed(self) -> i32x2 { unsafe { i32x2(vreinterpret_s32_u32(self.0)) } }
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

    // SAFETY: validation and the canonical operations act lane-wise, so a canonical input cannot
    // produce a mixed lane. With a canonical selector, `vbsl_s32` copies each complete physical
    // lane from one of the canonical inputs.
    unsafe impl MaskPrimitive for i32x2 {
        fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
        #[inline(always)]
        fn canonical_not(self) -> Self { unsafe { Self(vmvn_s32(self.0)) } }
        #[inline(always)]
        fn canonical_bitand(self, rhs: Self) -> Self { unsafe { Self(vand_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn canonical_bitor(self, rhs: Self) -> Self { unsafe { Self(vorr_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn canonical_bitxor(self, rhs: Self) -> Self { unsafe { Self(veor_s32(self.0, rhs.0)) } }
        #[inline(always)]
        fn canonical_select(self, true_values: Self, false_values: Self) -> Self {
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
    // Unlike x86's eight-byte pair, this is a NEON register that the comparison and bit
    // instructions take as they are, so storage and primitive are the same type.
    crate::utils::impl_mask_load!(i32x2);

    impl ArithPrimitive for f32x2 {
        type Scalar = f32;
        type F32 = f32x2;
        type F64 = f64x2;
        type I32 = i32x2;
        type I64 = i64x2;
        type U32 = u32x2;
        type U64 = u64x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0.; 2]);
        const ONE_: Self = Self::new([1.; 2]);
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
        fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self { a }
        #[inline(always)]
        fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self { kernels::cast::f32x2_from_f64(a) }
        #[inline(always)]
        fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self { kernels::cast::f32x2_from_i32(a) }
        #[inline(always)]
        fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self { kernels::cast::f32x2_from_i64(a) }
        #[inline(always)]
        fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self { kernels::cast::f32x2_from_u32(a) }
        #[inline(always)]
        fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self { kernels::cast::f32x2_from_u64(a) }
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
        fn div_(self, rhs: Self) -> Self { unsafe { Self(vdiv_f32(self.0, rhs.0)) } }
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
        fn sqrt_(self) -> Self { unsafe { Self(vsqrt_f32(self.0)) } }
        #[inline(always)]
        fn floor_(self) -> Self { unsafe { Self(vrndm_f32(self.0)) } }
        #[inline(always)]
        fn ceil_(self) -> Self { unsafe { Self(vrndp_f32(self.0)) } }
        #[inline(always)]
        fn round_(self) -> Self { unsafe { Self(vrnda_f32(self.0)) } }
        #[inline(always)]
        fn trunc_(self) -> Self { unsafe { Self(vrnd_f32(self.0)) } }
        #[inline(always)]
        fn fract_(self) -> Self { unsafe { Self(vsub_f32(self.0, vrnd_f32(self.0))) } }
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
        type F64 = f64x2;
        type I32 = i32x2;
        type I64 = i64x2;
        type U32 = u32x2;
        type U64 = u64x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0; 2]);
        const ONE_: Self = Self::new([1; 2]);
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
        fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self { kernels::cast::i32x2_from_f32(a) }
        #[inline(always)]
        fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self { kernels::cast::i32x2_from_f64(a) }
        #[inline(always)]
        fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self { a }
        #[inline(always)]
        fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self { kernels::cast::i32x2_from_i64(a) }
        #[inline(always)]
        fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self { kernels::cast::i32x2_from_u32(a) }
        #[inline(always)]
        fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self { kernels::cast::i32x2_from_u64(a) }
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
        fn ne_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_s32` produces an all-zero or all-one bit pattern in every lane,
            // `vmvn_u32` complements it (still all-zero or all-one), and `vreinterpret_s32_u32`
            // preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vmvn_u32(vceq_s32(
                    self.0, other.0,
                )))))
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
        fn bitand_(self, rhs: Self) -> Self { core::ops::BitAnd::bitand(self, rhs) }
        #[inline(always)]
        fn bitor_(self, rhs: Self) -> Self { core::ops::BitOr::bitor(self, rhs) }
        #[inline(always)]
        fn bitxor_(self, rhs: Self) -> Self { core::ops::BitXor::bitxor(self, rhs) }
        #[inline(always)]
        fn not_(self) -> Self { core::ops::Not::not(self) }
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
        type F64 = f64x2;
        type I32 = i32x2;
        type I64 = i64x2;
        type U32 = u32x2;
        type U64 = u64x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0; 2]);
        const ONE_: Self = Self::new([1; 2]);
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
        fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self { kernels::cast::u32x2_from_f32(a) }
        #[inline(always)]
        fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self { kernels::cast::u32x2_from_f64(a) }
        #[inline(always)]
        fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self { kernels::cast::u32x2_from_i32(a) }
        #[inline(always)]
        fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self { kernels::cast::u32x2_from_i64(a) }
        #[inline(always)]
        fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self { a }
        #[inline(always)]
        fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self { kernels::cast::u32x2_from_u64(a) }
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
        fn ne_(self, other: Self) -> MaskStorage<Self::Mask> {
            // SAFETY: `vceq_u32` produces an all-zero or all-one bit pattern in every lane,
            // `vmvn_u32` complements it (still all-zero or all-one), and `vreinterpret_s32_u32`
            // preserves those bits.
            unsafe {
                MaskStorage::new_unchecked(i32x2(vreinterpret_s32_u32(vmvn_u32(vceq_u32(
                    self.0, other.0,
                )))))
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
        fn bitand_(self, rhs: Self) -> Self { core::ops::BitAnd::bitand(self, rhs) }
        #[inline(always)]
        fn bitor_(self, rhs: Self) -> Self { core::ops::BitOr::bitor(self, rhs) }
        #[inline(always)]
        fn bitxor_(self, rhs: Self) -> Self { core::ops::BitXor::bitxor(self, rhs) }
        #[inline(always)]
        fn not_(self) -> Self { core::ops::Not::not(self) }
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
                #[inline(always)]
                fn $f_trait(self, rhs: Self) -> Self::Output { crate::utils::ArithPrimitive::$f(self, rhs) }
            })+
        };
    }
    impl_ops!(impl Add for [f32x2, i32x2, u32x2] { add => add_noexcept_ });
    impl_ops!(impl Sub for [f32x2, i32x2, u32x2] { sub => sub_noexcept_ });
    impl_ops!(impl Mul for [f32x2, i32x2, u32x2] { mul => mul_noexcept_ });
    impl core::ops::Div for f32x2 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output { unsafe { Self(vdiv_f32(self.0, rhs.0)) } }
    }
    impl core::ops::BitAnd for i32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self::Output { unsafe { Self(vand_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitAnd for u32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self::Output { unsafe { Self(vand_u32(self.0, rhs.0)) } }
    }
    impl core::ops::BitOr for i32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self::Output { unsafe { Self(vorr_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitOr for u32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self::Output { unsafe { Self(vorr_u32(self.0, rhs.0)) } }
    }
    impl core::ops::BitXor for i32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self::Output { unsafe { Self(veor_s32(self.0, rhs.0)) } }
    }
    impl core::ops::BitXor for u32x2 {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self::Output { unsafe { Self(veor_u32(self.0, rhs.0)) } }
    }
    impl core::ops::Not for i32x2 {
        type Output = Self;
        #[inline(always)]
        fn not(self) -> Self::Output { unsafe { Self(vmvn_s32(self.0)) } }
    }
    impl core::ops::Not for u32x2 {
        type Output = Self;
        #[inline(always)]
        fn not(self) -> Self::Output { unsafe { Self(vmvn_u32(self.0)) } }
    }
    impl MaskStorage<i32x2> {
        #[inline(always)]
        pub(crate) fn unpack(self) -> Self { self }
    }

    pub(crate) use f32x2 as compute_f32x2;
    pub(crate) use i32x2 as compute_i32x2;
    pub(crate) use u32x2 as compute_u32x2;
}

use crate::{
    simd::kernels,
    utils::{ArithPrimitive, MaskPrimitive},
};
#[allow(unused_imports)]
pub(crate) use _64bit_types::{compute_f32x2, compute_i32x2, compute_u32x2, f32x2, i32x2, u32x2};

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

/// Wasm's fused multiply-add, which `wide` does not reach for: it fuses on x86 with FMA and
/// on aarch64 NEON, and multiplies and adds separately everywhere else.
///
/// The names follow the operand type so that a caller holding `$float` can paste one
/// together. Two of the three widths are one `core::arch::wasm32` instruction each; the
/// four-lane `f64` is two registers here, so it is the one that needs writing out.
///
/// These instructions are *relaxed*: the runtime chooses whether to fuse the multiply and
/// the add, so one binary can round differently on two runtimes. That is a weaker promise
/// than the crate makes elsewhere, and it is what asking for `relaxed-simd` means.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
mod wasm_fma {
    use super::swizzle;
    pub(crate) use core::arch::wasm32::{
        f32x4_relaxed_madd,
        f32x4_relaxed_nmadd,
        f64x2_relaxed_madd,
        f64x2_relaxed_nmadd,
    };
    use wide::{bytemuck::cast, f64x2, f64x4};

    #[inline(always)]
    pub(crate) fn f64x4_relaxed_madd(a: f64x4, b: f64x4, c: f64x4) -> f64x4 {
        let [a_lo, a_hi] = cast::<f64x4, [f64x2; 2]>(a);
        let [b_lo, b_hi] = cast::<f64x4, [f64x2; 2]>(b);
        let [c_lo, c_hi] = cast::<f64x4, [f64x2; 2]>(c);
        let lo = f64x2::from(f64x2_relaxed_madd(a_lo.into(), b_lo.into(), c_lo.into()));
        let hi = f64x2::from(f64x2_relaxed_madd(a_hi.into(), b_hi.into(), c_hi.into()));
        swizzle!(lo, hi, @concat)
    }
    #[inline(always)]
    pub(crate) fn f64x4_relaxed_nmadd(a: f64x4, b: f64x4, c: f64x4) -> f64x4 {
        let [a_lo, a_hi] = cast::<f64x4, [f64x2; 2]>(a);
        let [b_lo, b_hi] = cast::<f64x4, [f64x2; 2]>(b);
        let [c_lo, c_hi] = cast::<f64x4, [f64x2; 2]>(c);
        let lo = f64x2::from(f64x2_relaxed_nmadd(a_lo.into(), b_lo.into(), c_lo.into()));
        let hi = f64x2::from(f64x2_relaxed_nmadd(a_hi.into(), b_hi.into(), c_hi.into()));
        swizzle!(lo, hi, @concat)
    }
}

macro_rules! impl_arith_primitive {
    ($self_ty:ident, scalar=$scalar:ident, mask=$mask:ident, [$f32:ident, $f64:ident, $i32:ident, $i64:ident, $u32:ident, $u64:ident] $(, $N:ident)? { $($item:item)* }) => {
        impl ArithPrimitive for $self_ty {
            type Scalar = $scalar;
            type F32 = $f32;
            type F64 = $f64;
            type I32 = $i32;
            type I64 = $i64;
            type U32 = $u32;
            type U64 = $u64;
            type Mask = $mask;
            const ZERO_: Self = Self::ZERO;
            const ONE_: Self = Self::ONE;
            #[inline(always)]
            fn filled_(a: Self::Scalar) -> Self { Self::splat(a) }
            #[inline(always)]
            fn as_array_(&self) -> &[Self::Scalar] { self.as_array() }
            #[inline(always)]
            fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { self.as_mut_array() }
            #[inline(always)]
            fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_f32>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_f64>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_i32>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_i64>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_u32>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self {
                paste::paste!(kernels::cast::[<$self_ty _from_u64>] $(::<$N>)? (a))
            }
            #[inline(always)]
            fn max_(self, other: Self) -> Self { self.max(other) }
            #[inline(always)]
            fn min_(self, other: Self) -> Self { self.min(other) }
            #[inline(always)]
            fn add_noexcept_(self, rhs: Self) -> Self { core::ops::Add::add(self, rhs) }
            #[inline(always)]
            fn sub_noexcept_(self, rhs: Self) -> Self { core::ops::Sub::sub(self, rhs) }
            #[inline(always)]
            fn mul_noexcept_(self, rhs: Self) -> Self { core::ops::Mul::mul(self, rhs) }
            $($item)*
        }
    };
}
macro_rules! impl_arith_primitive_int {
    ($self_ty:ident, scalar=$scalar:ident, mask=$int:ident, [$($t:ident),+] $(, $N:ident)? { $($item:item)* }) => {
        impl_arith_primitive! {
            $self_ty, scalar=$scalar, mask=$int, [$($t),+] $(, $N)? {
                #[inline(always)]
                fn bitand_(self, rhs: Self) -> Self { core::ops::BitAnd::bitand(self, rhs) }
                #[inline(always)]
                fn bitor_(self, rhs: Self) -> Self { core::ops::BitOr::bitor(self, rhs) }
                #[inline(always)]
                fn bitxor_(self, rhs: Self) -> Self { core::ops::BitXor::bitxor(self, rhs) }
                #[inline(always)]
                fn not_(self) -> Self { core::ops::Not::not(self) }
                #[inline(always)]
                fn shl_noexcept_(self, rhs: Self) -> Self { self << rhs }
                #[inline(always)]
                fn shr_noexcept_(self, rhs: Self) -> Self { self >> rhs }
                #[inline(always)]
                fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self << rhs }
                #[inline(always)]
                fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self >> rhs }
                #[inline(always)]
                fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { !self.eq_(other) }
                #[inline(always)]
                fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { !self.lt_(other) }
                #[inline(always)]
                fn le_(self, other: Self) -> MaskStorage<Self::Mask> { !self.gt_(other) }
                $($item)*
            }
        }
    }
}
macro_rules! impl_arith_primitive_all {
    ($float_scalar:ident: $float:ident, $int_scalar:ident: $int:ident, $uint_scalar:ident: $uint:ident, [$($t:ident),+] $(, $N:ident)?) => {
        impl_arith_primitive! {
            $float, scalar=$float_scalar, mask=$int, [$($t),+] $(, $N)? {
                #[inline(always)]
                fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_eq` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_eq(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn ne_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_ne` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_ne(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_gt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_gt(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_lt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_lt(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn ge_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_ge` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_ge(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn le_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_le` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.simd_le(other).to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
                    Self::from_bits(mask.into_inner().cast_unsigned()).select(true_values, false_values)
                }

                #[inline(always)]
                fn div_(self, rhs: Self) -> Self { core::ops::Div::div(self, rhs) }
                #[inline(always)]
                fn clamp_noexcept_(mut self, min: Self, max: Self) -> Self {
                    self = self.simd_lt(min).select(min, self);
                    self = self.simd_gt(max).select(max, self);
                    self
                }
                #[inline(always)]
                fn neg_noexcept_(self) -> Self { core::ops::Neg::neg(self) }
                #[inline(always)]
                fn abs_noexcept_(self) -> Self { self.abs() }
                #[inline(always)]
                fn signum_(self) -> Self { self.signum() }
                #[inline(always)]
                fn round_ties_even_(self) -> Self { paste::paste!(kernels::round::[<$float _round_ties_even>](self)) }
                #[inline(always)]
                fn sqrt_(self) -> Self { self.sqrt() }
                #[inline(always)]
                fn floor_(self) -> Self { self.floor() }
                #[inline(always)]
                fn ceil_(self) -> Self { self.ceil() }
                #[inline(always)]
                fn round_(self) -> Self { self.round() }
                #[inline(always)]
                fn trunc_(self) -> Self { self.trunc() }
                #[inline(always)]
                fn fract_(self) -> Self { self.fract() }
                #[inline(always)]
                fn is_nan_(self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `is_nan` produces an all-zero or all-one bit
                        // pattern in every lane. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.is_nan().to_bits().cast_signed())
                    }
                }
                // `wide` fuses these on x86 with FMA and on aarch64 NEON, and multiplies and adds
                // separately otherwise; on Wasm the relaxed instructions do it in one.
                #[inline(always)]
                fn mul_add_(a: Self, b: Self, c: Self) -> Self {
                    cfg_select! {
                        all(target_arch = "wasm32", target_feature = "relaxed-simd") => {
                            paste::paste!(wasm_fma::[<$float _relaxed_madd>](a.into(), b.into(), c.into())).into()
                        }
                        _ => a.mul_add(b, c),
                    }
                }
                #[inline(always)]
                fn mul_sub_(a: Self, b: Self, c: Self) -> Self {
                    cfg_select! {
                        all(target_arch = "wasm32", target_feature = "relaxed-simd") => {
                            paste::paste!(wasm_fma::[<$float _relaxed_madd>](a.into(), b.into(), (-c).into())).into()
                        }
                        _ => a.mul_sub(b, c),
                    }
                }
                #[inline(always)]
                fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self {
                    cfg_select! {
                        all(target_arch = "wasm32", target_feature = "relaxed-simd") => {
                            paste::paste!(wasm_fma::[<$float _relaxed_nmadd>](a.into(), b.into(), c.into())).into()
                        }
                        _ => a.mul_neg_add(b, c),
                    }
                }
            }
        }
        impl_arith_primitive_int! {
            $int, scalar=$int_scalar, mask=$int, [$($t),+] $(, $N)? {
                #[inline(always)]
                fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_eq` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_eq(other))
                    }
                }
                #[inline(always)]
                fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_gt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_gt(other))
                    }
                }
                #[inline(always)]
                fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_lt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_lt(other))
                    }
                }
                #[inline(always)]
                fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
                    mask.into_inner().select(true_values, false_values)
                }

                #[inline(always)]
                fn neg_noexcept_(self) -> Self { core::ops::Neg::neg(self) }
                #[inline(always)]
                fn abs_noexcept_(self) -> Self { self.abs() }
                #[inline(always)]
                fn signum_(self) -> Self {
                    // TODO(vector-extra-operations): implement SIMD signum or hide public signum APIs.
                    todo!()
                }
            }
        }
        impl_arith_primitive_int! {
            $uint, scalar=$uint_scalar, mask=$int, [$($t),+] $(, $N)? {
                #[inline(always)]
                fn eq_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_eq` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_eq(other).cast_signed())
                    }
                }
                #[inline(always)]
                fn gt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_gt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_gt(other).cast_signed())
                    }
                }
                #[inline(always)]
                fn lt_(self, other: Self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `simd_lt` produces an all-zero or all-one bit pattern in every
                        // lane, whatever the element type.
                        MaskStorage::new_unchecked(self.simd_lt(other).cast_signed())
                    }
                }
                #[inline(always)]
                fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
                    mask.into_inner().cast_unsigned().select(true_values, false_values)
                }
            }
        }
    }
}

impl_arith_primitive_all!(f32:f32x4, i32:i32x4, u32:u32x4, [f32x4, f64x4, i32x4, i64x4, u32x4, u64x4], N);
impl_arith_primitive_all!(f64:f64x4, i64:i64x4, u64:u64x4, [f32x4, f64x4, i32x4, i64x4, u32x4, u64x4], N);
impl_arith_primitive_all!(f64:f64x2, i64:i64x2, u64:u64x2, [f32x2, f64x2, i32x2, i64x2, u32x2, u64x2]);

// A vector mask is stored and operated on at the same width, so `MaskLoad` is the identity for
// every one of these. The two-lane x86 types are the exception, and implement it themselves.
crate::utils::impl_mask_load!(i32x4, i64x2, i64x4);

// SAFETY: validation and the relevant `ArithPrimitive` operations act lane-wise. Selection copies
// each complete physical lane from one of the canonical inputs.
unsafe impl MaskPrimitive for i32x4 {
    fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
    #[inline(always)]
    fn canonical_not(self) -> Self { !self }
    #[inline(always)]
    fn canonical_bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn canonical_bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn canonical_bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn canonical_select(self, true_values: Self, false_values: Self) -> Self {
        i32x4::select(self, true_values, false_values)
    }
    #[inline(always)]
    fn any<const N: usize>(self) -> bool {
        std::assert_matches!(N, 2..=4);
        if N == 4 {
            self.any()
        } else if N == 3 {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                    use core::arch::aarch64::*;
                    let clear_padding_lane: int32x4_t = core::mem::transmute([-1, -1, -1, 0i32]);
                    let masked = vandq_s32(self.into(), clear_padding_lane);
                    vminvq_s32(masked) < 0
                },
                _ => self.to_bitmask() & 0b0111 != 0,
            }
        } else if N == 2 {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => self.xy().any::<2>(),
                _ => self.to_bitmask() & 0b0011 != 0,
            }
        } else {
            unreachable!()
        }
    }
    #[inline(always)]
    fn all<const N: usize>(self) -> bool {
        std::assert_matches!(N, 2..=4);
        if N == 4 {
            self.all()
        } else if N == 3 {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                    use core::arch::aarch64::*;
                    let set_padding_lane: int32x4_t = core::mem::transmute([0, 0, 0, -1i32]);
                    let masked = vorrq_s32(self.into(), set_padding_lane);
                    vmaxvq_s32(masked) < 0
                },
                _ => self.to_bitmask() & 0b0111 == 0b0111,
            }
        } else if N == 2 {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => self.xy().all::<2>(),
                _ => self.to_bitmask() & 0b0011 == 0b0011,
            }
        } else {
            unreachable!()
        }
    }
}
// SAFETY: see `MaskPrimitive for i64x4`; a two-lane register has no padding to mask out.
unsafe impl MaskPrimitive for i64x2 {
    fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
    #[inline(always)]
    fn canonical_not(self) -> Self { !self }
    #[inline(always)]
    fn canonical_bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn canonical_bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn canonical_bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn canonical_select(self, true_values: Self, false_values: Self) -> Self {
        i64x2::select(self, true_values, false_values)
    }
    #[inline(always)]
    fn any<const N: usize>(self) -> bool {
        assert_eq!(N, 2);
        cfg_select! {
            // NEON has no bitmask instruction. A canonical 64-bit lane is all-zero or all-one, so
            // it stays canonical read as two 32-bit lanes -- a width NEON does reduce
            // horizontally, after which the same "least lane is negative" test as the 32-bit
            // types applies. `core::simd` lowers `Mask<i64, 2>::any` the same way; `wide`'s own
            // reduction folds the two lanes together instead and costs one instruction more.
            all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                use core::arch::aarch64::*;
                vminvq_s32(vreinterpretq_s32_s64(self.into())) < 0
            },
            _ => self.any(),
        }
    }
    #[inline(always)]
    fn all<const N: usize>(self) -> bool {
        assert_eq!(N, 2);
        cfg_select! {
            // See `any`.
            all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                use core::arch::aarch64::*;
                vmaxvq_s32(vreinterpretq_s32_s64(self.into())) < 0
            },
            _ => self.all(),
        }
    }
}
// SAFETY: validation and the relevant `ArithPrimitive` operations act lane-wise. Selection copies
// each complete physical lane from one of the canonical inputs.
unsafe impl MaskPrimitive for i64x4 {
    fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
    #[inline(always)]
    fn canonical_not(self) -> Self { !self }
    #[inline(always)]
    fn canonical_bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn canonical_bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn canonical_bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn canonical_select(self, true_values: Self, false_values: Self) -> Self {
        i64x4::select(self, true_values, false_values)
    }
    #[inline(always)]
    fn any<const N: usize>(self) -> bool {
        // Two lanes reach this type even though `i64x2` exists: a shape of `SealedElement<2, 3>`,
        // which is a 3x2 row-major or 2x3 column-major matrix, packs its three units of two lanes
        // into two four-lane ones, and the second of those has only two live lanes.
        std::assert_matches!(N, 2..=4);
        // AVX2 has a bitmask instruction spanning all four lanes, so the lanes in use are picked
        // out of its result. No other target has one: there a four-lane 64-bit value is a pair of
        // two-lane registers, so the pair is folded into a single register and handed to the
        // two-lane reduction, or the high register is dropped when it holds nothing but padding.
        if N == 4 {
            cfg_select! {
                target_feature = "avx2" => self.any(),
                _ => {
                    // SAFETY: without a 256-bit register `wide::i64x4` is `#[repr(C)] { a: i64x2,
                    // b: i64x2 }`, which has the same layout as `[i64x2; 2]`.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::any::<2>(low | high)
                }
            }
        } else if N == 3 {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0111 != 0,
                // Lane 3 is padding; clearing it stops it from making the answer true.
                _ => {
                    // SAFETY: see the four-lane branch.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::any::<2>(low | (high & i64x2::new([-1, 0])))
                }
            }
        } else {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0011 != 0,
                _ => {
                    // SAFETY: see the four-lane branch.
                    let [low, _]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::any::<2>(low)
                }
            }
        }
    }
    #[inline(always)]
    fn all<const N: usize>(self) -> bool {
        std::assert_matches!(N, 2..=4);
        // See `any` for how the two target families differ, and for why two lanes arrive here.
        if N == 4 {
            cfg_select! {
                target_feature = "avx2" => self.all(),
                _ => {
                    // SAFETY: see `any`.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::all::<2>(low & high)
                }
            }
        } else if N == 3 {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0111 == 0b0111,
                // Lane 3 is padding; filling it stops it from making the answer false.
                _ => {
                    // SAFETY: see `any`.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::all::<2>(low & (high | i64x2::new([0, -1])))
                }
            }
        } else {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0011 == 0b0011,
                _ => {
                    // SAFETY: see `any`.
                    let [low, _]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::all::<2>(low)
                }
            }
        }
    }
}
impl MaskStorage<i32x4> {
    #[inline(always)]
    pub(crate) fn unpack(self) -> Self { self }
}
impl MaskStorage<i64x2> {
    #[inline(always)]
    pub(crate) fn unpack(self) -> Self { self }
}
impl MaskStorage<i64x4> {
    #[inline(always)]
    pub(crate) fn unpack(self) -> Self { self }
}

// One impl per index list, and nothing else.
//
// The index list has to be enumerated: `_mm_shuffle_ps` and its counterparts take their control
// byte as a const-generic argument computed from the indices, and stable Rust cannot pass a
// computed const-generic argument, so the indices must still be literals where `swizzle!` is
// expanded. The element type and the source width are a different matter — `dispatch` mentions
// neither, it loads whatever compute vector the storage holds and stores the result back — so they
// are generic here. That is 336 impls rather than 336 x 3 widths x 6 element types.
//
// The bounds are the ones the body needs and they are discharged where `SealedElement::swizzle2`,
// `swizzle3` and `swizzle4` call this, with the element type and the source width both concrete.
// None of them reaches the element trait itself, which is what keeps the SIMD storage traits out
// of the backend-independent API.
//
// An index list that names a lane a source of width `M` does not have needs no special case: this
// impl is only instantiated when it is called, and `build.rs` pairs each accessor with the smallest
// `M` that has every lane it reads, so such a call is never generated. The previous shape needed
// `unimplemented!()` bodies here because it named `M` in the impl header and so had to write one
// out for every combination whether it could be called or not.
macro_rules! impl_swizzle_dispatch {
    (Indices2[$i0:tt, $i1:tt]) => {
        impl_swizzle_dispatch!(@impl 2, Indices2[$i0, $i1], Vector2, [$i0, $i1]);
    };
    (Indices3[$i0:tt, $i1:tt, $i2:tt]) => {
        impl_swizzle_dispatch!(@impl 3, Indices3[$i0, $i1, $i2], Vector4, [$i0, $i1, $i2]);
    };
    (Indices4[$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        impl_swizzle_dispatch!(@impl 4, Indices4[$i0, $i1, $i2, $i3], Vector4, [$i0, $i1, $i2, $i3]);
    };
    (@impl $n:tt, $kind:ident[$($parameter:tt),+], $result:ident, [$($index:tt),+]) => {
        impl<T, const M: usize> private::SwizzleDispatch<T, M, $n> for private::$kind<$($parameter),+>
        where
            T: private::SealedElement<M, 1> + private::SealedElement<$n, 1>,
            <T as private::SealedElement<M, 1>>::Storage: Load,
            <<T as private::SealedElement<M, 1>>::Storage as Load>::Output: Swizzle,
            <<<T as private::SealedElement<M, 1>>::Storage as Load>::Output as ComputeVector>::$result:
                Store<<T as private::SealedElement<$n, 1>>::Storage>,
        {
            #[inline(always)]
            fn dispatch(
                v: <T as private::SealedElement<M, 1>>::Storage,
            ) -> <T as private::SealedElement<$n, 1>>::Storage {
                swizzle!(v.load(), [$($index),+]).store()
            }
        }
    };
}

macro_rules! impl_swizzle2_for_i0 {
    ($i0:tt; $($i1:tt),*) => {$(impl_swizzle_dispatch!(Indices2[$i0, $i1]);)*};
}
macro_rules! impl_swizzle3_for_i0_i1 {
    ($i0:tt, $i1:tt; $($i2:tt),*) => {$(impl_swizzle_dispatch!(Indices3[$i0, $i1, $i2]);)*};
}
macro_rules! impl_swizzle3_for_i0 {
    ($i0:tt; $($i1:tt),*) => {
        $(impl_swizzle3_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0_i1_i2 {
    ($i0:tt, $i1:tt, $i2:tt; $($i3:tt),*) => {
        $(impl_swizzle_dispatch!(Indices4[$i0, $i1, $i2, $i3]);)*
    };
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
