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
pub(crate) use {sign, validate_lane4, validate_lane8};

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
        type F64 = f64x2;
        type I32 = i32x2;
        type I64 = i64x2;
        type U32 = u32x2;
        type U64 = u64x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0.; 2]);
        const ONE_: Self = Self::new([1.; 2]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }

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
    }
    impl crate::utils::ArithPrimitive for i32x2 {
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
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }

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
    }
    impl crate::utils::ArithPrimitive for u32x2 {
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
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }

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
    }

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
    impl<T: MaskPrimitive, const N: usize> Load for MaskStorage<[T; N]> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl<T: Load<Output = T>, const N: usize> Load for [T; N] {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }

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

impl<const N: usize> Store<MaskStorage<[i32x4; N]>> for [MaskStorage<i32x4>; N] {
    #[inline(always)]
    fn store(self) -> MaskStorage<[i32x4; N]> { self.into() }
}
impl<const N: usize> Store<MaskStorage<[i64x4; N]>> for [MaskStorage<i64x4>; N] {
    #[inline(always)]
    fn store(self) -> MaskStorage<[i64x4; N]> { self.into() }
}
impl<const N: usize> Store<MaskStorage<[i64x2; N]>> for [MaskStorage<i64x2>; N] {
    #[inline(always)]
    fn store(self) -> MaskStorage<[i64x2; N]> { self.into() }
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
                fn shl_noexcept_(self, rhs: Self) -> Self { self << rhs }
                #[inline(always)]
                fn shr_noexcept_(self, rhs: Self) -> Self { self >> rhs }
                #[inline(always)]
                fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self << rhs }
                #[inline(always)]
                fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self >> rhs }
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
                fn is_nan_(self) -> MaskStorage<Self::Mask> {
                    unsafe {
                        // SAFETY: `is_nan` produces an all-zero or all-one bit
                        // pattern in every lane. `to_bits` and `cast_signed` preserve those bits.
                        MaskStorage::new_unchecked(self.is_nan().to_bits().cast_signed())
                    }
                }
                #[inline(always)]
                fn mul_add_(a: Self, b: Self, c: Self) -> Self { a.mul_add(b, c) }
                #[inline(always)]
                fn mul_sub_(a: Self, b: Self, c: Self) -> Self { a.mul_sub(b, c) }
                #[inline(always)]
                fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self { a.mul_neg_add(b, c) }
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

// SAFETY: validation and `!` operate lane-wise. With a canonical selector,
// `select` copies each complete physical lane from one of the canonical inputs.
unsafe impl MaskPrimitive for i32x4 {
    fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
    #[inline(always)]
    fn not(self) -> Self { !self }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn select(self, true_values: Self, false_values: Self) -> Self {
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
    fn not(self) -> Self { !self }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn select(self, true_values: Self, false_values: Self) -> Self {
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
// SAFETY: validation and `not` operate lane-wise. With a canonical selector,
// `select` copies each complete physical lane from one of the canonical inputs.
unsafe impl MaskPrimitive for i64x4 {
    fn is_valid(self) -> bool { self.to_array().into_iter().all(MaskPrimitive::is_valid) }
    #[inline(always)]
    fn not(self) -> Self { !self }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self { self & rhs }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self { self | rhs }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self { self ^ rhs }
    #[inline(always)]
    fn select(self, true_values: Self, false_values: Self) -> Self {
        i64x4::select(self, true_values, false_values)
    }
    #[inline(always)]
    fn any<const N: usize>(self) -> bool {
        // Only three- and four-lane masks reach this type. The 32-bit types double as their own
        // two-lane compute vector everywhere but NEON, so `i32x4` also serves `N == 2`; `i64x2`
        // and `i64x4` are separate types on every target, so a two-lane 64-bit mask is stored in
        // the former and never here.
        std::assert_matches!(N, 3..=4);
        // AVX2 has a bitmask instruction spanning all four lanes, so the lanes in use are picked
        // out of its result. No other target has one: there a four-lane 64-bit value is a pair of
        // two-lane registers, so the pair is folded into a single register and handed to the
        // two-lane reduction.
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
        } else {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0111 != 0,
                // Lane 3 is padding; clearing it stops it from making the answer true.
                _ => {
                    // SAFETY: see the four-lane branch.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::any::<2>(low | (high & i64x2::new([-1, 0])))
                }
            }
        }
    }
    #[inline(always)]
    fn all<const N: usize>(self) -> bool {
        std::assert_matches!(N, 3..=4);
        // See `any` for how the two target families differ.
        if N == 4 {
            cfg_select! {
                target_feature = "avx2" => self.all(),
                _ => {
                    // SAFETY: see `any`.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::all::<2>(low & high)
                }
            }
        } else {
            cfg_select! {
                target_feature = "avx2" => self.to_bitmask() & 0b0111 == 0b0111,
                // Lane 3 is padding; filling it stops it from making the answer false.
                _ => {
                    // SAFETY: see `any`.
                    let [low, high]: [i64x2; 2] = unsafe { core::mem::transmute(self) };
                    MaskPrimitive::all::<2>(low & (high | i64x2::new([0, -1])))
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
        impl_swizzle_dispatch_one!(f64, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(i32, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(i64, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(u32, $m, $n, $kind[$($i),*]);
        impl_swizzle_dispatch_one!(u64, $m, $n, $kind[$($i),*]);
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
