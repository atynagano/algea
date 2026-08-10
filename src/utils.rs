// TODO(codegen-optimization): Compare balanced arithmetic trees on representative targets without
// FMA, and change the lowering only when codegen or benchmarks show a consistent improvement.
#[rustfmt::skip]
macro_rules! arith {
    // TODO(api-cleanup): fix macro parsing when higher-precedence operators appear on the right.
    ($a:tt + $b:tt * $c:tt) => { $crate::utils::ArithPrimitive::mul_add_($b, $c, $a) };
    ($a:tt - $b:tt * $c:tt) => { $crate::utils::ArithPrimitive::neg_mul_add_($b, $c, $a) };
    ($a:tt * $b:tt + $c:tt) => { $crate::utils::ArithPrimitive::mul_add_($a, $b, $c) };
    ($a:tt * $b:tt - $c:tt) => { $crate::utils::ArithPrimitive::mul_sub_($a, $b, $c) };
    ($a:tt * $b:tt + $($t:tt)+) => { arith!(($crate::utils::ArithPrimitive::mul_noexcept_($a, $b)) + $($t)+) };
    ($a:tt * $b:tt - $($t:tt)+) => { arith!(($crate::utils::ArithPrimitive::mul_noexcept_($a, $b)) - $($t)+) };
    ($a:tt + $b:tt * $c:tt + $($t:tt)+) => { arith!((arith!($a + $b * $c)) + $($t)+) };
    ($a:tt + $b:tt * $c:tt - $($t:tt)+) => { arith!((arith!($a + $b * $c)) - $($t)+) };
    ($a:tt - $b:tt * $c:tt + $($t:tt)+) => { arith!((arith!($a - $b * $c)) + $($t)+) };
    ($a:tt - $b:tt * $c:tt - $($t:tt)+) => { arith!((arith!($a - $b * $c)) - $($t)+) };
    // ($a:tt + $b:tt) => { $a + $b };
    // ($a:tt - $b:tt) => { $a - $b };
    // ($a:tt * $b:tt) => { $a * $b };
    // ($a:tt) => { $a };
}

macro_rules! if_ {
    (1 == 1 and 1 != 1 { $($then:tt)* }) => { };
    (1 == 1 and $_:tt != 1 { $($then:tt)* }) => { $($then)* };
    (1 == 1 { $($then:tt)* } else { $($else:tt)* }) => { $($then)* };
    (32 == 32 { $($then:tt)* }) => { $($then)* };
    ($_:tt == 1 { $($then:tt)* } else { $($else:tt)* }) => { $($else)* };
    (signed int == signed int { $($then:tt)* }) => { $($then)* };
    (float == float { $($then:tt)* }) => { $($then)* };
    (not_float == not_float { $($then:tt)* }) => { $($then)* };
    (signed == signed { $($then:tt)* }) => { $($then)* };
    (unsigned == unsigned { $($then:tt)* }) => { $($then)* };
    (int == int { $($then:tt)* }) => { $($then)* };
    (matrix == matrix { $($then:tt)* }) => { $($then)* };
    ($($_:tt)*) => {};
}

pub(crate) use arith;
pub(crate) use if_;

pub(crate) trait ArithPrimitive: Copy {
    type Scalar;
    type F32;
    type I32;
    type U32;
    type Mask: MaskPrimitive;
    const ZERO_: Self;
    const ONE_: Self;
    #[allow(dead_code)]
    fn filled_(a: Self::Scalar) -> Self;
    #[allow(dead_code)]
    fn as_array_(&self) -> &[Self::Scalar];
    #[allow(dead_code)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar];
    fn cast_from_f32_(_a: Self::F32) -> Self { unimplemented!() }
    fn cast_from_i32_(_a: Self::I32) -> Self { unimplemented!() }
    fn cast_from_u32_(_a: Self::U32) -> Self { unimplemented!() }

    fn max_(self, _other: Self) -> Self { unimplemented!() }
    fn min_(self, _other: Self) -> Self { unimplemented!() }
    #[inline(always)]
    fn clamp_noexcept_(self, min: Self, max: Self) -> Self { self.max_(min).min_(max) }
    fn add_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn sub_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn mul_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn eq_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    #[inline(always)]
    fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { !Self::eq_(self, other) }
    fn gt_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    fn lt_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    #[inline(always)]
    fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { !Self::lt_(self, other) }
    #[inline(always)]
    fn le_(self, other: Self) -> MaskStorage<Self::Mask> { !Self::gt_(self, other) }
    fn select_(_mask: MaskStorage<Self::Mask>, _true_values: Self, _false_values: Self) -> Self {
        unimplemented!()
    }

    // Signed operations.
    fn neg_noexcept_(self) -> Self { unimplemented!() }
    fn abs_noexcept_(self) -> Self { unimplemented!() }
    fn signum_(self) -> Self { unimplemented!() }

    // Floating-point operations.
    fn round_ties_even_(self) -> Self { unimplemented!() }
    fn is_nan_(self) -> MaskStorage<Self::Mask> { unimplemented!() }
    /// a * b + c
    fn mul_add_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    /// a * b - c
    fn mul_sub_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    /// -a * b + c
    fn neg_mul_add_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    // Integer operations.
    fn shl_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn shr_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    // LLVM already folds the `filled` implementation to `psrld`, so the scalar variant is unused.
    #[expect(dead_code)]
    fn shl_scalar_noexcept_(self, _rhs: Self::Scalar) -> Self { unimplemented!() }
    #[expect(dead_code)]
    fn shr_scalar_noexcept_(self, _rhs: Self::Scalar) -> Self { unimplemented!() }
}

impl ArithPrimitive for f32 {
    type Scalar = Self;
    type F32 = f32;
    type I32 = i32;
    type U32 = u32;
    type Mask = i32;
    const ZERO_: Self = 0.;
    const ONE_: Self = 1.;
    #[inline(always)]
    fn filled_(a: Self::Scalar) -> Self { a }
    #[inline(always)]
    fn as_array_(&self) -> &[Self::Scalar] { core::array::from_ref(self) }
    #[inline(always)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { core::array::from_mut(self) }
    #[inline(always)]
    fn cast_from_f32_(a: Self::F32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_i32_(a: Self::I32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_u32_(a: Self::U32) -> Self { a as _ }
    #[inline(always)]
    fn max_(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn min_(self, other: Self) -> Self { self.min(other) }
    #[inline(always)]
    fn clamp_noexcept_(mut self, min: Self, max: Self) -> Self {
        if self < min {
            self = min;
        }
        if self > max {
            self = max;
        }
        self
    }
    #[inline(always)]
    fn add_noexcept_(self, rhs: Self) -> Self { core::ops::Add::add(self, rhs) }
    #[inline(always)]
    fn sub_noexcept_(self, rhs: Self) -> Self { core::ops::Sub::sub(self, rhs) }
    #[inline(always)]
    fn mul_noexcept_(self, rhs: Self) -> Self { core::ops::Mul::mul(self, rhs) }
    #[inline(always)]
    fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self == other) }
    #[inline(always)]
    fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self != other) }
    #[inline(always)]
    fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self > other) }
    #[inline(always)]
    fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self < other) }
    #[inline(always)]
    fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self >= other) }
    #[inline(always)]
    fn le_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self <= other) }
    #[inline(always)]
    fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
        if mask.into_inner() < 0 { true_values } else { false_values }
    }
    #[inline(always)]
    fn neg_noexcept_(self) -> Self { core::ops::Neg::neg(self) }
    #[inline(always)]
    fn abs_noexcept_(self) -> Self { self.abs() }
    #[inline(always)]
    fn signum_(self) -> Self { self.signum() }
    #[inline(always)]
    fn round_ties_even_(self) -> Self { self.round_ties_even() }
    #[inline(always)]
    fn is_nan_(self) -> MaskStorage<Self::Mask> { MaskStorage::new(self.is_nan()) }
    #[inline(always)]
    fn mul_add_(a: Self, b: Self, c: Self) -> Self {
        cfg_select! {
            any(target_feature = "fma", all(target_feature = "neon", target_arch = "aarch64")) => {
                a.mul_add(b, c)
            }
            _ => a * b + c,
        }
    }
    // NOTE: LLVM lowers the following `mul_add` calls to the matching fused instructions.
    #[inline(always)]
    fn mul_sub_(a: Self, b: Self, c: Self) -> Self {
        cfg_select! {
            any(target_feature = "fma", all(target_feature = "neon", target_arch = "aarch64")) => {
                a.mul_add(b, -c)
            }
            _ => a * b - c,
        }
    }
    #[inline(always)]
    fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self {
        cfg_select! {
            any(target_feature = "fma", all(target_feature = "neon", target_arch = "aarch64")) => {
                (-a).mul_add(b, c)
            }
            _ => c - a * b,
        }
    }
}
impl ArithPrimitive for i32 {
    type Scalar = Self;
    type F32 = f32;
    type I32 = i32;
    type U32 = u32;
    type Mask = i32;
    const ZERO_: Self = 0;
    const ONE_: Self = 1;
    #[inline(always)]
    fn filled_(a: Self::Scalar) -> Self { a }
    #[inline(always)]
    fn as_array_(&self) -> &[Self::Scalar] { core::array::from_ref(self) }
    #[inline(always)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { core::array::from_mut(self) }
    #[inline(always)]
    fn cast_from_f32_(a: Self::F32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_i32_(a: Self::I32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_u32_(a: Self::U32) -> Self { a as _ }
    #[inline(always)]
    fn max_(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn min_(self, other: Self) -> Self { self.min(other) }

    #[inline(always)]
    fn add_noexcept_(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    #[inline(always)]
    fn sub_noexcept_(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    #[inline(always)]
    fn mul_noexcept_(self, rhs: Self) -> Self { self.wrapping_mul(rhs) }
    #[inline(always)]
    fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self == other) }
    #[inline(always)]
    fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self != other) }
    #[inline(always)]
    fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self > other) }
    #[inline(always)]
    fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self < other) }
    #[inline(always)]
    fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self >= other) }
    #[inline(always)]
    fn le_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self <= other) }
    #[inline(always)]
    fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
        if mask.into_inner() < 0 { true_values } else { false_values }
    }
    #[inline(always)]
    fn neg_noexcept_(self) -> Self { self.wrapping_neg() }
    #[inline(always)]
    fn abs_noexcept_(self) -> Self { self.wrapping_abs() }
    #[inline(always)]
    fn signum_(self) -> Self { self.signum() }
    #[inline(always)]
    fn shl_noexcept_(self, rhs: Self) -> Self { self.wrapping_shl(rhs as u32) }
    #[inline(always)]
    fn shr_noexcept_(self, rhs: Self) -> Self { self.wrapping_shr(rhs as u32) }
    #[inline(always)]
    fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shl(rhs as u32) }
    #[inline(always)]
    fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shr(rhs as u32) }
}
impl ArithPrimitive for u32 {
    type Scalar = Self;
    type F32 = f32;
    type I32 = i32;
    type U32 = u32;
    type Mask = i32;
    const ZERO_: Self = 0;
    const ONE_: Self = 1;
    #[inline(always)]
    fn filled_(a: Self::Scalar) -> Self { a }
    #[inline(always)]
    fn as_array_(&self) -> &[Self::Scalar] { core::array::from_ref(self) }
    #[inline(always)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { core::array::from_mut(self) }
    #[inline(always)]
    fn cast_from_f32_(a: Self::F32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_i32_(a: Self::I32) -> Self { a as _ }
    #[inline(always)]
    fn cast_from_u32_(a: Self::U32) -> Self { a as _ }
    #[inline(always)]
    fn max_(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn min_(self, other: Self) -> Self { self.min(other) }
    #[inline(always)]
    fn add_noexcept_(self, other: Self) -> Self { self.wrapping_add(other) }

    #[inline(always)]
    fn sub_noexcept_(self, other: Self) -> Self { self.wrapping_sub(other) }
    #[inline(always)]
    fn mul_noexcept_(self, other: Self) -> Self { self.wrapping_mul(other) }
    #[inline(always)]
    fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self == other) }
    #[inline(always)]
    fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self != other) }
    #[inline(always)]
    fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self > other) }
    #[inline(always)]
    fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self < other) }
    #[inline(always)]
    fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self >= other) }
    #[inline(always)]
    fn le_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::new(self <= other) }
    #[inline(always)]
    fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
        if mask.into_inner() < 0 { true_values } else { false_values }
    }
    #[inline(always)]
    fn shl_noexcept_(self, rhs: Self) -> Self { self.wrapping_shl(rhs) }
    #[inline(always)]
    fn shr_noexcept_(self, rhs: Self) -> Self { self.wrapping_shr(rhs) }
    #[inline(always)]
    fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shl(rhs) }
    #[inline(always)]
    fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shr(rhs) }
}

pub(super) trait Load {
    type Output;
    fn load(self) -> Self::Output;
}
pub(super) trait Store<T> {
    fn store(self) -> T;
}
impl<T> Store<T> for T {
    #[inline(always)]
    fn store(self) -> T { self }
}

#[allow(unused_macros)]
macro_rules! impl_default_load {
    () => {
        impl<T> crate::utils::Load for T {
            type Output = T;
            #[inline(always)]
            fn load(self) -> Self::Output { self }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use impl_default_load;

mod mask_utils {

    /// Storage whose physical lanes are all-zero or all-one bit patterns.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct CanonicalMaskStorage<T>(T);
    pub(crate) use CanonicalMaskStorage as MaskStorage;

    /// Primitive storage that can uphold the canonical mask invariant.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that:
    ///
    /// - `is_valid` returns `true` if and only if every physical lane, including
    ///   padding lanes, is either an all-zero or all-one bit pattern.
    /// - `not` maps every valid value to another valid value.
    /// - `select` maps a valid selector and two valid input values to a valid
    ///   output by selecting each physical lane in full from one of the inputs.
    /// - copying a value preserves its physical lane representation.
    pub unsafe trait MaskPrimitive: Copy {
        fn is_valid(self) -> bool;
        fn not(self) -> Self;
        fn bitand(self, rhs: Self) -> Self;
        fn bitor(self, rhs: Self) -> Self;
        fn bitxor(self, rhs: Self) -> Self;
        // same as `Primitive::select_`
        fn select(self, true_values: Self, false_values: Self) -> Self;
    }

    // SAFETY: `is_valid` accepts exactly 0 and -1, `!` swaps those values, and
    // `select` returns one of its two canonical inputs in full.
    unsafe impl MaskPrimitive for i32 {
        fn is_valid(self) -> bool { self == 0 || self == -1 }
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
            if self < 0 { true_values } else { false_values }
        }
    }
    // SAFETY: every array element is validated, transformed, and selected
    // through its `MaskPrimitive` implementation, including elements used as
    // padding.
    unsafe impl<T: MaskPrimitive, const N: usize> MaskPrimitive for [T; N] {
        fn is_valid(self) -> bool { self.into_iter().all(MaskPrimitive::is_valid) }
        #[inline(always)]
        fn not(self) -> Self { self.map(MaskPrimitive::not) }
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self {
            core::array::from_fn({
                #[inline(always)]
                |i| self[i].bitand(rhs[i])
            })
        }
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self {
            core::array::from_fn({
                #[inline(always)]
                |i| self[i].bitor(rhs[i])
            })
        }
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self {
            core::array::from_fn({
                #[inline(always)]
                |i| self[i].bitxor(rhs[i])
            })
        }
        #[inline(always)]
        fn select(self, true_values: Self, false_values: Self) -> Self {
            core::array::from_fn({
                #[inline(always)]
                |i| MaskPrimitive::select(self[i], true_values[i], false_values[i])
            })
        }
    }

    impl<T: MaskPrimitive> core::ops::Not for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn not(self) -> Self::Output { Self(self.0.not()) }
    }
    impl<T: MaskPrimitive> core::ops::BitAnd for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self::Output { Self(self.0.bitand(rhs.0)) }
    }
    impl<T: MaskPrimitive> core::ops::BitOr for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self::Output { Self(self.0.bitor(rhs.0)) }
    }
    impl<T: MaskPrimitive> core::ops::BitXor for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self::Output { Self(self.0.bitxor(rhs.0)) }
    }
    impl<T: MaskPrimitive, const N: usize> From<[MaskStorage<T>; N]> for MaskStorage<[T; N]> {
        #[inline(always)]
        fn from(value: [MaskStorage<T>; N]) -> Self {
            MaskStorage(value.map(MaskStorage::into_inner))
        }
    }
    impl<T: MaskPrimitive> MaskStorage<T> {
        /// Creates canonical mask storage without checking it in release builds.
        ///
        /// # Safety
        ///
        /// Every physical lane in `inner`, including padding lanes, must be
        /// either an all-zero or all-one bit pattern.
        #[inline(always)]
        pub(crate) unsafe fn new_unchecked(inner: T) -> Self {
            debug_assert!(inner.is_valid());
            Self(inner)
        }
        #[inline(always)]
        pub(crate) fn select(self, true_values: Self, false_values: Self) -> Self {
            // SAFETY: all three wrappers contain canonical physical lanes.
            // `MaskPrimitive::select` selects each output lane in full from one
            // of the canonical inputs and therefore preserves the invariant.
            unsafe { Self::new_unchecked(T::select(self.0, true_values.0, false_values.0)) }
        }
    }
    impl<T> MaskStorage<T> {
        #[inline(always)]
        pub(crate) fn into_inner(self) -> T { self.0 }
    }
    impl<T, const N: usize> MaskStorage<[T; N]> {
        #[inline(always)]
        pub(crate) fn unpack(self) -> [MaskStorage<T>; N] { self.0.map(MaskStorage) }
    }
    impl MaskStorage<i32> {
        #[inline(always)]
        #[allow(dead_code)]
        pub(crate) fn unpack(self) -> Self { self }
        #[expect(dead_code)]
        pub(crate) const TRUE: Self = Self(-1);
        #[expect(dead_code)]
        pub(crate) const FALSE: Self = Self(0);
        #[inline(always)]
        pub(crate) fn new(value: bool) -> Self {
            unsafe {
                // SAFETY: false is 0 and true is -1
                Self::new_unchecked(-(value as i32))
            }
        }
    }
}

pub(crate) use mask_utils::*;
