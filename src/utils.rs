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
    (2 == 2 and 1 == 1 { $($then:tt)* }) => { $($then)* };
    (3 == 3 and 1 == 1 { $($then:tt)* }) => { $($then)* };
    (4 == 4 and 1 == 1 { $($then:tt)* }) => { $($then)* };
    (1 == 1 and 1 == 1 { $($then:tt)* }) => { $($then)* };
    (2 == 2 and 2 == 2 { $($then:tt)* }) => { $($then)* };
    (3 == 3 and 3 == 3 { $($then:tt)* }) => { $($then)* };
    (4 == 4 and 4 == 4 { $($then:tt)* }) => { $($then)* };
    (1 == 1 { $($then:tt)* }) => { $($then)* };
    (1 == 1 { $($then:tt)* } else { $($else:tt)* }) => { $($then)* };
    (32 == 32 { $($then:tt)* }) => { $($then)* };
    ($_:tt == 1 { $($then:tt)* } else { $($else:tt)* }) => { $($else)* };
    (signed int == signed int { $($then:tt)* }) => { $($then)* };
    (unsigned int == unsigned int { $($then:tt)* }) => { $($then)* };
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

// TODO(lane-count-generics): consider `ArithPrimitive<const N: usize>` and the same for
// `MaskPrimitive`, so `any`/`all`/`cast`/`to_bitmask` can tell 2, 3 and 4 lanes apart.
pub(crate) trait ArithPrimitive: Copy {
    type Scalar;
    type F32;
    type F64;
    type I32;
    type I64;
    type U32;
    type U64;
    // NOTE: `Copy` is required to implement `Copy for Mask<T, D>`
    type Mask: Copy + MaskLoad;
    const ZERO_: Self;
    const ONE_: Self;
    #[allow(dead_code)]
    fn filled_(a: Self::Scalar) -> Self;
    #[allow(dead_code)]
    fn as_array_(&self) -> &[Self::Scalar];
    #[allow(dead_code)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar];
    fn cast_from_f32_<const N: usize>(_a: Self::F32) -> Self { unimplemented!() }
    fn cast_from_f64_<const N: usize>(_a: Self::F64) -> Self { unimplemented!() }
    fn cast_from_i32_<const N: usize>(_a: Self::I32) -> Self { unimplemented!() }
    fn cast_from_i64_<const N: usize>(_a: Self::I64) -> Self { unimplemented!() }
    fn cast_from_u32_<const N: usize>(_a: Self::U32) -> Self { unimplemented!() }
    fn cast_from_u64_<const N: usize>(_a: Self::U64) -> Self { unimplemented!() }

    fn max_(self, _other: Self) -> Self { unimplemented!() }
    fn min_(self, _other: Self) -> Self { unimplemented!() }
    #[inline(always)]
    fn clamp_noexcept_(self, min: Self, max: Self) -> Self { self.max_(min).min_(max) }
    fn add_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn sub_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn mul_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    // Only the floating-point types implement this. An integer division has to check its divisor
    // first, which needs the shape's valid lane count, so the element traits keep bodies of their
    // own for it.
    fn div_(self, _rhs: Self) -> Self { unimplemented!() }
    fn eq_(self, _other: Self) -> MaskStorage<Self::Mask>;
    fn ne_(self, _other: Self) -> MaskStorage<Self::Mask>;
    fn gt_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    fn lt_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    fn ge_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    fn le_(self, _other: Self) -> MaskStorage<Self::Mask> { unimplemented!() }
    fn select_(_mask: MaskStorage<Self::Mask>, _true_values: Self, _false_values: Self) -> Self {
        unimplemented!()
    }

    // Signed operations.
    fn neg_noexcept_(self) -> Self { unimplemented!() }
    fn abs_noexcept_(self) -> Self { unimplemented!() }
    fn signum_(self) -> Self { unimplemented!() }

    // Floating-point operations. An integer type inherits the `unimplemented!()` body: no public
    // operation reaches a rounding or a square root on one.
    fn sqrt_(self) -> Self { unimplemented!() }
    fn floor_(self) -> Self { unimplemented!() }
    fn ceil_(self) -> Self { unimplemented!() }
    fn round_(self) -> Self { unimplemented!() }
    fn trunc_(self) -> Self { unimplemented!() }
    fn fract_(self) -> Self { unimplemented!() }
    #[allow(dead_code)]
    fn round_ties_even_(self) -> Self { unimplemented!() }
    fn is_nan_(self) -> MaskStorage<Self::Mask> { unimplemented!() }
    /// a * b + c
    fn mul_add_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    /// a * b - c
    fn mul_sub_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    /// -a * b + c
    fn neg_mul_add_(_a: Self, _b: Self, _c: Self) -> Self { unimplemented!() }
    // Integer operations. These behave the same on a scalar and on a compute vector; they are here
    // so that the element traits can name one body instead of one per shape.
    fn bitand_(self, _rhs: Self) -> Self { unimplemented!() }
    fn bitor_(self, _rhs: Self) -> Self { unimplemented!() }
    fn bitxor_(self, _rhs: Self) -> Self { unimplemented!() }
    fn not_(self) -> Self { unimplemented!() }
    fn shl_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    fn shr_noexcept_(self, _rhs: Self) -> Self { unimplemented!() }
    // LLVM already folds the `filled` implementation to `psrld`, so the scalar variant is unused.
    #[expect(dead_code)]
    fn shl_scalar_noexcept_(self, _rhs: Self::Scalar) -> Self { unimplemented!() }
    #[expect(dead_code)]
    fn shr_scalar_noexcept_(self, _rhs: Self::Scalar) -> Self { unimplemented!() }
}

macro_rules! impl_arith_primitive {
    ($self_ty:ty, mask=$mask:ty { $($item:item)* }) => {
        impl ArithPrimitive for $self_ty {
            type Scalar = Self;
            type F32 = f32;
            type F64 = f64;
            type I32 = i32;
            type I64 = i64;
            type U32 = u32;
            type U64 = u64;
            type Mask = $mask;
            const ZERO_: Self = 0 as _;
            const ONE_: Self = 1 as _;
            #[inline(always)]
            fn filled_(a: Self::Scalar) -> Self { a }
            #[inline(always)]
            fn as_array_(&self) -> &[Self::Scalar] { core::array::from_ref(self) }
            #[inline(always)]
            fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { core::array::from_mut(self) }
            #[inline(always)]
            fn cast_from_f32_<const N: usize>(a: Self::F32) -> Self { a as _ }
            #[inline(always)]
            fn cast_from_f64_<const N: usize>(a: Self::F64) -> Self { a as _ }
            #[inline(always)]
            fn cast_from_i32_<const N: usize>(a: Self::I32) -> Self { a as _ }
            #[inline(always)]
            fn cast_from_i64_<const N: usize>(a: Self::I64) -> Self { a as _ }
            #[inline(always)]
            fn cast_from_u32_<const N: usize>(a: Self::U32) -> Self { a as _ }
            #[inline(always)]
            fn cast_from_u64_<const N: usize>(a: Self::U64) -> Self { a as _ }
            #[inline(always)]
            fn max_(self, other: Self) -> Self { self.max(other) }
            #[inline(always)]
            fn min_(self, other: Self) -> Self { self.min(other) }
            #[inline(always)]
            fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self == other) }
            #[inline(always)]
            fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self != other) }
            #[inline(always)]
            fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self > other) }
            #[inline(always)]
            fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self < other) }
            #[inline(always)]
            fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self >= other) }
            #[inline(always)]
            fn le_(self, other: Self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self <= other) }
            #[inline(always)]
            fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
                if mask.into_inner() < 0 { true_values } else { false_values }
            }
            $($item)*
        }
    };
}
macro_rules! impl_arith_primitive_int {
    ($self_ty:ty, mask=$mask:ty { $($item:item)* }) => {
        impl_arith_primitive! {
            $self_ty, mask=$mask {
                #[inline(always)]
                fn add_noexcept_(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
                #[inline(always)]
                fn sub_noexcept_(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
                #[inline(always)]
                fn mul_noexcept_(self, rhs: Self) -> Self { self.wrapping_mul(rhs) }

                #[inline(always)]
                fn bitand_(self, rhs: Self) -> Self { core::ops::BitAnd::bitand(self, rhs) }
                #[inline(always)]
                fn bitor_(self, rhs: Self) -> Self { core::ops::BitOr::bitor(self, rhs) }
                #[inline(always)]
                fn bitxor_(self, rhs: Self) -> Self { core::ops::BitXor::bitxor(self, rhs) }
                #[inline(always)]
                fn not_(self) -> Self { core::ops::Not::not(self) }
                #[inline(always)]
                fn shl_noexcept_(self, rhs: Self) -> Self { self.wrapping_shl(rhs as u32) }
                #[inline(always)]
                fn shr_noexcept_(self, rhs: Self) -> Self { self.wrapping_shr(rhs as u32) }
                #[inline(always)]
                fn shl_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shl(rhs as u32) }
                #[inline(always)]
                fn shr_scalar_noexcept_(self, rhs: Self::Scalar) -> Self { self.wrapping_shr(rhs as u32) }
                $($item)*
            }
        }
    }
}
macro_rules! impl_arith_primitive_all {
    ($float:ty, $int:ty, $uint:ty) => {
        impl_arith_primitive! {
            $float, mask=$int {
                #[inline(always)]
                fn add_noexcept_(self, rhs: Self) -> Self { core::ops::Add::add(self, rhs) }
                #[inline(always)]
                fn sub_noexcept_(self, rhs: Self) -> Self { core::ops::Sub::sub(self, rhs) }
                #[inline(always)]
                fn mul_noexcept_(self, rhs: Self) -> Self { core::ops::Mul::mul(self, rhs) }
                #[inline(always)]
                fn div_(self, rhs: Self) -> Self { core::ops::Div::div(self, rhs) }
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
                fn neg_noexcept_(self) -> Self { core::ops::Neg::neg(self) }
                #[inline(always)]
                fn abs_noexcept_(self) -> Self { self.abs() }
                #[inline(always)]
                fn signum_(self) -> Self { self.signum() }
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
                fn round_ties_even_(self) -> Self { self.round_ties_even() }
                #[inline(always)]
                fn is_nan_(self) -> MaskStorage<Self::Mask> { MaskStorage::<Self::Mask>::new(self.is_nan()) }
                // Wasm has no scalar fused multiply-add: the only one it has is the relaxed
                // vector instruction, and reaching that from a scalar costs an `f32x4.splat` per
                // operand that neither Wasmtime nor V8 removes. A scalar `f32.fma` has been asked
                // for since 2020 (WebAssembly/design issue 1391) but is not a proposal at any
                // phase; revisit if one lands.
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
        }
        impl_arith_primitive_int! {
            $int, mask=$int {
                #[inline(always)]
                fn neg_noexcept_(self) -> Self { self.wrapping_neg() }
                #[inline(always)]
                fn abs_noexcept_(self) -> Self { self.wrapping_abs() }
                #[inline(always)]
                fn signum_(self) -> Self { self.signum() }
            }
        }
        impl_arith_primitive_int! {
            $uint, mask=$int {}
        }
    }
}

impl_arith_primitive_all!(f32, i32, u32);
impl_arith_primitive_all!(f64, i64, u64);

// A storage value is an array of units -- compute vectors on the SIMD backend, scalars on the other
// -- so the lane-wise operations on it are the unit's operations applied elementwise. Both backends
// reach this through `SealedElement::Storage: Load<Output: ArithPrimitive>`.
impl<T: ArithPrimitive, const N: usize> ArithPrimitive for [T; N] {
    // The leaf scalar, as every other implementation reports: `f32` for `f32`, for `f32x4` and for
    // `[f32x4; N]` alike. That is what makes `filled_` reach the whole storage in one step.
    type Scalar = T::Scalar;
    type F32 = [T::F32; N];
    type F64 = [T::F64; N];
    type I32 = [T::I32; N];
    type I64 = [T::I64; N];
    type U32 = [T::U32; N];
    type U64 = [T::U64; N];
    // Never read. A mask only exists for a vector, and a vector's storage is one unit -- `f32`,
    // `f32x2`, `f32x4` and their integer counterparts on the SIMD backend, `[[T; M]; 1]` on the
    // other -- so an array of units is never the storage a mask is taken from. The associated type
    // has to name something, and this is the shape that would be right if it ever were read.
    type Mask = [T::Mask; N];

    // TODO(duplicate-constants): with these available through the storage type, `SealedElement`
    // need not carry `ZERO` and `ONE` of its own.
    const ZERO_: Self = [T::ZERO_; N];
    const ONE_: Self = [T::ONE_; N];

    #[inline(always)]
    fn filled_(a: Self::Scalar) -> Self { [T::filled_(a); N] }
    // The lanes of an array of units are not contiguous in the unit's scalar; nothing asks an array
    // for them. `reduce::sum` and `map2` in `simd.rs` ask a unit.
    #[inline(always)]
    fn as_array_(&self) -> &[Self::Scalar] { unimplemented!() }
    #[inline(always)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { unimplemented!() }

    #[inline(always)]
    fn max_(self, rhs: Self) -> Self { zip(self, rhs, T::max_) }
    #[inline(always)]
    fn min_(self, rhs: Self) -> Self { zip(self, rhs, T::min_) }
    #[inline(always)]
    fn add_noexcept_(self, rhs: Self) -> Self { zip(self, rhs, T::add_noexcept_) }
    #[inline(always)]
    fn sub_noexcept_(self, rhs: Self) -> Self { zip(self, rhs, T::sub_noexcept_) }
    #[inline(always)]
    fn mul_noexcept_(self, rhs: Self) -> Self { zip(self, rhs, T::mul_noexcept_) }
    #[inline(always)]
    fn div_(self, rhs: Self) -> Self { zip(self, rhs, T::div_) }
    #[inline(always)]
    fn eq_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::eq_) }
    #[inline(always)]
    fn ne_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::ne_) }
    #[inline(always)]
    fn lt_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::lt_) }
    #[inline(always)]
    fn le_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::le_) }
    #[inline(always)]
    fn gt_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::gt_) }
    #[inline(always)]
    fn ge_(self, other: Self) -> MaskStorage<Self::Mask> { zip_mask(self, other, T::ge_) }
    #[inline(always)]
    fn is_nan_(self) -> MaskStorage<Self::Mask> {
        MaskStorage::store_packed(core::array::from_fn(
            #[inline(always)]
            |i| T::is_nan_(self[i]),
        ))
    }
    #[inline(always)]
    fn shl_noexcept_(self, rhs: Self) -> Self { zip(self, rhs, T::shl_noexcept_) }
    #[inline(always)]
    fn shr_noexcept_(self, rhs: Self) -> Self { zip(self, rhs, T::shr_noexcept_) }

    #[inline(always)]
    fn neg_noexcept_(self) -> Self { map(self, T::neg_noexcept_) }
    #[inline(always)]
    fn abs_noexcept_(self) -> Self { map(self, T::abs_noexcept_) }
    #[inline(always)]
    fn signum_(self) -> Self { map(self, T::signum_) }
    #[inline(always)]
    fn round_ties_even_(self) -> Self { map(self, T::round_ties_even_) }
    #[inline(always)]
    fn sqrt_(self) -> Self { map(self, T::sqrt_) }
    #[inline(always)]
    fn floor_(self) -> Self { map(self, T::floor_) }
    #[inline(always)]
    fn ceil_(self) -> Self { map(self, T::ceil_) }
    #[inline(always)]
    fn round_(self) -> Self { map(self, T::round_) }
    #[inline(always)]
    fn trunc_(self) -> Self { map(self, T::trunc_) }
    #[inline(always)]
    fn fract_(self) -> Self { map(self, T::fract_) }
    #[inline(always)]
    fn not_(self) -> Self { map(self, T::not_) }
    #[inline(always)]
    fn bitand_(self, rhs: Self) -> Self { zip(self, rhs, T::bitand_) }
    #[inline(always)]
    fn bitor_(self, rhs: Self) -> Self { zip(self, rhs, T::bitor_) }
    #[inline(always)]
    fn bitxor_(self, rhs: Self) -> Self { zip(self, rhs, T::bitxor_) }
    #[inline(always)]
    fn select_(mask: MaskStorage<Self::Mask>, true_values: Self, false_values: Self) -> Self {
        let mask = mask.unpack();
        core::array::from_fn(
            #[inline(always)]
            |i| T::select_(mask[i], true_values[i], false_values[i]),
        )
    }

    #[inline(always)]
    fn mul_add_(a: Self, b: Self, c: Self) -> Self { zip3(a, b, c, T::mul_add_) }
    #[inline(always)]
    fn mul_sub_(a: Self, b: Self, c: Self) -> Self { zip3(a, b, c, T::mul_sub_) }
    #[inline(always)]
    fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self { zip3(a, b, c, T::neg_mul_add_) }
    #[inline(always)]
    fn clamp_noexcept_(self, min: Self, max: Self) -> Self {
        zip3(self, min, max, T::clamp_noexcept_)
    }
}

#[inline(always)]
fn map<T: Copy, const N: usize>(a: [T; N], mut f: impl FnMut(T) -> T) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| f(a[i]),
    )
}

#[inline(always)]
fn zip<T: Copy, const N: usize>(a: [T; N], b: [T; N], mut f: impl FnMut(T, T) -> T) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| f(a[i], b[i]),
    )
}

/// Applies a comparison to each unit and collects the results into one wrapper.
#[inline(always)]
fn zip_mask<T: ArithPrimitive, const N: usize>(
    a: [T; N],
    b: [T; N],
    mut f: impl FnMut(T, T) -> MaskStorage<T::Mask>,
) -> MaskStorage<[T::Mask; N]> {
    MaskStorage::store_packed(core::array::from_fn(
        #[inline(always)]
        |i| f(a[i], b[i]),
    ))
}

#[inline(always)]
fn zip3<T: Copy, const N: usize>(
    a: [T; N],
    b: [T; N],
    c: [T; N],
    mut f: impl FnMut(T, T, T) -> T,
) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| f(a[i], b[i], c[i]),
    )
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
    // TODO(mask-bitmask-storage): AVX-512 has dedicated mask registers whose bits carry no lane
    // width, so a bitmask representation would make the 32-bit/64-bit mask casts free and let
    // `any`/`all`/`to_bitmask` read the register directly. Keeping one lane per element instead
    // costs a shuffle on every cross-width `select` there. Not attempted: every other target
    // wants the wide form, so this would need a second storage type behind a target feature.

    // TODO(non-simd-bool-mask-storage): the non-SIMD backend could store `[[bool; M]; N]` for
    // every element type instead of mirroring the element's width in `i32`/`i64` lanes. Nothing
    // in the public API promises a storage layout, and without vector instructions there is no
    // reason for the lane width to match the values being selected: the width casts and the
    // canonical `0`/`-1` invariant would both disappear.

    use crate::private;

    /// Storage whose physical lanes are all-zero or all-one bit patterns.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct CanonicalMaskStorage<T>(T);
    pub(crate) use CanonicalMaskStorage as MaskStorage;

    /// The mask storage that goes with element type `T` at shape `M` x `N`.
    ///
    /// Spelled through `ArithPrimitive::Mask` rather than through `SealedElement::Storage` so that
    /// the width relationship holds for every element type, not only for the ones that are their
    /// own mask: `f32` at `(4, 1)` stores `f32x4` and masks it with `i32x4`.
    pub(crate) type MaskStorage2<T, const M: usize = 1, const N: usize = 1> = CanonicalMaskStorage<
        <<T as private::SealedElement<M, N>>::Storage as crate::utils::ArithPrimitive>::Mask,
    >;

    /// Primitive storage that can uphold the canonical mask invariant.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that:
    ///
    /// - `is_valid` returns `true` if and only if every physical lane, including
    ///   padding lanes, is either an all-zero or all-one bit pattern.
    /// - `canonical_not`, `canonical_bitand`, `canonical_bitor` and `canonical_bitxor` map
    ///   canonical values to canonical values, combining every physical lane in full.
    /// - `canonical_select` maps a canonical selector and two canonical input values to a
    ///   canonical output by selecting each physical lane in full from one of the inputs.
    /// - copying a value preserves its physical lane representation.
    pub unsafe trait MaskPrimitive: Copy {
        fn is_valid(self) -> bool;
        fn canonical_not(self) -> Self;
        fn canonical_bitand(self, rhs: Self) -> Self;
        fn canonical_bitor(self, rhs: Self) -> Self;
        fn canonical_bitxor(self, rhs: Self) -> Self;
        // same as `Primitive::select_`
        fn canonical_select(self, true_values: Self, false_values: Self) -> Self;
        // Only the SIMD backend's `SealedElement::any`/`all` (see `simd.rs`) calls these; the
        // non-SIMD backend implements `any`/`all` directly over its flat array storage instead.
        #[allow(dead_code)]
        fn any<const N: usize>(self) -> bool;
        #[allow(dead_code)]
        fn all<const N: usize>(self) -> bool;
    }

    /// Mask storage, paired with the `MaskPrimitive` its operations are performed on.
    ///
    /// A mask is stored at the width of the element it selects and computed at the width the
    /// target's instructions use. On x86 those differ for a two-lane shape, whose storage is an
    /// eight-byte pair that widens to a four-lane vector to be operated on; everywhere else the
    /// two are the same type and both directions are the identity.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that:
    ///
    /// - `is_valid_storage` returns `true` if and only if every physical lane, including padding
    ///   lanes, is either an all-zero or an all-one bit pattern.
    /// - `__load` maps a value accepted by `is_valid_storage` to one accepted by
    ///   `MaskPrimitive::is_valid`, and `__store` maps one back, so that neither direction can
    ///   turn a canonical value into a mixed lane.
    pub(crate) unsafe trait MaskLoad: Copy {
        type Primitive: MaskPrimitive;
        fn is_valid_storage(self) -> bool;
        fn __load(self) -> Self::Primitive;
        fn __store(v: Self::Primitive) -> Self;
    }
    /// Implements `MaskLoad` for a type that is its own primitive.
    macro_rules! impl_mask_load {
        ($($ty:ty),+) => {$(
            // SAFETY: both directions are the identity, and `is_valid_storage` is the primitive's
            // own `is_valid`, so the two agree on which values are canonical.
            unsafe impl $crate::utils::MaskLoad for $ty {
                type Primitive = $ty;
                #[inline(always)]
                fn is_valid_storage(self) -> bool {
                    $crate::utils::MaskPrimitive::is_valid(self)
                }
                #[inline(always)]
                fn __load(self) -> Self::Primitive { self }
                #[inline(always)]
                fn __store(v: Self::Primitive) -> Self { v }
            }
        )+};
    }
    // The non-SIMD backend has no vector mask primitive to implement this for.
    #[allow(unused_imports)]
    pub(crate) use impl_mask_load;

    impl_mask_load!(i32, i64);

    // SAFETY: every element, including the ones used as padding, is validated and converted
    // through its own implementation, so neither direction can produce a mixed lane.
    unsafe impl<T: MaskLoad, const N: usize> MaskLoad for [T; N] {
        type Primitive = [T::Primitive; N];
        #[inline(always)]
        fn is_valid_storage(self) -> bool { self.into_iter().all(T::is_valid_storage) }
        #[inline(always)]
        fn __load(self) -> Self::Primitive { self.map(T::__load) }
        #[inline(always)]
        fn __store(v: Self::Primitive) -> Self { v.map(T::__store) }
    }

    impl<T: MaskLoad> MaskStorage<T> {
        #[inline(always)]
        pub(crate) fn load_mask(self) -> MaskStorage<<T as MaskLoad>::Primitive> {
            // SAFETY: the wrapper holds canonical physical lanes, and `MaskLoad` guarantees that
            // `__load` maps those to canonical lanes of the primitive.
            unsafe { MaskStorage::new_unchecked(T::__load(self.into_inner())) }
        }
        #[inline(always)]
        pub(crate) fn store_mask(mask: MaskStorage<<T as MaskLoad>::Primitive>) -> Self {
            // `MaskLoad` guarantees that `__store` maps canonical lanes back to canonical lanes.
            // The debug assertion checks that rather than taking it on trust.
            let inner = T::__store(mask.into_inner());
            debug_assert!(inner.is_valid_storage());
            MaskStorage(inner)
        }
    }
    impl<T, const N: usize> MaskStorage<[T; N]> {
        /// Collects one wrapper per unit into a single wrapper over the array.
        ///
        /// No bound is needed and no lane is inspected: every element is already canonical, and an
        /// array of canonical units is canonical.
        #[inline(always)]
        pub(crate) fn store_packed(mask: [MaskStorage<T>; N]) -> Self {
            MaskStorage(mask.map(MaskStorage::into_inner))
        }
    }

    // SAFETY: `is_valid` accepts exactly 0 and -1. The relevant `ArithPrimitive` operations act on
    // the value's complete bit pattern and preserve those two canonical values.
    unsafe impl MaskPrimitive for i32 {
        fn is_valid(self) -> bool { self == 0 || self == -1 }
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
            if self < 0 { true_values } else { false_values }
        }
        #[inline(always)]
        fn any<const N: usize>(self) -> bool {
            assert_eq!(N, 1);
            self < 0
        }
        #[inline(always)]
        fn all<const N: usize>(self) -> bool {
            assert_eq!(N, 1);
            self < 0
        }
    }
    unsafe impl MaskPrimitive for i64 {
        fn is_valid(self) -> bool { self == 0 || self == -1 }
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
            if self < 0 { true_values } else { false_values }
        }
        #[inline(always)]
        fn any<const N: usize>(self) -> bool {
            assert_eq!(N, 1);
            self < 0
        }
        #[inline(always)]
        fn all<const N: usize>(self) -> bool {
            assert_eq!(N, 1);
            self < 0
        }
    }
    // SAFETY: every array element, including padding, is validated through its `MaskPrimitive`
    // implementation. Its `ArithPrimitive` operations are applied elementwise and preserve the
    // invariant by `T`'s guarantee.
    unsafe impl<T: MaskPrimitive, const N: usize> MaskPrimitive for [T; N] {
        fn is_valid(self) -> bool { self.into_iter().all(MaskPrimitive::is_valid) }
        #[inline(always)]
        fn canonical_not(self) -> Self { self.map(MaskPrimitive::canonical_not) }
        #[inline(always)]
        fn canonical_bitand(self, rhs: Self) -> Self {
            core::array::from_fn(
                #[inline(always)]
                |i| self[i].canonical_bitand(rhs[i]),
            )
        }
        #[inline(always)]
        fn canonical_bitor(self, rhs: Self) -> Self {
            core::array::from_fn(
                #[inline(always)]
                |i| self[i].canonical_bitor(rhs[i]),
            )
        }
        #[inline(always)]
        fn canonical_bitxor(self, rhs: Self) -> Self {
            core::array::from_fn(
                #[inline(always)]
                |i| self[i].canonical_bitxor(rhs[i]),
            )
        }
        #[inline(always)]
        fn canonical_select(self, true_values: Self, false_values: Self) -> Self {
            core::array::from_fn(
                #[inline(always)]
                |i| self[i].canonical_select(true_values[i], false_values[i]),
            )
        }
        fn any<const M: usize>(self) -> bool { unimplemented!() }
        fn all<const M: usize>(self) -> bool { unimplemented!() }
    }

    impl<T: MaskPrimitive> core::ops::Not for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn not(self) -> Self::Output {
            unsafe { Self::new_unchecked(self.into_inner().canonical_not()) }
        }
    }
    impl<T: MaskPrimitive> core::ops::BitAnd for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self::Output {
            unsafe { Self::new_unchecked(self.into_inner().canonical_bitand(rhs.into_inner())) }
        }
    }
    impl<T: MaskPrimitive> core::ops::BitOr for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self::Output {
            unsafe { Self::new_unchecked(self.into_inner().canonical_bitor(rhs.into_inner())) }
        }
    }
    impl<T: MaskPrimitive> core::ops::BitXor for MaskStorage<T> {
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self::Output {
            unsafe { Self::new_unchecked(self.into_inner().canonical_bitxor(rhs.into_inner())) }
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
            // `MaskPrimitive` guarantees that `ArithPrimitive::select_` selects each output lane
            // in full from one of the canonical inputs and therefore preserves the invariant.
            unsafe {
                Self::new_unchecked(T::canonical_select(self.0, true_values.0, false_values.0))
            }
        }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn any<const N: usize>(self) -> bool { self.0.any::<N>() }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn all<const N: usize>(self) -> bool { self.0.all::<N>() }
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
    impl MaskStorage<i64> {
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
                Self::new_unchecked(-(value as i64))
            }
        }
    }
}

pub(crate) use mask_utils::*;
