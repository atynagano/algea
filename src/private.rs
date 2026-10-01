pub(crate) use crate::definitions::{SealedStorageElement, SealedSupportedDimension};
use crate::{
    impl_marker_trait,
    support::{
        ColumnMajor,
        Dimension,
        Float,
        Int,
        RowMajor,
        StoredVerbatim,
        SupportedDimension,
        SupportedElement,
    },
    utils::{self, ArithOps, CanonicalMask, DimMaskStorage},
};

pub(crate) trait Fmt {
    fn fmt<T, const M: usize, const N: usize>(
        storage: impl utils::Store<ConstStorage<T, M, N>>,
    ) -> impl core::fmt::Debug
    where
        T: SealedSupportedElement + core::fmt::Debug,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension;
}

pub(crate) enum VectorFmt {}

pub(crate) enum Indices2<const I0: usize, const I1: usize> {}
pub(crate) enum Indices3<const I0: usize, const I1: usize, const I2: usize> {}
pub(crate) enum Indices4<const I0: usize, const I1: usize, const I2: usize, const I3: usize> {}

pub(crate) trait SwizzleDispatch<T, const M: usize, const N: usize> {
    // The non-SIMD backend never calls this (see `src/non_simd/utils.rs`), so it is unused
    // under that backend.
    #[allow(dead_code)]
    fn dispatch(v: ConstStorage<T, M>) -> ConstStorage<T, N>
    where
        T: SealedSupportedElement,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension;
}
pub(crate) trait SwizzleDispatchAny<const N: usize>:
    SwizzleDispatch<f32, 2, N>
    + SwizzleDispatch<f32, 3, N>
    + SwizzleDispatch<f32, 4, N>
    + SwizzleDispatch<i32, 2, N>
    + SwizzleDispatch<i32, 3, N>
    + SwizzleDispatch<i32, 4, N>
    + SwizzleDispatch<u32, 2, N>
    + SwizzleDispatch<u32, 3, N>
    + SwizzleDispatch<u32, 4, N>
    + SwizzleDispatch<f64, 2, N>
    + SwizzleDispatch<f64, 3, N>
    + SwizzleDispatch<f64, 4, N>
    + SwizzleDispatch<i64, 2, N>
    + SwizzleDispatch<i64, 3, N>
    + SwizzleDispatch<i64, 4, N>
    + SwizzleDispatch<u64, 2, N>
    + SwizzleDispatch<u64, 3, N>
    + SwizzleDispatch<u64, 4, N>
{
}
impl<T, const N: usize> SwizzleDispatchAny<N> for T where
    T: SwizzleDispatch<f32, 2, N>
        + SwizzleDispatch<f32, 3, N>
        + SwizzleDispatch<f32, 4, N>
        + SwizzleDispatch<i32, 2, N>
        + SwizzleDispatch<i32, 3, N>
        + SwizzleDispatch<i32, 4, N>
        + SwizzleDispatch<u32, 2, N>
        + SwizzleDispatch<u32, 3, N>
        + SwizzleDispatch<u32, 4, N>
        + SwizzleDispatch<f64, 2, N>
        + SwizzleDispatch<f64, 3, N>
        + SwizzleDispatch<f64, 4, N>
        + SwizzleDispatch<i64, 2, N>
        + SwizzleDispatch<i64, 3, N>
        + SwizzleDispatch<i64, 4, N>
        + SwizzleDispatch<u64, 2, N>
        + SwizzleDispatch<u64, 3, N>
        + SwizzleDispatch<u64, 4, N>
{
}

impl Fmt for VectorFmt {
    #[inline(never)]
    fn fmt<T, const M: usize, const N: usize>(
        storage: impl utils::Store<ConstStorage<T, M, N>>,
    ) -> impl core::fmt::Debug
    where
        T: SealedSupportedElement + core::fmt::Debug,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension,
    {
        let array = ConstStorage::<T, M, N>::to_array(storage.store());
        assert_eq!(array.len(), 1);
        array.into_iter().next().unwrap()
    }
}

pub(crate) enum Type {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

// TODO(trait-consolidation): evaluate making `Sealed` inherit
// `ArithOps`. Their scalar capability and arithmetic boundaries still
// overlap; consolidate them once their marker bounds and all backend call
// sites can be unified without broadening the public API.
pub(crate) trait Sealed: Sized {
    // Scalar implementations used through `SealedSupportedElement` must override `TYPE`
    // so that it exactly identifies the concrete scalar type. Non-scalar
    // implementations used only to seal helper traits may keep this default;
    // evaluating it then fails during const evaluation.
    const TYPE: Type = panic!("TYPE is only defined for sealed scalar types");

    fn sqrt(self) -> Self { unimplemented!() }
}

impl_marker_trait!(Sealed for [
    f32 {
        const TYPE: Type = Type::F32;
        #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    },
    f64 {
        const TYPE: Type = Type::F64;
        #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    },
    i32 {
        const TYPE: Type = Type::I32;
        #[inline(always)] fn sqrt(self) -> Self { self.isqrt() }
    },
    i64 {
        const TYPE: Type = Type::I64;
        #[inline(always)] fn sqrt(self) -> Self { self.isqrt() }
    },
    u32 {
        const TYPE: Type = Type::U32;
        #[inline(always)] fn sqrt(self) -> Self { self.isqrt() }
    },
    u64 {
        const TYPE: Type = Type::U64;
        #[inline(always)] fn sqrt(self) -> Self { self.isqrt() }
    },
]);

// TODO: Revisit this witness after Rust issue #100177; the new trait solver may allow a
// simpler formulation.
pub(crate) trait SealedDimensionWitness<const D: usize>: SupportedElement {
    type Dimension;
}
impl<T, const D: usize> SealedDimensionWitness<D> for T
where
    T: SupportedElement,
    Dimension<D>: SupportedDimension,
{
    type Dimension = Dimension<D>;
}

pub(crate) trait DimensionTypes {
    type Array<T>;
    type Vector<T>
    where
        T: SupportedElement,
        Self: SupportedDimension;
}
impl<const D: usize> DimensionTypes for Dimension<D> {
    type Array<T> = [T; D];
    type Vector<T>
        = crate::Vector<T, D>
    where
        T: SupportedElement,
        Self: SupportedDimension;
}

pub(crate) type DimStorage<T, R, C = Dimension<1>> =
    <R as SealedSupportedDimension>::StorageNxC<T, C>;
pub(crate) type ConstStorage<T, const R: usize, const C: usize = 1> =
    DimStorage<T, Dimension<R>, Dimension<C>>;
pub(crate) type DimArray<T, D> = <D as DimensionTypes>::Array<T>;
pub(crate) type DimVector<T, D> = <D as DimensionTypes>::Vector<T>;

pub(crate) trait SealedSupportedElement: Sealed + SealedStorageElement {
    fn vector_concat_1_1(
        a: ConstStorage<Self, 1>,
        b: ConstStorage<Self, 1>,
    ) -> ConstStorage<Self, 2> {
        let [[a]] = crate::api::vector::call!(<Self, 1>::to_array(a));
        let [[b]] = crate::api::vector::call!(<Self, 1>::to_array(b));
        crate::api::vector::call!(<Self, 2>::from_array([[a, b]]))
    }
    fn vector_concat_1_2(
        a: ConstStorage<Self, 1>,
        b: ConstStorage<Self, 2>,
    ) -> ConstStorage<Self, 3> {
        let [[a]] = crate::api::vector::call!(<Self, 1>::to_array(a));
        let [[b, c]] = crate::api::vector::call!(<Self, 2>::to_array(b));
        crate::api::vector::call!(<Self, 3>::from_array([[a, b, c]]))
    }
    fn vector_concat_2_1(
        a: ConstStorage<Self, 2>,
        b: ConstStorage<Self, 1>,
    ) -> ConstStorage<Self, 3> {
        let [[a0, a1]] = crate::api::vector::call!(<Self, 2>::to_array(a));
        let [[b0]] = crate::api::vector::call!(<Self, 1>::to_array(b));
        crate::api::vector::call!(<Self, 3>::from_array([[a0, a1, b0]]))
    }
}

pub(crate) trait StorageOps<
    T: SealedSupportedElement,
    R: SealedSupportedDimension,
    C: SealedSupportedDimension = Dimension<1>,
>: ArithOps<Scalar = T>
{
    const ZERO: Self = ArithOps::ZERO_;
    const ONE: Self = ArithOps::ONE_;
    const IDENTITY: Self = unimplemented!();
    const POS_X: Self = unimplemented!();
    const POS_Y: Self = unimplemented!();
    const POS_Z: Self = unimplemented!();
    const POS_W: Self = unimplemented!();
    const NEG_X: Self = unimplemented!();
    const NEG_Y: Self = unimplemented!();
    const NEG_Z: Self = unimplemented!();
    const NEG_W: Self = unimplemented!();

    fn map2(a: Self, b: Self, f: impl FnMut(T, T) -> T) -> Self;
    fn index(_a: &Self, _index: (usize, usize)) -> Option<&T> { unimplemented!() }
    fn index_mut(_a: &mut Self, _index: (usize, usize)) -> Option<&mut T> { unimplemented!() }
    //noinspection RsSelfConvention
    fn as_array_first(_a: &Self) -> &DimArray<T, R>
    where
        R: DimensionTypes,
    {
        unimplemented!()
    }
    //noinspection RsSelfConvention
    fn as_mut_array_first(_a: &mut Self) -> &mut DimArray<T, R>
    where
        R: DimensionTypes,
    {
        unimplemented!()
    }
    //noinspection RsSelfConvention
    fn to_array(_a: Self) -> DimArray<DimArray<T, R>, C>
    where
        R: DimensionTypes,
        C: DimensionTypes,
    {
        unimplemented!()
    }
    fn from_array(_a: DimArray<DimArray<T, R>, C>) -> Self
    where
        R: DimensionTypes,
        C: DimensionTypes,
    {
        unimplemented!()
    }
    fn from_vecs(_a: DimArray<DimVector<T, R>, C>) -> Self
    where
        T: SupportedElement,
        R: DimensionTypes + SupportedDimension,
        C: DimensionTypes,
    {
        unimplemented!()
    }
    #[inline(always)]
    fn filled(value: T) -> Self {
        // Named rather than inferred: `filled_` takes only the scalar, so nothing else
        // pins which type produces the storage.
        ArithOps::filled_(value)
    }
    fn substantiate_f32(_a: Self) -> DimStorage<f32, R, C> { unimplemented!() }
    fn substantiate_f64(_a: Self) -> DimStorage<f64, R, C> { unimplemented!() }
    fn substantiate_i32(_a: Self) -> DimStorage<i32, R, C> { unimplemented!() }
    fn substantiate_i64(_a: Self) -> DimStorage<i64, R, C> { unimplemented!() }
    fn substantiate_u32(_a: Self) -> DimStorage<u32, R, C> { unimplemented!() }
    fn substantiate_u64(_a: Self) -> DimStorage<u64, R, C> { unimplemented!() }
    fn cast_from_f32(_a: DimStorage<f32, R, C>) -> Self { unimplemented!() }
    fn cast_from_f64(_a: DimStorage<f64, R, C>) -> Self { unimplemented!() }
    fn cast_from_i32(_a: DimStorage<i32, R, C>) -> Self { unimplemented!() }
    fn cast_from_i64(_a: DimStorage<i64, R, C>) -> Self { unimplemented!() }
    fn cast_from_u32(_a: DimStorage<u32, R, C>) -> Self { unimplemented!() }
    fn cast_from_u64(_a: DimStorage<u64, R, C>) -> Self { unimplemented!() }
    fn cast_from<U: SealedSupportedElement>(_a: DimStorage<U, R, C>) -> Self { unimplemented!() }

    fn cast_signed(_a: Self) -> DimStorage<<T as Int>::Signed, R, C>
    where
        T: Int<Signed: SealedSupportedElement>,
    {
        unimplemented!()
    }
    fn cast_unsigned(_a: Self) -> DimStorage<<T as Int>::Unsigned, R, C>
    where
        T: Int<Unsigned: SealedSupportedElement>,
    {
        unimplemented!()
    }
    fn swizzle2<const I0: usize, const I1: usize>(_a: Self) -> ConstStorage<T, 2, 1>
    where
        Indices2<I0, I1>: SwizzleDispatchAny<2>,
    {
        unimplemented!()
    }
    fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(
        _a: Self,
    ) -> ConstStorage<T, 3, 1>
    where
        Indices3<I0, I1, I2>: SwizzleDispatchAny<3>,
    {
        unimplemented!()
    }
    fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        _a: Self,
    ) -> ConstStorage<T, 4, 1>
    where
        Indices4<I0, I1, I2, I3>: SwizzleDispatchAny<4>,
    {
        unimplemented!()
    }

    /// Reinterprets a mask produced at `Self`'s width as one belonging to `Self::Mask`.
    ///
    /// Both sides are the same concrete type -- `f32` at `(4, 1)` and its mask element `i32`
    /// both mask with `i32x4` -- but only an implementation, where the element type is
    /// concrete, can see that. The comparison defaults below leave this one step to the
    /// backend and keep the comparison itself in one body.
    // TODO: Could proving `Self = DimStorage<T, R, C>` simplify this implementation?
    // In particular, could it establish equality between their `ArithOps::Mask` types?
    fn substantiate_mask(
        _mask: CanonicalMask<<Self as ArithOps>::Mask>,
    ) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        unimplemented!()
    }
    /// Unwraps a mask into the vector storage of the same element type, for `Mask::to_vector`.
    ///
    /// Implemented for the mask element types alone, where the two are the same type.
    fn from_mask(_mask: DimMaskStorage<T::Mask, R, C>) -> Self
    where
        T: SupportedElement,
    {
        unimplemented!()
    }
    fn select_mask(
        _mask: DimMaskStorage<T::Mask, R, C>,
        _true_values: Self,
        _false_values: Self,
    ) -> Self
    where
        T: SupportedElement,
    {
        unimplemented!()
    }
    fn select_any_mask<Mask: SealedSupportedElement>(
        _mask: DimMaskStorage<Mask, R, C>,
        _true_values: Self,
        _false_values: Self,
    ) -> Self
    where
        T: SupportedElement,
    {
        unimplemented!()
    }
    fn select_bitmask(bitmask: u8, true_values: Self, false_values: Self) -> Self
    where
        T: SupportedElement,
    {
        let mask = <DimStorage<T::Mask, R, C> as StorageOps<T::Mask, R, C>>::from_bitmask(bitmask);
        Self::select_mask(mask, true_values, false_values)
    }

    fn cast_i32(_mask: DimMaskStorage<T, R, C>) -> DimMaskStorage<i32, R, C> { unimplemented!() }
    fn cast_i64(_mask: DimMaskStorage<T, R, C>) -> DimMaskStorage<i64, R, C> { unimplemented!() }
    fn cast_mask<U: SealedSupportedElement>(
        _mask: DimMaskStorage<U, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        unimplemented!()
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_bool_array(_mask: DimMaskStorage<T, R, C>) -> DimArray<DimArray<bool, R>, C>
    where
        R: DimensionTypes,
        C: DimensionTypes,
    {
        unimplemented!()
    }
    fn from_bool_array(_array: DimArray<DimArray<bool, R>, C>) -> DimMaskStorage<T, R, C>
    where
        R: DimensionTypes,
        C: DimensionTypes,
    {
        unimplemented!()
    }
    fn all(_mask: DimMaskStorage<T, R, C>) -> bool { unimplemented!() }
    fn any(_mask: DimMaskStorage<T, R, C>) -> bool { unimplemented!() }
    #[allow(clippy::wrong_self_convention)]
    fn to_bitmask(_mask: DimMaskStorage<T, R, C>) -> u8 { unimplemented!() }
    fn from_bitmask(_bitmask: u8) -> DimMaskStorage<T, R, C> { unimplemented!() }

    // The four operations below take a mask and return a mask of the same element type, so
    // widening and narrowing both happen here and the backends need no hook. The comparisons
    // further down cannot do the same: they take `Self` and return `Self::Mask`, and nothing
    // at this level relates the two storage types, which is what `substantiate_mask` is for.
    #[inline(always)]
    fn mask_not(mask: DimMaskStorage<T, R, C>) -> DimMaskStorage<T, R, C> {
        CanonicalMask::store_mask(!mask.load_mask())
    }
    #[inline(always)]
    fn mask_bitand(
        a: DimMaskStorage<T, R, C>,
        b: DimMaskStorage<T, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        CanonicalMask::store_mask(a.load_mask() & b.load_mask())
    }
    #[inline(always)]
    fn mask_bitor(
        a: DimMaskStorage<T, R, C>,
        b: DimMaskStorage<T, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        CanonicalMask::store_mask(a.load_mask() | b.load_mask())
    }
    #[inline(always)]
    fn mask_bitxor(
        a: DimMaskStorage<T, R, C>,
        b: DimMaskStorage<T, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        CanonicalMask::store_mask(a.load_mask() ^ b.load_mask())
    }
    fn mask_select_any<Mask: SealedSupportedElement>(
        _mask: DimMaskStorage<Mask, R, C>,
        _true_values: DimMaskStorage<T, R, C>,
        _false_values: DimMaskStorage<T, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        unimplemented!()
    }
    fn mask_select_bitmask(
        _mask: u8,
        _true_values: DimMaskStorage<T, R, C>,
        _false_values: DimMaskStorage<T, R, C>,
    ) -> DimMaskStorage<T, R, C> {
        unimplemented!()
    }

    fn each_eq(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::eq_(a, b))
    }
    fn each_ne(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::ne_(a, b))
    }
    fn each_lt(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::lt_(a, b))
    }
    fn each_le(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::le_(a, b))
    }
    fn each_gt(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::gt_(a, b))
    }
    fn each_ge(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::ge_(a, b))
    }
    #[expect(dead_code)]
    fn is_nan(_a: Self) -> DimMaskStorage<T::Mask, R, C>
    where
        T: SupportedElement,
    {
        Self::substantiate_mask(ArithOps::is_nan_(_a))
    }

    // The public API exposes these on `Vector` alone, but the `each_max` and `each_min`
    // defaults cover every supported storage shape.
    #[inline(always)]
    fn each_max(a: Self, b: Self) -> Self { ArithOps::max_(a, b) }
    #[inline(always)]
    fn each_min(a: Self, b: Self) -> Self { ArithOps::min_(a, b) }
    fn each_clamp<F: Fmt>(_a: Self, _min: Self, _max: Self) -> Self { unimplemented!() }
    fn eq(a: Self, b: Self) -> bool;
    fn ne(a: Self, b: Self) -> bool;
    // The lane-wise operations below have one body for every backend and every shape: the
    // unit's operation applied to each unit, which `ArithOps for [T; N]` in `utils.rs`
    // expresses once. They stay named here rather than moving to the call sites so that the
    // arithmetic family reads as one -- `div` and `rem` still need bodies of their own for the
    // zero check -- and so that a shape or a backend can override one without the operation
    // having to move back onto the trait first.
    //
    // `#[inline(always)]` is load-bearing, not decoration. A matrix multiplied by a scalar has
    // to fold the scalar broadcast into the multiply, and the two-lane widths are eight-byte
    // aggregates that Rust passes in a general-purpose register: leaving a call boundary in
    // between costs the broadcast its `vbroadcastss`.
    #[inline(always)]
    fn add(a: Self, b: Self) -> Self { ArithOps::add_noexcept_(a, b) }
    #[inline(always)]
    fn sub(a: Self, b: Self) -> Self { ArithOps::sub_noexcept_(a, b) }
    #[inline(always)]
    fn mul(a: Self, b: Self) -> Self { ArithOps::mul_noexcept_(a, b) }
    // Integer division overrides this to reject a zero divisor first.
    #[inline(always)]
    fn div(a: Self, b: Self) -> Self { ArithOps::div_(a, b) }
    // TODO(integer-vector): separate sqrt and isqrt semantics in public traits.
    #[inline(always)]
    fn sqrt(a: Self) -> Self { ArithOps::sqrt_(a) }
    fn transpose(a: Self) -> DimStorage<T, C, R>;

    #[allow(dead_code)]
    fn substantiate_1x(_a: Self) -> DimStorage<T, Dimension<1>, C> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_2x(_a: Self) -> DimStorage<T, Dimension<2>, C> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_3x(_a: Self) -> DimStorage<T, Dimension<3>, C> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_4x(_a: Self) -> DimStorage<T, Dimension<4>, C> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_x1(_a: Self) -> DimStorage<T, R, Dimension<1>> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_x2(_a: Self) -> DimStorage<T, R, Dimension<2>> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_x3(_a: Self) -> DimStorage<T, R, Dimension<3>> { unimplemented!() }
    #[allow(dead_code)]
    fn substantiate_x4(_a: Self) -> DimStorage<T, R, Dimension<4>> { unimplemented!() }
    // TODO(integer-products): After the numeric element semantics are defined, generalize
    // matrix products to integers, retain non-FMA integer kernels, and add debug-mode
    // overflow tests through every public product operation.
    fn matmul<const K: usize>(
        _a: DimStorage<T, R, Dimension<K>>,
        _b: DimStorage<T, Dimension<K>, C>,
    ) -> Self
    where
        Dimension<K>: SealedSupportedDimension,
    {
        unimplemented!()
    }

    fn from_bits(_a: DimStorage<<T as Float>::Bits, R, C>) -> Self
    where
        T: Float<Bits: SealedSupportedElement>,
    {
        unimplemented!()
    }
    #[allow(clippy::wrong_self_convention)]
    fn to_bits(_a: Self) -> DimStorage<<T as Float>::Bits, R, C>
    where
        T: Float<Bits: SealedSupportedElement>,
    {
        unimplemented!()
    }
    #[inline(always)]
    fn floor(a: Self) -> Self { ArithOps::floor_(a) }
    #[inline(always)]
    fn ceil(a: Self) -> Self { ArithOps::ceil_(a) }
    #[inline(always)]
    fn round(a: Self) -> Self { ArithOps::round_(a) }
    #[inline(always)]
    fn round_ties_even(a: Self) -> Self { ArithOps::round_ties_even_(a) }
    #[inline(always)]
    fn trunc(a: Self) -> Self { ArithOps::trunc_(a) }
    #[inline(always)]
    fn fract(a: Self) -> Self { ArithOps::fract_(a) }
    #[inline(always)]
    fn neg(a: Self) -> Self { ArithOps::neg_noexcept_(a) }
    #[inline(always)]
    fn abs(a: Self) -> Self { ArithOps::abs_noexcept_(a) }
    // No public operation reaches this yet; the vocabulary is here for when one does.
    #[expect(dead_code)]
    #[inline(always)]
    fn signum(a: Self) -> Self { ArithOps::signum_(a) }
    fn rem(_a: Self, _b: Self) -> Self { unimplemented!() }
    #[inline(always)]
    fn not(a: Self) -> Self { ArithOps::not_(a) }
    #[inline(always)]
    fn bitand(a: Self, b: Self) -> Self { ArithOps::bitand_(a, b) }
    #[inline(always)]
    fn bitor(a: Self, b: Self) -> Self { ArithOps::bitor_(a, b) }
    #[inline(always)]
    fn bitxor(a: Self, b: Self) -> Self { ArithOps::bitxor_(a, b) }
    #[inline(always)]
    fn shl(a: Self, b: Self) -> Self { ArithOps::shl_noexcept_(a, b) }
    #[inline(always)]
    fn shr(a: Self, b: Self) -> Self { ArithOps::shr_noexcept_(a, b) }

    fn reduce_sum(_a: Self) -> T { unimplemented!() }
    #[inline(always)]
    fn dot(a: Self, b: Self) -> T { Self::reduce_sum(Self::mul(a, b)) }

    fn diagonal(_a: Self) -> DimStorage<T, R> { unimplemented!() }
    // Only regular 1x1 through 4x4 square shapes are supported; shapes of 5x5 or larger are
    // outside the planned scope. Floating-point and boolean forms are reversible, while
    // integer and integer-backed mask forms are not.
    fn inverse(_a: Self) -> Self { unimplemented!() }
    #[expect(dead_code)]
    fn try_inverse(_a: Self) -> Option<Self> { unimplemented!() }
    fn determinant(_a: Self) -> T { unimplemented!() }
}

pub(crate) trait SealedMatrixLayout: Sized {
    type StorageRxC<T, const R: usize, const C: usize>: OrientedStorageOps<T, R, C, Self>
    where
        T: SealedSupportedElement,
        Dimension<R>: SealedSupportedDimension,
        Dimension<C>: SealedSupportedDimension;
}

impl SealedMatrixLayout for ColumnMajor {
    type StorageRxC<T, const R: usize, const C: usize>
        = ConstStorage<T, R, C>
    where
        T: SealedSupportedElement,
        Dimension<R>: SealedSupportedDimension,
        Dimension<C>: SealedSupportedDimension;
}

impl SealedMatrixLayout for RowMajor {
    type StorageRxC<T, const R: usize, const C: usize>
        = ConstStorage<T, C, R>
    where
        T: SealedSupportedElement,
        Dimension<R>: SealedSupportedDimension,
        Dimension<C>: SealedSupportedDimension;
}

pub(crate) trait OrientedStorageOps<T, const R: usize, const C: usize, L>: Copy
where
    T: SealedSupportedElement,
    Dimension<R>: SealedSupportedDimension,
    Dimension<C>: SealedSupportedDimension,
    L: SealedMatrixLayout,
{
    const ZERO: Self;
    const ONE: Self;
    const IDENTITY: Self;

    fn filled(value: T) -> Self;
    fn transpose(a: Self) -> L::StorageRxC<T, C, R>;
    fn cast_from<U: SealedSupportedElement>(_a: L::StorageRxC<U, R, C>) -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn sub(a: Self, b: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
    fn div(a: Self, b: Self) -> Self;
    fn rem(a: Self, b: Self) -> Self;
    fn neg(a: Self) -> Self;
    fn not(a: Self) -> Self;
    fn bitand(a: Self, b: Self) -> Self;
    fn bitor(a: Self, b: Self) -> Self;
    fn bitxor(a: Self, b: Self) -> Self;
    fn shl(a: Self, b: Self) -> Self;
    fn shr(a: Self, b: Self) -> Self;
    fn eq(a: Self, b: Self) -> bool;
    fn ne(a: Self, b: Self) -> bool;
    fn index(a: &Self, index: (usize, usize)) -> Option<&T>;
    fn index_mut(a: &mut Self, index: (usize, usize)) -> Option<&mut T>;
    fn with_major_slices<U>(a: Self, f: impl FnOnce(&[&[T]]) -> U) -> U;
    fn matmul<const K: usize>(a: L::StorageRxC<T, R, K>, b: L::StorageRxC<T, K, C>) -> Self
    where
        Dimension<K>: SealedSupportedDimension;
    fn diagonal<const D: usize>(a: L::StorageRxC<T, D, D>) -> ConstStorage<T, D>
    where
        Dimension<D>: SealedSupportedDimension;
    fn inverse(a: Self) -> Self;
    fn determinant(a: Self) -> T;
}

macro_rules! impl_oriented_storage_ops {
    () => {
        const ZERO: Self = <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::ZERO;
        const ONE: Self = <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::ONE;
        const IDENTITY: Self = <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::IDENTITY;

        #[inline(always)]
        fn filled(value: T) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::filled(value)
        }
        #[inline(always)]
        fn transpose(a: Self) -> ConstStorage<T, C, R> {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::transpose(a)
        }
        #[inline(always)]
        fn cast_from<U: SealedSupportedElement>(a: ConstStorage<U, R, C>) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::cast_from(a)
        }
        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::add(a, b)
        }
        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::sub(a, b)
        }
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::mul(a, b)
        }
        #[inline(always)]
        fn div(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::div(a, b)
        }
        #[inline(always)]
        fn rem(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::rem(a, b)
        }
        #[inline(always)]
        fn neg(a: Self) -> Self { <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::neg(a) }
        #[inline(always)]
        fn not(a: Self) -> Self { <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::not(a) }
        #[inline(always)]
        fn bitand(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::bitand(a, b)
        }
        #[inline(always)]
        fn bitor(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::bitor(a, b)
        }
        #[inline(always)]
        fn bitxor(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::bitxor(a, b)
        }
        #[inline(always)]
        fn shl(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::shl(a, b)
        }
        #[inline(always)]
        fn shr(a: Self, b: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::shr(a, b)
        }
        #[inline(always)]
        fn eq(a: Self, b: Self) -> bool {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::eq(a, b)
        }
        #[inline(always)]
        fn ne(a: Self, b: Self) -> bool {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::ne(a, b)
        }
        fn with_major_slices<U>(a: Self, f: impl FnOnce(&[&[T]]) -> U) -> U {
            let major_vectors: [[T; R]; C] =
                <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::to_array(a);
            let slices: [&[T]; C] = major_vectors.each_ref().map(|vector| vector.as_slice());
            f(&slices)
        }
        #[inline(always)]
        fn diagonal<const D: usize>(a: ConstStorage<T, D, D>) -> ConstStorage<T, D>
        where
            Dimension<D>: SealedSupportedDimension,
        {
            <ConstStorage<T, D, D> as StorageOps<T, Dimension<D>, Dimension<D>>>::diagonal(a)
        }
        #[inline(always)]
        fn inverse(a: Self) -> Self {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::inverse(a)
        }
        #[inline(always)]
        fn determinant(a: Self) -> T {
            <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::determinant(a)
        }
    };
}

impl<S, T, const R: usize, const C: usize> OrientedStorageOps<T, R, C, ColumnMajor> for S
where
    S: StorageOps<T, Dimension<R>, Dimension<C>>,
    T: SealedSupportedElement,
    Dimension<R>: SealedSupportedDimension,
    Dimension<C>: SealedSupportedDimension,
{
    impl_oriented_storage_ops!();

    #[inline(always)]
    fn index(a: &Self, index: (usize, usize)) -> Option<&T> {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::index(a, index)
    }
    #[inline(always)]
    fn index_mut(a: &mut Self, index: (usize, usize)) -> Option<&mut T> {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::index_mut(a, index)
    }
    #[inline(always)]
    fn matmul<const K: usize>(a: ConstStorage<T, R, K>, b: ConstStorage<T, K, C>) -> Self
    where
        Dimension<K>: SealedSupportedDimension,
    {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::matmul::<K>(a, b)
    }
}
impl<S, T, const R: usize, const C: usize> OrientedStorageOps<T, C, R, RowMajor> for S
where
    S: StorageOps<T, Dimension<R>, Dimension<C>>,
    T: SealedSupportedElement,
    Dimension<R>: SealedSupportedDimension,
    Dimension<C>: SealedSupportedDimension,
{
    impl_oriented_storage_ops!();

    #[inline(always)]
    fn index(a: &Self, (row, column): (usize, usize)) -> Option<&T> {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::index(a, (column, row))
    }
    #[inline(always)]
    fn index_mut(a: &mut Self, (row, column): (usize, usize)) -> Option<&mut T> {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::index_mut(a, (column, row))
    }
    #[inline(always)]
    fn matmul<const K: usize>(a: ConstStorage<T, K, C>, b: ConstStorage<T, R, K>) -> Self
    where
        Dimension<K>: SealedSupportedDimension,
    {
        <Self as StorageOps<T, Dimension<R>, Dimension<C>>>::matmul::<K>(b, a)
    }
}

#[repr(C)]
pub struct XY<T: StoredVerbatim> {
    pub x: T,
    pub y: T,
}

#[repr(C)]
#[allow(clippy::upper_case_acronyms)] // Coordinate views intentionally mirror `.x/.y/.z` naming.
pub struct XYZ<T: StoredVerbatim> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[repr(C)]
#[allow(clippy::upper_case_acronyms)] // Coordinate views intentionally mirror `.x/.y/.z/.w` naming.
pub struct XYZW<T: StoredVerbatim> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T: StoredVerbatim> XY<T> {
    #[inline(always)]
    pub(crate) const fn from_array(array: &[T; 2]) -> &Self {
        const {
            assert!(size_of::<XY<T>>() == size_of::<[T; 2]>());
            assert!(align_of::<XY<T>>() <= align_of::<[T; 2]>());
        }
        // SAFETY: `XY<T>` is `repr(C)` and contains exactly two consecutive
        // `T` fields in array order. The assertions above establish equal
        // size and a compatible alignment for every monomorphization.
        // `StoredVerbatim` is sealed to scalar types with identical value
        // validity. The returned lifetime is inherited from `array`.
        unsafe { &*(array as *const [T; 2] as *const Self) }
    }
    #[inline(always)]
    pub(crate) const fn from_mut_array(array: &mut [T; 2]) -> &mut Self {
        const {
            assert!(size_of::<XY<T>>() == size_of::<[T; 2]>());
            assert!(align_of::<XY<T>>() <= align_of::<[T; 2]>());
        }
        // SAFETY: The layout and validity argument is the same as in
        // `from_array`. The exclusive reference is derived from `array`,
        // so its lifetime cannot outlive or be used alongside that borrow.
        unsafe { &mut *(array as *mut [T; 2] as *mut Self) }
    }
}

impl<T: StoredVerbatim> XYZ<T> {
    #[inline(always)]
    pub(crate) const fn from_array(array: &[T; 3]) -> &Self {
        const {
            assert!(size_of::<XYZ<T>>() == size_of::<[T; 3]>());
            assert!(align_of::<XYZ<T>>() <= align_of::<[T; 3]>());
        }
        // SAFETY: `XYZ<T>` is `repr(C)` and contains exactly three
        // consecutive `T` fields in array order. Size, alignment, validity,
        // and lifetime are guaranteed as described by `XY::from_array`.
        unsafe { &*(array as *const [T; 3] as *const Self) }
    }
    #[inline(always)]
    pub(crate) const fn from_mut_array(array: &mut [T; 3]) -> &mut Self {
        const {
            assert!(size_of::<XYZ<T>>() == size_of::<[T; 3]>());
            assert!(align_of::<XYZ<T>>() <= align_of::<[T; 3]>());
        }
        // SAFETY: The layout and validity argument is the same as in
        // `from_array`; borrowing `array` mutably preserves exclusivity.
        unsafe { &mut *(array as *mut [T; 3] as *mut Self) }
    }
}

impl<T: StoredVerbatim> XYZW<T> {
    #[inline(always)]
    pub(crate) const fn from_array(array: &[T; 4]) -> &Self {
        const {
            assert!(size_of::<XYZW<T>>() == size_of::<[T; 4]>());
            assert!(align_of::<XYZW<T>>() <= align_of::<[T; 4]>());
        }
        // SAFETY: `XYZW<T>` is `repr(C)` and contains exactly four
        // consecutive `T` fields in array order. Size, alignment, validity,
        // and lifetime are guaranteed as described by `XY::from_array`.
        unsafe { &*(array as *const [T; 4] as *const Self) }
    }
    #[inline(always)]
    pub(crate) const fn from_mut_array(array: &mut [T; 4]) -> &mut Self {
        const {
            assert!(size_of::<XYZW<T>>() == size_of::<[T; 4]>());
            assert!(align_of::<XYZW<T>>() <= align_of::<[T; 4]>());
        }
        // SAFETY: The layout and validity argument is the same as in
        // `from_array`; borrowing `array` mutably preserves exclusivity.
        unsafe { &mut *(array as *mut [T; 4] as *mut Self) }
    }
}
