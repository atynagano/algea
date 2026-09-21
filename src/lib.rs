#![doc = include_str!("../README.md")]

pub use support::{Element, FloatElement, IntElement, MaskElement, SintElement, UintElement};

#[doc(hidden)]
pub mod __internal;
mod api;
mod arch;
/// Column-major matrices and their multiplication traits.
pub mod column_major;
/// Row-major matrices and their multiplication traits.
pub mod row_major;
mod swizzle;
mod utils;

#[cfg(all(algea_force_simd = "true", algea_force_simd = "false",))]
compile_error!("`algea_force_simd` cannot be both `true` and `false`");

#[cfg(all(algea_force_simd, not(any(algea_force_simd = "true", algea_force_simd = "false",)),))]
compile_error!("`algea_force_simd` must be `true` or `false`");

cfg_select! {
    // Keep this cfg name in sync if the crate is renamed.
    algea_force_simd = "true" => {
        mod simd;
        use simd::{definitions, kernels};
    }
    algea_force_simd = "false" => {
        mod non_simd;
        use non_simd::{definitions, kernels};
    }
    any(
        target_feature = "sse2",
        all(target_feature = "neon", target_arch = "aarch64"),
        target_feature = "simd128",
    ) => {
        mod simd;
        use simd::{definitions, kernels};
    }
    _ => {
        mod non_simd;
        use non_simd::{definitions, kernels};
    }
}

// TODO(vector-casts): redesign f32x4 from_bits/to_bits around integer vector types.

/// A fixed-size, orientation-independent vector.
///
/// In column-major expressions it acts as a column vector; in row-major
/// expressions it acts as a row vector. The supported lane types are `f32`, `f64`,
/// `i32`, `i64`, `u32`, and `u64`.
///
pub struct Vector<T: Element<D>, const D: usize> {
    pub(crate) storage: private::ConstStorage<T, D>,
}

/// A lane mask represented by signed integer lanes.
///
/// Boolean arrays can be converted into masks and read back.
///
/// ```
/// use algea::Mask;
/// let mask: Mask<i32, 2> = [true, false].into();
/// assert_eq!(mask.to_array(), [true, false]);
/// ```
pub struct Mask<T: MaskElement<D>, const D: usize> {
    // Mask lanes use the width of `T` and contain either all one bits or all zero bits.
    pub(crate) storage: utils::ConstMaskStorage<T, D>,
}

/// Constructs a vector by concatenating scalar and vector expressions.
#[macro_export]
macro_rules! vector {
    ($elem:expr; $n:expr) => {
        $crate::Vector::<_, $n>::splat($elem)
    };
    ($a:expr $(,)?) => {
        $crate::__internal::__IntoVector::__into_vector($a)
    };
    ($a:expr, $b:expr $(,)?) => {
        <$crate::__internal::__ConcatDispatch as $crate::__internal::__Concat2<_, _>>::__concat(
            $crate::__internal::__IntoVector::__into_vector($a),
            $crate::__internal::__IntoVector::__into_vector($b),
        )
    };
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        <$crate::__internal::__ConcatDispatch as $crate::__internal::__Concat3<_, _, _>>::__concat(
            $crate::__internal::__IntoVector::__into_vector($a),
            $crate::__internal::__IntoVector::__into_vector($b),
            $crate::__internal::__IntoVector::__into_vector($c),
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        <$crate::__internal::__ConcatDispatch as $crate::__internal::__Concat4<_, _, _, _>>::__concat(
            $crate::__internal::__IntoVector::__into_vector($a),
            $crate::__internal::__IntoVector::__into_vector($b),
            $crate::__internal::__IntoVector::__into_vector($c),
            $crate::__internal::__IntoVector::__into_vector($d),
        )
    };
}

fn _assert_vector_macro_compile<T: Element>(
    s: T,
    a: Vector<T, 1>,
    b: Vector<T, 2>,
    c: Vector<T, 3>,
) {
    _ = vector![s];
    _ = vector![s, s];
    _ = vector![s, s, s];
    _ = vector![s, s, s, s];
    _ = vector![a];
    _ = vector![a, a];
    _ = vector![a, a, a];
    _ = vector![a, a, a, a];
    _ = vector![b];
    _ = vector![c];
    _ = vector![c, a];
    _ = vector![a, c];
    _ = vector![b, a];
    _ = vector![a, b];
    _ = vector![b, a, a];
    _ = vector![a, b, a];
    _ = vector![a, a, b];
}

// Floating-point lanes require dedicated min/max semantics because `f32` does not implement
// `Ord`, while integer min/max comes from `Ord`.
/// Provides lane-wise ordering operations.
pub trait EachOrd {
    /// Returns the lane-wise maximum of `self` and `other`.
    fn each_max(self, other: Self) -> Self;
    /// Returns the lane-wise minimum of `self` and `other`.
    fn each_min(self, other: Self) -> Self;
    /// Restricts every lane to the corresponding inclusive range.
    ///
    /// # Panics
    ///
    /// Panics if any lane of `min` is greater than the corresponding lane of
    /// `max`.
    fn each_clamp(self, min: Self, max: Self) -> Self;
}

// std::simd::Select
/// Selects lanes from two values according to a mask.
pub trait Select<T> {
    /// Chooses a lane from `true_values` when the corresponding mask lane is
    /// true, and from `false_values` otherwise.
    fn select(self, true_values: T, false_values: T) -> T;
}

macro_rules! impl_marker_trait {
    ($trait:ident for [$($t:ty $({ $($item:item)* })?),+ $(,)?]) => {
        $(
            impl $trait for $t {
                $($($item)*)?
            }
        )+
    };
}

macro_rules! impl_cast_from {
    ($self:ident from [$($t:ty),+]) => {
        $(impl CastFrom<$t> for $self {})+
    };
}

/// Marker traits describing scalar lane capabilities.
pub mod marker {
    use crate::{private, support::SupportedElement};

    // This models Rust `as` conversions rather than `std::simd::SimdCast`: conversions involving
    // `bool` or `char` can be one-way rather than forming a symmetric pair.
    /// Marks a scalar conversion accepted by [`Vector::cast`](crate::Vector::cast)
    /// and the corresponding matrix methods.
    #[expect(private_bounds)]
    pub trait CastFrom<T>: private::Sealed {}
    impl_cast_from!(f32 from [f32, f64, i32, i64, u32, u64]);
    impl_cast_from!(f64 from [f32, f64, i32, i64, u32, u64]);
    impl_cast_from!(i32 from [f32, f64, i32, i64, u32, u64]);
    impl_cast_from!(i64 from [f32, f64, i32, i64, u32, u64]);
    impl_cast_from!(u32 from [f32, f64, i32, i64, u32, u64]);
    impl_cast_from!(u64 from [f32, f64, i32, i64, u32, u64]);

    // TODO(extra-type-support): Separate comparison bounds from numeric operations before adding
    // `char`, which is ordered but not numeric.
    /// Groups the scalar arithmetic operations required by numeric lanes.
    #[expect(private_bounds)]
    pub trait NumOps:
        Sized
        + private::Sealed
        + core::cmp::PartialEq
        + core::cmp::PartialOrd
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Mul<Output = Self>
        + core::ops::Div<Output = Self>
        + core::ops::Rem<Output = Self>
    {
    }
    impl<T> NumOps for T where
        T: private::Sealed
            + core::cmp::PartialEq
            + core::cmp::PartialOrd
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::Div<Output = T>
            + core::ops::Rem<Output = T>
    {
    }
    /// Groups the scalar bitwise operations required by integer lanes.
    #[expect(private_bounds)]
    pub trait BitOps:
        Sized
        + private::Sealed
        + core::ops::Not<Output = Self>
        + core::ops::BitAnd<Output = Self>
        + core::ops::BitOr<Output = Self>
        + core::ops::BitXor<Output = Self>
        + core::ops::Shr<Output = Self>
        + core::ops::Shl<Output = Self>
    {
    }
    impl<T> BitOps for T where
        T: private::Sealed
            + core::ops::Not<Output = Self>
            + core::ops::BitAnd<Output = Self>
            + core::ops::BitOr<Output = Self>
            + core::ops::BitXor<Output = Self>
            + core::ops::Shr<Output = Self>
            + core::ops::Shl<Output = Self>
    {
    }

    /// Marks signed scalar lane types.
    #[expect(private_bounds)]
    pub trait Signed: private::Sealed + Copy + core::ops::Neg<Output = Self> {}
    /// Marks unsigned scalar lane types.
    #[expect(private_bounds)]
    pub trait Unsigned: private::Sealed + Copy {}
    /// Associates an integer lane type with its signed and unsigned forms.
    #[expect(private_bounds)]
    pub trait Int: private::Sealed + NumOps + BitOps {
        /// The signed type with the same lane width.
        type Signed: Sint;
        /// The unsigned type with the same lane width.
        type Unsigned: Uint;
    }
    /// Marks signed integer lane types.
    pub trait Sint: Signed + Int<Signed = Self, Unsigned: Unsigned + Int<Signed = Self>> {}
    /// Marks unsigned integer lane types.
    pub trait Uint: Unsigned + Int<Unsigned = Self, Signed: Signed + Int<Unsigned = Self>> {}
    /// Associates a floating-point lane type with its unsigned bit representation.
    pub trait Float: Signed + NumOps {
        /// The unsigned integer type containing this type's representation bits.
        type Bits: Uint;
    }
    impl_marker_trait!(Signed for [f32, f64, i32, i64]);
    impl_marker_trait!(Unsigned for [u32, u64]);
    impl_marker_trait!(Int for [
        i32 { type Signed = Self; type Unsigned = u32; },
        i64 { type Signed = Self; type Unsigned = u64; },
        u32 { type Signed = i32; type Unsigned = Self; },
        u64 { type Signed = i64; type Unsigned = Self; },
    ]);
    impl_marker_trait!(Sint for [i32, i64]);
    impl_marker_trait!(Uint for [u32, u64]);
    impl_marker_trait!(Float for [
        f32 { type Bits = u32; },
        f64 { type Bits = u64; },
    ]);

    /// Marks scalar types stored in a directly referenceable form.
    ///
    /// This trait is required by reference-based APIs such as `AsRef`, [`Deref`],
    /// and [`Index`], including their mutable counterparts. It guarantees that
    /// active vector and matrix elements are retained as properly aligned,
    /// contiguous `Self` values that can be safely borrowed.
    ///
    /// Every currently supported element type implements this trait. Keeping it
    /// separate from [`Element`](crate::Element) allows future element types to use
    /// packed or otherwise non-referenceable storage while still supporting APIs
    /// that operate by value.
    ///
    /// [`Deref`]: core::ops::Deref
    /// [`Index`]: core::ops::Index
    #[expect(private_bounds)]
    pub trait StoredVerbatim: private::Sealed {}
    impl_marker_trait!(StoredVerbatim for [f32, f64, i32, i64, u32, u64]);

    // TODO(api-cleanup): Audit every public trait for a sealed boundary before release.
    // TODO(api-cleanup): Define a consistent rule for requiring `Copy` only when operations need
    // value duplication, rather than solely for downstream convenience.
    /// Associates a scalar lane with the scalar type used by its comparison mask.
    #[expect(private_bounds)]
    pub trait Lane: Copy + private::Sealed {
        /// The signed integer scalar used to represent mask lanes.
        type Mask: MaskLane + SupportedElement;
    }
    /// Marks a signed integer lane that serves as its own comparison mask type.
    pub trait MaskLane: Sint + Lane<Mask = Self> {}
    // The required bound depends on the shape, so it cannot be expressed as
    // `Float: HasBits<Bits: SimdElement<D>>`; `FloatElement<D>` carries it instead.
    impl_marker_trait!(Lane for [
        f32 { type Mask = i32; },
        f64 { type Mask = i64; },
        i32 { type Mask = i32; },
        i64 { type Mask = i64; },
        u32 { type Mask = i32; },
        u64 { type Mask = i64; },
    ]);
    impl_marker_trait!(MaskLane for [i32, i64]);
}

/// Dimension-dependent traits used to express supported vector and matrix types.
pub mod support {
    use crate::{Vector, marker::*, private};

    /// Marks a scalar lane supported by the library's vectors and matrices.
    #[expect(private_bounds)]
    pub trait SupportedElement: Lane + private::SealedSupportedElement {}

    /// Marks a dimension supported by the library's vectors and matrices.
    #[expect(private_bounds)]
    pub trait SupportedDimension: private::SealedSupportedDimension {}

    /// Represents a vector or matrix dimension as a type.
    pub enum Dimension<const D: usize> {}

    impl_marker_trait!(SupportedElement for [f32, f64, i32, i64, u32, u64]);
    impl_marker_trait! {
        SupportedDimension for [
            Dimension<1> {},
            Dimension<2> {},
            Dimension<3> {},
            Dimension<4> {},
        ]
    }
    /// Marks a scalar lane supported for the given vector dimension or matrix
    /// shape.
    ///
    /// `D0` is the vector length when `D1` is left at its default. For matrices,
    /// `D0` and `D1` are the row and column counts.
    #[expect(private_bounds)]
    pub trait Element<const D0: usize = 1, const D1: usize = 1>:
        SupportedElement
        + private::SealedDimensionWitness<
            D0,
            Dimension = Dimension<D0>,
            Dimension: SupportedDimension,
        > + private::SealedDimensionWitness<
            D1,
            Dimension = Dimension<D1>,
            Dimension: SupportedDimension,
        >
    {
    }
    /// Marks a signed integer scalar supported as a mask for the given shape.
    #[expect(private_bounds)]
    pub trait MaskElement<const D0: usize = 1, const D1: usize = 1>:
        SupportedElement
        + MaskLane
        + private::SealedDimensionWitness<
            D0,
            Dimension = Dimension<D0>,
            Dimension: SupportedDimension,
        > + private::SealedDimensionWitness<
            D1,
            Dimension = Dimension<D1>,
            Dimension: SupportedDimension,
        >
    {
    }
    /// Marks a floating-point scalar supported for the given shape, including
    /// its corresponding integer representation.
    pub trait FloatElement<const D0: usize = 1, const D1: usize = 1>:
        Element<D0, D1> + Float<Bits: SupportedElement + Uint<Signed: SupportedElement>>
    {
    }
    // `Mask` could be fixed to an integer type's signed counterpart, but doing the same through a
    // float's bit type would overconstrain this marker, so the association remains explicit.
    /// Marks an integer scalar supported for the given shape.
    pub trait IntElement<const D0: usize = 1, const D1: usize = 1>:
        Element<D0, D1>
        + Int<
            Signed: SupportedElement + Sint<Unsigned: SupportedElement>,
            Unsigned: SupportedElement + Uint<Signed: SupportedElement>,
        >
    {
    }
    /// Marks a signed integer scalar supported for the given shape.
    pub trait SintElement<const D0: usize = 1, const D1: usize = 1>:
        Element<D0, D1> + Sint<Unsigned: SupportedElement>
    {
    }
    /// Marks an unsigned integer scalar supported for the given shape.
    pub trait UintElement<const D0: usize = 1, const D1: usize = 1>:
        Element<D0, D1> + Uint<Signed: SupportedElement>
    {
    }

    impl<T, const D0: usize, const D1: usize> Element<D0, D1> for T
    where
        T: SupportedElement,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }
    impl<T, const D0: usize, const D1: usize> MaskElement<D0, D1> for T
    where
        T: SupportedElement + MaskLane,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }
    impl<T, const D0: usize, const D1: usize> FloatElement<D0, D1> for T
    where
        T: SupportedElement + Float<Bits: SupportedElement + Uint<Signed: SupportedElement>>,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }
    impl<T, const D0: usize, const D1: usize> IntElement<D0, D1> for T
    where
        T: SupportedElement
            + Int<
                Signed: SupportedElement + Sint<Unsigned: SupportedElement>,
                Unsigned: SupportedElement + Uint<Signed: SupportedElement>,
            >,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }
    impl<T, const D0: usize, const D1: usize> SintElement<D0, D1> for T
    where
        T: SupportedElement + Sint<Unsigned: SupportedElement>,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }
    impl<T, const D0: usize, const D1: usize> UintElement<D0, D1> for T
    where
        T: SupportedElement + Uint<Signed: SupportedElement>,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }

    mod integer_element_compile_checks {
        use super::*;

        fn _assert_element_mask_relationship<T: Element>() {
            _assert_mask_element_relationship::<T::Mask>();
        }
        fn _assert_mask_element_relationship<T: MaskElement>() {
            _assert_element_mask_relationship::<T>();
        }

        fn _assert_unsigned_cast_relationships<T: UintElement>(a: Vector<T, 1>) {
            // Verify that signed and unsigned casts preserve the corresponding element bounds.
            _ = a.cast_signed().cast_unsigned().cast_signed();
            _accept_integer_vector(a);
            _accept_integer_vector(a.cast_signed());
            _accept_integer_vector(a.cast_signed().cast_unsigned());
        }
        fn _accept_integer_vector<T: IntElement>(a: Vector<T, 1>) {
            _assert_unsigned_cast_relationships(a.abs_diff(a))
        }
        fn _assert_float_bit_pattern_is_unsigned<T: FloatElement>(a: Vector<T, 1>) {
            _assert_unsigned_cast_relationships(a.to_bits())
        }

        // TODO(mask-integer-boundary): reconsider whether `Mask::to_vector`
        // should provide the signed-integer API when its lane is known only as a
        // `MaskElement`. Making `MaskElement` imply `SintElement` currently
        // complicates the associated type bounds substantially, and future mask
        // element types may intentionally have no `SintElement` implementation.
    }
}

pub(crate) mod private {
    pub(crate) use crate::definitions::{SealedStorageElement, SealedSupportedDimension};
    use crate::{
        marker::{Float, Int, Lane, StoredVerbatim},
        support::{Dimension, SupportedDimension, SupportedElement},
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
    >: Copy + ArithOps<Scalar = T>
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
        fn cast_from<U: SealedSupportedElement>(_a: DimStorage<U, R, C>) -> Self {
            unimplemented!()
        }

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
            T: Lane<Mask: SealedSupportedElement>,
        {
            unimplemented!()
        }
        /// Unwraps a mask into the vector storage of the same element type, for `Mask::to_vector`.
        ///
        /// Implemented for the mask element types alone, where the two are the same type.
        fn from_mask(_mask: DimMaskStorage<T::Mask, R, C>) -> Self
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            unimplemented!()
        }
        fn select_mask(
            _mask: DimMaskStorage<T::Mask, R, C>,
            _true_values: Self,
            _false_values: Self,
        ) -> Self
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            unimplemented!()
        }
        fn select_any_mask<Mask: SealedSupportedElement>(
            _mask: DimMaskStorage<Mask, R, C>,
            _true_values: Self,
            _false_values: Self,
        ) -> Self
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            unimplemented!()
        }
        #[expect(dead_code)]
        fn select_u64(_mask: u64, _true_values: Self, _false_values: Self) -> Self {
            unimplemented!()
        }

        fn cast_i32(_mask: DimMaskStorage<T, R, C>) -> DimMaskStorage<i32, R, C> {
            unimplemented!()
        }
        fn cast_i64(_mask: DimMaskStorage<T, R, C>) -> DimMaskStorage<i64, R, C> {
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
        #[expect(dead_code)]
        fn to_bitmask(_mask: DimMaskStorage<T, R, C>) -> u64 { unimplemented!() }

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

        fn each_eq(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::eq_(a, b))
        }
        fn each_ne(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::ne_(a, b))
        }
        fn each_lt(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::lt_(a, b))
        }
        fn each_le(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::le_(a, b))
        }
        fn each_gt(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::gt_(a, b))
        }
        fn each_ge(a: Self, b: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
        {
            Self::substantiate_mask(ArithOps::ge_(a, b))
        }
        #[expect(dead_code)]
        fn is_nan(_a: Self) -> DimMaskStorage<T::Mask, R, C>
        where
            T: Lane<Mask: SealedSupportedElement>,
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
}
