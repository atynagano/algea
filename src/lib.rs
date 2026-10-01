#![doc = include_str!("../README.md")]

pub use support::{Element, MaskElement};

#[doc(hidden)]
pub mod __internal;
mod api;
mod arch;
/// Column-major matrices and their multiplication traits.
pub mod column_major;
pub(crate) mod private;
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

/// A fixed-size matrix whose storage orientation is selected by `L`.
///
/// The [`row_major::Matrix`] and [`column_major::Matrix`] aliases provide the
/// customary public names with `L` fixed to one orientation.
pub struct Matrix<T: Element<R, C>, const R: usize, const C: usize, L: MatrixLayout> {
    pub(crate) storage: <L as private::SealedMatrixLayout>::StorageRxC<T, R, C>,
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
use crate::support::MatrixLayout;
pub(crate) use impl_marker_trait;

macro_rules! impl_cast_from {
    ($self:ident from [$($t:ty),+]) => {
        $(impl CastFrom<$t> for $self {})+
    };
}

/// Marker traits describing scalar lane capabilities.
pub mod support {
    use crate::private;

    /// Selects the physical storage orientation of a [`Matrix`](crate::Matrix).
    #[expect(private_bounds)]
    pub trait MatrixLayout: private::SealedMatrixLayout {}

    /// Column-major matrix storage.
    pub enum ColumnMajor {}
    /// Row-major matrix storage.
    pub enum RowMajor {}

    impl MatrixLayout for ColumnMajor {}
    impl MatrixLayout for RowMajor {}

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
    pub trait Num:
        SupportedElement
        + core::cmp::PartialEq
        + core::cmp::PartialOrd
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Mul<Output = Self>
        + core::ops::Div<Output = Self>
        + core::ops::Rem<Output = Self>
    {
    }
    /// Groups the scalar bitwise operations required by integer lanes.
    pub trait Bitwise:
        SupportedElement
        + core::ops::Not<Output = Self>
        + core::ops::BitAnd<Output = Self>
        + core::ops::BitOr<Output = Self>
        + core::ops::BitXor<Output = Self>
        + core::ops::Shr<Output = Self>
        + core::ops::Shl<Output = Self>
    {
    }
    /// Marks signed scalar lane types.
    pub trait Signed: SupportedElement + core::ops::Neg<Output = Self> {}
    /// Marks unsigned scalar lane types.
    pub trait Unsigned: SupportedElement {}
    /// Associates an integer lane type with its signed and unsigned forms.
    pub trait Int: Num + Bitwise + core::cmp::Eq + core::cmp::Ord + core::hash::Hash {
        /// The signed type with the same lane width.
        type Signed: Sint<Unsigned = Self::Unsigned>;
        /// The unsigned type with the same lane width.
        type Unsigned: Uint<Signed = Self::Signed>;
    }
    /// Marks signed integer lane types.
    pub trait Sint: Signed + Int<Signed = Self> {}
    /// Marks unsigned integer lane types.
    pub trait Uint: Unsigned + Int<Unsigned = Self> {}
    /// Associates a floating-point lane type with its unsigned bit representation.
    pub trait Float: Signed + Num {
        /// The unsigned integer type containing this type's representation bits.
        type Bits: Uint;
    }
    impl_marker_trait!(Num for [f32, f64, i32, i64, u32, u64]);
    impl_marker_trait!(Bitwise for [i32, i64, u32, u64]);
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
    pub trait SupportedElement:
        Copy + Default + core::fmt::Debug + private::SealedSupportedElement
    {
        /// The signed integer scalar used to represent mask lanes.
        type Mask: SupportedMaskElement;
    }
    /// Marks a signed integer lane that serves as its own comparison mask type.
    pub trait SupportedMaskElement: Sint<Mask = Self> {}
    impl_marker_trait!(SupportedElement for [
        f32 { type Mask = i32; },
        f64 { type Mask = i64; },
        i32 { type Mask = i32; },
        i64 { type Mask = i64; },
        u32 { type Mask = i32; },
        u64 { type Mask = i64; },
    ]);
    impl_marker_trait!(SupportedMaskElement for [i32, i64]);

    /// Marks a dimension supported by the library's vectors and matrices.
    #[expect(private_bounds)]
    pub trait SupportedDimension: private::SealedSupportedDimension {}

    /// Represents a vector or matrix dimension as a type.
    pub enum Dimension<const D: usize> {}

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
    pub trait MaskElement<const D0: usize = 1, const D1: usize = 1>:
        SupportedMaskElement + Element<D0, D1>
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
        T: SupportedElement + SupportedMaskElement,
        Dimension<D0>: SupportedDimension,
        Dimension<D1>: SupportedDimension,
    {
    }

    mod integer_element_compile_checks {
        use super::*;
        use crate::Vector;

        fn _assert_element_mask_relationship<T: Element>() {
            _assert_mask_element_relationship::<T::Mask>();
        }
        fn _assert_mask_element_relationship<T: MaskElement>() {
            _assert_element_mask_relationship::<T>();
        }

        fn _assert_unsigned_cast_relationships<T: Uint + Element>(a: Vector<T, 1>) {
            // Verify that signed and unsigned casts preserve the corresponding element bounds.
            _ = a.cast_signed().cast_unsigned().cast_signed();
            _accept_integer_vector(a);
            _accept_integer_vector(a.cast_signed());
            _accept_integer_vector(a.cast_signed().cast_unsigned());
        }
        fn _accept_integer_vector<T: Int + Element>(a: Vector<T, 1>) {
            _assert_unsigned_cast_relationships(a.abs_diff(a))
        }
        fn _assert_float_bit_pattern_is_unsigned<T: Float + Element>(a: Vector<T, 1>) {
            _assert_unsigned_cast_relationships(a.to_bits())
        }
    }
}
