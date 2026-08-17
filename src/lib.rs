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
        use simd::kernels;
    }
    algea_force_simd = "false" => {
        mod non_simd;
        use non_simd::kernels;
    }
    any(
        target_feature = "sse2",
        all(target_feature = "neon", target_arch = "aarch64"),
        target_feature = "simd128",
    ) => {
        mod simd;
        use simd::kernels;
    }
    _ => {
        mod non_simd;
        use non_simd::kernels;
    }
}

// TODO(vector-casts): redesign f32x4 from_bits/to_bits around integer vector types.

/// A fixed-size, orientation-independent vector.
///
/// In column-major expressions it acts as a column vector; in row-major
/// expressions it acts as a row vector. Types other than `f32`, `i32`, and `u32`
/// do not currently implement [`Element`].
///
pub struct Vector<T: Element<D>, const D: usize> {
    pub(crate) storage: <T as private::SealedElement<D, 1>>::Storage,
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
    pub(crate) storage: utils::MaskStorage<<T as private::SealedElement<D, 1>>::Storage>,
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
    use crate::private;

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
        type Mask;
    }
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
}

/// Dimension-dependent traits used to express supported vector and matrix types.
pub mod support {
    use crate::{Vector, column_major, marker::*, private, row_major};

    /// Marks a scalar supported by both matrix storage orientations for the given
    /// shape.
    #[expect(private_bounds)]
    pub trait SupportedMatrixElement<const R: usize, const C: usize>:
        private::SealedElement<R, C> + private::SealedElement<C, R>
    {
    }
    macro_rules! impl_element {
        ([$($t:ty),+], $r:tt) => {
            $(impl_element!{ type=$t, $r, $r })+
        };
        (type=$t:ty, [$($r:literal),+], $c:tt) => {
            $(impl_element!{ type=$t, row=$r, $c })+
        };
        (type=$t:ty, row=$r:literal, [$($c:literal),+]) => {
            $(impl SupportedMatrixElement<$r, $c> for $t {})+
        };
    }
    impl_element!([f32, f64, i32, i64, u32, u64], [1, 2, 3, 4]);

    macro_rules! impl_element2 {
        ({$($acc:tt)*}, $r:tt,) => {
            impl_element2! { @a { $($acc)* }, $r, $r }
        };
        (@a {$($acc:tt)*}, [], $c:tt) => {
            $($acc)* {}
        };
        (@a {$($acc:tt)*}, [$r0:expr $(, $r:expr)*], $c:tt) => {
            impl_element2! { @b {$($acc)*}, $r0, [$($r),*], $c }
        };
        (@b {$($acc:tt)*}, $r0:expr, $r:tt, [$($c:expr),*]) => {
            impl_element2!{ @a {
                $($acc)* $(+ SupportedMatrixElement<$r0, $c>)*
            }, $r, [$($c),*] }
        };
    }
    macro_rules! impl_element3 {
        ({$($acc:tt)*}, $dims:tt $(,)?) => {
            impl_element3! { @a {$($acc)*}, $dims, $dims, $dims }
        };
        (@a {$($acc:tt)*}, [], $bs:tt, $cs:tt) => {
            $($acc)* {}
        };
        (@a {$($acc:tt)*}, [$a:expr $(, $as:expr)*], $bs:tt, $cs:tt) => {
            impl_element3! { @b {$($acc)*}, $a, [$($as),*], $bs, $bs, $cs}
        };
        (@b {$($acc:tt)*}, $a:tt, $as:tt, [], $bs:tt, $cs:tt) => {
            impl_element3! { @a {$($acc)*}, $as, $bs, $cs }
        };
        (@b {$($acc:tt)*}, $a:expr, $as:tt, [$b:expr $(, $bs_remaining:expr)*], $bs:tt, [$($c:expr),*]) => {
            impl_element3! { @b {
                    $($acc)* $(+ row_major::MatrixProduct<$a, $b, $c> + column_major::MatrixProduct<$a, $b, $c>)*
                }, $a, $as, [$($bs_remaining),*], $bs, [$($c),*]
            }
        };
    }

    impl_element2! {{
        /// Marks a scalar lane supported for the given vector dimension or matrix
        /// shape.
        ///
        /// `D0` is the vector length when `D1` is left at its default. For matrices,
        /// `D0` and `D1` are the row and column counts.
        pub trait Element<const D0: usize = 1, const D1: usize = 1>:
            Lane<Mask: MaskElement<D0, D1>>
        }, [1, 2, 3, 4, D0, D1],
    }
    impl_element2! {{
        impl<T, const D0: usize, const D1: usize> Element<D0, D1> for T
            where T: Lane<Mask: MaskElement<D0, D1>>
        }, [1, 2, 3, 4, D0, D1],
    }

    // TODO(mask-representation): Reconsider the `SintElement` requirement if a future mask lane
    // has no signed-integer representation.
    // `Element` cannot be a supertrait here because that creates a recursive trait dependency.
    impl_element2! {{
        /// Marks a signed integer scalar supported as a mask for the given shape.
        pub trait MaskElement<const D0: usize = 1, const D1: usize = 1>:
            Sint + Lane<Mask = Self>
        }, [1, 2, 3, 4, D0, D1],
    }
    impl_element2! {{
        impl<T, const D0: usize, const D1: usize> MaskElement<D0, D1> for T
            where T: Sint + Lane<Mask = Self>
        }, [1, 2, 3, 4, D0, D1],
    }

    impl_element3! {{
        /// Marks a floating-point scalar supported for the given shape, including
        /// its corresponding integer representation.
        pub trait FloatElement<const D0: usize = 1, const D1: usize = 1>:
            Element<D0, D1> + Float<Bits: Uint<Signed: Element<D0, D1>> + Element<D0, D1>>
        }, [1, 2, 3, 4, D0, D1]
    }
    impl_element3! {{
        impl<T, const D0: usize, const D1: usize> FloatElement<D0, D1> for T
            where T: Element<D0, D1> + Float<Bits: Uint<Signed: Element<D0, D1>> + Element<D0, D1>>
        }, [1, 2, 3, 4, D0, D1]
    }

    // `Mask` could be fixed to an integer type's signed counterpart, but doing the same through a
    // float's bit type would overconstrain this marker, so the association remains explicit.
    /// Marks an integer scalar supported for the given shape.
    pub trait IntElement<const D0: usize = 1, const D1: usize = 1>:
        Int<Signed: SintElement<D0, D1>, Unsigned: UintElement<D0, D1>> + Element<D0, D1>
    {
    }
    /// Marks a signed integer scalar supported for the given shape.
    pub trait SintElement<const D0: usize = 1, const D1: usize = 1>:
        Sint<Unsigned: Element<D0, D1>> + Element<D0, D1>
    {
    }
    /// Marks an unsigned integer scalar supported for the given shape.
    pub trait UintElement<const D0: usize = 1, const D1: usize = 1>:
        Uint<Signed: Element<D0, D1>> + Element<D0, D1>
    {
    }
    impl<T, const D0: usize, const D1: usize> IntElement<D0, D1> for T where
        T: Int<Signed: SintElement<D0, D1>, Unsigned: UintElement<D0, D1>> + Element<D0, D1>
    {
    }
    impl<T, const D0: usize, const D1: usize> SintElement<D0, D1> for T where
        T: Sint<Unsigned: Element<D0, D1>> + Element<D0, D1>
    {
    }
    impl<T, const D0: usize, const D1: usize> UintElement<D0, D1> for T where
        T: Uint<Signed: Element<D0, D1>> + Element<D0, D1>
    {
    }

    mod integer_element_compile_checks {
        use super::*;

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
    use crate::{
        marker::{Float, Lane, StoredVerbatim},
        utils::MaskStorage,
    };

    pub(crate) trait Fmt {
        fn fmt<T: SealedElement<M, N> + core::fmt::Debug, const M: usize, const N: usize>(
            storage: impl crate::utils::Store<T::Storage>,
        ) -> impl core::fmt::Debug;
    }

    pub(crate) enum VectorFmt {}

    pub(crate) enum Indices2<const I0: usize, const I1: usize> {}
    pub(crate) enum Indices3<const I0: usize, const I1: usize, const I2: usize> {}
    pub(crate) enum Indices4<const I0: usize, const I1: usize, const I2: usize, const I3: usize> {}

    pub(crate) trait SwizzleDispatch<T, const M: usize, const N: usize> {
        // The non-SIMD backend never calls this (see `src/non_simd/utils.rs`), so it is unused
        // under that backend.
        #[allow(dead_code)]
        fn dispatch(v: <T as SealedElement<M, 1>>::Storage) -> <T as SealedElement<N, 1>>::Storage
        where
            T: SealedElement<M, 1> + SealedElement<N, 1>;
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
        fn fmt<T: SealedElement<M, N> + core::fmt::Debug, const M: usize, const N: usize>(
            storage: impl crate::utils::Store<T::Storage>,
        ) -> impl core::fmt::Debug {
            let array = T::to_array(storage.store());
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
    // `ArithPrimitive`. Their scalar capability and arithmetic boundaries still
    // overlap; consolidate them once their marker bounds and all backend call
    // sites can be unified without broadening the public API.
    pub(crate) trait Sealed: Sized {
        // Scalar implementations used through `SealedElement` must override `TYPE`
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

    // M, N describe the row and column dimensions of the private column-major storage.
    pub(crate) trait SealedElement<const M: usize, const N: usize>: Sealed {
        type Storage: Copy;

        const ZERO: Self::Storage;
        const ONE: Self::Storage;
        // Defaults are traps for operation and shape combinations that are not
        // exposed by the public marker bounds. Implementations in `simd` and
        // `non_simd` override every member reachable by a released API; tests
        // cover all supported scalar types and dimensions.
        const IDENTITY: Self::Storage = unimplemented!();
        const POS_X: Self::Storage = unimplemented!();
        const POS_Y: Self::Storage = unimplemented!();
        const POS_Z: Self::Storage = unimplemented!();
        const POS_W: Self::Storage = unimplemented!();
        const NEG_X: Self::Storage = unimplemented!();
        const NEG_Y: Self::Storage = unimplemented!();
        const NEG_Z: Self::Storage = unimplemented!();
        const NEG_W: Self::Storage = unimplemented!();

        fn map2(
            a: Self::Storage,
            b: Self::Storage,
            f: impl FnMut(Self, Self) -> Self,
        ) -> Self::Storage;
        fn index(_a: &Self::Storage, _index: (usize, usize)) -> Option<&Self> { unimplemented!() }
        fn index_mut(_a: &mut Self::Storage, _index: (usize, usize)) -> Option<&mut Self> {
            unimplemented!()
        }
        //noinspection RsSelfConvention
        fn as_array_first(_a: &Self::Storage) -> &[Self; M] { unimplemented!() }
        //noinspection RsSelfConvention
        fn as_mut_array_first(_a: &mut Self::Storage) -> &mut [Self; M] { unimplemented!() }
        //noinspection RsSelfConvention
        fn to_array(a: Self::Storage) -> [[Self; M]; N];
        fn from_array(array: [[Self; M]; N]) -> Self::Storage;
        fn from_vecs(array: [crate::Vector<Self, M>; N]) -> <Self as SealedElement<M, N>>::Storage
        where
            Self: crate::Element<M>;
        fn filled(value: Self) -> Self::Storage;
        fn substantiate_f32(_a: Self::Storage) -> <f32 as SealedElement<M, N>>::Storage
        where
            f32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn substantiate_f64(_a: Self::Storage) -> <f64 as SealedElement<M, N>>::Storage
        where
            f64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn substantiate_i32(_a: Self::Storage) -> <i32 as SealedElement<M, N>>::Storage
        where
            i32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn substantiate_i64(_a: Self::Storage) -> <i64 as SealedElement<M, N>>::Storage
        where
            i64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn substantiate_u32(_a: Self::Storage) -> <u32 as SealedElement<M, N>>::Storage
        where
            u32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn substantiate_u64(_a: Self::Storage) -> <u64 as SealedElement<M, N>>::Storage
        where
            u64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_f32(_a: <f32 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            f32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_f64(_a: <f64 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            f64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_i32(_a: <i32 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            i32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_i64(_a: <i64 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            i64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_u32(_a: <u32 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            u32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from_u64(_a: <u64 as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            u64: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_from<U: SealedElement<M, N>>(
            a: <U as SealedElement<M, N>>::Storage,
        ) -> Self::Storage;

        fn cast_signed(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self::Signed as SealedElement<M, N>>::Storage
        where
            Self: crate::IntElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_unsigned(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self::Unsigned as SealedElement<M, N>>::Storage
        where
            Self: crate::IntElement<M, N>,
        {
            unimplemented!()
        }
        fn swizzle2<const I0: usize, const I1: usize>(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<2, 1>>::Storage
        where
            Self: SealedElement<2, 1>,
            Indices2<I0, I1>: SwizzleDispatchAny<2>,
        {
            unimplemented!()
        }
        fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<3, 1>>::Storage
        where
            Self: SealedElement<3, 1>,
            Indices3<I0, I1, I2>: SwizzleDispatchAny<3>,
        {
            unimplemented!()
        }
        fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<4, 1>>::Storage
        where
            Self: SealedElement<4, 1>,
            Indices4<I0, I1, I2, I3>: SwizzleDispatchAny<4>,
        {
            unimplemented!()
        }

        fn select_mask(
            _mask: MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>,
            _true_values: <Self as SealedElement<M, N>>::Storage,
            _false_values: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<M, N>>::Storage
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn select_any_mask<Mask>(
            _mask: MaskStorage<<Mask as SealedElement<M, N>>::Storage>,
            _true_values: <Self as SealedElement<M, N>>::Storage,
            _false_values: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<M, N>>::Storage
        where
            Mask: SealedElement<M, N>,
        {
            unimplemented!()
        }
        #[expect(dead_code)]
        fn select_u64(
            _mask: u64,
            _true_values: Self::Storage,
            _false_values: Self::Storage,
        ) -> Self::Storage {
            unimplemented!()
        }

        fn cast_i32(
            _a: MaskStorage<Self::Storage>,
        ) -> MaskStorage<<i32 as SealedElement<M, N>>::Storage>
        where
            i32: SealedElement<M, N>,
        {
            unimplemented!()
        }
        fn cast_i64(
            _a: MaskStorage<Self::Storage>,
        ) -> MaskStorage<<i64 as SealedElement<M, N>>::Storage>
        where
            i64: SealedElement<M, N>,
        {
            unimplemented!()
        }

        #[allow(clippy::wrong_self_convention)]
        fn to_bool_array(_a: MaskStorage<Self::Storage>) -> [[bool; M]; N] { unimplemented!() }
        fn from_bool_array(_array: [[bool; M]; N]) -> MaskStorage<Self::Storage> {
            unimplemented!()
        }
        fn all(_mask: MaskStorage<Self::Storage>) -> bool { unimplemented!() }
        fn any(_mask: MaskStorage<Self::Storage>) -> bool { unimplemented!() }
        #[allow(clippy::wrong_self_convention)]
        #[expect(dead_code)]
        fn to_bitmask(_mask: MaskStorage<Self::Storage>) -> u64 { unimplemented!() }

        fn canonical_not(_a: MaskStorage<Self::Storage>) -> MaskStorage<Self::Storage> {
            unimplemented!()
        }
        fn canonical_bitand(
            _a: MaskStorage<Self::Storage>,
            _b: MaskStorage<Self::Storage>,
        ) -> MaskStorage<Self::Storage> {
            unimplemented!()
        }
        fn canonical_bitor(
            _a: MaskStorage<Self::Storage>,
            _b: MaskStorage<Self::Storage>,
        ) -> MaskStorage<Self::Storage> {
            unimplemented!()
        }
        fn canonical_bitxor(
            _a: MaskStorage<Self::Storage>,
            _b: MaskStorage<Self::Storage>,
        ) -> MaskStorage<Self::Storage> {
            unimplemented!()
        }
        fn canonical_select_any_mask<Mask>(
            _mask: MaskStorage<<Mask as SealedElement<M, N>>::Storage>,
            _true_values: MaskStorage<<Self as SealedElement<M, N>>::Storage>,
            _false_values: MaskStorage<<Self as SealedElement<M, N>>::Storage>,
        ) -> MaskStorage<<Self as SealedElement<M, N>>::Storage>
        where
            Mask: SealedElement<M, N>,
        {
            unimplemented!()
        }

        fn each_eq(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn each_ne(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn each_lt(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn each_le(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn each_gt(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn each_ge(
            _a: Self::Storage,
            _b: Self::Storage,
        ) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }

        #[expect(dead_code)]
        fn is_nan(_a: Self::Storage) -> MaskStorage<<Self::Mask as SealedElement<M, N>>::Storage>
        where
            Self: Lane<Mask: SealedElement<M, N>>,
        {
            unimplemented!()
        }

        fn each_max(a: Self::Storage, b: Self::Storage) -> Self::Storage;
        fn each_min(a: Self::Storage, b: Self::Storage) -> Self::Storage;
        fn each_clamp<F: Fmt>(
            a: Self::Storage,
            min: Self::Storage,
            max: Self::Storage,
        ) -> Self::Storage;
        fn eq(a: Self::Storage, b: Self::Storage) -> bool;
        fn ne(a: Self::Storage, b: Self::Storage) -> bool;
        fn add(a: Self::Storage, b: Self::Storage) -> Self::Storage;
        fn sub(a: Self::Storage, b: Self::Storage) -> Self::Storage;
        fn mul(a: Self::Storage, b: Self::Storage) -> Self::Storage;
        fn div(_a: Self::Storage, _b: Self::Storage) -> Self::Storage;
        // TODO(integer-vector): separate sqrt and isqrt semantics in public traits.
        fn sqrt(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn transpose(
            a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<N, M>>::Storage
        where
            Self: SealedElement<N, M>;

        fn from_bits(_a: <<Self as Float>::Bits as SealedElement<M, N>>::Storage) -> Self::Storage
        where
            Self: Float<Bits: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        #[allow(clippy::wrong_self_convention)]
        fn to_bits(_a: Self::Storage) -> <<Self as Float>::Bits as SealedElement<M, N>>::Storage
        where
            Self: Float<Bits: SealedElement<M, N>>,
        {
            unimplemented!()
        }
        fn floor(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn ceil(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn round(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn round_ties_even(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn trunc(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn fract(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn neg(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn abs(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        #[expect(dead_code)]
        fn signum(_a: Self::Storage) -> Self::Storage { unimplemented!() }

        fn rem(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }
        fn not(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        fn bitand(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }
        fn bitor(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }
        fn bitxor(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }
        fn shl(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }
        fn shr(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }

        fn reduce_sum(_a: Self::Storage) -> Self { unimplemented!() }
        #[inline(always)]
        fn dot(a: Self::Storage, b: Self::Storage) -> Self { Self::reduce_sum(Self::mul(a, b)) }
        #[expect(dead_code)]
        fn cross(_a: Self::Storage, _b: Self::Storage) -> Self::Storage { unimplemented!() }

        fn vector_concat_1_1(
            a: <Self as SealedElement<1, 1>>::Storage,
            b: <Self as SealedElement<1, 1>>::Storage,
        ) -> <Self as SealedElement<2, 1>>::Storage
        where
            Self: SealedElement<1, 1> + SealedElement<2, 1>,
        {
            let [[a]] = <Self as SealedElement<1, 1>>::to_array(a);
            let [[b]] = <Self as SealedElement<1, 1>>::to_array(b);
            SealedElement::<2, 1>::from_array([[a, b]])
        }

        fn vector_concat_1_2(
            a: <Self as SealedElement<1, 1>>::Storage,
            b: <Self as SealedElement<2, 1>>::Storage,
        ) -> <Self as SealedElement<3, 1>>::Storage
        where
            Self: SealedElement<1, 1> + SealedElement<2, 1> + SealedElement<3, 1>,
        {
            let [[a]] = <Self as SealedElement<1, 1>>::to_array(a);
            let [[b, c]] = <Self as SealedElement<2, 1>>::to_array(b);
            SealedElement::<3, 1>::from_array([[a, b, c]])
        }

        fn vector_concat_2_1(
            a: <Self as SealedElement<2, 1>>::Storage,
            b: <Self as SealedElement<1, 1>>::Storage,
        ) -> <Self as SealedElement<3, 1>>::Storage
        where
            Self: SealedElement<1, 1> + SealedElement<2, 1> + SealedElement<3, 1>,
        {
            let [[a0, a1]] = <Self as SealedElement<2, 1>>::to_array(a);
            let [[b0]] = <Self as SealedElement<1, 1>>::to_array(b);
            SealedElement::<3, 1>::from_array([[a0, a1, b0]])
        }

        fn diagonal(
            _a: <Self as SealedElement<M, N>>::Storage,
        ) -> <Self as SealedElement<N, 1>>::Storage
        where
            Self: SealedElement<N, 1>,
        {
            unimplemented!()
        }

        // Only regular 1x1 through 4x4 square shapes are supported; shapes of 5x5 or larger are
        // outside the planned scope. Floating-point and boolean forms are reversible, while
        // integer and integer-backed mask forms are not.
        fn inverse(_a: Self::Storage) -> Self::Storage { unimplemented!() }
        #[expect(dead_code)]
        fn try_inverse(_a: Self::Storage) -> Option<Self::Storage> { unimplemented!() }
        fn determinant(_a: Self::Storage) -> Self { unimplemented!() }
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
