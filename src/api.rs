use crate::{
    __internal,
    Mask,
    Select,
    Vector,
    column_major,
    marker::{CastFrom, Lane, Signed, StoredVerbatim},
    private,
    row_major,
    support::{Element, FloatElement, IntElement, MaskElement, SintElement, UintElement},
};

pub(crate) mod vector {
    macro_rules! call {
        (<$t:ty, $r:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?) => {
            <$t as $crate::private::SealedElement<$r, 1>>::$f $(::<$gen>)? $(($($arg),*))?
        };
        ($w:ident(<$t:ty, $r:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?)) => {
            $w { storage: $crate::api::vector::call!(<$t, $r>::$f $(::<$gen>)? $(($($arg),*))?) }
        };
        (<$t:ty, $r:tt>::$f:ident $(::<$($gen:tt),+>)? $(($($arg:expr),*))?) => {
            <$t as $crate::private::SealedElement<$r, 1>>::$f $(::<$($gen),+>)? $(($($arg),*))?
        };
        ($w:ident(<$t:ty, $r:tt>::$f:ident $(::<$($gen:tt),+>)? $(($($arg:expr),*))?)) => {
            $w { storage: $crate::api::vector::call!(<$t, $r>::$f $(::<$($gen),+>)? $(($($arg),*))?) }
        };
    }
    pub(crate) use crate::Vector;
    pub(crate) use call;
}

macro_rules! impl_from_array {
    ($t:ty, $D:expr, $array:expr, [$($d:literal),*]) => {
        match $D {
            $($d => {
                let a = core::mem::transmute_copy::<[T; $D], [$t; $d]>(& $array);
                let a = paste::paste!(crate::kernels::from_array::$t:: [<_ $d x1>])([a]);
                core::mem::transmute_copy::<
                    <$t as private::SealedElement<$d, 1>>::Storage,
                    <T as private::SealedElement<$D, 1>>::Storage,
                >(&a)
            },)*
            _ => unreachable!(),
        }
    }
}

impl<T: Element<D>, const D: usize> Vector<T, D> {
    /// A vector with all lanes set to zero.
    pub const ZERO: Self = vector::call!(Self(<T, D>::ZERO));
    /// A vector with all lanes set to one.
    pub const ONE: Self = vector::call!(Self(<T, D>::ONE));

    /// Constructs a vector with every lane set to `value`.
    #[inline]
    pub fn splat(value: T) -> Self { vector::call!(Self(<T, D>::filled(value))) }
    #[inline(always)]
    pub(crate) fn filled(value: T) -> Self { vector::call!(Self(<T, D>::filled(value))) }

    /// Constructs a vector from an array of lanes.
    #[inline]
    pub const fn from_array(array: [T; D]) -> Self {
        let mut out =
            core::mem::MaybeUninit::<<T as private::SealedElement<D, 1>>::Storage>::uninit();

        // SAFETY: `Element<D>` is sealed to `f32`, `i32`, and `u32`, with `D`
        // restricted to 1..=4. `Sealed::TYPE` exactly identifies `T`, so the
        // selected type arm has `$t == T`; likewise, the selected dimension arm
        // has `$d == D`. Consequently, the first `transmute_copy` copies between
        // identical array types, and the second copies between identical
        // `SealedElement<D, 1>::Storage` types. Both sources are fully initialized
        // and `Copy`, so the copies preserve layout, validity, and ownership.
        let storage = unsafe {
            match <T as private::Sealed>::TYPE {
                private::Type::F32 => impl_from_array!(f32, D, &array, [1, 2, 3, 4]),
                private::Type::F64 => impl_from_array!(f64, D, &array, [1, 2, 3, 4]),
                private::Type::I32 => impl_from_array!(i32, D, &array, [1, 2, 3, 4]),
                private::Type::I64 => impl_from_array!(i64, D, &array, [1, 2, 3, 4]),
                private::Type::U32 => impl_from_array!(u32, D, &array, [1, 2, 3, 4]),
                private::Type::U64 => impl_from_array!(u64, D, &array, [1, 2, 3, 4]),
            }
        };

        Self { storage: *out.write(storage) }
    }

    /// Returns the vector's lanes as an array.
    #[inline]
    pub fn to_array(self) -> [T; D] { vector::call!(<T, D>::to_array(self.storage))[0] }

    /// Converts each lane to `U` using Rust's `as` conversion semantics.
    #[inline]
    pub fn cast<U: Element<D> + CastFrom<T>>(self) -> Vector<U, D> {
        vector::call!(Vector(<U, D>::cast_from::<T>(self.storage)))
    }
}

impl<T: FloatElement<D>, const D: usize> Vector<T, D> {
    /// Returns the dot product of `self` and `rhs`.
    #[inline]
    pub fn dot(self, rhs: Self) -> T { vector::call!(<T, D>::dot(self.storage, rhs.storage)) }
    // TODO(codegen-optimization): Compare composite two-lane methods with a backend implementation
    // that loads into `f32x4` only once, and specialize only if loads or stores remain after inlining.
    /// Returns the Euclidean norm of the vector.
    #[inline]
    pub fn norm(self) -> T { <T as private::Sealed>::sqrt(self.norm_squared()) }
    /// Returns the square of the Euclidean norm.
    #[inline]
    pub fn norm_squared(self) -> T { self.dot(self) }
    /// Returns the Euclidean distance between `self` and `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> T {
        <T as private::Sealed>::sqrt(self.distance_squared(rhs))
    }
    /// Returns the square of the Euclidean distance between `self` and `rhs`.
    #[inline]
    pub fn distance_squared(self, rhs: Self) -> T { (self - rhs).norm_squared() }
    /// Returns a vector with the same direction and a norm of one.
    ///
    /// A zero vector produces non-finite lanes according to floating-point
    /// division rules.
    #[inline]
    pub fn normalize(self) -> Self { self / self.norm() }

    /// Returns the largest integer less than or equal to each lane.
    #[inline]
    pub fn floor(self) -> Self { vector::call!(Self(<T, D>::floor(self.storage))) }
    /// Returns the smallest integer greater than or equal to each lane.
    #[inline]
    pub fn ceil(self) -> Self { vector::call!(Self(<T, D>::ceil(self.storage))) }
    /// Rounds each lane to the nearest integer, with halfway cases away from zero.
    #[inline]
    pub fn round(self) -> Self { vector::call!(Self(<T, D>::round(self.storage))) }
    /// Rounds each lane to the nearest integer, with halfway cases toward the even
    /// integer.
    #[inline]
    pub fn round_ties_even(self) -> Self {
        vector::call!(Self(<T, D>::round_ties_even(self.storage)))
    }
    /// Returns the integer part of each lane by rounding toward zero.
    #[inline]
    pub fn trunc(self) -> Self { vector::call!(Self(<T, D>::trunc(self.storage))) }
    /// Returns the fractional part of each lane.
    #[inline]
    pub fn fract(self) -> Self { vector::call!(Self(<T, D>::fract(self.storage))) }
    /// Returns the square root of each lane.
    #[inline]
    pub fn sqrt(self) -> Self { vector::call!(Self(<T, D>::sqrt(self.storage))) }
    /// Returns the reciprocal of each lane.
    #[inline]
    pub fn recip(self) -> Self { Self::ONE / self }
    /// Constructs a floating-point vector from the raw bits of each lane.
    #[inline]
    pub fn from_bits(v: Vector<T::Bits, D>) -> Self {
        vector::call!(Self(<T, D>::from_bits(v.storage)))
    }
    /// Returns the raw bits of each floating-point lane.
    #[inline]
    pub fn to_bits(self) -> Vector<T::Bits, D> {
        vector::call!(Vector(<T, D>::to_bits(self.storage)))
    }
}

impl<T: Signed + Element<D>, const D: usize> Vector<T, D> {
    /// Computes the absolute value of each lane.
    #[inline]
    pub fn abs(self) -> Self { vector::call!(Self(<T, D>::abs(self.storage))) }
}

impl<T: Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<1>,
{
    /// The positive unit vector along the x-axis.
    pub const POS_X: Self = vector::call!(Self(<T, D>::POS_X));
}
impl<T: Signed + Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<1>,
{
    /// The negative unit vector along the x-axis.
    pub const NEG_X: Self = vector::call!(Self(<T, D>::NEG_X));
}
impl<T: Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<2>,
{
    /// The positive unit vector along the y-axis.
    pub const POS_Y: Self = vector::call!(Self(<T, D>::POS_Y));
}
impl<T: Signed + Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<2>,
{
    /// The negative unit vector along the y-axis.
    pub const NEG_Y: Self = vector::call!(Self(<T, D>::NEG_Y));
}
impl<T: Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<3>,
{
    /// The positive unit vector along the z-axis.
    pub const POS_Z: Self = vector::call!(Self(<T, D>::POS_Z));
}
impl<T: Signed + Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<3>,
{
    /// The negative unit vector along the z-axis.
    pub const NEG_Z: Self = vector::call!(Self(<T, D>::NEG_Z));
}
impl<T: Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<4>,
{
    /// The positive unit vector along the w-axis.
    pub const POS_W: Self = vector::call!(Self(<T, D>::POS_W));
}
impl<T: Signed + Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<4>,
{
    /// The negative unit vector along the w-axis.
    pub const NEG_W: Self = vector::call!(Self(<T, D>::NEG_W));
}

impl<T: IntElement<D>, const D: usize> Vector<T, D> {
    /// Computes the absolute difference between corresponding lanes.
    ///
    /// The result uses the unsigned counterpart of `T`, so every difference is
    /// representable, including the distance between the signed extrema.
    #[inline]
    pub fn abs_diff(self, other: Self) -> Vector<T::Unsigned, D> {
        use crate::Select;

        // The comparison deliberately uses the original signedness, while subtraction uses the
        // corresponding unsigned bit patterns. Vector subtraction is wrapping, so each candidate
        // is computed modulo 2^N. If `self < other`, `right - left` has the same low N bits as the
        // mathematical value `other - self`; otherwise `left - right` has the same low N bits as
        // `self - other`. The selected difference is non-negative and at most 2^N - 1, so reducing
        // it modulo 2^N does not alter its value. This also covers signed endpoints such as
        // `MIN.abs_diff(MAX)`, whose result cannot be represented by the signed element type.
        let left = vector::call!(Vector(<T, D>::cast_unsigned(self.storage)));
        let right = vector::call!(Vector(<T, D>::cast_unsigned(other.storage)));
        let a = left - right;
        let b = right - left;
        self.each_lt(other).select(b, a)
    }
}

impl<T: SintElement<D>, const D: usize> Vector<T, D> {
    /// Reinterprets each lane as the corresponding unsigned integer type.
    #[inline]
    pub fn cast_unsigned(self) -> Vector<T::Unsigned, D> {
        vector::call!(Vector(<T, D>::cast_unsigned(self.storage)))
    }
}

impl<T: UintElement<D>, const D: usize> Vector<T, D> {
    /// Reinterprets each lane as the corresponding signed integer type.
    #[inline]
    pub fn cast_signed(self) -> Vector<T::Signed, D> {
        vector::call!(Vector(<T, D>::cast_signed(self.storage)))
    }
}

macro_rules! impl_matrix_from_array {
    ($t:ty, $M:expr, $N:expr, $array:expr, $ms:tt, [$($n:literal),*]) => {
        match $N {
            $($n => impl_matrix_from_array!(@m $t, $M, $N, $array, $n, $ms),)*
            _ => unreachable!(),
        }
    };
    (@m $t:ty, $M:expr, $N:expr, $array:expr, $n:literal, [$($m:literal),*]) => {
        match $M {
            $($m => {
                let a = core::mem::transmute_copy::<
                    [[T; $M]; $N],
                    [[$t; $m]; $n],
                >($array);
                let a = paste::paste!(crate::kernels::from_array::$t:: [<_ $m x $n>])(a);
                core::mem::transmute_copy::<
                    <$t as private::SealedElement<$m, $n>>::Storage,
                    <T as private::SealedElement<$M, $N>>::Storage,
                >(&a)
            },)*
            _ => unreachable!(),
        }
    };
}

impl<T: Element<R, C>, const R: usize, const C: usize> row_major::Matrix<T, R, C> {
    /// A matrix with all elements set to zero.
    pub const ZERO: Self = row_major::call!(Self(<T, R, C>::ZERO));

    /// A matrix with all elements set to one.
    pub const ONE: Self = row_major::call!(Self(<T, R, C>::ONE));

    /// Constructs a matrix with every element set to `value`.
    #[inline]
    pub fn filled(value: T) -> row_major::Matrix<T, R, C> {
        row_major::call!(Self(<T, R, C>::filled(value)))
    }

    /// Constructs a matrix from its logical rows.
    #[inline]
    pub const fn from_rows(rows: [[T; C]; R]) -> Self {
        let mut out =
            core::mem::MaybeUninit::<<T as private::SealedElement<C, R>>::Storage>::uninit();

        // SAFETY: The type and dimension argument is the same as in
        // `column_major::Matrix::from_columns`. Row-major storage uses the
        // transposed private shape `(C, R)`, matching the shape of `rows`.
        let storage = unsafe {
            match <T as private::Sealed>::TYPE {
                private::Type::F32 => {
                    impl_matrix_from_array!(f32, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::F64 => {
                    impl_matrix_from_array!(f64, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::I32 => {
                    impl_matrix_from_array!(i32, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::I64 => {
                    impl_matrix_from_array!(i64, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::U32 => {
                    impl_matrix_from_array!(u32, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::U64 => {
                    impl_matrix_from_array!(u64, C, R, &rows, [1, 2, 3, 4], [1, 2, 3, 4])
                }
            }
        };

        Self { storage: *out.write(storage) }
    }

    /// Constructs a row-major matrix from row vectors.
    #[inline]
    pub fn from_row_vecs(rows: [Vector<T, C>; R]) -> Self {
        row_major::call!(Self(<T, R, C>::from_vecs(rows)))
    }

    /// Returns the logical rows of the matrix.
    #[inline]
    pub fn to_rows(self) -> [[T; C]; R] { row_major::call!(<T, R, C>::to_array(self.storage)) }

    /// Returns the logical rows as vectors.
    #[inline]
    pub fn to_row_vecs(self) -> [Vector<T, C>; R] { self.to_rows().map(Vector::from) }

    /// Returns the transpose of this matrix in the same storage orientation.
    #[inline]
    pub fn transpose(self) -> row_major::Matrix<T, C, R> {
        use row_major::Matrix;
        row_major::call!(Matrix(<T, R, C>::transpose(self.storage)))
    }

    /// Reinterprets the storage as a column-major matrix representing the
    /// transpose of `self`.
    #[inline]
    pub const fn to_column_major_transposed(self) -> column_major::Matrix<T, C, R> {
        column_major::Matrix { storage: self.storage }
    }

    /// Converts each matrix element to `U` using Rust's `as` conversion semantics.
    #[inline]
    pub fn cast<U: Element<R, C> + CastFrom<T>>(self) -> row_major::Matrix<U, R, C> {
        use row_major::Matrix;
        row_major::call!(Matrix(<U, R, C>::cast_from::<T>(self.storage)))
    }
}
impl<T: Element<R, C>, const R: usize, const C: usize> column_major::Matrix<T, R, C> {
    /// A matrix with all elements set to zero.
    pub const ZERO: Self = column_major::call!(Self(<T, R, C>::ZERO));

    /// A matrix with all elements set to one.
    pub const ONE: Self = column_major::call!(Self(<T, R, C>::ONE));

    /// Constructs a matrix with every element set to `value`.
    #[inline]
    pub fn filled(value: T) -> column_major::Matrix<T, R, C> {
        column_major::call!(Self(<T, R, C>::filled(value)))
    }

    /// Constructs a matrix from its logical columns.
    #[inline]
    pub const fn from_columns(columns: [[T; R]; C]) -> Self {
        let mut out =
            core::mem::MaybeUninit::<<T as private::SealedElement<R, C>>::Storage>::uninit();

        // SAFETY: `Element<R, C>` is sealed to `f32`, `i32`, and `u32`, with
        // both dimensions restricted to 1..=4. `Sealed::TYPE` identifies `T`,
        // and the selected dimension arms identify `R` and `C`. The two
        // `transmute_copy` calls therefore copy between identical array and
        // storage types, respectively. Both sources are initialized and `Copy`.
        let storage = unsafe {
            match <T as private::Sealed>::TYPE {
                private::Type::F32 => {
                    impl_matrix_from_array!(f32, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::F64 => {
                    impl_matrix_from_array!(f64, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::I32 => {
                    impl_matrix_from_array!(i32, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::I64 => {
                    impl_matrix_from_array!(i64, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::U32 => {
                    impl_matrix_from_array!(u32, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
                private::Type::U64 => {
                    impl_matrix_from_array!(u64, R, C, &columns, [1, 2, 3, 4], [1, 2, 3, 4])
                }
            }
        };

        Self { storage: *out.write(storage) }
    }

    /// Constructs a column-major matrix from column vectors.
    #[inline]
    pub fn from_column_vecs(columns: [Vector<T, R>; C]) -> Self {
        column_major::call!(Self(<T, R, C>::from_vecs(columns)))
    }

    /// Returns the logical columns of the matrix.
    #[inline]
    pub fn to_columns(self) -> [[T; R]; C] {
        column_major::call!(<T, R, C>::to_array(self.storage))
    }

    // TODO(codegen-optimization): Compare array conversion with direct packed-storage swizzles,
    // starting with the 2x2 `f32x4` representation, before specializing vector extraction.
    /// Returns the logical columns as vectors.
    #[inline]
    pub fn to_column_vecs(self) -> [Vector<T, R>; C] { self.to_columns().map(Vector::from) }

    /// Returns the transpose of this matrix in the same storage orientation.
    #[inline]
    pub fn transpose(self) -> column_major::Matrix<T, C, R> {
        use column_major::Matrix;
        column_major::call!(Matrix(<T, R, C>::transpose(self.storage)))
    }

    /// Reinterprets the storage as a row-major matrix representing the transpose
    /// of `self`.
    #[inline]
    pub const fn to_row_major_transposed(self) -> row_major::Matrix<T, C, R> {
        row_major::Matrix { storage: self.storage }
    }

    /// Converts each matrix element to `U` using Rust's `as` conversion semantics.
    #[inline]
    pub fn cast<U: Element<R, C> + CastFrom<T>>(self) -> column_major::Matrix<U, R, C> {
        use column_major::Matrix;
        column_major::call!(Matrix(<U, R, C>::cast_from::<T>(self.storage)))
    }
}

// TODO(codegen-optimization): Compare scalar-specific backend members with the current lowering
// through `filled`, and add them only if representative codegen retains redundant splats,
// loads, or stores.
// In particular, a scalar RHS for an arithmetic right shift can use `_mm_sra_epi32`, whereas the
// filled vector RHS follows the lane-wise `_mm_srav_epi32` form; preserve this distinction in the
// codegen comparison instead of considering only the number of loads and stores.

macro_rules! if_match {
    (($_:tt) { $($then:tt)* } else { $($else:tt)* }) => { $($then)* };
    (() { $($then:tt)* } else { $($else:tt)* }) => { $($else)* };
}

macro_rules! impl_binop {
    (
        docs: [$doc:literal, $assign_doc:literal],
        $mod:tt::$Tensor:tt,
        $(any:$T:tt,)? // self_tensor
        $(spec:$t:tt,)? // self_scalar
        $(rhs_tensor:$rhs_tensor:tt,)?
        $(rhs_scalar:$rhs_scalar:tt,)?
        [$($N:tt),+],
        $trait:ident::$method:ident
        $(, $trait_assign:ident::$method_assign:ident)?
    ) => {
        impl<$($T,)? $(const $N: usize),+> core::ops::$trait<
            if_match!(($($rhs_tensor)?) { $mod::$Tensor<$($T)? $($t)?, $($N),+> } else { $($T)? })
        > for if_match!(($($T)?) { $mod::$Tensor<$($T)?, $($N),+> } else { $($t)? } )
            where $($T)? $($t)?: Element<$($N),+> $(+ core::ops::$trait<Output = $T>)?
        {
            type Output = $mod::$Tensor<$($T)? $($t)?, $($N),+>;
            #[doc = $doc]
            #[inline]
            fn $method(
                self,
                rhs: if_match!(($($rhs_tensor)?) { $mod::$Tensor<$($T)? $($t)?, $($N),+> } else { $($T)? } ),
            ) -> Self::Output {
                use $mod::$Tensor;
                let lhs = if_match!(($($T)?)          { self.storage } else { $Tensor::filled(self).storage });
                let rhs = if_match!(($($rhs_tensor)?) { rhs.storage }  else { $Tensor::filled(rhs).storage });
                $mod::call!($Tensor(<$($T)? $($t)?, $($N),+>::$method(lhs, rhs)))
            }
        }
        if_match!{ ($($trait_assign)?) {
            impl<$($T,)? $(const $N: usize),+> core::ops::$($trait_assign)?<
                if_match!(($($rhs_tensor)?) { $mod::$Tensor<$($T)? $($t)?, $($N),+> } else { $($T)? })
            > for $mod::$Tensor<$($T)?, $($N),+>
                where $($T)? $($t)?: Element<$($N),+> $(+ core::ops::$trait<Output = $T>)?
            {
                #[doc = $assign_doc]
                #[inline]
                fn $($method_assign)?(
                    &mut self,
                    rhs: if_match!(($($rhs_tensor)?) { $mod::$Tensor<$($T)? $($t)?, $($N),+> } else { $($T)? }),
                ) {
                    *self = core::ops::$trait::$method(*self, rhs);
                }
            }
        } else {}}
    };
}

macro_rules! impl_binop_all {
    (@a [$docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt, not_matrix]) => {
        // Matrix-to-matrix multiplication and division are omitted because multiplication is
        // reserved for the matrix product and division has no corresponding matrix operation.
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_tensor:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, row_major::Matrix, any:T, rhs_scalar:_, [R, C], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, column_major::Matrix, any:T, rhs_scalar:_, [R, C], $trait::$method, $trait_assign::$method_assign);
    };
    (@a [$docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt, vector_only]) => {
        // This branch intentionally generates vector operations only; matrix multiplication uses
        // the matrix-product implementation and matrix division is not defined.
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_tensor:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
    };
    (@a [$docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt, vector_and_mask]) => {
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_tensor:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
    };
    (@a [$docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt]) => {
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_tensor:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, vector::Vector, any:T, rhs_scalar:_, [D], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, row_major::Matrix, any:T, rhs_tensor:_, [R, C], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, row_major::Matrix, any:T, rhs_scalar:_, [R, C], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, column_major::Matrix, any:T, rhs_tensor:_, [R, C], $trait::$method, $trait_assign::$method_assign);
        impl_binop!(docs: $docs, column_major::Matrix, any:T, rhs_scalar:_, [R, C], $trait::$method, $trait_assign::$method_assign);
    };
    // Generate scalar-left implementations only for concrete scalar types: the orphan rules do
    // not permit implementing an external operator trait for an arbitrary type parameter.
    (@b [$($scalar:tt),+], [$docs:tt, $trait:tt::$method:tt], vector_only) => {
        $(
            impl_binop!(docs: $docs, vector::Vector,spec:$scalar,rhs_tensor:_,[D],$trait::$method);
        )+
    };
    (@b [$($scalar:tt),+], [$docs:tt, $trait:tt::$method:tt] $(,$option:tt)?) => {
        $(
            impl_binop!(docs: $docs, vector::Vector,spec:$scalar,rhs_tensor:_,[D],$trait::$method);
            impl_binop!(docs: $docs, row_major::Matrix,spec:$scalar,rhs_tensor:_,[R, C],$trait::$method);
            impl_binop!(docs: $docs, column_major::Matrix,spec:$scalar,rhs_tensor:_,[R, C],$trait::$method);
        )+
    };
    (arithmetic, [$([$generic_docs:tt, $float_docs:tt, $integer_docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt $(, $option:tt)?],)+]) => {
        $(
            impl_binop_all!(@a [$generic_docs, $trait::$method, $trait_assign::$method_assign $(, $option)?]);
            impl_binop_all!(@b [f32], [$float_docs, $trait::$method] $(, $option)?);
            impl_binop_all!(@b [i32, u32], [$integer_docs, $trait::$method] $(, $option)?);
        )+
    };
    ($scalar:tt, [$([$docs:tt, $trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt $(, $option:tt)?],)+]) => {
        $(
            impl_binop_all!(@a [$docs, $trait::$method, $trait_assign::$method_assign $(, $option)?]);
            impl_binop_all!(@b $scalar, [$docs, $trait::$method] $(, $option)?);
        )+
    };
}

impl_binop_all!(arithmetic, [
    [
        [
            "Performs component-wise addition.\n\nFor integer element types, the result wraps on overflow.",
            "Performs component-wise addition assignment.\n\nFor integer element types, the result wraps on overflow."
        ],
        ["Performs component-wise addition.", "Performs component-wise addition assignment."],
        [
            "Performs component-wise addition.\n\nThe result wraps on overflow.",
            "Performs component-wise addition assignment.\n\nThe result wraps on overflow."
        ],
        Add::add,
        AddAssign::add_assign
    ],
    [
        [
            "Performs component-wise subtraction.\n\nFor integer element types, the result wraps on overflow.",
            "Performs component-wise subtraction assignment.\n\nFor integer element types, the result wraps on overflow."
        ],
        ["Performs component-wise subtraction.", "Performs component-wise subtraction assignment."],
        [
            "Performs component-wise subtraction.\n\nThe result wraps on overflow.",
            "Performs component-wise subtraction assignment.\n\nThe result wraps on overflow."
        ],
        Sub::sub,
        SubAssign::sub_assign
    ],
    [
        [
            "Performs component-wise multiplication.\n\nFor integer element types, the result wraps on overflow.",
            "Performs component-wise multiplication assignment.\n\nFor integer element types, the result wraps on overflow."
        ],
        [
            "Performs component-wise multiplication.",
            "Performs component-wise multiplication assignment."
        ],
        [
            "Performs component-wise multiplication.\n\nThe result wraps on overflow.",
            "Performs component-wise multiplication assignment.\n\nThe result wraps on overflow."
        ],
        Mul::mul,
        MulAssign::mul_assign,
        not_matrix
    ],
    [
        [
            "Performs component-wise division.\n\nInteger division rounds toward zero and wraps on overflow.\n\n# Panics\n\nFor integer element types, panics if any active divisor element is zero.",
            "Performs component-wise division assignment.\n\nInteger division rounds toward zero and wraps on overflow.\n\n# Panics\n\nFor integer element types, panics if any active divisor element is zero."
        ],
        ["Performs component-wise division.", "Performs component-wise division assignment."],
        [
            "Performs component-wise division.\n\nDivision rounds toward zero and wraps on overflow.\n\n# Panics\n\nPanics if any active divisor element is zero.",
            "Performs component-wise division assignment.\n\nDivision rounds toward zero and wraps on overflow.\n\n# Panics\n\nPanics if any active divisor element is zero."
        ],
        Div::div,
        DivAssign::div_assign,
        not_matrix
    ],
    [
        [
            "Performs component-wise remainder.\n\nFor integer element types, the result uses wrapping remainder semantics.\n\n# Panics\n\nFor integer element types, panics if any active divisor element is zero.",
            "Performs component-wise remainder assignment.\n\nFor integer element types, the result uses wrapping remainder semantics.\n\n# Panics\n\nFor integer element types, panics if any active divisor element is zero."
        ],
        ["Performs component-wise remainder.", "Performs component-wise remainder assignment."],
        [
            "Performs component-wise remainder.\n\nThe result uses wrapping integer remainder semantics.\n\n# Panics\n\nPanics if any active divisor element is zero.",
            "Performs component-wise remainder assignment.\n\nThe result uses wrapping integer remainder semantics.\n\n# Panics\n\nPanics if any active divisor element is zero."
        ],
        Rem::rem,
        RemAssign::rem_assign,
        vector_only
    ],
]);
impl_binop_all!([i32, u32], [
    [
        ["Performs component-wise bitwise AND.", "Performs component-wise bitwise AND assignment."],
        BitAnd::bitand,
        BitAndAssign::bitand_assign,
        vector_only
    ],
    [
        ["Performs component-wise bitwise OR.", "Performs component-wise bitwise OR assignment."],
        BitOr::bitor,
        BitOrAssign::bitor_assign,
        vector_only
    ],
    [
        ["Performs component-wise bitwise XOR.", "Performs component-wise bitwise XOR assignment."],
        BitXor::bitxor,
        BitXorAssign::bitxor_assign,
        vector_only
    ],
    [
        ["Performs component-wise left shift.", "Performs component-wise left-shift assignment."],
        Shl::shl,
        ShlAssign::shl_assign,
        vector_only
    ],
    [
        ["Performs component-wise right shift.", "Performs component-wise right-shift assignment."],
        Shr::shr,
        ShrAssign::shr_assign,
        vector_only
    ],
]);

impl<T: core::ops::Neg + Element<R, C>, const R: usize, const C: usize> core::ops::Neg
    for row_major::Matrix<T, R, C>
{
    type Output = Self;
    /// Performs component-wise negation.
    ///
    /// For `i32` elements, negation wraps on overflow.
    #[inline]
    fn neg(self) -> Self::Output { row_major::call!(Self(<T, R, C>::neg(self.storage))) }
}
impl<T: core::ops::Neg + Element<R, C>, const R: usize, const C: usize> core::ops::Neg
    for column_major::Matrix<T, R, C>
{
    type Output = Self;
    /// Performs component-wise negation.
    ///
    /// For `i32` elements, negation wraps on overflow.
    #[inline]
    fn neg(self) -> Self::Output { column_major::call!(Self(<T, R, C>::neg(self.storage))) }
}
impl<T: core::ops::Neg + Element<D>, const D: usize> core::ops::Neg for Vector<T, D> {
    type Output = Self;
    /// Performs component-wise negation.
    ///
    /// For `i32` elements, negation wraps on overflow.
    #[inline]
    fn neg(self) -> Self::Output { vector::call!(Self(<T, D>::neg(self.storage))) }
}

impl<T: core::ops::Not + Element<D>, const D: usize> core::ops::Not for Vector<T, D> {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output { vector::call!(Self(<T, D>::not(self.storage))) }
}
impl<T: MaskElement<D>, const D: usize> core::ops::Not for Mask<T, D> {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output { vector::call!(Self(<T, D>::canonical_not(self.storage))) }
}

macro_rules! impl_mask_binop {
    (@a $scalar:tt, [$([$trait:tt::$method:tt, $trait_assign:tt::$method_assign:tt, $f:tt],)+]) => {
        $(
            impl<T: MaskElement<D>, const D: usize> core::ops::$trait for Mask<T, D> {
                type Output = Self;
                #[inline]
                fn $method(self, rhs: Self) -> Self::Output {
                    vector::call!(Self(<T, D>::$f(self.storage, rhs.storage)))
                }
            }
            impl<T: MaskElement<D>, const D: usize> core::ops::$trait_assign for Mask<T, D> {
                #[inline]
                fn $method_assign(&mut self, rhs: Self) {
                    *self = <Self as core::ops::$trait>::$method(*self, rhs)
                }
            }
        )+
    };
    ([$($scalar:tt),+], $remaining:tt) => {
        $(impl_mask_binop!(@a $scalar, $remaining);)+
    };
}

impl_mask_binop! {
    [i32],
    [
        [BitAnd::bitand, BitAndAssign::bitand_assign, canonical_bitand],
        [BitOr::bitor, BitOrAssign::bitor_assign, canonical_bitor],
        [BitXor::bitxor, BitXorAssign::bitxor_assign, canonical_bitxor],
    ]
}

impl<T: row_major::MatrixProduct<N, N, N>, const N: usize> core::iter::Product
    for row_major::Matrix<T, N, N>
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc * x
            })
        } else {
            Self::IDENTITY
        }
    }
}
impl<T: column_major::MatrixProduct<N, N, N>, const N: usize> core::iter::Product
    for column_major::Matrix<T, N, N>
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc * x
            })
        } else {
            Self::IDENTITY
        }
    }
}
impl<T: Element<D> + core::ops::Mul<Output = T>, const D: usize> core::iter::Product
    for Vector<T, D>
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc * x
            })
        } else {
            Self::ONE
        }
    }
}

impl<T: Element<R, C> + core::ops::Add<Output = T>, const R: usize, const C: usize> core::iter::Sum
    for row_major::Matrix<T, R, C>
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc + x
            })
        } else {
            Self::ZERO
        }
    }
}
impl<T: Element<R, C> + core::ops::Add<Output = T>, const R: usize, const C: usize> core::iter::Sum
    for column_major::Matrix<T, R, C>
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc + x
            })
        } else {
            Self::ZERO
        }
    }
}
impl<T: Element<D> + core::ops::Add<Output = T>, const D: usize> core::iter::Sum for Vector<T, D> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, {
                #[inline(always)]
                |acc, x| acc + x
            })
        } else {
            Self::ZERO
        }
    }
}

impl<T: Element<R, C>, const R: usize, const C: usize> Default for row_major::Matrix<T, R, C> {
    /// Returns the zero matrix.
    #[inline]
    fn default() -> Self { Self::ZERO }
}
impl<T: Element<R, C>, const R: usize, const C: usize> Default for column_major::Matrix<T, R, C> {
    /// Returns the zero matrix.
    #[inline]
    fn default() -> Self { Self::ZERO }
}
impl<T: Element<D>, const D: usize> Default for Vector<T, D> {
    #[inline]
    fn default() -> Self { Self::ZERO }
}
impl<T: MaskElement<D>, const D: usize> Default for Mask<T, D> {
    #[inline]
    fn default() -> Self { Self::splat(false) }
}

struct CompactRow<'a, T>(&'a [T]);

impl<T: core::fmt::Debug> core::fmt::Debug for CompactRow<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[")?;
        for (index, value) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            core::fmt::Debug::fmt(value, f)?;
        }
        f.write_str("]")
    }
}

impl<T: core::fmt::Debug + Element<R, C>, const R: usize, const C: usize> core::fmt::Debug
    for row_major::Matrix<T, R, C>
{
    #[allow(clippy::missing_inline_in_public_items)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rows = self.to_rows();
        f.debug_list().entries(rows.iter().map(|row| CompactRow(row.as_slice()))).finish()
    }
}
impl<T: core::fmt::Debug + Element<R, C>, const R: usize, const C: usize> core::fmt::Debug
    for column_major::Matrix<T, R, C>
{
    #[allow(clippy::missing_inline_in_public_items)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let columns = self.to_columns();
        f.debug_list().entries(columns.iter().map(|column| CompactRow(column.as_slice()))).finish()
    }
}
impl<T: core::fmt::Debug + Element<D>, const D: usize> core::fmt::Debug for Vector<T, D> {
    #[allow(clippy::missing_inline_in_public_items)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = self.to_array();
        core::fmt::Debug::fmt(&CompactRow(array.as_slice()), f)
    }
}
impl<T: MaskElement<D>, const D: usize> core::fmt::Debug for Mask<T, D> {
    #[allow(clippy::missing_inline_in_public_items)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let array = self.to_array();
        core::fmt::Debug::fmt(&CompactRow(array.as_slice()), f)
    }
}

#[allow(clippy::partialeq_ne_impl)] // Keep the dedicated SIMD `ne` reduction instead of negating `eq`.
impl<T: PartialEq + Element<R, C>, const R: usize, const C: usize> PartialEq
    for row_major::Matrix<T, R, C>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        row_major::call!(<T, R, C>::eq(self.storage, other.storage))
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        row_major::call!(<T, R, C>::ne(self.storage, other.storage))
    }
}
#[allow(clippy::partialeq_ne_impl)]
impl<T: PartialEq + Element<R, C>, const R: usize, const C: usize> PartialEq
    for column_major::Matrix<T, R, C>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        column_major::call!(<T, R, C>::eq(self.storage, other.storage))
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        column_major::call!(<T, R, C>::ne(self.storage, other.storage))
    }
}
#[allow(clippy::partialeq_ne_impl)] // Keep the dedicated SIMD `ne` reduction instead of negating `eq`.
impl<T: PartialEq + Element<D>, const D: usize> PartialEq for Vector<T, D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool { vector::call!(<T, D>::eq(self.storage, other.storage)) }
    #[inline]
    fn ne(&self, other: &Self) -> bool { vector::call!(<T, D>::ne(self.storage, other.storage)) }
}
impl<T: Eq + Element<R, C>, const R: usize, const C: usize> Eq for row_major::Matrix<T, R, C> {}
impl<T: Eq + Element<R, C>, const R: usize, const C: usize> Eq for column_major::Matrix<T, R, C> {}
impl<T: Eq + Element<D>, const D: usize> Eq for Vector<T, D> {}

impl<T: PartialEq + Element<D>, const D: usize> Vector<T, D> {
    /// Tests each lane for equality.
    #[inline]
    pub fn each_eq(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_eq(self.storage, rhs.storage)))
    }
    /// Tests each lane for inequality.
    #[inline]
    pub fn each_ne(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_ne(self.storage, rhs.storage)))
    }
}
impl<T: PartialOrd + Element<D>, const D: usize> Vector<T, D> {
    /// Tests whether each lane is less than the corresponding lane of `rhs`.
    #[inline]
    pub fn each_lt(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_lt(self.storage, rhs.storage)))
    }
    /// Tests whether each lane is less than or equal to the corresponding lane of
    /// `rhs`.
    #[inline]
    pub fn each_le(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_le(self.storage, rhs.storage)))
    }
    /// Tests whether each lane is greater than the corresponding lane of `rhs`.
    #[inline]
    pub fn each_gt(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_gt(self.storage, rhs.storage)))
    }
    /// Tests whether each lane is greater than or equal to the corresponding lane
    /// of `rhs`.
    #[inline]
    pub fn each_ge(self, rhs: Self) -> Mask<<T as Lane>::Mask, D> {
        vector::call!(Mask(<T, D>::each_ge(self.storage, rhs.storage)))
    }
}
impl<T: Ord + Element<D>, const D: usize> crate::EachOrd for Vector<T, D> {
    #[inline]
    fn each_max(self, rhs: Self) -> Self {
        vector::call!(Self(<T, D>::each_max(self.storage, rhs.storage)))
    }
    #[inline]
    fn each_min(self, rhs: Self) -> Self {
        vector::call!(Self(<T, D>::each_min(self.storage, rhs.storage)))
    }
    #[inline]
    fn each_clamp(self, min: Self, max: Self) -> Self {
        vector::call!(Self(<T, D>::each_clamp::<private::VectorFmt>(self.storage, min.storage, max.storage)))
    }
}
impl<T: FloatElement<D>, const D: usize> Vector<T, D> {
    /// Returns the lane-wise maximum of `self` and `rhs`.
    #[inline]
    pub fn each_max(self, rhs: Self) -> Self {
        vector::call!(Self(<T, D>::each_max(self.storage, rhs.storage)))
    }
    /// Returns the lane-wise minimum of `self` and `rhs`.
    #[inline]
    pub fn each_min(self, rhs: Self) -> Self {
        vector::call!(Self(<T, D>::each_min(self.storage, rhs.storage)))
    }
    /// Restricts every lane to the corresponding inclusive range.
    ///
    /// # Panics
    ///
    /// Panics if any minimum is greater than its corresponding maximum, or if a
    /// bound is NaN.
    #[inline]
    pub fn each_clamp(self, min: Self, max: Self) -> Self {
        vector::call!(Self(<T, D>::each_clamp::<private::VectorFmt>(self.storage, min.storage, max.storage)))
    }
}

impl<T: Element<R, C>, const R: usize, const C: usize> Clone for row_major::Matrix<T, R, C> {
    #[inline]
    fn clone(&self) -> Self { *self }
}
impl<T: Element<R, C>, const R: usize, const C: usize> Clone for column_major::Matrix<T, R, C> {
    #[inline]
    fn clone(&self) -> Self { *self }
}
impl<T: Element<D>, const D: usize> Clone for Vector<T, D> {
    #[inline]
    fn clone(&self) -> Self { *self }
}
impl<T: MaskElement<D>, const D: usize> Clone for Mask<T, D> {
    #[inline]
    fn clone(&self) -> Self { *self }
}
impl<T: Element<R, C>, const R: usize, const C: usize> Copy for row_major::Matrix<T, R, C> {}
impl<T: Element<R, C>, const R: usize, const C: usize> Copy for column_major::Matrix<T, R, C> {}
impl<T: Element<D>, const D: usize> Copy for Vector<T, D> {}
impl<T: MaskElement<D>, const D: usize> Copy for Mask<T, D> {}

impl<T: Element<D> + StoredVerbatim, const D: usize> Vector<T, D> {
    /// Borrows the vector's active lanes as an array.
    ///
    /// SIMD padding lanes, when present, are not included in the returned view.
    #[inline]
    pub fn as_array(&self) -> &[T; D] { vector::call!(<T, D>::as_array_first(&self.storage)) }
    /// Mutably borrows the vector's active lanes as an array.
    ///
    /// SIMD padding lanes, when present, are not included. The returned borrow
    /// is tied to `self`, so Rust's borrowing rules prevent overlapping access
    /// through indexing or component fields while it remains live.
    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [T; D] {
        vector::call!(<T, D>::as_mut_array_first(&mut self.storage))
    }
}

impl<T: Element<R, C>, const R: usize, const C: usize> From<[[T; C]; R]>
    for row_major::Matrix<T, R, C>
{
    #[inline]
    fn from(value: [[T; C]; R]) -> Self { row_major::call!(Self(<T, R, C>::from_array(value))) }
}
impl<T: Element<R, C>, const R: usize, const C: usize> From<row_major::Matrix<T, R, C>>
    for [[T; C]; R]
{
    #[inline]
    fn from(value: row_major::Matrix<T, R, C>) -> Self {
        row_major::call!(<T, R, C>::to_array(value.storage))
    }
}
impl<T: Element<R, C>, const R: usize, const C: usize> From<[[T; R]; C]>
    for column_major::Matrix<T, R, C>
{
    #[inline]
    fn from(value: [[T; R]; C]) -> Self { column_major::call!(Self(<T, R, C>::from_array(value))) }
}
impl<T: Element<R, C>, const R: usize, const C: usize> From<column_major::Matrix<T, R, C>>
    for [[T; R]; C]
{
    #[inline]
    fn from(value: column_major::Matrix<T, R, C>) -> Self {
        column_major::call!(<T, R, C>::to_array(value.storage))
    }
}
impl<T: Element<D>, const D: usize> From<[T; D]> for Vector<T, D> {
    #[inline]
    fn from(value: [T; D]) -> Self { vector::call!(Self(<T, D>::from_array([value]))) }
}
impl<T: Element<D>, const D: usize> From<Vector<T, D>> for [T; D] {
    #[inline]
    fn from(value: Vector<T, D>) -> Self { value.to_array() }
}
impl<T: MaskElement<D>, const D: usize> From<[bool; D]> for Mask<T, D> {
    #[inline]
    fn from(value: [bool; D]) -> Self { vector::call!(Self(<T, D>::from_bool_array([value]))) }
}

impl<T: Element<2> + StoredVerbatim> core::ops::Deref for Vector<T, 2> {
    type Target = private::XY<T>;
    #[inline]
    fn deref(&self) -> &Self::Target { private::XY::from_array(self.as_array()) }
}
impl<T: Element<3> + StoredVerbatim> core::ops::Deref for Vector<T, 3> {
    type Target = private::XYZ<T>;
    #[inline]
    fn deref(&self) -> &Self::Target { private::XYZ::from_array(self.as_array()) }
}
impl<T: Element<4> + StoredVerbatim> core::ops::Deref for Vector<T, 4> {
    type Target = private::XYZW<T>;
    #[inline]
    fn deref(&self) -> &Self::Target { private::XYZW::from_array(self.as_array()) }
}
impl<T: Element<2> + StoredVerbatim> core::ops::DerefMut for Vector<T, 2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        private::XY::from_mut_array(self.as_mut_array())
    }
}
impl<T: Element<3> + StoredVerbatim> core::ops::DerefMut for Vector<T, 3> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        private::XYZ::from_mut_array(self.as_mut_array())
    }
}
impl<T: Element<4> + StoredVerbatim> core::ops::DerefMut for Vector<T, 4> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        private::XYZW::from_mut_array(self.as_mut_array())
    }
}

impl<T: Element<R, C> + StoredVerbatim, const R: usize, const C: usize>
    core::ops::Index<(usize, usize)> for row_major::Matrix<T, R, C>
{
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        // `StoredVerbatim` guarantees that active storage lanes are referenceable;
        // the storage implementation returns `None` rather than a padding lane.
        if let Some(value) = row_major::call!(<T, R, C>::index(&self.storage, (j, i))) {
            value
        } else {
            std::hint::cold_path();
            panic!(
                "matrix index out of bounds: the dimensions are {R}x{C} but the index is ({i}, {j})"
            )
        }
    }
}
impl<T: Element<R, C> + StoredVerbatim, const R: usize, const C: usize>
    core::ops::IndexMut<(usize, usize)> for row_major::Matrix<T, R, C>
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        // The reference is derived from the exclusive borrow of `self`, which
        // prevents another storage reference from overlapping its lifetime.
        if let Some(value) = row_major::call!(<T, R, C>::index_mut(&mut self.storage, (j, i))) {
            value
        } else {
            std::hint::cold_path();
            panic!(
                "matrix index out of bounds: the dimensions are {R}x{C} but the index is ({i}, {j})"
            )
        }
    }
}
impl<T: Element<R, C> + StoredVerbatim, const R: usize, const C: usize>
    core::ops::Index<(usize, usize)> for column_major::Matrix<T, R, C>
{
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        if let Some(value) = column_major::call!(<T, R, C>::index(&self.storage, (i, j))) {
            value
        } else {
            std::hint::cold_path();
            panic!(
                "matrix index out of bounds: the dimensions are {R}x{C} but the index is ({i}, {j})"
            )
        }
    }
}
impl<T: Element<R, C> + StoredVerbatim, const R: usize, const C: usize>
    core::ops::IndexMut<(usize, usize)> for column_major::Matrix<T, R, C>
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        if let Some(value) = column_major::call!(<T, R, C>::index_mut(&mut self.storage, (i, j))) {
            value
        } else {
            std::hint::cold_path();
            panic!(
                "matrix index out of bounds: the dimensions are {R}x{C} but the index is ({i}, {j})"
            )
        }
    }
}
impl<T: Element<D> + StoredVerbatim, const D: usize> core::ops::Index<usize> for Vector<T, D> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        // `StoredVerbatim` guarantees that active storage lanes are referenceable;
        // the storage implementation returns `None` rather than a padding lane.
        if let Some(value) = vector::call!(<T, D>::index(&self.storage, (index, 0))) {
            value
        } else {
            std::hint::cold_path();
            panic!("index out of bounds: the len is {D} but the index is {index}")
        }
    }
}
impl<T: Element<D> + StoredVerbatim, const D: usize> core::ops::IndexMut<usize> for Vector<T, D> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // The reference is derived from the exclusive borrow of `self`, which
        // prevents array, component, or index views from overlapping it.
        if let Some(value) = vector::call!(<T, D>::index_mut(&mut self.storage, (index, 0))) {
            value
        } else {
            std::hint::cold_path();
            panic!("index out of bounds: the len is {D} but the index is {index}")
        }
    }
}

impl<T: MaskElement<D>, const D: usize> Mask<T, D> {
    /// Constructs a mask with every lane set to `value`.
    #[inline]
    pub fn splat(value: bool) -> Self { Self::from([value; D]) }
    /// Returns the mask lanes as a boolean array.
    #[inline]
    pub fn to_array(self) -> [bool; D] { vector::call!(<T, D>::to_bool_array(self.storage))[0] }
    /// Converts the mask to its signed integer vector representation.
    ///
    /// True lanes contain all one bits and false lanes contain zero.
    #[inline]
    pub fn to_vector(self) -> Vector<T, D> { Vector { storage: self.storage.into_inner() } }
    /// Returns `true` if every lane is true.
    #[inline]
    pub fn all(self) -> bool { vector::call!(<T, D>::all(self.storage)) }
    /// Returns `true` if any lane is true.
    #[inline]
    pub fn any(self) -> bool { vector::call!(<T, D>::any(self.storage)) }
}

impl<T, U, const D: usize> Select<Mask<T, D>> for Mask<U, D>
where
    T: MaskElement<D>,
    U: MaskElement<D>,
{
    #[inline]
    fn select(self, true_values: Mask<T, D>, false_values: Mask<T, D>) -> Mask<T, D> {
        Mask {
            storage: <T as private::SealedElement<D, 1>>::canonical_select_any_mask::<U>(
                self.storage,
                true_values.storage,
                false_values.storage,
            ),
        }
    }
}
impl<T, U, const D: usize> Select<Vector<T, D>> for Mask<U, D>
where
    T: Element<D>,
    U: MaskElement<D>,
{
    #[inline]
    fn select(self, true_values: Vector<T, D>, false_values: Vector<T, D>) -> Vector<T, D> {
        Vector {
            storage: <T as private::SealedElement<D, 1>>::select_any_mask::<U>(
                self.storage,
                true_values.storage,
                false_values.storage,
            ),
        }
    }
}
