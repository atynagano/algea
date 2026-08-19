use super::{Element, private};
use crate::{
    Vector,
    kernels::matmul,
    utils::{Load, Store},
};

/// A fixed-size matrix stored as column vectors.
///
/// For a four-by-four matrix, the storage is organized as follows, where each
/// `cN` is a [`Vector`] column:
///
/// ```text
///       c0    c1    c2    c3
///     ┌                         ┐
///     │ c0[0] c1[0] c2[0] c3[0] │
///     │ c0[1] c1[1] c2[1] c3[1] │
///     │ c0[2] c1[2] c2[2] c3[2] │
///     │ c0[3] c1[3] c2[3] c3[3] │
///     └                         ┘
/// ```
pub struct Matrix<T: Element<R, C>, const R: usize, const C: usize> {
    pub(crate) storage: <T as private::SealedElement<R, C>>::Storage,
}

#[rustfmt::skip]
#[allow(clippy::duplicate_mod)]
#[path = "../row_major/impls.rs"]
mod impls;

macro_rules! call {
    (<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?) => {
        <$t as $crate::private::SealedElement<$r, $c>>::$f $(::<$gen>)? $(($($arg),*))?
    };
    ($w:ident(<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?)) => {
        $w { storage: $crate::column_major::call!(<$t, $r, $c>::$f $(::<$gen>)? $(($($arg),*))?) }
    };
}
pub(crate) use call;

/// Enables `R × K` by `K × C` matrix multiplication with the `*` operator.
///
/// When `T` implements this trait, `Matrix<T, R, K>` implements
/// [`core::ops::Mul`]`<Matrix<T, K, C>, Output = Matrix<T, R, C>>`.
/// This allows generic code to require that a particular matrix product is available:
///
/// ```text
/// ┌ a00 a01 a02 a03 ┐   ┌ b00 b01 b02 b03 ┐   ┌ c00 c01 c02 c03 ┐
/// │ a10 a11 a12 a13 │ × │ b10 b11 b12 b13 │ = │ c10 c11 c12 c13 │
/// │ a20 a21 a22 a23 │   │ b20 b21 b22 b23 │   │ c20 c21 c22 c23 │
/// └ a30 a31 a32 a33 ┘   └ b30 b31 b32 b33 ┘   └ c30 c31 c32 c33 ┘
/// ```
///
/// ```
/// use algea::column_major::{Matrix, MatrixProduct};
///
/// fn multiply<T: MatrixProduct<4, 4, 4>>(
///     a: Matrix<T, 4, 4>,
///     b: Matrix<T, 4, 4>,
/// ) -> Matrix<T, 4, 4> {
///     a * b
/// }
/// ```
pub trait MatrixProduct<const R: usize, const K: usize, const C: usize>:
    MatrixVectorProduct<R, K> + OuterProduct<R, C> + Element<K, C>
{
    #[doc(hidden)]
    fn __matrix_product(lhs: Matrix<Self, R, K>, rhs: Matrix<Self, K, C>) -> Matrix<Self, R, C>;
}
/// Enables multiplication of an `R × C` matrix by a `C`-lane column vector with the
/// `*` operator.
///
/// When `T` implements this trait, `Matrix<T, R, C>` implements
/// [`core::ops::Mul`]`<Vector<T, C>, Output = Vector<T, R>>`.
///
/// ```text
/// ┌ a00 a01 a02 a03 ┐   ┌ x0 ┐   ┌ y0 ┐
/// │ a10 a11 a12 a13 │ × │ x1 │ = │ y1 │
/// │ a20 a21 a22 a23 │   │ x2 │   │ y2 │
/// └ a30 a31 a32 a33 ┘   └ x3 ┘   └ y3 ┘
/// ```
///
/// ```
/// use algea::{Vector, column_major::{Matrix, MatrixVectorProduct}};
///
/// fn multiply<T: MatrixVectorProduct<4, 4>>(
///     matrix: Matrix<T, 4, 4>,
///     vector: Vector<T, 4>,
/// ) -> Vector<T, 4> {
///     matrix * vector
/// }
/// ```
pub trait MatrixVectorProduct<const R: usize, const C: usize>: Element<R, C> {
    #[doc(hidden)]
    fn __matrix_vector_product(lhs: Matrix<Self, R, C>, rhs: Vector<Self, C>) -> Vector<Self, R>;
}
/// Enables an outer product between an `R`-lane column vector and a `1 × C` matrix
/// with the `*` operator.
///
/// When `T` implements this trait, `Vector<T, R>` implements
/// [`core::ops::Mul`]`<Matrix<T, 1, C>, Output = Matrix<T, R, C>>`.
///
/// ```text
/// ┌ x0 ┐                     ┌ x0*y0 x0*y1 x0*y2 x0*y3 ┐
/// │ x1 │                     │ x1*y0 x1*y1 x1*y2 x1*y3 │
/// │ x2 │ × [ y0 y1 y2 y3 ] = │ x2*y0 x2*y1 x2*y2 x2*y3 │
/// └ x3 ┘                     └ x3*y0 x3*y1 x3*y2 x3*y3 ┘
/// ```
///
/// ```
/// use algea::{Vector, column_major::{Matrix, OuterProduct}};
///
/// fn outer_product<T: OuterProduct<4, 4>>(
///     column: Vector<T, 4>,
///     row: Matrix<T, 1, 4>,
/// ) -> Matrix<T, 4, 4> {
///     column * row
/// }
/// ```
pub trait OuterProduct<const R: usize, const C: usize>: Element<R, C> {
    #[doc(hidden)]
    fn __outer_product(lhs: Vector<Self, R>, rhs: Matrix<Self, 1, C>) -> Matrix<Self, R, C>;
}

impl<T: MatrixProduct<R, K, C>, const R: usize, const K: usize, const C: usize>
    core::ops::Mul<Matrix<T, K, C>> for Matrix<T, R, K>
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, K, C>) -> Self::Output {
        MatrixProduct::__matrix_product(self, rhs)
    }
}
impl<T: MatrixVectorProduct<R, C>, const R: usize, const C: usize> core::ops::Mul<Vector<T, C>>
    for Matrix<T, R, C>
{
    type Output = Vector<T, R>;
    #[inline]
    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        MatrixVectorProduct::__matrix_vector_product(self, rhs)
    }
}
impl<T: OuterProduct<R, C>, const R: usize, const C: usize> core::ops::Mul<Matrix<T, 1, C>>
    for Vector<T, R>
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, 1, C>) -> Self::Output { OuterProduct::__outer_product(self, rhs) }
}

// TODO(matrix-element-generalization): Replace the concrete scalar implementations with a sealed
// matrix-element bound only after every integer and floating-point shape is verified; integer
// kernels must retain a non-FMA path.
macro_rules! impl_mat_mul_mat {
    ($self:ident, [$($a:literal),*]; $b:tt; $c:tt) => {
        $(impl_mat_mul_mat!(@a $self, $a; $b; $c);)*
    };
    (@a $self:ident, $a:literal; [$($b:literal),*]; $c:tt) => {
        $(impl_mat_mul_mat!(@ab $self, $a; $b; $c);)*
    };
    (@ab $self:ident, $a:literal; $b:literal; [$($c:literal),*]) => {
        $(
            paste::paste! {
                impl_mat_mul_mat!(@c $self, $a, $b, $c, [<matmul $a x $b x $c>]);
            }
        )*
    };
    (@c $self:ident, $a:literal, $b:literal, $c:literal, $f:ident) => {
        impl MatrixProduct<$a, $b, $c> for $self {
            #[doc(hidden)]
            #[inline(always)]
            fn __matrix_product(
                lhs: Matrix<Self, $a, $b>,
                rhs: Matrix<Self, $b, $c>,
            ) -> Matrix<Self, $a, $c> {
                Matrix { storage: matmul::$self::$f(lhs.storage.load(), rhs.storage.load()).store() }
            }
        }
    };
}

macro_rules! impl_vec_mul_mat {
    ($self:ident, $a:literal, $b:literal, $f:ident) => {
        impl MatrixVectorProduct<$a, $b> for $self {
            #[doc(hidden)]
            #[inline(always)]
            fn __matrix_vector_product(
                lhs: Matrix<Self, $a, $b>,
                rhs: Vector<Self, $b>,
            ) -> Vector<Self, $a> {
                Vector {
                    storage: matmul::$self::$f(lhs.storage.load(), rhs.storage.load()).store(),
                }
            }
        }
    };
}

macro_rules! impl_mat_mul_vec {
    ($self:ident, $a:literal, $b:literal, $f:ident) => {
        impl OuterProduct<$a, $b> for $self {
            #[doc(hidden)]
            #[inline(always)]
            fn __outer_product(
                lhs: Vector<Self, $a>,
                rhs: Matrix<Self, 1, $b>,
            ) -> Matrix<Self, $a, $b> {
                Matrix {
                    storage: matmul::$self::$f(lhs.storage.load(), rhs.storage.load()).store(),
                }
            }
        }
    };
}

macro_rules! impl_mat_mul_float {
    ($($self:ident),+) => {
        $(
            impl_mat_mul_mat!($self, [1, 2, 3, 4]; [1, 2, 3, 4]; [1, 2, 3, 4]);

            impl_vec_mul_mat!($self, 1, 1, matmul1x1x1);
            impl_vec_mul_mat!($self, 1, 2, matmul1x2x1);
            impl_vec_mul_mat!($self, 1, 3, matmul1x3x1);
            impl_vec_mul_mat!($self, 1, 4, matmul1x4x1);
            impl_vec_mul_mat!($self, 2, 1, matmul2x1x1);
            impl_vec_mul_mat!($self, 2, 2, matmul2x2x1);
            impl_vec_mul_mat!($self, 2, 3, matmul2x3x1);
            impl_vec_mul_mat!($self, 2, 4, matmul2x4x1);
            impl_vec_mul_mat!($self, 3, 1, matmul3x1x1);
            impl_vec_mul_mat!($self, 3, 2, matmul3x2x1);
            impl_vec_mul_mat!($self, 3, 3, matmul3x3x1);
            impl_vec_mul_mat!($self, 3, 4, matmul3x4x1);
            impl_vec_mul_mat!($self, 4, 1, matmul4x1x1);
            impl_vec_mul_mat!($self, 4, 2, matmul4x2x1);
            impl_vec_mul_mat!($self, 4, 3, matmul4x3x1);
            impl_vec_mul_mat!($self, 4, 4, matmul4x4x1);

            impl_mat_mul_vec!($self, 1, 1, matmul1x1x1);
            impl_mat_mul_vec!($self, 1, 2, matmul1x1x2);
            impl_mat_mul_vec!($self, 1, 3, matmul1x1x3);
            impl_mat_mul_vec!($self, 1, 4, matmul1x1x4);
            impl_mat_mul_vec!($self, 2, 1, matmul2x1x1);
            impl_mat_mul_vec!($self, 2, 2, matmul2x1x2);
            impl_mat_mul_vec!($self, 2, 3, matmul2x1x3);
            impl_mat_mul_vec!($self, 2, 4, matmul2x1x4);
            impl_mat_mul_vec!($self, 3, 1, matmul3x1x1);
            impl_mat_mul_vec!($self, 3, 2, matmul3x1x2);
            impl_mat_mul_vec!($self, 3, 3, matmul3x1x3);
            impl_mat_mul_vec!($self, 3, 4, matmul3x1x4);
            impl_mat_mul_vec!($self, 4, 1, matmul4x1x1);
            impl_mat_mul_vec!($self, 4, 2, matmul4x1x2);
            impl_mat_mul_vec!($self, 4, 3, matmul4x1x3);
            impl_mat_mul_vec!($self, 4, 4, matmul4x1x4);
        )+
    };
}
impl_mat_mul_float!(f32, f64);
