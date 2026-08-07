use super::{Element, private};
use crate::{
    Vector,
    kernels::matmul::f32::*,
    utils::{Load, Store},
};

/// A fixed-size matrix stored as row vectors.
///
/// For a four-by-four matrix, the storage is organized as follows, where each
/// `rN` is a [`Vector`] row:
///
/// ```text
///     ┌                         ┐
/// r0  │ r0[0] r0[1] r0[2] r0[3] │
/// r1  │ r1[0] r1[1] r1[2] r1[3] │
/// r2  │ r2[0] r2[1] r2[2] r2[3] │
/// r3  │ r3[0] r3[1] r3[2] r3[3] │
///     └                         ┘
/// ```
pub struct Matrix<T: Element<R, C>, const R: usize, const C: usize> {
    pub(crate) storage: <T as private::SealedElement<C, R>>::Storage,
}

mod impls;

macro_rules! call {
    (<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?) => {
        <$t as $crate::private::SealedElement<$c, $r>>::$f $(::<$gen>)? $(($($arg),*))?
    };
    ($w:ident(<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:ty>)? $(($($arg:expr),*))?)) => {
        $w { storage: $crate::row_major::call!(<$t, $r, $c>::$f $(::<$gen>)? $(($($arg),*))?) }
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
/// use algea::row_major::{Matrix, MatrixProduct};
///
/// fn multiply<T: MatrixProduct<4, 4, 4>>(
///     a: Matrix<T, 4, 4>,
///     b: Matrix<T, 4, 4>,
/// ) -> Matrix<T, 4, 4> {
///     a * b
/// }
/// ```
pub trait MatrixProduct<const R: usize, const K: usize, const C: usize>:
    Element<R, K> + OuterProduct<R, C> + VectorMatrixProduct<K, C>
{
    #[doc(hidden)]
    fn __matrix_product(lhs: Matrix<Self, R, K>, rhs: Matrix<Self, K, C>) -> Matrix<Self, R, C>;
}
/// Enables multiplication of an `R`-lane row vector by an `R × C` matrix with the
/// `*` operator.
///
/// When `T` implements this trait, `Vector<T, R>` implements
/// [`core::ops::Mul`]`<Matrix<T, R, C>, Output = Vector<T, C>>`.
///
/// ```text
///                   ┌ a00 a01 a02 a03 ┐
/// [ x0 x1 x2 x3 ] × │ a10 a11 a12 a13 │ = [ y0 y1 y2 y3 ]
///                   │ a20 a21 a22 a23 │
///                   └ a30 a31 a32 a33 ┘
/// ```
///
/// ```
/// use algea::{Vector, row_major::{Matrix, VectorMatrixProduct}};
///
/// fn multiply<T: VectorMatrixProduct<4, 4>>(
///     vector: Vector<T, 4>,
///     matrix: Matrix<T, 4, 4>,
/// ) -> Vector<T, 4> {
///     vector * matrix
/// }
/// ```
pub trait VectorMatrixProduct<const R: usize, const C: usize>: Element<R, C> {
    #[doc(hidden)]
    fn __vector_matrix_product(lhs: Vector<Self, R>, rhs: Matrix<Self, R, C>) -> Vector<Self, C>;
}
/// Enables an outer product between an `R × 1` matrix and a `C`-lane row vector with
/// the `*` operator.
///
/// When `T` implements this trait, `Matrix<T, R, 1>` implements
/// [`core::ops::Mul`]`<Vector<T, C>, Output = Matrix<T, R, C>>`.
///
/// ```text
/// ┌ x0 ┐                     ┌ x0*y0 x0*y1 x0*y2 x0*y3 ┐
/// │ x1 │                     │ x1*y0 x1*y1 x1*y2 x1*y3 │
/// │ x2 │ × [ y0 y1 y2 y3 ] = │ x2*y0 x2*y1 x2*y2 x2*y3 │
/// └ x3 ┘                     └ x3*y0 x3*y1 x3*y2 x3*y3 ┘
/// ```
///
/// ```
/// use algea::{Vector, row_major::{Matrix, OuterProduct}};
///
/// fn outer_product<T: OuterProduct<4, 4>>(
///     column: Matrix<T, 4, 1>,
///     row: Vector<T, 4>,
/// ) -> Matrix<T, 4, 4> {
///     column * row
/// }
/// ```
pub trait OuterProduct<const R: usize, const C: usize>: Element<R, C> {
    #[doc(hidden)]
    fn __outer_product(lhs: Matrix<Self, R, 1>, rhs: Vector<Self, C>) -> Matrix<Self, R, C>;
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
impl<T: VectorMatrixProduct<R, C>, const R: usize, const C: usize> core::ops::Mul<Matrix<T, R, C>>
    for Vector<T, R>
{
    type Output = Vector<T, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, R, C>) -> Self::Output {
        VectorMatrixProduct::__vector_matrix_product(self, rhs)
    }
}
impl<T: OuterProduct<R, C>, const R: usize, const C: usize> core::ops::Mul<Vector<T, C>>
    for Matrix<T, R, 1>
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn mul(self, rhs: Vector<T, C>) -> Self::Output { OuterProduct::__outer_product(self, rhs) }
}

// Assignment is available only for the row-major vector-times-matrix orientation.
impl<T: VectorMatrixProduct<N, N>, const N: usize> core::ops::MulAssign<Matrix<T, N, N>>
    for Vector<T, N>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) { *self = *self * rhs; }
}

// TODO(matrix-element-generalization): Replace the concrete scalar implementations with a sealed
// matrix-element bound only after every integer and floating-point shape is verified; integer
// kernels must retain a non-FMA path.
macro_rules! impl_mat_mul_mat {
    ([$($a:literal),*]; $b:tt; $c:tt) => {
        $(impl_mat_mul_mat!(@a $a; $b; $c);)*
    };
    (@a $a:literal; [$($b:literal),*]; $c:tt) => {
        $(impl_mat_mul_mat!(@ab $a; $b; $c);)*
    };
    (@ab $a:literal; $b:literal; [$($c:literal),*]) => {
        $(
            paste::paste! {
                impl_mat_mul_mat!(@c $a, $b, $c, [<matmul $c x $b x $a>]);
            }
        )*
    };
    (@c $a:literal, $b:literal, $c:literal, $f:ident) => {
        impl MatrixProduct<$a, $b, $c> for f32 {
            #[doc(hidden)]
            #[inline(always)]
            fn __matrix_product(
                lhs: Matrix<Self, $a, $b>,
                rhs: Matrix<Self, $b, $c>,
            ) -> Matrix<Self, $a, $c> {
                Matrix { storage: $f(rhs.storage.load(), lhs.storage.load()).store() }
            }
        }
    };
}

impl_mat_mul_mat!([1, 2, 3, 4]; [1, 2, 3, 4]; [1, 2, 3, 4]);

macro_rules! impl_vec_mul_mat {
    ($a:literal, $b:literal, $f:ident) => {
        impl VectorMatrixProduct<$a, $b> for f32 {
            #[doc(hidden)]
            #[inline(always)]
            fn __vector_matrix_product(
                lhs: Vector<Self, $a>,
                rhs: Matrix<Self, $a, $b>,
            ) -> Vector<Self, $b> {
                Vector { storage: $f(rhs.storage.load(), lhs.storage.load()).store() }
            }
        }
    };
}
impl_vec_mul_mat!(1, 1, matmul1x1x1);
impl_vec_mul_mat!(1, 2, matmul2x1x1);
impl_vec_mul_mat!(1, 3, matmul3x1x1);
impl_vec_mul_mat!(1, 4, matmul4x1x1);
impl_vec_mul_mat!(2, 1, matmul1x2x1);
impl_vec_mul_mat!(2, 2, matmul2x2x1);
impl_vec_mul_mat!(2, 3, matmul3x2x1);
impl_vec_mul_mat!(2, 4, matmul4x2x1);
impl_vec_mul_mat!(3, 1, matmul1x3x1);
impl_vec_mul_mat!(3, 2, matmul2x3x1);
impl_vec_mul_mat!(3, 3, matmul3x3x1);
impl_vec_mul_mat!(3, 4, matmul4x3x1);
impl_vec_mul_mat!(4, 1, matmul1x4x1);
impl_vec_mul_mat!(4, 2, matmul2x4x1);
impl_vec_mul_mat!(4, 3, matmul3x4x1);
impl_vec_mul_mat!(4, 4, matmul4x4x1);

macro_rules! impl_mat_mul_vec {
    ($a:literal, $b:literal, $f:ident) => {
        impl OuterProduct<$a, $b> for f32 {
            #[doc(hidden)]
            #[inline(always)]
            fn __outer_product(
                lhs: Matrix<Self, $a, 1>,
                rhs: Vector<Self, $b>,
            ) -> Matrix<Self, $a, $b> {
                Matrix { storage: $f(rhs.storage.load(), lhs.storage.load()).store() }
            }
        }
    };
}
impl_mat_mul_vec!(1, 1, matmul1x1x1);
impl_mat_mul_vec!(1, 2, matmul2x1x1);
impl_mat_mul_vec!(1, 3, matmul3x1x1);
impl_mat_mul_vec!(1, 4, matmul4x1x1);
impl_mat_mul_vec!(2, 1, matmul1x1x2);
impl_mat_mul_vec!(2, 2, matmul2x1x2);
impl_mat_mul_vec!(2, 3, matmul3x1x2);
impl_mat_mul_vec!(2, 4, matmul4x1x2);
impl_mat_mul_vec!(3, 1, matmul1x1x3);
impl_mat_mul_vec!(3, 2, matmul2x1x3);
impl_mat_mul_vec!(3, 3, matmul3x1x3);
impl_mat_mul_vec!(3, 4, matmul4x1x3);
impl_mat_mul_vec!(4, 1, matmul1x1x4);
impl_mat_mul_vec!(4, 2, matmul2x1x4);
impl_mat_mul_vec!(4, 3, matmul3x1x4);
impl_mat_mul_vec!(4, 4, matmul4x1x4);
