use crate::{
    Vector,
    support::{Element, Float, RowMajor},
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
pub type Matrix<T, const R: usize, const C: usize> = crate::Matrix<T, R, C, RowMajor>;

macro_rules! call {
    (<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:tt>)? $(($($arg:expr),*))?) => {
        <$crate::private::ConstStorage<$t, $c, $r> as $crate::private::StorageOps<
            $t,
            $crate::support::Dimension<$c>,
            $crate::support::Dimension<$r>,
        >>::$f $(::<$gen>)? $(($($arg),*))?
    };
    ($w:ident(<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:tt>)? $(($($arg:expr),*))?)) => {
        $w { storage: $crate::row_major::call!(<$t, $r, $c>::$f $(::<$gen>)? $(($($arg),*))?) }
    };
}
pub(crate) use call;

/// Enables multiplication of an `R`-lane row vector by an `R × C` matrix with the
/// `*` operator.
///
/// ```text
///                   ┌ a00 a01 a02 a03 ┐
/// [ x0 x1 x2 x3 ] × │ a10 a11 a12 a13 │ = [ y0 y1 y2 y3 ]
///                   │ a20 a21 a22 a23 │
///                   └ a30 a31 a32 a33 ┘
/// ```
///
/// ```
/// use algea::{Vector, row_major::Matrix};
///
/// let vector = Vector::<f32, 2>::from([2.0, 3.0]);
/// let matrix = Matrix::<f32, 2, 3>::from_rows([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// assert_eq!((vector * matrix).to_array(), [29.0, 34.0, 39.0]);
/// ```
impl<T: Float + Element<R, C>, const R: usize, const C: usize> core::ops::Mul<Matrix<T, R, C>>
    for Vector<T, R>
{
    type Output = Vector<T, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, R, C>) -> Self::Output {
        call!(Vector(<T, 1, C>::matmul::<R>(rhs.storage, self.storage)))
    }
}

/// Computes an outer product between an `R × 1` matrix and a `C`-lane row vector with the
/// `*` operator.
///
/// ```text
/// ┌ x0 ┐                     ┌ x0*y0 x0*y1 x0*y2 x0*y3 ┐
/// │ x1 │                     │ x1*y0 x1*y1 x1*y2 x1*y3 │
/// │ x2 │ × [ y0 y1 y2 y3 ] = │ x2*y0 x2*y1 x2*y2 x2*y3 │
/// └ x3 ┘                     └ x3*y0 x3*y1 x3*y2 x3*y3 ┘
/// ```
///
/// ```
/// use algea::{Vector, row_major::Matrix};
///
/// let column = Matrix::<f32, 2, 1>::from_rows([[2.0], [3.0]]);
/// let row = Vector::<f32, 3>::from([4.0, 5.0, 6.0]);
/// assert_eq!((column * row).to_rows(), [[8.0, 10.0, 12.0], [12.0, 15.0, 18.0]]);
/// ```
impl<T: Float + Element<R, C>, const R: usize, const C: usize> core::ops::Mul<Vector<T, C>>
    for Matrix<T, R, 1>
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        call!(Matrix(<T, R, C>::matmul::<1>(rhs.storage, self.storage)))
    }
}

// Vector assignment follows the row-vector orientation. Column-vector multiplication has the
// matrix on the left and therefore has no symmetric `MulAssign` form.
impl<T: Float + Element<N>, const N: usize> core::ops::MulAssign<Matrix<T, N, N>> for Vector<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) { *self = *self * rhs; }
}
