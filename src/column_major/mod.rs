use crate::{FloatElement, Vector};

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
pub type Matrix<T, const R: usize, const C: usize> = crate::Matrix<T, R, C, ColumnMajor>;

macro_rules! call {
    (<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:tt>)? $(($($arg:expr),*))?) => {
        <$crate::private::ConstStorage<$t, $r, $c> as $crate::private::StorageOps<
            $t,
            $crate::support::Dimension<$r>,
            $crate::support::Dimension<$c>,
        >>::$f $(::<$gen>)? $(($($arg),*))?
    };
    ($w:ident(<$t:ty, $r:tt, $c:tt>::$f:ident $(::<$gen:tt>)? $(($($arg:expr),*))?)) => {
        $w { storage: $crate::column_major::call!(<$t, $r, $c>::$f $(::<$gen>)? $(($($arg),*))?) }
    };
}
use crate::marker::ColumnMajor;
pub(crate) use call;

/// Enables multiplication of an `R × C` matrix by a `C`-lane column vector with the
/// `*` operator.
///
/// ```text
/// ┌ a00 a01 a02 a03 ┐   ┌ x0 ┐   ┌ y0 ┐
/// │ a10 a11 a12 a13 │ × │ x1 │ = │ y1 │
/// │ a20 a21 a22 a23 │   │ x2 │   │ y2 │
/// └ a30 a31 a32 a33 ┘   └ x3 ┘   └ y3 ┘
/// ```
///
/// ```
/// use algea::{Vector, column_major::Matrix};
///
/// let matrix = Matrix::<f32, 3, 2>::from_columns([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let vector = Vector::<f32, 2>::from([2.0, 3.0]);
/// assert_eq!((matrix * vector).to_array(), [29.0, 34.0, 39.0]);
/// ```
impl<T: FloatElement<R, C>, const R: usize, const C: usize> core::ops::Mul<Vector<T, C>>
    for Matrix<T, R, C>
{
    type Output = Vector<T, R>;
    #[inline]
    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        call!(Vector(<T, R, 1>::matmul::<C>(self.storage, rhs.storage)))
    }
}

/// Computes an outer product between an `R`-lane column vector and a `1 × C` matrix with the
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
/// use algea::{Vector, column_major::Matrix};
///
/// let column = Vector::<f32, 2>::from([2.0, 3.0]);
/// let row = Matrix::<f32, 1, 3>::from_columns([[4.0], [5.0], [6.0]]);
/// assert_eq!((column * row).to_columns(), [[8.0, 12.0], [10.0, 15.0], [12.0, 18.0]]);
/// ```
impl<T: FloatElement<R, C>, const R: usize, const C: usize> core::ops::Mul<Matrix<T, 1, C>>
    for Vector<T, R>
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, 1, C>) -> Self::Output {
        call!(Matrix(<T, R, C>::matmul::<1>(self.storage, rhs.storage)))
    }
}
