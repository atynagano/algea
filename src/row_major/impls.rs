use super::{Matrix, call};
use crate::{__internal, Element, FloatElement, Vector};

impl<T: Element<D, D>, const D: usize> Matrix<T, D, D> {
    /// The identity matrix.
    pub const IDENTITY: Self = call!(Self(<T, D, D>::IDENTITY));

    /// Returns the main diagonal as a vector.
    #[inline]
    pub fn diagonal(self) -> Vector<T, D> { call!(Vector(<T, D, D>::diagonal(self.storage))) }
}

impl<T: FloatElement<D>, const D: usize> Matrix<T, D, D>
where
    __internal::Dimension<D>: __internal::AtMost<4>,
{
    /// Returns the multiplicative inverse of the matrix.
    #[inline]
    pub fn inverse(self) -> Self { call!(Self(<T, D, D>::inverse(self.storage))) }
    /// Returns the determinant of the matrix.
    #[inline]
    pub fn determinant(self) -> T { call!(<T, D, D>::determinant(self.storage)) }
}
