use crate::{Element, Vector, api::vector, private};

pub enum Dimension<const D: usize> {}

#[expect(private_bounds)]
pub trait AtLeast<const MIN: usize>: private::Sealed {}
#[expect(private_bounds)]
pub trait AtMost<const MAX: usize>: private::Sealed {}

impl<const N: usize> private::Sealed for Dimension<N> {}

impl AtLeast<1> for Dimension<1> {}
impl AtLeast<1> for Dimension<2> {}
impl AtLeast<2> for Dimension<2> {}
impl AtLeast<1> for Dimension<3> {}
impl AtLeast<2> for Dimension<3> {}
impl AtLeast<3> for Dimension<3> {}
impl AtLeast<1> for Dimension<4> {}
impl AtLeast<2> for Dimension<4> {}
impl AtLeast<3> for Dimension<4> {}
impl AtLeast<4> for Dimension<4> {}

impl AtMost<1> for Dimension<1> {}
impl AtMost<2> for Dimension<1> {}
impl AtMost<2> for Dimension<2> {}
impl AtMost<3> for Dimension<1> {}
impl AtMost<3> for Dimension<2> {}
impl AtMost<3> for Dimension<3> {}
impl AtMost<4> for Dimension<1> {}
impl AtMost<4> for Dimension<2> {}
impl AtMost<4> for Dimension<3> {}
impl AtMost<4> for Dimension<4> {}

pub enum __ConcatDispatch {}

impl private::Sealed for __ConcatDispatch {}

#[expect(private_bounds)]
pub trait __IntoVector<T: Element<D>, const D: usize>: private::Sealed {
    fn __into_vector(value: Self) -> Vector<T, D>;
}

impl<T: Element> __IntoVector<T, 1> for T {
    #[inline]
    fn __into_vector(value: Self) -> Vector<T, 1> { Vector::from_array([value]) }
}

impl<T: Element<D>, const D: usize> private::Sealed for Vector<T, D> {}

impl<T: Element<D>, const D: usize> __IntoVector<T, D> for Vector<T, D> {
    #[inline]
    fn __into_vector(value: Self) -> Vector<T, D> { value }
}

#[expect(private_bounds)]
pub trait __Concat4<A, B, C, D>: private::Sealed {
    type __Output;
    fn __concat(a: A, b: B, c: C, d: D) -> Self::__Output;
}

#[expect(private_bounds)]
pub trait __Concat3<A, B, C>: private::Sealed {
    type __Output;
    fn __concat(a: A, b: B, c: C) -> Self::__Output;
}

#[expect(private_bounds)]
pub trait __Concat2<A, B>: private::Sealed {
    type __Output;
    fn __concat(a: A, b: B) -> Self::__Output;
}

impl<T> __Concat4<Vector<T, 1>, Vector<T, 1>, Vector<T, 1>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(
        a: Vector<T, 1>,
        b: Vector<T, 1>,
        c: Vector<T, 1>,
        d: Vector<T, 1>,
    ) -> Self::__Output {
        let [a0] = a.to_array();
        let [b0] = b.to_array();
        let [c0] = c.to_array();
        let [d0] = d.to_array();
        Vector::from_array([a0, b0, c0, d0])
    }
}

impl<T> __Concat3<Vector<T, 1>, Vector<T, 1>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<3>,
{
    type __Output = Vector<T, 3>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 1>, c: Vector<T, 1>) -> Self::__Output {
        let [a0] = a.to_array();
        let [b0] = b.to_array();
        let [c0] = c.to_array();
        Vector::from_array([a0, b0, c0])
    }
}

impl<T> __Concat3<Vector<T, 1>, Vector<T, 1>, Vector<T, 2>> for __ConcatDispatch
where
    T: Element<1> + Element<2> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 1>, c: Vector<T, 2>) -> Self::__Output {
        let [a0] = a.to_array();
        let [b0] = b.to_array();
        let [c0, c1] = c.to_array();
        Vector::from_array([a0, b0, c0, c1])
    }
}

impl<T> __Concat3<Vector<T, 1>, Vector<T, 2>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<2> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 2>, c: Vector<T, 1>) -> Self::__Output {
        let [a0] = a.to_array();
        let [b0, b1] = b.to_array();
        let [c0] = c.to_array();
        Vector::from_array([a0, b0, b1, c0])
    }
}

impl<T> __Concat3<Vector<T, 2>, Vector<T, 1>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<2> + Element<4>,
{
    type __Output = Vector<T, 4>;

    #[inline]
    fn __concat(a: Vector<T, 2>, b: Vector<T, 1>, c: Vector<T, 1>) -> Self::__Output {
        let [a0, a1] = a.to_array();
        let [b0] = b.to_array();
        let [c0] = c.to_array();
        Vector::from_array([a0, a1, b0, c0])
    }
}

impl<T> __Concat2<Vector<T, 1>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<2>,
{
    type __Output = Vector<T, 2>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 1>) -> Self::__Output {
        Vector { storage: vector::call!(<T, 2>::vector_concat_1_1(a.storage, b.storage)) }
    }
}

impl<T> __Concat2<Vector<T, 1>, Vector<T, 2>> for __ConcatDispatch
where
    T: Element<1> + Element<2> + Element<3>,
{
    type __Output = Vector<T, 3>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 2>) -> Self::__Output {
        Vector { storage: vector::call!(<T, 3>::vector_concat_1_2(a.storage, b.storage)) }
    }
}

impl<T> __Concat2<Vector<T, 1>, Vector<T, 3>> for __ConcatDispatch
where
    T: Element<1> + Element<3> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(a: Vector<T, 1>, b: Vector<T, 3>) -> Self::__Output {
        let [a0] = a.to_array();
        let [b0, b1, b2] = b.to_array();
        Vector::from_array([a0, b0, b1, b2])
    }
}

impl<T> __Concat2<Vector<T, 2>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<2> + Element<3>,
{
    type __Output = Vector<T, 3>;
    #[inline]
    fn __concat(a: Vector<T, 2>, b: Vector<T, 1>) -> Self::__Output {
        Vector { storage: vector::call!(<T, 2>::vector_concat_2_1(a.storage, b.storage)) }
    }
}

impl<T> __Concat2<Vector<T, 2>, Vector<T, 2>> for __ConcatDispatch
where
    T: Element<2> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(a: Vector<T, 2>, b: Vector<T, 2>) -> Self::__Output {
        let [a0, a1] = a.to_array();
        let [b0, b1] = b.to_array();
        Vector::from_array([a0, a1, b0, b1])
    }
}

impl<T> __Concat2<Vector<T, 3>, Vector<T, 1>> for __ConcatDispatch
where
    T: Element<1> + Element<3> + Element<4>,
{
    type __Output = Vector<T, 4>;
    #[inline]
    fn __concat(a: Vector<T, 3>, b: Vector<T, 1>) -> Self::__Output {
        let [a0, a1, a2] = a.to_array();
        let [b0] = b.to_array();
        Vector::from_array([a0, a1, a2, b0])
    }
}
