use crate::{
    private::{Indices2, Indices3, Indices4, SealedElement, SwizzleDispatch},
    utils::ArithPrimitive,
};

impl<T: ArithPrimitive<Scalar = T>, const N: usize> ArithPrimitive for [T; N] {
    type Scalar = T;
    type F32 = [T::F32; N];
    type F64 = [T::F64; N];
    type I32 = [T::I32; N];
    type I64 = [T::I64; N];
    type U32 = [T::U32; N];
    type U64 = [T::U64; N];
    type Mask = [T::Mask; N];

    const ZERO_: Self = [T::ZERO_; N];
    const ONE_: Self = [T::ONE_; N];

    #[inline(always)]
    fn filled_(a: Self::Scalar) -> Self { [a; N] }
    #[inline(always)]
    fn as_array_(&self) -> &[Self::Scalar] { self }
    #[inline(always)]
    fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { self }

    #[inline(always)]
    fn add_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::add_noexcept_(self[i], rhs[i]),
        )
    }
    #[inline(always)]
    fn sub_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::sub_noexcept_(self[i], rhs[i]),
        )
    }
    #[inline(always)]
    fn mul_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::mul_noexcept_(self[i], rhs[i]),
        )
    }
    #[inline(always)]
    fn mul_add_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::mul_add_(a[i], b[i], c[i]),
        )
    }
    #[inline(always)]
    fn mul_sub_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::mul_sub_(a[i], b[i], c[i]),
        )
    }
    #[inline(always)]
    fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn(
            #[inline(always)]
            |i| T::neg_mul_add_(a[i], b[i], c[i]),
        )
    }
}

// `SwizzleDispatch::dispatch` is never actually called on this backend: `swizzle2`/`swizzle3`/
// `swizzle4` in `non_simd.rs` swizzle directly by indexing the `[[T; M]; N]` array (e.g.
// `[[a[0][I0], a[0][I1]]]`) instead of going through `SwizzleDispatch`. These blanket impls exist
// only so that `SwizzleDispatchAny<N>` — which the `D`-generic `Vector<T, D>` swizzle accessors in
// `src/swizzle.rs` bound on — has something to be satisfied by under this backend too; the SIMD
// backend is the one that actually dispatches through `SwizzleDispatch` (see `src/simd/utils.rs`).
impl<T, const M: usize, const N: usize, const I0: usize, const I1: usize> SwizzleDispatch<T, M, N>
    for Indices2<I0, I1>
{
    fn dispatch(_v: <T as SealedElement<M, 1>>::Storage) -> <T as SealedElement<N, 1>>::Storage
    where
        T: SealedElement<M, 1> + SealedElement<N, 1>,
    {
        unimplemented!()
    }
}
impl<T, const M: usize, const N: usize, const I0: usize, const I1: usize, const I2: usize>
    SwizzleDispatch<T, M, N> for Indices3<I0, I1, I2>
{
    fn dispatch(_v: <T as SealedElement<M, 1>>::Storage) -> <T as SealedElement<N, 1>>::Storage
    where
        T: SealedElement<M, 1> + SealedElement<N, 1>,
    {
        unimplemented!()
    }
}
impl<
    T,
    const M: usize,
    const N: usize,
    const I0: usize,
    const I1: usize,
    const I2: usize,
    const I3: usize,
> SwizzleDispatch<T, M, N> for Indices4<I0, I1, I2, I3>
{
    fn dispatch(_v: <T as SealedElement<M, 1>>::Storage) -> <T as SealedElement<N, 1>>::Storage
    where
        T: SealedElement<M, 1> + SealedElement<N, 1>,
    {
        unimplemented!()
    }
}
