use crate::{
    private::{
        ConstStorage,
        Indices2,
        Indices3,
        Indices4,
        SealedSupportedDimension,
        SealedSupportedElement,
        SwizzleDispatch,
    },
    support::Dimension,
};

// `SwizzleDispatch::dispatch` is never actually called on this backend: `swizzle2`/`swizzle3`/
// `swizzle4` in `non_simd.rs` swizzle directly by indexing the `[[T; M]; N]` array (e.g.
// `[[a[0][I0], a[0][I1]]]`) instead of going through `SwizzleDispatch`. These blanket impls exist
// only so that `SwizzleDispatchAny<N>` — which the `D`-generic `Vector<T, D>` swizzle accessors in
// `src/swizzle.rs` bound on — has something to be satisfied by under this backend too; the SIMD
// backend is the one that actually dispatches through `SwizzleDispatch` (see `src/simd/utils.rs`).
impl<T, const M: usize, const N: usize, const I0: usize, const I1: usize> SwizzleDispatch<T, M, N>
    for Indices2<I0, I1>
{
    fn dispatch(_v: ConstStorage<T, M>) -> ConstStorage<T, N>
    where
        T: SealedSupportedElement,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension,
    {
        unimplemented!()
    }
}
impl<T, const M: usize, const N: usize, const I0: usize, const I1: usize, const I2: usize>
    SwizzleDispatch<T, M, N> for Indices3<I0, I1, I2>
{
    fn dispatch(_v: ConstStorage<T, M>) -> ConstStorage<T, N>
    where
        T: SealedSupportedElement,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension,
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
    fn dispatch(_v: ConstStorage<T, M>) -> ConstStorage<T, N>
    where
        T: SealedSupportedElement,
        Dimension<M>: SealedSupportedDimension,
        Dimension<N>: SealedSupportedDimension,
    {
        unimplemented!()
    }
}
