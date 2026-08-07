use crate::utils::ArithPrimitive;

impl<T: ArithPrimitive<Scalar = T>, const N: usize> ArithPrimitive for [T; N] {
    type Scalar = T;
    type F32 = [T::F32; N];
    type I32 = [T::I32; N];
    type U32 = [T::U32; N];
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
    fn cast_from_f32_(a: Self::F32) -> Self { a.map(T::cast_from_f32_) }
    #[inline(always)]
    fn cast_from_i32_(a: Self::I32) -> Self { a.map(T::cast_from_i32_) }
    #[inline(always)]
    fn cast_from_u32_(a: Self::U32) -> Self { a.map(T::cast_from_u32_) }
    #[inline(always)]
    fn add_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::add_noexcept_(self[i], rhs[i])
        })
    }
    #[inline(always)]
    fn sub_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::sub_noexcept_(self[i], rhs[i])
        })
    }
    #[inline(always)]
    fn mul_noexcept_(self, rhs: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::mul_noexcept_(self[i], rhs[i])
        })
    }
    #[inline(always)]
    fn mul_add_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::mul_add_(a[i], b[i], c[i])
        })
    }
    #[inline(always)]
    fn mul_sub_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::mul_sub_(a[i], b[i], c[i])
        })
    }
    #[inline(always)]
    fn neg_mul_add_(a: Self, b: Self, c: Self) -> Self {
        core::array::from_fn({
            #[inline(always)]
            |i| T::neg_mul_add_(a[i], b[i], c[i])
        })
    }
}
