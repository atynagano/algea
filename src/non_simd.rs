pub(crate) mod kernels;
mod utils;

use crate::{
    private::{self, ConstStorage, DimArray, DimVector, SealedSupportedElement},
    support::{Dimension, Float, Int, SupportedDimension, SupportedElement},
    utils::{ArithOps, CanonicalMask, ConstMaskStorage, MaskOps, if_, impl_default_load},
};
use definitions::SealedStorageElement;

type ConstArray<T, const N: usize> = DimArray<T, Dimension<N>>;
type ConstVector<T, const N: usize> = DimVector<T, Dimension<N>>;

impl_default_load!();

pub(crate) mod definitions {
    use crate::{
        private::{SealedSupportedElement, StorageOps},
        support::Dimension,
    };

    pub(crate) trait SealedStorageElement {
        type Storage<const R: usize, const C: usize>: StorageOps<Self, Dimension<R>, Dimension<C>>
        where
            Self: SealedSupportedElement;
    }

    pub(crate) trait SealedSupportedDimension: Sized {
        type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension>: StorageOps<T, Self, C>;
        type StorageRxN<T: SealedSupportedElement, const R: usize>: StorageOps<T, Dimension<R>, Self>;
    }

    impl<const N: usize> SealedSupportedDimension for Dimension<N> {
        type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension> =
            C::StorageRxN<T, N>;
        type StorageRxN<T: SealedSupportedElement, const R: usize> =
            <T as SealedStorageElement>::Storage<R, N>;
    }
}

#[inline(always)]
fn map1<U, T: Copy, const M: usize, const N: usize>(
    a: [[T; M]; N],
    mut f: impl FnMut(T) -> U,
) -> [[U; M]; N] {
    a.map(
        #[inline(always)]
        |b| b.map(&mut f),
    )
}
#[inline(always)]
fn map2<U, T0: Copy, T1: Copy, const M: usize, const N: usize>(
    a: [[T0; M]; N],
    b: [[T1; M]; N],
    mut f: impl FnMut(T0, T1) -> U,
) -> [[U; M]; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| {
            core::array::from_fn(
                #[inline(always)]
                |j| f(a[i][j], b[i][j]),
            )
        },
    )
}
#[inline(always)]
fn map3<U, T0: Copy, T1: Copy, T2: Copy, const M: usize, const N: usize>(
    a: [[T0; M]; N],
    b: [[T1; M]; N],
    c: [[T2; M]; N],
    mut f: impl FnMut(T0, T1, T2) -> U,
) -> [[U; M]; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| {
            core::array::from_fn(
                #[inline(always)]
                |j| f(a[i][j], b[i][j], c[i][j]),
            )
        },
    )
}

#[inline(always)]
fn map3_with_mask<U, T0: MaskOps, T1: Copy, T2: Copy, const M: usize, const N: usize>(
    mask: CanonicalMask<[[T0; M]; N]>,
    b: [[T1; M]; N],
    c: [[T2; M]; N],
    mut f: impl FnMut(CanonicalMask<T0>, T1, T2) -> U,
) -> [[U; M]; N] {
    let mask = mask.into_parts().map(
        #[inline(always)]
        |column| column.into_parts(),
    );
    core::array::from_fn(
        #[inline(always)]
        |i| {
            core::array::from_fn(
                #[inline(always)]
                |j| f(mask[i][j], b[i][j], c[i][j]),
            )
        },
    )
}

impl<const M: usize, const N: usize> CanonicalMask<[[i32; M]; N]> {
    #[inline(always)]
    pub(crate) fn cast_i64(self) -> CanonicalMask<[[i64; M]; N]> {
        // SAFETY: sign extension maps `0` to `0` and `-1` to `-1`.
        unsafe { CanonicalMask::new_unchecked(map1(self.into_inner(), i64::from)) }
    }
}
impl<const M: usize, const N: usize> CanonicalMask<[[i64; M]; N]> {
    #[inline(always)]
    pub(crate) fn cast_i32(self) -> CanonicalMask<[[i32; M]; N]> {
        // SAFETY: truncation keeps the low bits, mapping `0` to `0` and `-1` to `-1`.
        unsafe {
            CanonicalMask::new_unchecked(map1(
                self.into_inner(),
                #[inline(always)]
                |x| x as i32,
            ))
        }
    }
}

macro_rules! impl_layout {
    ((
        self: $t:ident,
        feature: [$float:tt, $int:tt, $signed:tt, $bits:tt],
    ) => {
        $($item:item)*
    }) => {
        impl SealedSupportedElement for $t {}

        impl SealedStorageElement for $t {
            type Storage<const R: usize, const C: usize> = [[$t; R]; C];
        }

        impl<const M: usize, const N: usize> private::StorageOps<$t, Dimension<M>, Dimension<N>> for [[$t; M]; N] {
            const POS_X: Self = {
                let mut a = [[0 as _; M]; N];
                a[0][0] = 1 as _;
                a
            };
            const POS_Y: Self = {
                let mut a = [[0 as _; M]; N];
                a[0][1] = 1 as _;
                a
            };
            const POS_Z: Self = {
                let mut a = [[0 as _; M]; N];
                a[0][2] = 1 as _;
                a
            };
            const POS_W: Self = {
                let mut a = [[0 as _; M]; N];
                a[0][3] = 1 as _;
                a
            };
            if_! { $signed == signed {
                const NEG_X: Self = {
                    let mut a = [[0 as _; M]; N];
                    a[0][0] = -1 as _;
                    a
                };
                const NEG_Y: Self = {
                    let mut a = [[0 as _; M]; N];
                    a[0][1] = -1 as _;
                    a
                };
                const NEG_Z: Self = {
                    let mut a = [[0 as _; M]; N];
                    a[0][2] = -1 as _;
                    a
                };
                const NEG_W: Self = {
                    let mut a = [[0 as _; M]; N];
                    a[0][3] = -1 as _;
                    a
                };
            }}
            const IDENTITY: Self = {
                let mut a = [[0 as _; M]; N];
                let mut i = 0;
                while i < M {
                    a[i][i] = 1 as _;
                    i += 1;
                }
                a
            };

            #[inline(always)]
            fn map2(a: Self, b: Self, f: impl FnMut($t, $t) -> $t) -> Self { map2(a, b, f) }
            #[inline(always)]
            fn index(a: &Self, (i, j): (usize, usize)) -> Option<&$t> {
                a.get(j).and_then(#[inline(always)] |vec| vec.get(i))
            }
            #[inline(always)]
            fn index_mut(a: &mut Self, (i, j): (usize, usize)) -> Option<&mut $t> {
                a.get_mut(j).and_then(#[inline(always)] |vec| vec.get_mut(i))
            }
            // `Vector::as_array` and `Vector::as_mut_array` are the only callers, and both
            // name a one-column shape.
            #[inline(always)]
            fn as_array_first(a: &Self) -> &ConstArray<$t, M> { &a[0] }
            #[inline(always)]
            fn as_mut_array_first(a: &mut Self) -> &mut ConstArray<$t, M> { &mut a[0] }
            #[inline(always)]
            fn to_array(a: Self) -> ConstArray<ConstArray<$t, M>, N> { a }
            #[inline(always)]
            fn from_array(a: ConstArray<ConstArray<$t, M>, N>) -> Self { a }
            #[inline(always)]
            fn from_vecs(a: ConstArray<ConstVector<$t, M>, N>) -> Self
            where
                Dimension<M>: SupportedDimension
            {
                a.map(
                    #[inline(always)]
                    |vec| private::StorageOps::to_array(vec.storage)[0]
                )
            }
            #[inline(always)]
            fn cast_from_f32(a: ConstStorage<f32, M, N>) -> Self { map1(a, $t::cast_from_f32_::<1>) }
            #[inline(always)]
            fn cast_from_i32(a: ConstStorage<i32, M, N>) -> Self { map1(a, $t::cast_from_i32_::<1>) }
            #[inline(always)]
            fn cast_from_u32(a: ConstStorage<u32, M, N>) -> Self { map1(a, $t::cast_from_u32_::<1>) }
            #[inline(always)]
            fn cast_from_f64(a: ConstStorage<f64, M, N>) -> Self { map1(a, $t::cast_from_f64_::<1>) }
            #[inline(always)]
            fn cast_from_i64(a: ConstStorage<i64, M, N>) -> Self { map1(a, $t::cast_from_i64_::<1>) }
            #[inline(always)]
            fn cast_from_u64(a: ConstStorage<u64, M, N>) -> Self { map1(a, $t::cast_from_u64_::<1>) }
            #[inline(always)]
            fn cast_from<U: SealedSupportedElement>(a: ConstStorage<U, M, N>) -> Self {
                match U::TYPE {
                    private::Type::F32 => Self::cast_from_f32(ConstStorage::<U, M, N>::substantiate_f32(a)),
                    private::Type::F64 => Self::cast_from_f64(ConstStorage::<U, M, N>::substantiate_f64(a)),
                    private::Type::I32 => Self::cast_from_i32(ConstStorage::<U, M, N>::substantiate_i32(a)),
                    private::Type::I64 => Self::cast_from_i64(ConstStorage::<U, M, N>::substantiate_i64(a)),
                    private::Type::U32 => Self::cast_from_u32(ConstStorage::<U, M, N>::substantiate_u32(a)),
                    private::Type::U64 => Self::cast_from_u64(ConstStorage::<U, M, N>::substantiate_u64(a)),
                }
            }

            #[inline(always)]
            fn select_mask(
                mask: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>,
                true_values: Self,
                false_values: Self,
            ) -> Self {
                map3_with_mask(mask, true_values, false_values, ArithOps::select_)
            }
            #[inline(always)]
            fn select_any_mask<Mask: SealedSupportedElement>(
                mask: ConstMaskStorage<Mask, M, N>,
                true_values: Self,
                false_values: Self,
            ) -> Self {
                Self::select_mask(
                    paste::paste!(ConstStorage::<Mask, M, N>::[<cast_i $bits>](mask)),
                    true_values,
                    false_values,
                )
            }
            #[inline(always)]
            fn select_bitmask(mask: u8, true_values: Self, false_values: Self) -> Self {
                kernels::select::select_bitmask(mask, true_values, false_values)
            }
            // A mask is stored at the width of the element it selects, which without vector
            // instructions means one `i32` or `i64` per lane in the same shape as the storage.
            // That is the same type the shared comparison bodies produce, so this is the identity
            // and they need nothing else from this backend.
            #[inline(always)]
            fn substantiate_mask(
                mask: CanonicalMask<<Self as ArithOps>::Mask>,
            ) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> {
                mask
            }
            // Lane-wise comparisons and the clamp. `src/api.rs` exposes these on `Vector` alone, so a
            // matrix shape would carry a body nothing can call. `each_eq` above is the exception: the
            // integer `div` uses it to find a zero divisor, and a matrix divided by a scalar reaches
            // `div`.
            #[inline(always)]
            fn each_clamp<F: private::Fmt>(a: Self, min: Self, max: Self) -> Self {
                let valid = Self::each_le(min, max);
                assert!(
                    ConstStorage::<<$t as SupportedElement>::Mask, M, N>::all(valid),
                    "each element in `min` must be less than or equal to the corresponding element in `max`. \
                    min = {min:?}, max = {max:?}",
                    min = F::fmt::<$t, M, N>(min),
                    max = F::fmt::<$t, M, N>(max),
                );
                map3(a, min, max, ArithOps::clamp_noexcept_)
            }
            #[inline(always)]
            fn eq(a: Self, b: Self) -> bool { a.as_flattened().iter().zip(b.as_flattened()).all(|(a, b)| a == b) }
            #[inline(always)]
            fn ne(a: Self, b: Self) -> bool { a.as_flattened().iter().zip(b.as_flattened()).any(|(a, b)| a != b) }
            #[inline(always)]
            fn transpose(a: Self) -> ConstStorage<$t, N, M> { kernels::transpose(a) }

            if_! { $float == float {
                #[inline(always)]
                fn matmul<const K: usize>(
                    a: ConstStorage<$t, M, K>,
                    b: ConstStorage<$t, K, N>,
                ) -> Self {
                    kernels::matmul::matmul(
                        private::StorageOps::to_array(a),
                        private::StorageOps::to_array(b),
                    )
                }
            }}

            if_! { $signed $int == signed int {
                #[inline(always)]
                fn from_mask(mask: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> Self { mask.into_inner() }
                #[inline(always)]
                fn all(mask: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> bool {
                    mask.into_inner().as_flattened().iter().copied().all($t::is_negative)
                }
                #[inline(always)]
                fn any(mask: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> bool {
                    mask.into_inner().as_flattened().iter().copied().any($t::is_negative)
                }
                #[inline(always)]
                fn mask_not(a: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> { !a }
                #[inline(always)]
                fn mask_bitand(a: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>, b: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> { a & b }
                #[inline(always)]
                fn mask_bitor(a: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>, b: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> { a | b }
                #[inline(always)]
                fn mask_bitxor(a: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>, b: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> { a ^ b }
                #[inline(always)]
                fn to_bool_array(a: ConstMaskStorage<<$t as SupportedElement>::Mask, M, N>) -> ConstArray<ConstArray<bool, M>, N> {
                    a.into_inner().map(
                        #[inline(always)]
                        |column| column.map($t::is_negative)
                    )
                }
                #[inline(always)]
                fn from_bool_array(a: ConstArray<ConstArray<bool, M>, N>) -> ConstMaskStorage<<$t as SupportedElement>::Mask, M, N> {
                    CanonicalMask::from_parts(a.map(
                        #[inline(always)]
                        |column| CanonicalMask::from_parts(column.map(CanonicalMask::<$t>::new)),
                    ))
                }
                // `Vector::cast_signed`, `Vector::cast_unsigned` and `Vector::abs_diff`
                // are the only callers, and all name a one-column shape.
                #[inline(always)]
                fn cast_signed(a: Self) -> ConstStorage<<$t as Int>::Signed, M, N> { a }
                #[inline(always)]
                fn cast_unsigned(a: Self) -> ConstStorage<<$t as Int>::Unsigned, M, N> {
                    map1(a, $t::cast_unsigned)
                }
                #[inline(always)]
                fn to_bitmask(mask: ConstMaskStorage<$t, M, N>) -> u8 {
                    let col = mask.into_inner()[0];
                    let mut bitmask = 0u8;
                    for i in 0..M {
                        bitmask |= u8::from(col[i] < 0) << i;
                    }
                    bitmask
                }
            }}
            if_! { $signed $int == unsigned int {
                // `Vector::cast_signed`, `Vector::cast_unsigned` and `Vector::abs_diff`
                // are the only callers, and all name a one-column shape.
                #[inline(always)]
                fn cast_signed(a: Self) -> ConstStorage<<$t as Int>::Signed, M, N> {
                    map1(a, $t::cast_signed)
                }
                #[inline(always)]
                fn cast_unsigned(a: Self) -> ConstStorage<<$t as Int>::Unsigned, M, N> { a }
            }}
            if_! { $int == int {
                #[inline(always)]
                fn div(a: Self, b: Self) -> Self {
                    let mask = ArithOps::eq_(b, ArithOps::ZERO_);
                    assert!(
                        !ConstStorage::<<$t as SupportedElement>::Mask, M, N>::any(mask),
                        "attempt to divide by zero",
                    );
                    Self::map2(a, b, #[inline(always)] |x, y| x.wrapping_div(y))
                }
                #[inline(always)]
                fn rem(a: Self, b: Self) -> Self {
                    let mask = ArithOps::eq_(b, ArithOps::ZERO_);
                    assert!(
                        !ConstStorage::<<$t as SupportedElement>::Mask, M, N>::any(mask),
                        "attempt to calculate the remainder with a divisor of zero",
                    );
                    Self::map2(a, b, #[inline(always)] |x, y| x.wrapping_rem(y))
                }
            }}
            if_! { $float == float {
                // `Vector::from_bits` and `Vector::to_bits` are the only public callers, but
                // these storage methods cover every shape.
                #[inline(always)]
                fn from_bits(a: ConstStorage<<$t as Float>::Bits, M, N>) -> Self {
                    map1(a, $t::from_bits)
                }
                #[allow(clippy::wrong_self_convention)]
                #[inline(always)]
                fn to_bits(a: Self) -> ConstStorage<<$t as Float>::Bits, M, N> {
                    map1(a, $t::to_bits)
                }
                // TODO(integer-vector): split div/sqrt requirements for integer and float element traits.
                #[inline(always)]
                fn rem(a: Self, b: Self) -> Self { map2(a, b, core::ops::Rem::rem) }
            }}
            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> ConstStorage<$t, 2, 1> {
                [[a[0][I0], a[0][I1]]]
            }
            #[inline(always)]
            fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(a: Self) -> ConstStorage<$t, 3, 1> {
                [[a[0][I0], a[0][I1], a[0][I2]]]
            }
            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(a: Self) -> ConstStorage<$t, 4, 1> {
                [[a[0][I0], a[0][I1], a[0][I2], a[0][I3]]]
            }
            #[inline(always)]
            fn reduce_sum(a: Self) -> $t { kernels::reduce::sum::<$t, M>(a[0]) }
            #[inline(always)]
            fn diagonal(a: Self) -> ConstStorage<$t, M> {
                [core::array::from_fn(#[inline(always)] |i| a[i][i])]
            }
            if_! { $float == float {
                #[inline(always)]
                fn dot(a: Self, b: Self) -> $t {
                    kernels::matmul::matmul(kernels::transpose(a), b)[0][0]
                }
                #[inline(always)]
                fn inverse(a: Self) -> Self { kernels::inverse::inverse(a) }
                #[inline(always)]
                fn determinant(a: Self) -> $t { kernels::determinant::determinant(a) }
            }}

            $($item)*
        }
    };
}

impl_layout!((
    self: f32,
    feature: [float, not_int, signed, 32],
) => {
    #[inline(always)]
    fn substantiate_f32(a: Self) -> ConstStorage<f32, M, N> { a }
});
impl_layout!((
    self: f64,
    feature: [float, not_int, signed, 64],
) => {
    #[inline(always)]
    fn substantiate_f64(a: Self) -> ConstStorage<f64, M, N> { a }
});
impl_layout!((
    self: i32,
    feature: [not_float, int, signed, 32],
) => {
    #[inline(always)]
    fn substantiate_i32(a: Self) -> ConstStorage<i32, M, N> { a }

    #[inline(always)]
    fn from_bitmask(bitmask: u8) -> ConstMaskStorage<i32, M, N> {
        CanonicalMask::from_parts(core::array::from_fn(
            #[inline(always)]
            |column| {
                CanonicalMask::from_parts(core::array::from_fn(
                    #[inline(always)]
                    |row| CanonicalMask::<i32>::new(column == 0 && bitmask & (1 << row) != 0),
                ))
            },
        ))
    }

    #[inline(always)]
    fn cast_i32(a: CanonicalMask<Self>) -> ConstMaskStorage<i32, M, N> { a }
    #[inline(always)]
    fn cast_i64(a: CanonicalMask<Self>) -> ConstMaskStorage<i64, M, N> { a.cast_i64() }
    #[inline(always)]
    fn cast_mask<U: SealedSupportedElement>(
        mask: ConstMaskStorage<U, M, N>,
    ) -> ConstMaskStorage<i32, M, N> {
        ConstStorage::<U, M, N>::cast_i32(mask)
    }
    #[inline(always)]
    fn mask_select_any<Mask: SealedSupportedElement>(
        mask: ConstMaskStorage<Mask, M, N>,
        true_values: CanonicalMask<Self>,
        false_values: CanonicalMask<Self>,
    ) -> CanonicalMask<Self> {
        ConstStorage::<Mask, M, N>::cast_i32(mask).select(true_values, false_values)
    }
    #[inline(always)]
    fn mask_select_bitmask(
        mask: u8,
        true_values: CanonicalMask<Self>,
        false_values: CanonicalMask<Self>,
    ) -> CanonicalMask<Self> {
        // SAFETY: selecting between canonical masks preserves canonical lanes.
        unsafe {
            CanonicalMask::new_unchecked(kernels::select::select_bitmask(
                mask,
                true_values.into_inner(),
                false_values.into_inner(),
            ))
        }
    }
});
impl_layout!((
    self: i64,
    feature: [not_float, int, signed, 64],
) => {
    #[inline(always)]
    fn substantiate_i64(a: Self) -> ConstStorage<i64, M, N> { a }

    #[inline(always)]
    fn from_bitmask(bitmask: u8) -> ConstMaskStorage<i64, M, N> {
        CanonicalMask::from_parts(core::array::from_fn(
            #[inline(always)]
            |column| {
                CanonicalMask::from_parts(core::array::from_fn(
                    #[inline(always)]
                    |row| CanonicalMask::<i64>::new(column == 0 && bitmask & (1 << row) != 0),
                ))
            },
        ))
    }

    #[inline(always)]
    fn cast_i32(a: CanonicalMask<Self>) -> ConstMaskStorage<i32, M, N> { a.cast_i32() }
    #[inline(always)]
    fn cast_i64(a: CanonicalMask<Self>) -> ConstMaskStorage<i64, M, N> { a }
    #[inline(always)]
    fn cast_mask<U: SealedSupportedElement>(
        mask: ConstMaskStorage<U, M, N>,
    ) -> ConstMaskStorage<i64, M, N> {
        ConstStorage::<U, M, N>::cast_i64(mask)
    }
    #[inline(always)]
    fn mask_select_any<Mask: SealedSupportedElement>(
        mask: ConstMaskStorage<Mask, M, N>,
        true_values: CanonicalMask<Self>,
        false_values: CanonicalMask<Self>,
    ) -> CanonicalMask<Self> {
        ConstStorage::<Mask, M, N>::cast_i64(mask).select(true_values, false_values)
    }
    #[inline(always)]
    fn mask_select_bitmask(
        mask: u8,
        true_values: CanonicalMask<Self>,
        false_values: CanonicalMask<Self>,
    ) -> CanonicalMask<Self> {
        // SAFETY: selecting between canonical masks preserves canonical lanes.
        unsafe {
            CanonicalMask::new_unchecked(kernels::select::select_bitmask(
                mask,
                true_values.into_inner(),
                false_values.into_inner(),
            ))
        }
    }
});
impl_layout!((
    self: u32,
    feature: [not_float, int, unsigned, 32],
) => {
    #[inline(always)]
    fn substantiate_u32(a: Self) -> ConstStorage<u32, M, N> { a }
});
impl_layout!((
    self: u64,
    feature: [not_float, int, unsigned, 64],
) => {
    #[inline(always)]
    fn substantiate_u64(a: Self) -> ConstStorage<u64, M, N> { a }
});
