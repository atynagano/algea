pub(crate) mod kernels;
mod utils;

use super::{
    Vector,
    marker::{Float, Int, Lane},
    private,
    private::{Indices2, Indices3, Indices4, SwizzleDispatch},
};
use crate::utils::{ArithPrimitive, MaskPrimitive, MaskStorage, if_, impl_default_load};

impl_default_load!();

#[inline(always)]
fn map1<U, T: Copy, const M: usize, const N: usize>(
    a: [[T; M]; N],
    mut f: impl FnMut(T) -> U,
) -> [[U; M]; N] {
    a.map({
        #[inline(always)]
        |b| b.map(&mut f)
    })
}
#[inline(always)]
fn map2<U, T0: Copy, T1: Copy, const M: usize, const N: usize>(
    a: [[T0; M]; N],
    b: [[T1; M]; N],
    mut f: impl FnMut(T0, T1) -> U,
) -> [[U; M]; N] {
    core::array::from_fn({
        #[inline(always)]
        |i| {
            core::array::from_fn({
                #[inline(always)]
                |j| f(a[i][j], b[i][j])
            })
        }
    })
}
#[inline(always)]
fn map3<U, T0: Copy, T1: Copy, T2: Copy, const M: usize, const N: usize>(
    a: [[T0; M]; N],
    b: [[T1; M]; N],
    c: [[T2; M]; N],
    mut f: impl FnMut(T0, T1, T2) -> U,
) -> [[U; M]; N] {
    core::array::from_fn({
        #[inline(always)]
        |i| {
            core::array::from_fn({
                #[inline(always)]
                |j| f(a[i][j], b[i][j], c[i][j])
            })
        }
    })
}

#[inline(always)]
fn map1_mask<U: MaskPrimitive, T: Copy, const M: usize, const N: usize>(
    a: [[T; M]; N],
    f: impl FnMut(T) -> MaskStorage<U>,
) -> MaskStorage<[[U; M]; N]> {
    map1(a, f).map(Into::into).into()
}

#[inline(always)]
fn map2_mask<U: MaskPrimitive, T0: Copy, T1: Copy, const M: usize, const N: usize>(
    a: [[T0; M]; N],
    b: [[T1; M]; N],
    f: impl FnMut(T0, T1) -> MaskStorage<U>,
) -> MaskStorage<[[U; M]; N]> {
    map2(a, b, f).map(Into::into).into()
}

#[inline(always)]
fn map3_with_mask<U, T0: MaskPrimitive, T1: Copy, T2: Copy, const M: usize, const N: usize>(
    mask: MaskStorage<[[T0; M]; N]>,
    b: [[T1; M]; N],
    c: [[T2; M]; N],
    mut f: impl FnMut(MaskStorage<T0>, T1, T2) -> U,
) -> [[U; M]; N] {
    let mask = mask.unpack().map({
        #[inline(always)]
        |column| column.unpack()
    });
    core::array::from_fn({
        #[inline(always)]
        |i| {
            core::array::from_fn({
                #[inline(always)]
                |j| f(mask[i][j], b[i][j], c[i][j])
            })
        }
    })
}

macro_rules! impl_layout {
    ((
        size: [$m:tt, $n:tt],
        self: $self_ty:ty,
        storage: $primitive:tt x $len:tt,
        feature: [$float:tt, $int:tt, $signed:tt, $bits:tt],
    ) => {
        $($item:item)*
    }) => {
        impl private::SealedElement<$m, $n> for $self_ty {
            type Storage = [[Self; $m]; $n];

            const ZERO: Self::Storage = [[Self::ZERO_; $m]; $n];
            const ONE: Self::Storage = [[Self::ONE_; $m]; $n];

            #[inline(always)]
            fn map2(
                a: Self::Storage,
                b: Self::Storage,
                f: impl FnMut(Self, Self) -> Self,
            ) -> Self::Storage {
                crate::non_simd::map2(a, b, f)
            }
            #[inline(always)]
            fn index(a: &Self::Storage, (i, j): (usize, usize)) -> Option<&Self> {
                a.get(j).and_then({
                    #[inline(always)]
                    |vec| vec.get(i)
                })
            }
            #[inline(always)]
            fn index_mut(a: &mut Self::Storage, (i, j): (usize, usize)) -> Option<&mut Self> {
                a.get_mut(j).and_then({
                    #[inline(always)]
                    |vec| vec.get_mut(i)
                })
            }
            #[inline(always)]
            fn as_array_first(a: &Self::Storage) -> &[Self; $m] { &a[0] }
            #[inline(always)]
            fn as_mut_array_first(a: &mut Self::Storage) -> &mut [Self; $m] { &mut a[0] }
            #[inline(always)]
            fn to_array(a: Self::Storage) -> [[Self; $m]; $n] { a }
            #[inline(always)]
            fn from_array(a: [[Self; $m]; $n]) -> Self::Storage { a }
            #[inline(always)]
            fn from_vecs(a: [Vector<Self, $m>; $n]) -> Self::Storage {
                a.map({
                    #[inline(always)]
                    |vec| vec.storage[0]
                })
            }
            #[inline(always)]
            fn filled(a: Self) -> Self::Storage { [[a; $m]; $n] }
            #[inline(always)]
            fn cast_from_f32(a: <f32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage {
                map1(a, Self::cast_from_f32_)
            }
            #[inline(always)]
            fn cast_from_i32(a: <i32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage {
                map1(a, Self::cast_from_i32_)
            }
            #[inline(always)]
            fn cast_from_u32(a: <u32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage {
                map1(a, Self::cast_from_u32_)
            }
            #[inline(always)]
            fn cast_from<U: private::SealedElement<$m, $n>>(
                a: <U as private::SealedElement<$m, $n>>::Storage,
            ) -> Self::Storage {
                match U::TYPE {
                    private::Type::F32 => <Self as private::SealedElement<$m, $n>>::cast_from_f32(
                        <U as private::SealedElement<$m, $n>>::substantiate_f32(a),
                    ),
                    private::Type::I32 => <Self as private::SealedElement<$m, $n>>::cast_from_i32(
                        <U as private::SealedElement<$m, $n>>::substantiate_i32(a),
                    ),
                    private::Type::U32 => <Self as private::SealedElement<$m, $n>>::cast_from_u32(
                        <U as private::SealedElement<$m, $n>>::substantiate_u32(a),
                    ),
                }
            }

            #[inline(always)]
            fn select_mask(
                mask: MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage>,
                true_values: <Self as private::SealedElement<$m, $n>>::Storage,
                false_values: <Self as private::SealedElement<$m, $n>>::Storage,
            ) -> <Self as private::SealedElement<$m, $n>>::Storage {
                map3_with_mask(mask, true_values, false_values, Self::select_)
            }
            if_! { $bits == 32 {
                #[inline(always)]
                fn select_any_mask<Mask>(
                    mask: MaskStorage<<Mask as private::SealedElement<$m, $n>>::Storage>,
                    true_values: <Self as private::SealedElement<$m, $n>>::Storage,
                    false_values: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<$m, $n>>::Storage
                where
                    Mask: private::SealedElement<$m, $n>,
                {
                    <Self as private::SealedElement<$m, $n>>::select_mask(
                        <Mask as private::SealedElement<$m, $n>>::cast_i32(mask),
                        true_values,
                        false_values,
                    )
                }
            }}
            #[inline(always)]
            fn each_eq(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::eq_)
            }
            #[inline(always)]
            fn each_ne(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::ne_)
            }
            #[inline(always)]
            fn each_lt(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::lt_)
            }
            #[inline(always)]
            fn each_le(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::le_)
            }
            #[inline(always)]
            fn each_gt(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::gt_)
            }
            #[inline(always)]
            fn each_ge(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                map2_mask(a, b, Self::ge_)
            }

            #[inline(always)]
            fn each_max(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                map2(a, b, Self::max_)
            }
            #[inline(always)]
            fn each_min(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                map2(a, b, Self::min_)
            }
            #[inline(always)]
            fn each_clamp<F: private::Fmt>(
                a: Self::Storage,
                min: Self::Storage,
                max: Self::Storage,
            ) -> Self::Storage {
                let valid = <Self as private::SealedElement<$m, $n>>::each_le(min, max);
                assert!(
                    <<Self as Lane>::Mask as private::SealedElement<$m, $n>>::all(valid),
                    "each element in `min` must be less than or equal to the corresponding element in `max`. \
                    min = {min:?}, max = {max:?}",
                    min = F::fmt::<Self, $m, $n>(min),
                    max = F::fmt::<Self, $m, $n>(max),
                );
                map3(a, min, max, Self::clamp_noexcept_)
            }
            #[inline(always)]
            fn eq(a: Self::Storage, b: Self::Storage) -> bool { a.as_flattened().iter().zip(b.as_flattened()).all(|(a, b)| a == b) }
            #[inline(always)]
            fn ne(a: Self::Storage, b: Self::Storage) -> bool { a.as_flattened().iter().zip(b.as_flattened()).any(|(a, b)| a != b) }
            #[inline(always)]
            fn transpose(
                a: <Self as private::SealedElement<$m, $n>>::Storage,
            ) -> <Self as private::SealedElement<$n, $m>>::Storage {
                kernels::transpose::transpose(a)
            }

            if_! { $signed $int == signed int {
                #[inline(always)]
                fn all(mask: MaskStorage<Self::Storage>) -> bool {
                    mask.into_inner().as_flattened().iter().all({
                        #[inline(always)]
                        |&a| a < 0
                    })
                }
                #[inline(always)]
                fn any(mask: MaskStorage<Self::Storage>) -> bool {
                    mask.into_inner().as_flattened().iter().any({
                        #[inline(always)]
                        |&a| a < 0
                    })
                }
                #[inline(always)]
                fn canonical_not(
                    a: MaskStorage<Self::Storage>,
                ) -> MaskStorage<Self::Storage> {
                    !a
                }
                #[inline(always)]
                fn canonical_bitand(
                    a: MaskStorage<Self::Storage>,
                    b: MaskStorage<Self::Storage>,
                ) -> MaskStorage<Self::Storage> {
                    a & b
                }
                #[inline(always)]
                fn canonical_bitor(
                    a: MaskStorage<Self::Storage>,
                    b: MaskStorage<Self::Storage>,
                ) -> MaskStorage<Self::Storage> {
                    a | b
                }
                #[inline(always)]
                fn canonical_bitxor(
                    a: MaskStorage<Self::Storage>,
                    b: MaskStorage<Self::Storage>,
                ) -> MaskStorage<Self::Storage> {
                    a ^ b
                }
                #[inline(always)]
                fn to_bool_array(a: MaskStorage<Self::Storage>) -> [[bool; $m]; $n] {
                    a.into_inner().map({
                        #[inline(always)]
                        |column| column.map({
                            #[inline(always)]
                            |value| value < 0
                        })
                    })
                }
                #[inline(always)]
                fn from_bool_array(a: [[bool; $m]; $n]) -> MaskStorage<Self::Storage> {
                    a.map({
                        #[inline(always)]
                        |column| column.map(MaskStorage::new).into()
                    })
                    .into()
                }
            }}
            if_! { $int == int {
                #[inline(always)]
                fn add(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::add_noexcept_) }
                #[inline(always)]
                fn sub(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::sub_noexcept_) }
                #[inline(always)]
                fn mul(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::mul_noexcept_) }
                #[inline(always)]
                fn div(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    let zero = <Self as private::SealedElement<$m, $n>>::ZERO;
                    let mask = <Self as private::SealedElement<$m, $n>>::each_eq(b, zero);
                    if <<Self as Lane>::Mask as private::SealedElement::<$m, $n>>::any(mask) {
                        panic!("attempt to divide by zero")
                    } else {
                        <Self as private::SealedElement<$m, $n>>::map2(
                            a,
                            b,
                            #[inline(always)]
                            |x, y| x.wrapping_div(y),
                        )
                    }
                }
                #[inline(always)]
                fn rem(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    let zero = <Self as private::SealedElement<$m, $n>>::ZERO;
                    let mask = <Self as private::SealedElement<$m, $n>>::each_eq(b, zero);
                    assert!(
                        !<<Self as Lane>::Mask as private::SealedElement::<$m, $n>>::any(mask),
                        "attempt to calculate the remainder with a divisor of zero",
                    );
                    <Self as private::SealedElement<$m, $n>>::map2(
                        a,
                        b,
                        #[inline(always)]
                        |x, y| x.wrapping_rem(y),
                    )
                }
                #[inline(always)]
                fn bitand(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    map2(a, b, core::ops::BitAnd::bitand)
                }
                #[inline(always)]
                fn bitor(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    map2(a, b, core::ops::BitOr::bitor)
                }
                #[inline(always)]
                fn bitxor(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    map2(a, b, core::ops::BitXor::bitxor)
                }
                #[inline(always)]
                fn shl(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    map2(a, b, ArithPrimitive::shl_noexcept_)
                }
                #[inline(always)]
                fn shr(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    map2(a, b, ArithPrimitive::shr_noexcept_)
                }
            }}
            if_! { $float == not_float {
                #[inline(always)]
                fn not(a: Self::Storage) -> Self::Storage { map1(a, core::ops::Not::not) }
            }}
            // TODO(extra-type-support): add f64 layouts after the initial f32/i32/u32 release.
            if_! { $float == float {
                #[inline(always)]
                fn from_bits(
                    a: <<Self as Float>::Bits as private::SealedElement<$m, $n>>::Storage,
                ) -> Self::Storage {
                    map1(a, f32::from_bits)
                }
                #[allow(clippy::wrong_self_convention)]
                #[inline(always)]
                fn to_bits(
                    a: Self::Storage,
                ) -> <<Self as Float>::Bits as private::SealedElement<$m, $n>>::Storage {
                    map1(a, f32::to_bits)
                }
                // TODO(integer-vector): split div/sqrt requirements for integer and float element traits.
                #[inline(always)]
                fn add(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::add_noexcept_) }
                #[inline(always)]
                fn sub(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::sub_noexcept_) }
                #[inline(always)]
                fn mul(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, Self::mul_noexcept_) }
                #[inline(always)]
                fn div(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, core::ops::Div::div) }
                #[inline(always)]
                fn rem(a: Self::Storage, b: Self::Storage) -> Self::Storage { map2(a, b, core::ops::Rem::rem) }
                #[inline(always)]
                fn sqrt(a: Self::Storage) -> Self::Storage { map1(a, Self::sqrt) }

                #[inline(always)]
                fn floor(a: Self::Storage) -> Self::Storage { map1(a, Self::floor) }
                #[inline(always)]
                fn ceil(a: Self::Storage) -> Self::Storage { map1(a, Self::ceil) }
                #[inline(always)]
                fn round(a: Self::Storage) -> Self::Storage { map1(a, f32::round) }
                #[inline(always)]
                fn round_ties_even(a: Self::Storage) -> Self::Storage { map1(a, ArithPrimitive::round_ties_even_) }
                #[inline(always)]
                fn trunc(a: Self::Storage) -> Self::Storage { map1(a, Self::trunc) }
                #[inline(always)]
                fn fract(a: Self::Storage) -> Self::Storage { map1(a, Self::fract) }
                #[inline(always)]
                fn is_nan(a: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                    map1_mask(a, Self::is_nan_)
                }
            }}
            if_! { $signed == signed {
                #[inline(always)]
                fn neg(a: Self::Storage) -> Self::Storage { map1(a, ArithPrimitive::neg_noexcept_) }
                #[inline(always)]
                fn abs(a: Self::Storage) -> Self::Storage { map1(a, ArithPrimitive::abs_noexcept_) }
                #[inline(always)]
                fn signum(a: Self::Storage) -> Self::Storage { map1(a, ArithPrimitive::signum_) }
            }}
            if_! { $n == 1 and $m != 1 {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<2, 1>>::Storage
                where
                    Indices2<I0, I1>: SwizzleDispatch,
                {
                    <Self as private::SealedElement<2, 1>>::from_array([[a[0][I0], a[0][I1]]])
                }
                #[inline(always)]
                fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<3, 1>>::Storage
                where
                    Indices3<I0, I1, I2>: SwizzleDispatch,
                {
                    <Self as private::SealedElement<3, 1>>::from_array([[a[0][I0], a[0][I1], a[0][I2]]])
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<4, 1>>::Storage
                where
                    Indices4<I0, I1, I2, I3>: SwizzleDispatch,
                {
                    <Self as private::SealedElement<4, 1>>::from_array([
                        [a[0][I0], a[0][I1], a[0][I2], a[0][I3]],
                    ])
                }
            }}

            $($item)*
        }
    };
}

macro_rules! impl_layouts_f32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: f32,
            storage: $primitive x $len,
            feature: [float, not_int, signed, 32],
        ) => {
            #[inline(always)]
            fn substantiate_f32(a: Self::Storage) -> Self::Storage { a }
            $($item)*
        });)*
    };
}

macro_rules! impl_layouts_i32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: i32,
            storage: $primitive x $len,
            feature: [not_float, int, signed, 32],
        ) => {
            #[inline(always)]
            fn substantiate_i32(a: Self::Storage) -> Self::Storage { a }
            #[inline(always)]
            fn cast_i32(
                a: MaskStorage<Self::Storage>,
            ) -> MaskStorage<<i32 as private::SealedElement<$m, $n>>::Storage> {
                a
            }
            #[inline(always)]
            fn canonical_select_any_mask<Mask>(
                mask: MaskStorage<<Mask as private::SealedElement<$m, $n>>::Storage>,
                true_values: MaskStorage<Self::Storage>,
                false_values: MaskStorage<Self::Storage>,
            ) -> MaskStorage<Self::Storage>
            where
                Mask: private::SealedElement<$m, $n>,
            {
                <Mask as private::SealedElement<$m, $n>>::cast_i32(mask)
                    .select(true_values, false_values)
            }
            #[inline(always)]
            fn cast_signed(a: Self::Storage) -> <<Self as Int>::Signed as private::SealedElement<$m, $n>>::Storage {
                a
            }
            #[inline(always)]
            fn cast_unsigned(a: Self::Storage) -> <<Self as Int>::Unsigned as private::SealedElement<$m, $n>>::Storage {
                map1(a, Self::cast_unsigned)
            }
            $($item)*
        });)*
    };
}

macro_rules! impl_layouts_u32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: u32,
            storage: $primitive x $len,
            feature: [not_float, int, unsigned, 32],
        ) => {
            #[inline(always)]
            fn substantiate_u32(a: Self::Storage) -> Self::Storage { a }
            #[inline(always)]
            fn cast_signed(a: Self::Storage) -> <<Self as Int>::Signed as private::SealedElement<$m, $n>>::Storage {
                map1(a, Self::cast_signed)
            }
            #[inline(always)]
            fn cast_unsigned(a: Self::Storage) -> <<Self as Int>::Unsigned as private::SealedElement<$m, $n>>::Storage {
                a
            }
            $($item)*
        });)*
    };
}

impl_layouts_f32! {
    (1, 1; f32 x 1) => {
        const IDENTITY: Self::Storage = [[1.]];
        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            if mask & 1 != 0 { true_values } else { false_values }
        }

        #[inline(always)]
        fn sum(a: Self::Storage) -> Self { a[0][0] }

        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<1, 1>>::Storage,
        ) -> <Self as private::SealedElement<1, 1>>::Storage {
            a
        }
        #[inline(always)]
        fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::f32_1x1(a) }
        #[inline(always)]
        fn determinant(a: Self::Storage) -> Self { a[0][0] }
    },
    (2, 1; f32x4 x 1) => {
        const POS_X: Self::Storage = [[1., 0.]];
        const POS_Y: Self::Storage = [[0., 1.]];
        const NEG_X: Self::Storage = [[-1., 0.]];
        const NEG_Y: Self::Storage = [[0., -1.]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }

        #[inline(always)]
        fn dot(a: Self::Storage, b: Self::Storage) -> Self {
            kernels::matmul::f32::matmul1x2x1(kernels::transpose::transpose(a), b)[0][0]
        }
        #[inline(always)]
        fn sum([[x, y]]: Self::Storage) -> Self { x + y }
    },
    (3, 1; f32x4 x 1) => {
        const POS_X: Self::Storage = [[1., 0., 0.]];
        const POS_Y: Self::Storage = [[0., 1., 0.]];
        const POS_Z: Self::Storage = [[0., 0., 1.]];
        const NEG_X: Self::Storage = [[-1., 0., 0.]];
        const NEG_Y: Self::Storage = [[0., -1., 0.]];
        const NEG_Z: Self::Storage = [[0., 0., -1.]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }

        #[inline(always)]
        fn dot(a: Self::Storage, b: Self::Storage) -> Self {
            kernels::matmul::f32::matmul1x3x1(kernels::transpose::transpose(a), b)[0][0]
        }
        #[inline(always)]
        fn sum([[x, y, z]]: Self::Storage) -> Self { x + y + z }
        #[inline(always)]
        fn cross(a: Self::Storage, b: Self::Storage) -> Self::Storage {
            kernels::cross::f32_3d(a, b)
        }
    },
    (4, 1; f32x4 x 1) => {
        const POS_X: Self::Storage = [[1., 0., 0., 0.]];
        const POS_Y: Self::Storage = [[0., 1., 0., 0.]];
        const POS_Z: Self::Storage = [[0., 0., 1., 0.]];
        const POS_W: Self::Storage = [[0., 0., 0., 1.]];
        const NEG_X: Self::Storage = [[-1., 0., 0., 0.]];
        const NEG_Y: Self::Storage = [[0., -1., 0., 0.]];
        const NEG_Z: Self::Storage = [[0., 0., -1., 0.]];
        const NEG_W: Self::Storage = [[0., 0., 0., -1.]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }

        #[inline(always)]
        fn dot(a: Self::Storage, b: Self::Storage) -> Self {
            kernels::matmul::f32::matmul1x4x1(kernels::transpose::transpose(a), b)[0][0]
        }
        #[inline(always)]
        fn sum([[x, y, z, w]]: Self::Storage) -> Self { (x + z) + (y + w) }
    },
    (1, 2; f32x4 x 1) => {},
    (2, 2; f32x4 x 1) => {
        const IDENTITY: Self::Storage = [[1., 0.], [0., 1.]];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<2, 2>>::Storage
        ) -> <Self as private::SealedElement<2, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
        #[inline(always)]
        fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::f32_2x2(a) }
        #[inline(always)]
        fn determinant(a: Self::Storage) -> Self { kernels::determinant::f32_2x2(a) }
    },
    (3, 2; f32x4 x 2) => {},
    (4, 2; f32x4 x 2) => {},
    (1, 3; f32x4 x 1) => {},
    (2, 3; f32x4 x 2) => {},
    (3, 3; f32x4 x 3) => {
        const IDENTITY: Self::Storage = [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<3, 3>>::Storage
        ) -> <Self as private::SealedElement<3, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
        #[inline(always)]
        fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::f32_3x3(a) }
        #[inline(always)]
        fn determinant(a: Self::Storage) -> Self { kernels::determinant::f32_3x3(a) }
    },
    (4, 3; f32x4 x 3) => {},
    (1, 4; f32x4 x 1) => {},
    (2, 4; f32x4 x 2) => {},
    (3, 4; f32x4 x 4) => {},
    (4, 4; f32x4 x 4) => {
        const IDENTITY: Self::Storage = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<4, 4>>::Storage
        ) -> <Self as private::SealedElement<4, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
        #[inline(always)]
        fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::f32_4x4(a) }
        #[inline(always)]
        fn determinant(a: Self::Storage) -> Self { kernels::determinant::f32_4x4(a) }
    },
}

impl_layouts_i32! {
    (1, 1; i32 x 1) => {
        const IDENTITY: Self::Storage = [[1]];
        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            if mask & 1 != 0 { true_values } else { false_values }
        }

        #[inline(always)]
        fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
            let mask = mask.into_inner();
            u64::from(mask[0][0] < 0)
        }

        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<1, 1>>::Storage,
        ) -> <Self as private::SealedElement<1, 1>>::Storage {
            a
        }
    },
    (2, 1; i32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0]];
        const POS_Y: Self::Storage = [[0, 1]];
        const NEG_X: Self::Storage = [[-1, 0]];
        const NEG_Y: Self::Storage = [[0, -1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
        #[inline(always)]
        fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
            let mask = mask.into_inner();
            u64::from(mask[0][0] < 0) | u64::from(mask[0][1] < 0) << 1
        }
    },
    (3, 1; i32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0, 0]];
        const POS_Y: Self::Storage = [[0, 1, 0]];
        const POS_Z: Self::Storage = [[0, 0, 1]];
        const NEG_X: Self::Storage = [[-1, 0, 0]];
        const NEG_Y: Self::Storage = [[0, -1, 0]];
        const NEG_Z: Self::Storage = [[0, 0, -1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
        #[inline(always)]
        fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
            let mask = mask.into_inner();
            u64::from(mask[0][0] < 0) | u64::from(mask[0][1] < 0) << 1 | u64::from(mask[0][2] < 0) << 2
        }
    },
    (4, 1; i32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0, 0, 0]];
        const POS_Y: Self::Storage = [[0, 1, 0, 0]];
        const POS_Z: Self::Storage = [[0, 0, 1, 0]];
        const POS_W: Self::Storage = [[0, 0, 0, 1]];
        const NEG_X: Self::Storage = [[-1, 0, 0, 0]];
        const NEG_Y: Self::Storage = [[0, -1, 0, 0]];
        const NEG_Z: Self::Storage = [[0, 0, -1, 0]];
        const NEG_W: Self::Storage = [[0, 0, 0, -1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
        #[inline(always)]
        fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
            let mask = mask.into_inner();
            u64::from(mask[0][0] < 0) | u64::from(mask[0][1] < 0) << 1 | u64::from(mask[0][2] < 0) << 2 | u64::from(mask[0][3] < 0) << 3
        }
    },
    (1, 2; i32x4 x 1) => {},
    (2, 2; i32x4 x 1) => {
        const IDENTITY: Self::Storage = [[1, 0], [0, 1]];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<2, 2>>::Storage
        ) -> <Self as private::SealedElement<2, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
    (3, 2; i32x4 x 2) => {},
    (4, 2; i32x4 x 2) => {},
    (1, 3; i32x4 x 1) => {},
    (2, 3; i32x4 x 2) => {},
    (3, 3; i32x4 x 3) => {
        const IDENTITY: Self::Storage = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<3, 3>>::Storage
        ) -> <Self as private::SealedElement<3, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
    (4, 3; i32x4 x 3) => {},
    (1, 4; i32x4 x 1) => {},
    (2, 4; i32x4 x 2) => {},
    (3, 4; i32x4 x 4) => {},
    (4, 4; i32x4 x 4) => {
        const IDENTITY: Self::Storage = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<4, 4>>::Storage
        ) -> <Self as private::SealedElement<4, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
}

impl_layouts_u32! {
    (1, 1; u32 x 1) => {
        const IDENTITY: Self::Storage = [[1]];
        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            if mask & 1 != 0 { true_values } else { false_values }
        }

        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<1, 1>>::Storage,
        ) -> <Self as private::SealedElement<1, 1>>::Storage {
            a
        }
    },
    (2, 1; u32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0]];
        const POS_Y: Self::Storage = [[0, 1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
    },
    (3, 1; u32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0, 0]];
        const POS_Y: Self::Storage = [[0, 1, 0]];
        const POS_Z: Self::Storage = [[0, 0, 1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
    },
    (4, 1; u32x4 x 1) => {
        const POS_X: Self::Storage = [[1, 0, 0, 0]];
        const POS_Y: Self::Storage = [[0, 1, 0, 0]];
        const POS_Z: Self::Storage = [[0, 0, 1, 0]];
        const POS_W: Self::Storage = [[0, 0, 0, 1]];

        #[inline(always)]
        fn select_u64(
            mask: u64,
            true_values: Self::Storage,
            false_values: Self::Storage,
        ) -> Self::Storage {
            kernels::select::select_u64(mask, true_values, false_values)
        }
    },
    (1, 2; u32x4 x 1) => {},
    (2, 2; u32x4 x 1) => {
        const IDENTITY: Self::Storage = [[1, 0], [0, 1]];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<2, 2>>::Storage
        ) -> <Self as private::SealedElement<2, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
    (3, 2; u32x4 x 2) => {},
    (4, 2; u32x4 x 2) => {},
    (1, 3; u32x4 x 1) => {},
    (2, 3; u32x4 x 2) => {},
    (3, 3; u32x4 x 3) => {
        const IDENTITY: Self::Storage = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<3, 3>>::Storage
        ) -> <Self as private::SealedElement<3, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
    (4, 3; u32x4 x 3) => {},
    (1, 4; u32x4 x 1) => {},
    (2, 4; u32x4 x 2) => {},
    (3, 4; u32x4 x 4) => {},
    (4, 4; u32x4 x 4) => {
        const IDENTITY: Self::Storage = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ];
        #[inline(always)]
        fn diagonal(
            a: <Self as private::SealedElement<4, 4>>::Storage
        ) -> <Self as private::SealedElement<4, 1>>::Storage {
            kernels::diagonal::diagonal(a)
        }
    },
}

macro_rules! impl_swizzle2_for_i0 {
    ($i0:literal; $($i1:literal),*) => {$(
        impl SwizzleDispatch for Indices2<$i0, $i1> {
            #[inline(always)]
            fn dispatch<T: Copy + crate::utils::Swizzle>(v: T) -> T { v }
        }
    )*};
}
macro_rules! impl_swizzle3_for_i0_i1 {
    ($i0:literal, $i1:literal; $($i2:literal),*) => {$(
        impl SwizzleDispatch for Indices3<$i0, $i1, $i2> {
            #[inline(always)]
            fn dispatch<T: Copy + crate::utils::Swizzle>(v: T) -> T { v }
        }
    )*};
}
macro_rules! impl_swizzle3_for_i0 {
    ($i0:literal; $($i1:literal),*) => {
        $(impl_swizzle3_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0_i1_i2 {
    ($i0:literal, $i1:literal, $i2:literal; $($i3:literal),*) => {$(
        impl SwizzleDispatch for Indices4<$i0, $i1, $i2, $i3> {
            #[inline(always)]
            fn dispatch<T: Copy + crate::utils::Swizzle>(v: T) -> T { v }
        }
    )*};
}
macro_rules! impl_swizzle4_for_i0_i1 {
    ($i0:literal, $i1:literal; $($i2:literal),*) => {
        $(impl_swizzle4_for_i0_i1_i2!($i0, $i1, $i2; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0 {
    ($i0:literal; $($i1:literal),*) => {
        $(impl_swizzle4_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}

impl_swizzle2_for_i0!(0; 0, 1, 2, 3);
impl_swizzle2_for_i0!(1; 0, 1, 2, 3);
impl_swizzle2_for_i0!(2; 0, 1, 2, 3);
impl_swizzle2_for_i0!(3; 0, 1, 2, 3);
impl_swizzle3_for_i0!(0; 0, 1, 2, 3);
impl_swizzle3_for_i0!(1; 0, 1, 2, 3);
impl_swizzle3_for_i0!(2; 0, 1, 2, 3);
impl_swizzle3_for_i0!(3; 0, 1, 2, 3);
impl_swizzle4_for_i0!(0; 0, 1, 2, 3);
impl_swizzle4_for_i0!(1; 0, 1, 2, 3);
impl_swizzle4_for_i0!(2; 0, 1, 2, 3);
impl_swizzle4_for_i0!(3; 0, 1, 2, 3);
