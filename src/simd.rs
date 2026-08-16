pub(crate) mod kernels;
#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
mod swizzle_arm;
#[cfg(target_feature = "simd128")]
mod swizzle_wasm;
#[cfg(target_feature = "sse2")]
mod swizzle_x86;
pub(crate) mod utils;

use crate::{
    Vector,
    marker::{Float, Int, Lane},
    private,
    private::{Indices2, Indices3, Indices4, SwizzleDispatch, SwizzleDispatchAny},
    utils::{ArithPrimitive, Load, MaskStorage, Store, if_},
};
use utils::{Simd2Ext, f32x2, i32x2, u32x2};
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

macro_rules! arg_or_value {
    ($arg:ident) => {
        $arg
    };
    ($arg:ident = $value:expr) => {
        $value
    };
}

macro_rules! unpack_array {
    ([($($arg:ident $(=$value:expr)?),+) $f:expr; 4]) => {{
        $(let $arg = arg_or_value!($arg $(=$value)?);)+
        [($f)($($arg[0]),+), ($f)($($arg[1]),+), ($f)($($arg[2]),+), ($f)($($arg[3]),+)]
    }};
    ([($($arg:ident $(=$value:expr)?),+) $f:expr; 3]) => {{
        $(let $arg = arg_or_value!($arg $(=$value)?);)+
        [($f)($($arg[0]),+), ($f)($($arg[1]),+), ($f)($($arg[2]),+)]
    }};
    ([($($arg:ident $(=$value:expr)?),+) $f:expr; 2]) => {{
        $(let $arg = arg_or_value!($arg $(=$value)?);)+
        [($f)($($arg[0]),+), ($f)($($arg[1]),+)]
    }};
    ([($($arg:ident $(=$value:expr)?),+) $f:expr; 1]) => {{
        $(let $arg = arg_or_value!($arg $(=$value)?);)+
        ($f)($($arg),+)
    }};

    ([($self:ident $(=$self_value:expr)?) . $f:ident ($($arg:ident $(=$value:expr)?),*); 4]) => {{
        let $self = arg_or_value!($self $(=$self_value)?);
        $(let $arg = arg_or_value!($arg $(=$value)?);)*
        [$self[0].$f($($arg[0]),*), $self[1].$f($($arg[1]),*), $self[2].$f($($arg[2]),*), $self[3].$f($($arg[3]),*)]
    }};
    ([($self:ident $(=$self_value:expr)?) . $f:ident ($($arg:ident $(=$value:expr)?),*); 3]) => {{
        let $self = arg_or_value!($self $(=$self_value)?);
        $(let $arg = arg_or_value!($arg $(=$value)?);)*
        [$self[0].$f($($arg[0]),*), $self[1].$f($($arg[1]),*), $self[2].$f($($arg[2]),*)]
    }};
    ([($self:ident $(=$self_value:expr)?) . $f:ident ($($arg:ident $(=$value:expr)?),*); 2]) => {{
        let $self = arg_or_value!($self $(=$self_value)?);
        $(let $arg = arg_or_value!($arg $(=$value)?);)*
        [$self[0].$f($($arg[0]),*), $self[1].$f($($arg[1]),*)]
    }};
    ([($self:ident $(=$self_value:expr)?) . $f:ident ($($arg:ident $(=$value:expr)?),*); 1]) => {{
        let $self = arg_or_value!($self $(=$self_value)?);
        $(let $arg = arg_or_value!($arg $(=$value)?);)*
        $self.$f($($arg),*)
    }};

    ([$value:tt; 1]) => { $value };
    ([$value:tt; $len:literal]) => { [$value; $len] };

    (ref: $value:expr; 1) => { core::array::from_ref($value) };
    (ref: $value:expr; $_len:literal) => { $value };
    (mut: $value:expr; 1) => { core::array::from_mut($value) };
    (mut: $value:expr; $_len:literal) => { $value };
}

macro_rules! reduce_array {
    (&&: [$($value:expr),+]) => { $($value)&&+ };
    (||: [$($value:expr),+]) => { $($value)||+ };
}

macro_rules! impl_layout {
    ((
        size: [$m:tt, $n:tt],
        self: $self_ty:ident,
        storage: $primitive:tt x $len:tt,
        // How many of each physical `$primitive` unit's lanes hold real elements versus
        // padding. Not yet consumed here; `map2`/`any`/`all`/etc. still rely on `$len` alone
        // via `unpack_array!`'s arity dispatch.
        valid: [$($valid:tt),+ $(,)?],
        feature: [$float:tt, $int:tt, $signed:tt, $bits:tt],
    ) => {
        $($item:item)*
    }) => {
        if_! { $signed $int == signed int {
            paste::paste! {
                #[inline(always)]
                fn [<mask $bits x $m x $n _all>](
                    mask: MaskStorage<<<$self_ty as private::SealedElement<$m, $n>>::Storage as Load>::Output>
                ) -> bool {
                    let mask = mask.unpack();
                    let mask = unpack_array!(ref: &mask; $len);
                    let mut _index = 0;
                    reduce_array!(&&: [$({
                        let all = mask[_index].all::<$valid>();
                        _index += 1;
                        all
                    }),+])
                }
                #[inline(always)]
                fn [<mask $bits x $m x $n _any>](
                    mask: MaskStorage<<<$self_ty as private::SealedElement<$m, $n>>::Storage as Load>::Output>
                ) -> bool {
                    let mask = mask.unpack();
                    let mask = unpack_array!(ref: &mask; $len);
                    let mut _index = 0;
                    reduce_array!(||: [$({
                        let any = mask[_index].any::<$valid>();
                        _index += 1;
                        any
                    }),+])
                }
            }
        }}

        impl private::SealedElement<$m, $n> for $self_ty {
            type Storage = unpack_array!([$primitive; $len]);

            const ZERO: Self::Storage = unpack_array!([($primitive::ZERO_); $len]);
            const ONE: Self::Storage = unpack_array!([($primitive::ONE_); $len]);

            #[inline(always)]
            fn map2(mut a: Self::Storage, b: Self::Storage, mut f: impl FnMut(Self, Self) -> Self) -> Self::Storage {
                let array_a = unpack_array!(mut: &mut a; $len);
                let array_b = unpack_array!(ref: &b; $len);
                let valid = [$($valid),+];
                for i in 0..$len {
                    let col_a = array_a[i].as_mut_array_();
                    let col_b = array_b[i].as_array_();
                    for j in 0..valid[i] {
                        col_a[j] = f(col_a[j], col_b[j]);
                    }
                }
                a
            }
            #[inline(always)]
            fn index(a: &Self::Storage, index: (usize, usize)) -> Option<&Self> { paste::paste!(kernels::index:: [<_ $m x $n>])(a, index) }
            #[inline(always)]
            fn index_mut(a: &mut Self::Storage, index: (usize, usize)) -> Option<&mut Self> { paste::paste!(kernels::index_mut:: [<_ $m x $n>])(a, index) }
            #[inline(always)]
            fn as_array_first(a: &Self::Storage) -> &[Self; $m] { unpack_array!(ref: a; $len)[0].as_array_().first_chunk().unwrap() }
            #[inline(always)]
            fn as_mut_array_first(a: &mut Self::Storage) -> &mut [Self; $m] { unpack_array!(mut: a; $len)[0].as_mut_array_().first_chunk_mut().unwrap() }
            #[inline(always)]
            fn to_array(a: Self::Storage) -> [[Self; $m]; $n] { paste::paste!(kernels::to_array:: [<_ $m x $n>])(a) }
            #[inline(always)]
            fn from_array(a: [[Self; $m]; $n]) -> Self::Storage { paste::paste!(kernels::from_array::$self_ty:: [<_ $m x $n>])(a) }
            #[inline(always)]
            fn from_vecs(a: [Vector<Self, $m>; $n]) -> Self::Storage { paste::paste!(kernels::from_vecs::$self_ty:: [<_ $m x $n>])(a) }
            #[inline(always)]
            fn filled(a: Self) -> Self::Storage { unpack_array!([($primitive::filled_(a)); $len]) }
            #[inline(always)]
            fn cast_from_f32(a: <f32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_f32_; $len]) }
            #[inline(always)]
            fn cast_from_i32(a: <i32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_i32_; $len]) }
            #[inline(always)]
            fn cast_from_u32(a: <u32 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_u32_; $len]) }
            #[inline(always)]
            fn cast_from_f64(a: <f64 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_f64_; $len]) }
            #[inline(always)]
            fn cast_from_i64(a: <i64 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_i64_; $len]) }
            #[inline(always)]
            fn cast_from_u64(a: <u64 as private::SealedElement<$m, $n>>::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::cast_from_u64_; $len]) }
            #[inline(always)]
            fn cast_from<U: private::SealedElement<$m, $n>>(
                a: <U as private::SealedElement<$m, $n>>::Storage,
            ) -> Self::Storage {
                match U::TYPE {
                    private::Type::F32 => <Self as private::SealedElement<$m, $n>>::cast_from_f32(<U as private::SealedElement<$m, $n>>::substantiate_f32(a)),
                    private::Type::F64 => <Self as private::SealedElement<$m, $n>>::cast_from_f64(<U as private::SealedElement<$m, $n>>::substantiate_f64(a)),
                    private::Type::I32 => <Self as private::SealedElement<$m, $n>>::cast_from_i32(<U as private::SealedElement<$m, $n>>::substantiate_i32(a)),
                    private::Type::I64 => <Self as private::SealedElement<$m, $n>>::cast_from_i64(<U as private::SealedElement<$m, $n>>::substantiate_i64(a)),
                    private::Type::U32 => <Self as private::SealedElement<$m, $n>>::cast_from_u32(<U as private::SealedElement<$m, $n>>::substantiate_u32(a)),
                    private::Type::U64 => <Self as private::SealedElement<$m, $n>>::cast_from_u64(<U as private::SealedElement<$m, $n>>::substantiate_u64(a)),
                }
            }

            #[inline(always)]
            fn select_mask(
                mask: MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage>,
                true_values: <Self as private::SealedElement<$m, $n>>::Storage,
                false_values: <Self as private::SealedElement<$m, $n>>::Storage,
            ) -> <Self as private::SealedElement<$m, $n>>::Storage {
                unpack_array!([(mask=mask.load().unpack(), t=true_values.load(), f=false_values.load()) ArithPrimitive::select_; $len]).store()
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
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::eq_; $len]).store()
            }
            #[inline(always)]
            fn each_ne(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::ne_; $len]).store()
            }
            #[inline(always)]
            fn each_lt(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::lt_; $len]).store()
            }
            #[inline(always)]
            fn each_le(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::le_; $len]).store()
            }
            #[inline(always)]
            fn each_gt(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::gt_; $len]).store()
            }
            #[inline(always)]
            fn each_ge(a: Self::Storage, b: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::ge_; $len]).store()
            }

            #[inline(always)]
            fn each_max(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::max_; $len]).store()
            }
            #[inline(always)]
            fn each_min(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::min_; $len]).store()
            }
            #[inline(always)]
            fn each_clamp<F: private::Fmt>(a: Self::Storage, min: Self::Storage, max: Self::Storage) -> Self::Storage {
                let a = a.load();
                let min = min.load();
                let max = max.load();
                let valid = unpack_array!([(min, max) ArithPrimitive::le_; $len]);
                let valid_all = paste::paste!([<mask $bits x $m x $n _all>])(valid.into());
                assert!(
                    valid_all,
                    "each element in `min` must be less than or equal to the corresponding element in `max`. \
                    min = {min:?}, max = {max:?}",
                    min = F::fmt::<Self, $m, $n>(min),
                    max = F::fmt::<Self, $m, $n>(max),
                );
                unpack_array!([(a, min, max) ArithPrimitive::clamp_noexcept_; $len]).store()
            }
            #[inline(always)]
            fn eq(a: Self::Storage, b: Self::Storage) -> bool {
                let mask = unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::eq_; $len]);
                paste::paste!([<mask $bits x $m x $n _all>])(mask.into())
            }
            #[inline(always)]
            fn ne(a: Self::Storage, b: Self::Storage) -> bool {
                let mask = unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::ne_; $len]);
                paste::paste!([<mask $bits x $m x $n _any>])(mask.into())
            }
            #[inline(always)]
            fn transpose(
                a: <Self as private::SealedElement<$m, $n>>::Storage,
            ) -> <Self as private::SealedElement<$n, $m>>::Storage {
                paste::paste!(crate::kernels::transpose::[<transpose $m x $n>])(a)
            }
            #[inline(always)]
            fn add(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::add_noexcept_; $len]).store()
            }
            #[inline(always)]
            fn sub(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::sub_noexcept_; $len]).store()
            }
            #[inline(always)]
            fn mul(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::mul_noexcept_; $len]).store()
            }

            if_! { $signed $int == signed int {
                #[inline(always)]
                fn all(mask: MaskStorage<Self::Storage>) -> bool {
                    paste::paste!([<mask $bits x $m x $n _all>])(mask.load())
                }
                #[inline(always)]
                fn any(mask: MaskStorage<Self::Storage>) -> bool {
                    paste::paste!([<mask $bits x $m x $n _any>])(mask.load())
                }
                #[inline(always)]
                fn canonical_not(a: MaskStorage<Self::Storage>) -> MaskStorage<Self::Storage> {
                    (!a.load()).store()
                }
                #[inline(always)]
                fn canonical_bitand(a: MaskStorage<Self::Storage>, b: MaskStorage<Self::Storage>) -> MaskStorage<Self::Storage> {
                    (a.load() & b.load()).store()
                }
                #[inline(always)]
                fn canonical_bitor(a: MaskStorage<Self::Storage>, b: MaskStorage<Self::Storage>) -> MaskStorage<Self::Storage> {
                    (a.load() | b.load()).store()
                }
                #[inline(always)]
                fn canonical_bitxor(a: MaskStorage<Self::Storage>, b: MaskStorage<Self::Storage>) -> MaskStorage<Self::Storage> {
                    (a.load() ^ b.load()).store()
                }
                #[inline(always)]
                fn to_bool_array(a: MaskStorage<Self::Storage>) -> [[bool; $m]; $n] {
                    paste::paste!(kernels::mask::$self_ty::[<to_array_ $m x $n>](a.load()))
                }
                #[inline(always)]
                fn from_bool_array(a: [[bool; $m]; $n]) -> MaskStorage<Self::Storage> {
                    paste::paste!(kernels::mask::$self_ty::[<from_array_ $m x $n>](a)).store()
                }
                #[inline(always)]
                fn cast_signed(a: Self::Storage) -> <<Self as Int>::Signed as private::SealedElement<$m, $n>>::Storage { a }
                #[inline(always)]
                fn cast_unsigned(a: Self::Storage) -> <<Self as Int>::Unsigned as private::SealedElement<$m, $n>>::Storage {
                    unpack_array!([(a) $primitive::cast_unsigned; $len])
                }
            }}
            if_! { $signed $int == unsigned int {
                #[inline(always)]
                fn cast_signed(a: Self::Storage) -> <<Self as Int>::Signed as private::SealedElement<$m, $n>>::Storage {
                    unpack_array!([(a) $primitive::cast_signed; $len])
                }
                #[inline(always)]
                fn cast_unsigned(a: Self::Storage) -> <<Self as Int>::Unsigned as private::SealedElement<$m, $n>>::Storage { a }
            }}
            if_! { $int == int {
                #[inline(always)]
                fn div(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    let zero = <Self as private::SealedElement<$m, $n>>::ZERO;
                    let mask = <Self as private::SealedElement<$m, $n>>::each_eq(b, zero);
                    assert!(
                        !<<Self as Lane>::Mask as private::SealedElement::<$m, $n>>::any(mask),
                        "attempt to divide by zero",
                    );
                    <Self as private::SealedElement<$m, $n>>::map2(a, b, #[inline(always)] |x, y| x.wrapping_div(y))
                }
                #[inline(always)]
                fn rem(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    let zero = <Self as private::SealedElement<$m, $n>>::ZERO;
                    let mask = <Self as private::SealedElement<$m, $n>>::each_eq(b, zero);
                    assert!(
                        !<<Self as Lane>::Mask as private::SealedElement::<$m, $n>>::any(mask),
                        "attempt to calculate the remainder with a divisor of zero",
                    );
                    <Self as private::SealedElement<$m, $n>>::map2(a, b, #[inline(always)] |x, y| x.wrapping_rem(y))
                }
                #[inline(always)]
                fn bitand(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    unpack_array!([(a=a.load(), b=b.load()) core::ops::BitAnd::bitand; $len]).store()
                }
                #[inline(always)]
                fn bitor(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    unpack_array!([(a=a.load(), b=b.load()) core::ops::BitOr::bitor; $len]).store()
                }
                #[inline(always)]
                fn bitxor(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    unpack_array!([(a=a.load(), b=b.load()) core::ops::BitXor::bitxor; $len]).store()
                }
                #[inline(always)]
                fn shl(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::shl_noexcept_; $len]).store()
                }
                #[inline(always)]
                fn shr(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    unpack_array!([(a=a.load(), b=b.load()) ArithPrimitive::shr_noexcept_; $len]).store()
                }
            }}
            if_! { $float == not_float {
                #[inline(always)]
                fn not(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()) core::ops::Not::not; $len]).store() }
            }}
            // TODO(extra-type-support): add f64 layouts after the initial f32/i32/u32 release.
            if_! { $float == float {
                #[inline(always)]
                fn from_bits(
                    a: <<Self as Float>::Bits as private::SealedElement<$m, $n>>::Storage,
                ) -> Self::Storage {
                    unpack_array!([(a) $primitive::from_bits; $len])
                }
                #[allow(clippy::wrong_self_convention)]
                #[inline(always)]
                fn to_bits(
                    a: Self::Storage,
                ) -> <<Self as Float>::Bits as private::SealedElement<$m, $n>>::Storage {
                    unpack_array!([(a) $primitive::to_bits; $len])
                }
                // TODO(integer-vector): split div/sqrt requirements for integer and float element traits.
                #[inline(always)]
                fn div(a: Self::Storage, b: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load(), b=b.load()) core::ops::Div::div; $len]).store() }
                #[inline(always)]
                fn rem(a: Self::Storage, b: Self::Storage) -> Self::Storage {
                    // TODO(codegen-optimization): Vectorize `fmodf` only with exact special-value
                    // and error-bound tests; `std::simd::Simd<f32, N>` delegates to Windows UCRT
                    // scalar `fmodf` calls on x86-64, while libm provides a possible implementation.
                    <Self as private::SealedElement<$m, $n>>::map2(a, b, core::ops::Rem::rem)
                }
                #[inline(always)]
                fn sqrt(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).sqrt(); $len]).store() }

                #[inline(always)]
                fn floor(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).floor(); $len]).store() }
                #[inline(always)]
                fn ceil(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).ceil(); $len]).store() }
                #[inline(always)]
                fn round(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).round(); $len]).store() }
                #[inline(always)]
                fn round_ties_even(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()) ArithPrimitive::round_ties_even_; $len]).store() }
                #[inline(always)]
                fn trunc(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).trunc(); $len]).store() }
                #[inline(always)]
                fn fract(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()).fract(); $len]).store() }
                #[inline(always)]
                fn is_nan(a: Self::Storage) -> MaskStorage<<<Self as Lane>::Mask as private::SealedElement<$m, $n>>::Storage> {
                    unpack_array!([(a=a.load()).is_nan_(); $len]).store()
                }
            }}
            if_! { $signed == signed {
                #[inline(always)]
                fn neg(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()) ArithPrimitive::neg_noexcept_; $len]).store() }
                #[inline(always)]
                fn abs(a: Self::Storage) -> Self::Storage { unpack_array!([(a=a.load()) ArithPrimitive::abs_noexcept_; $len]).store() }
                #[inline(always)]
                fn signum(a: Self::Storage) -> Self::Storage { unpack_array!([(a) ArithPrimitive::signum_; $len]) }
            }}
            if_! { $n == 1 and $m != 1 {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<2, 1>>::Storage
                where
                    Indices2<I0, I1>: SwizzleDispatchAny<2>,
                {
                    <Indices2<I0, I1> as SwizzleDispatch<Self, $m, 2>>::dispatch(a)
                }
                #[inline(always)]
                fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<3, 1>>::Storage
                where
                    Indices3<I0, I1, I2>: SwizzleDispatchAny<3>,
                {
                    <Indices3<I0, I1, I2> as SwizzleDispatch<Self, $m, 3>>::dispatch(a)
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
                    a: <Self as private::SealedElement<$m, $n>>::Storage,
                ) -> <Self as private::SealedElement<4, 1>>::Storage
                where
                    Indices4<I0, I1, I2, I3>: SwizzleDispatchAny<4>,
                {
                    <Indices4<I0, I1, I2, I3> as SwizzleDispatch<Self, $m, 4>>::dispatch(a)
                }
            }}

            // This needs only `Sealed` because it does not use the shape, but retaining the
            // generated implementation avoids another macro branch.
            #[inline(always)]
            fn vector_concat_1_1(
                a: <Self as private::SealedElement<1, 1>>::Storage,
                b: <Self as private::SealedElement<1, 1>>::Storage,
            ) -> <Self as private::SealedElement<2, 1>>::Storage {
                let [[a]] = <Self as private::SealedElement<1, 1>>::to_array(a);
                let [[b]] = <Self as private::SealedElement<1, 1>>::to_array(b);
                // NEON can swizzle straight from a 64-bit (2-lane) width without first widening
                // to 128-bit, but this path stays shared with SSE (which has no such shortcut)
                // for implementation simplicity, leaving that codegen-level optimization to LLVM
                // rather than hand-writing a NEON-specific 64-bit-first version: for these fixed
                // concat patterns, LLVM already collapses the zero-padded 128-bit form down to
                // the same instruction count as a hand-written 64-bit-first version would need.
                let zero = <Self as crate::utils::ArithPrimitive>::ZERO_;
                crate::simd::utils::swizzle!(
                    <Self as private::SealedElement<4, 1>>::Storage::new([a, zero, zero, zero]),
                    <Self as private::SealedElement<4, 1>>::Storage::new([b, zero, zero, zero]),
                    [0, 4]
                ).store()
            }

            #[inline(always)]
            fn vector_concat_1_2(
                a: <Self as private::SealedElement<1, 1>>::Storage,
                b: <Self as private::SealedElement<2, 1>>::Storage,
            ) -> <Self as private::SealedElement<3, 1>>::Storage {
                let [[a]] = <Self as private::SealedElement<1, 1>>::to_array(a);
                let zero = <Self as crate::utils::ArithPrimitive>::ZERO_;
                // See the comment in `vector_concat_1_1`: NEON could combine `a`/`b` at their
                // natural (64-bit) width without widening to 128-bit first, but this path stays
                // shared with SSE for implementation simplicity and leaves that optimization to
                // LLVM, which generates equivalent code either way for these fixed patterns.
                crate::simd::utils::swizzle!(
                    <Self as private::SealedElement<4, 1>>::Storage::new([a, zero, zero, zero]),
                    b.load().widen(),
                    [0, 4, 5]
                )
            }

            #[inline(always)]
            fn vector_concat_2_1(
                a: <Self as private::SealedElement<2, 1>>::Storage,
                b: <Self as private::SealedElement<1, 1>>::Storage,
            ) -> <Self as private::SealedElement<3, 1>>::Storage {
                let [[b]] = <Self as private::SealedElement<1, 1>>::to_array(b);
                let zero = <Self as crate::utils::ArithPrimitive>::ZERO_;
                // See the comment in `vector_concat_1_1`: NEON could combine `a`/`b` at their
                // natural (64-bit) width without widening to 128-bit first, but this path stays
                // shared with SSE for implementation simplicity and leaves that optimization to
                // LLVM, which generates equivalent code either way for these fixed patterns.
                crate::simd::utils::swizzle!(
                    a.load().widen(),
                    <Self as private::SealedElement<4, 1>>::Storage::new([b, zero, zero, zero]),
                    [0, 1, 4]
                )
            }

            if_! { $n == 1 {
                #[inline(always)]
                fn reduce_sum(a: Self::Storage) -> Self { kernels::reduce::sum::<Self::Storage, $m>(a) }
                if_! { $float == float {
                    #[inline(always)]
                    fn dot(a: Self::Storage, b: Self::Storage) -> Self {
                        paste::paste!(kernels::matmul::$self_ty:: [<matmul1x $m x1>] (a.load(), b.load()).store())
                    }
                }}
            }}
            if_! { $m == 1 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
                        u64::from(mask.into_inner() < 0)
                    }
                }}
            }}
            if_! { $m == 2 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
                        // TODO(to-bitmask-lane-width): NEON and the 64-bit types hold two
                        // lanes outright, so masking is only needed elsewhere.
                        u64::from(mask.into_inner().load().to_bitmask() & 0b11)
                    }
                }}
                const POS_X: Self::Storage = $primitive::new([1 as _, 0 as _]);
                const POS_Y: Self::Storage = $primitive::new([0 as _, 1 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self::Storage = $primitive::new([-1 as _, 0 as _]);
                    const NEG_Y: Self::Storage = $primitive::new([0 as _, -1 as _]);
                }}
            }}
            if_! { $m == 3 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
                        u64::from(mask.into_inner().to_bitmask() & 0b111)
                    }
                }}
                const POS_X: Self::Storage = $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]);
                const POS_Y: Self::Storage = $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]);
                const POS_Z: Self::Storage = $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self::Storage = $primitive::new([-1 as _, 0 as _, 0 as _, 0 as _]);
                    const NEG_Y: Self::Storage = $primitive::new([0 as _, -1 as _, 0 as _, 0 as _]);
                    const NEG_Z: Self::Storage = $primitive::new([0 as _, 0 as _, -1 as _, 0 as _]);
                }}
            }}
            if_! { $m == 4 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: MaskStorage<Self::Storage>) -> u64 {
                        u64::from(mask.into_inner().to_bitmask())
                    }
                }}
                const POS_X: Self::Storage = $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]);
                const POS_Y: Self::Storage = $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]);
                const POS_Z: Self::Storage = $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]);
                const POS_W: Self::Storage = $primitive::new([0 as _, 0 as _, 0 as _, 1 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self::Storage = $primitive::new([-1 as _, 0 as _, 0 as _, 0 as _]);
                    const NEG_Y: Self::Storage = $primitive::new([0 as _, -1 as _, 0 as _, 0 as _]);
                    const NEG_Z: Self::Storage = $primitive::new([0 as _, 0 as _, -1 as _, 0 as _]);
                    const NEG_W: Self::Storage = $primitive::new([0 as _, 0 as _, 0 as _, -1 as _]);
                }}
            }}
            if_! { $m == 1 and $n == 1 {
                const IDENTITY: Self::Storage = 1 as _;
                #[inline(always)]
                fn diagonal(a: Self::Storage) -> Self::Storage { a }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::$self_ty::_1x1(a) }
                    #[inline(always)]
                    fn determinant(a: Self::Storage) -> Self { a }
                }}
            }}
            if_! { $m == 2 and $n == 2 {
                const IDENTITY: Self::Storage = $primitive::new([1 as _, 0 as _, 0 as _, 1 as _]);
                #[inline(always)]
                fn diagonal(a: Self::Storage) -> <Self as private::SealedElement<2, 1>>::Storage {
                    kernels::diagonal::diagonal2x2(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::$self_ty::_2x2(a) }
                    #[inline(always)]
                    fn determinant(a: Self::Storage) -> Self { kernels::determinant::$self_ty::_2x2(a) }
                }}
            }}
            if_! { $m == 3 and $n == 3 {
                const IDENTITY: Self::Storage = [
                    $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]),
                ];
                #[inline(always)]
                fn diagonal(a: Self::Storage) -> <Self as private::SealedElement<3, 1>>::Storage {
                    kernels::diagonal::diagonal3x3(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::$self_ty::_3x3(a) }
                    #[inline(always)]
                    fn determinant(a: Self::Storage) -> Self { kernels::determinant::$self_ty::_3x3(a) }
                }}
            }}
            if_! { $m == 4 and $n == 4 {
                const IDENTITY: Self::Storage = [
                    $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 0 as _, 1 as _]),
                ];
                #[inline(always)]
                fn diagonal(a: Self::Storage) -> <Self as private::SealedElement<4, 1>>::Storage {
                    kernels::diagonal::diagonal4x4(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self::Storage) -> Self::Storage { kernels::inverse::$self_ty::_4x4(a) }
                    #[inline(always)]
                    fn determinant(a: Self::Storage) -> Self { kernels::determinant::$self_ty::_4x4(a) }
                }}
            }}

            $($item)*
        }
    };
}

macro_rules! impl_layouts_f32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: f32,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [float, not_int, signed, 32],
        ) => {
            #[inline(always)]
            fn substantiate_f32(a: Self::Storage) -> Self::Storage { a }
            $($item)*
        });)*
    };
}
macro_rules! impl_layouts_f64 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: f64,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [float, not_int, signed, 64],
        ) => {
            #[inline(always)]
            fn substantiate_f64(a: Self::Storage) -> Self::Storage { a }
            $($item)*
        });)*
    };
}
macro_rules! impl_layouts_i32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: i32,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [not_float, int, signed, 32],
        ) => {
            #[inline(always)]
            fn substantiate_i32(a: Self::Storage) -> Self::Storage { a }
            #[inline(always)]
            fn cast_i32(a: MaskStorage<Self::Storage>) -> MaskStorage<<i32 as private::SealedElement<$m, $n>>::Storage> {
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
                // TODO(mask-representation): When adding non-i32 mask lanes, compare casting before
                // and after loading for each lane width and target instead of assuming one order.
                <Mask as private::SealedElement<$m, $n>>::cast_i32(mask)
                    .load()
                    .select(true_values.load(), false_values.load())
                    .store()
            }
            $($item)*
        });)*
    };
}
// TODO(i64-mask-casts): supply `cast_i32`, `cast_i64`, `canonical_select_any_mask` and the
// `bits == 64` form of `select_any_mask`.
macro_rules! impl_layouts_i64 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: i64,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [not_float, int, signed, 64],
        ) => {
            #[inline(always)]
            fn substantiate_i64(a: Self::Storage) -> Self::Storage { a }
            // TODO: cast_i32, cast_i64
            $($item)*
        });)*
    };
}
macro_rules! impl_layouts_u32 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: u32,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [not_float, int, unsigned, 32],
        ) => {
            #[inline(always)]
            fn substantiate_u32(a: Self::Storage) -> Self::Storage { a }
            $($item)*
        });)*
    };
}
macro_rules! impl_layouts_u64 {
    ($(($m:tt, $n:tt; $primitive:tt x $len:tt valid [$($valid:tt),+ $(,)?]) => {$($item:item)*}),* $(,)?) => {
        $(impl_layout!((
            size: [$m, $n],
            self: u64,
            storage: $primitive x $len,
            valid: [$($valid),+],
            feature: [not_float, int, unsigned, 64],
        ) => {
            #[inline(always)]
            fn substantiate_u64(a: Self::Storage) -> Self::Storage { a }
            $($item)*
        });)*
    };
}

macro_rules! call_layouts {
    ($macro_name:ident ($scalar:tt, $vec2:tt, $vec4:tt)) => {
        $macro_name! {
            (1, 1; $scalar x 1 valid [1]) => {},
            (2, 1; $vec2 x 1 valid [2]) => {},
            (3, 1; $vec4 x 1 valid [3]) => {},
            (4, 1; $vec4 x 1 valid [4]) => {},
            (1, 2; $vec2 x 1 valid [2]) => {},
            (2, 2; $vec4 x 1 valid [4]) => {},
            (3, 2; $vec4 x 2 valid [3, 3]) => {},
            (4, 2; $vec4 x 2 valid [4, 4]) => {},
            (1, 3; $vec4 x 1 valid [3]) => {},
            (2, 3; $vec4 x 2 valid [4, 2]) => {},
            (3, 3; $vec4 x 3 valid [3, 3, 3]) => {},
            (4, 3; $vec4 x 3 valid [4, 4, 4]) => {},
            (1, 4; $vec4 x 1 valid [4]) => {},
            (2, 4; $vec4 x 2 valid [4, 4]) => {},
            (3, 4; $vec4 x 4 valid [3, 3, 3, 3]) => {},
            (4, 4; $vec4 x 4 valid [4, 4, 4, 4]) => {},
        }
    };
}

call_layouts!(impl_layouts_f32(f32, f32x2, f32x4));
call_layouts!(impl_layouts_f64(f64, f64x2, f64x4));
call_layouts!(impl_layouts_i32(i32, i32x2, i32x4));
call_layouts!(impl_layouts_i64(i64, i64x2, i64x4));
call_layouts!(impl_layouts_u32(u32, u32x2, u32x4));
call_layouts!(impl_layouts_u64(u64, u64x2, u64x4));
