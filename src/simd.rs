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
    private::{
        ConstStorage,
        Indices2,
        Indices3,
        Indices4,
        SealedSupportedDimension,
        SealedSupportedElement,
        StorageOps,
        SwizzleDispatch,
        SwizzleDispatchAny,
    },
    support::{Dimension, SupportedElement},
    utils::{ArithOps, CanonicalMask, ConstMaskStorage, Load, Store, if_},
};
use utils::{Simd2Ext, f32x2, i32x2, u32x2};
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

impl SealedSupportedDimension for Dimension<1> {
    type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension> = C::Storage1xN<T>;
    type Storage1xN<T: SealedSupportedElement> = <T as private::SealedElement<1, 1>>::Storage;
    type Storage2xN<T: SealedSupportedElement> = <T as private::SealedElement<2, 1>>::Storage;
    type Storage3xN<T: SealedSupportedElement> = <T as private::SealedElement<3, 1>>::Storage;
    type Storage4xN<T: SealedSupportedElement> = <T as private::SealedElement<4, 1>>::Storage;
}
impl SealedSupportedDimension for Dimension<2> {
    type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension> = C::Storage2xN<T>;
    type Storage1xN<T: SealedSupportedElement> = <T as private::SealedElement<1, 2>>::Storage;
    type Storage2xN<T: SealedSupportedElement> = <T as private::SealedElement<2, 2>>::Storage;
    type Storage3xN<T: SealedSupportedElement> = <T as private::SealedElement<3, 2>>::Storage;
    type Storage4xN<T: SealedSupportedElement> = <T as private::SealedElement<4, 2>>::Storage;
}
impl SealedSupportedDimension for Dimension<3> {
    type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension> = C::Storage3xN<T>;
    type Storage1xN<T: SealedSupportedElement> = <T as private::SealedElement<1, 3>>::Storage;
    type Storage2xN<T: SealedSupportedElement> = <T as private::SealedElement<2, 3>>::Storage;
    type Storage3xN<T: SealedSupportedElement> = <T as private::SealedElement<3, 3>>::Storage;
    type Storage4xN<T: SealedSupportedElement> = <T as private::SealedElement<4, 3>>::Storage;
}
impl SealedSupportedDimension for Dimension<4> {
    type StorageNxC<T: SealedSupportedElement, C: SealedSupportedDimension> = C::Storage4xN<T>;
    type Storage1xN<T: SealedSupportedElement> = <T as private::SealedElement<1, 4>>::Storage;
    type Storage2xN<T: SealedSupportedElement> = <T as private::SealedElement<2, 4>>::Storage;
    type Storage3xN<T: SealedSupportedElement> = <T as private::SealedElement<3, 4>>::Storage;
    type Storage4xN<T: SealedSupportedElement> = <T as private::SealedElement<4, 4>>::Storage;
}

#[cfg(false)]
impl StorageOps<f32, Dimension<1>, Dimension<1>> for f32 {
    const IDENTITY: Self = 1.;
    const POS_X: Self = 1.;
    const NEG_X: Self = -1.;

    fn from_array(array: [[f32; 1]; 1]) -> Self { crate::kernels::from_array::f32::_1x1(array) }
    fn eq(a: Self, b: Self) -> bool { todo!() }
    fn ne(a: Self, b: Self) -> bool { todo!() }
    fn transpose(a: Self) -> ConstStorage<f32, 1, 1> { crate::kernels::transpose::transpose1x1(a) }
}

macro_rules! arg_or_value {
    ($arg:ident) => {
        $arg
    };
    ($arg:ident = $value:expr) => {
        $value
    };
}

macro_rules! unpack_array {
    ([($arg:ident) $trait:ident :: $f:ident; [$valid:tt]]) => {{
        $trait::$f::<$valid>($arg)
    }};
    ([($arg:ident) $trait:ident :: $f:ident; [$($valid:tt),+]]) => {{
        let mut _index = 0;
        [$({
            let result = $trait::$f::<$valid>($arg[_index]);
            _index += 1;
            result
        }),+]
    }};
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
        self: $t:ident,
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
                    mask: CanonicalMask<<<$t as private::SealedElement<$m, $n>>::Storage as Load>::Output>
                ) -> bool {
                    let mask = mask.into_parts();
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
                    mask: CanonicalMask<<<$t as private::SealedElement<$m, $n>>::Storage as Load>::Output>
                ) -> bool {
                    let mask = mask.into_parts();
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

        impl private::SealedElement<$m, $n> for $t {
            type Storage = unpack_array!([$primitive; $len]);
        }

        impl private::StorageOps<$t, Dimension<$m>, Dimension<$n>> for unpack_array!([$primitive; $len]) {
            #[inline(always)]
            fn map2(mut a: Self, b: Self, mut f: impl FnMut($t, $t) -> $t) -> Self {
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
            fn index(a: &Self, index: (usize, usize)) -> Option<&$t> {
                paste::paste!(kernels::index::[<_ $bits bit>]::[<_ $m x $n>])(a, index)
            }
            #[inline(always)]
            fn index_mut(a: &mut Self, index: (usize, usize)) -> Option<&mut $t> {
                paste::paste!(kernels::index_mut::[<_ $bits bit>]::[<_ $m x $n>])(a, index)
            }

            // `Vector::as_array` and `Vector::as_mut_array` are the only callers, and both
            // name a one-column shape.
            if_! { $n == 1 {
                #[inline(always)]
                fn as_array_first(a: &Self) -> &[$t; $m] {
                    unpack_array!(ref: a; $len)[0].as_array_().first_chunk().unwrap()
                }
                #[inline(always)]
                fn as_mut_array_first(a: &mut Self) -> &mut [$t; $m] {
                    unpack_array!(mut: a; $len)[0].as_mut_array_().first_chunk_mut().unwrap()
                }
            }}
            #[inline(always)]
            fn to_array(a: Self) -> [[$t; $m]; $n] {
                paste::paste!(kernels::to_array::[<_ $bits bit>]::[<_ $m x $n>])(a)
            }
            #[inline(always)]
            fn from_array(a: [[$t; $m]; $n]) -> Self {
                // TODO: Dispatch this kernel by `$bits` instead of `$t`.
                paste::paste!(kernels::from_array::$t::[<_ $m x $n>])(a)
            }
            #[inline(always)]
            fn from_vecs(a: [Vector<$t, $m>; $n]) -> Self {
                // TODO: Dispatch this kernel by `$bits` instead of `$t`.
                paste::paste!(kernels::from_vecs::$t:: [<_ $m x $n>])(a)
            }
            #[inline(always)]
            fn cast_from_f32(a: <f32 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_f32_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from_i32(a: <i32 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_i32_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from_u32(a: <u32 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_u32_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from_f64(a: <f64 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_f64_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from_i64(a: <i64 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_i64_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from_u64(a: <u64 as private::SealedElement<$m, $n>>::Storage) -> Self {
                let a = RelayoutStorage::<$m, $n, $bits>::relayout_storage(a);
                unpack_array!([(a) ArithOps::cast_from_u64_; [$($valid),+]])
            }
            #[inline(always)]
            fn cast_from<U: SealedSupportedElement>(a: ConstStorage<U, $m, $n>) -> Self {
                match U::TYPE {
                    private::Type::F32 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_f32(<U as private::SealedElement<$m, $n>>::Storage::substantiate_f32(a)),
                    private::Type::F64 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_f64(<U as private::SealedElement<$m, $n>>::Storage::substantiate_f64(a)),
                    private::Type::I32 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_i32(<U as private::SealedElement<$m, $n>>::Storage::substantiate_i32(a)),
                    private::Type::I64 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_i64(<U as private::SealedElement<$m, $n>>::Storage::substantiate_i64(a)),
                    private::Type::U32 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_u32(<U as private::SealedElement<$m, $n>>::Storage::substantiate_u32(a)),
                    private::Type::U64 => <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::cast_from_u64(<U as private::SealedElement<$m, $n>>::Storage::substantiate_u64(a)),
                }
            }

            // Lane-wise comparisons and the clamp. `src/api.rs` exposes these on `Vector` alone, so a
            // matrix shape would carry a body nothing can call. `each_eq` above is the exception: the
            // integer `div` uses it to find a zero divisor, and a matrix divided by a scalar reaches
            // `div`.
            if_! { $n == 1 {
                #[inline(always)]
                fn substantiate_mask(
                    mask: CanonicalMask<<Self as ArithOps>::Mask>,
                ) -> ConstMaskStorage<<$t as Lane>::Mask, $m, $n> {
                    mask
                }
                #[inline(always)]
                fn select_mask(
                    mask: ConstMaskStorage<<$t as Lane>::Mask, $m, $n>,
                    true_values: Self,
                    false_values: Self,
                ) -> Self {
                    unpack_array!([(mask=mask.load_mask().into_parts(), t=true_values.load(), f=false_values.load()) ArithOps::select_; $len]).store()
                }
                #[inline(always)]
                fn select_any_mask<Mask: SupportedElement>(
                    mask: ConstMaskStorage<Mask, $m, $n>,
                    true_values: Self,
                    false_values: Self,
                ) -> Self {
                    <Self as private::StorageOps<$t, Dimension<$m>, Dimension<$n>>>::select_mask(
                        paste::paste!(<Mask as private::SealedElement<$m, $n>>::Storage::[<cast_i $bits>](mask)),
                        true_values,
                        false_values,
                    )
                }

                #[inline(always)]
                fn each_clamp<F: private::Fmt>(a: Self, min: Self, max: Self) -> Self {
                    let a = a.load();
                    let min = min.load();
                    let max = max.load();
                    let valid = unpack_array!([(min, max) ArithOps::le_; $len]);
                    let valid_all = paste::paste!([<mask $bits x $m x $n _all>])(valid.into());
                    assert!(
                        valid_all,
                        "each element in `min` must be less than or equal to the corresponding element in `max`. \
                        min = {min:?}, max = {max:?}",
                        min = F::fmt::<$t, $m, $n>(min),
                        max = F::fmt::<$t, $m, $n>(max),
                    );
                    unpack_array!([(a, min, max) ArithOps::clamp_noexcept_; $len]).store()
                }
            }}
            #[inline(always)]
            fn eq(a: Self, b: Self) -> bool {
                // let mask = unpack_array!([(a=a.load(), b=b.load()) ArithOps::eq_; $len]);
                let mask = ArithOps::eq_(a.load(), b.load());
                paste::paste!([<mask $bits x $m x $n _all>])(mask)
            }
            #[inline(always)]
            fn ne(a: Self, b: Self) -> bool {
                // let mask = unpack_array!([(a=a.load(), b=b.load()) ArithOps::ne_; $len]);
                let mask = ArithOps::ne_(a.load(), b.load());
                paste::paste!([<mask $bits x $m x $n _any>])(mask)
            }
            #[inline(always)]
            fn transpose(a: Self) -> ConstStorage<$t, $n, $m> {
                paste::paste!(crate::kernels::transpose::[<_ $bits bit>]::[<transpose $m x $n>])(a)
            }
            if_! { $signed $int == signed int {
                #[inline(always)]
                fn from_mask(mask: ConstMaskStorage<<$t as Lane>::Mask, $m, $n>) -> Self { mask.into_inner() }
                #[inline(always)]
                fn all(mask: ConstMaskStorage<<$t as Lane>::Mask, $m, $n>) -> bool {
                    paste::paste!([<mask $bits x $m x $n _all>])(mask.load_mask())
                }
                #[inline(always)]
                fn any(mask: ConstMaskStorage<<$t as Lane>::Mask, $m, $n>) -> bool {
                    paste::paste!([<mask $bits x $m x $n _any>])(mask.load_mask())
                }
                #[inline(always)]
                fn to_bool_array(mask: ConstMaskStorage<<$t as Lane>::Mask, $m, $n>) -> [[bool; $m]; $n] {
                    paste::paste!(kernels::mask::$t::[<to_array_ $m x $n>](mask.load_mask()))
                }
                #[inline(always)]
                fn from_bool_array(a: [[bool; $m]; $n]) -> ConstMaskStorage<<$t as Lane>::Mask, $m, $n> {
                    CanonicalMask::store_mask(paste::paste!(kernels::mask::$t::[<from_array_ $m x $n>](a)))
                }
                // `Vector::cast_signed`, `Vector::cast_unsigned` and `Vector::abs_diff`
                // are the only callers, and all name a one-column shape.
                if_! { $n == 1 {
                    #[inline(always)]
                    fn cast_signed(a: Self) -> ConstStorage<<$t as Int>::Signed, $m, $n> { a }
                    #[inline(always)]
                    fn cast_unsigned(a: Self) -> ConstStorage<<$t as Int>::Unsigned, $m, $n> {
                        unpack_array!([(a) $primitive::cast_unsigned; $len])
                    }
                }}
            }}
            if_! { $signed $int == unsigned int {
                // `Vector::cast_signed`, `Vector::cast_unsigned` and `Vector::abs_diff`
                // are the only callers, and all name a one-column shape.
                if_! { $n == 1 {
                    #[inline(always)]
                    fn cast_signed(a: Self) -> ConstStorage<<$t as Int>::Signed, $m, $n> {
                        unpack_array!([(a) $primitive::cast_signed; $len])
                    }
                    #[inline(always)]
                    fn cast_unsigned(a: Self) -> ConstStorage<<$t as Int>::Unsigned, $m, $n> { a }
                }}
            }}
            if_! { $int == int {
                #[inline(always)]
                fn div(a: Self, b: Self) -> Self {
                    // The reduction kernel is named directly rather than reached through `each_eq`
                    // and `SealedElement::any`, which would store the comparison at the element's
                    // width and load it back to fold it.
                    let mask = ArithOps::eq_(b.load(), ArithOps::ZERO_);
                    assert!(
                        !paste::paste!([<mask $bits x $m x $n _any>])(mask),
                        "attempt to divide by zero",
                    );
                    <Self as StorageOps<$t, Dimension<$m>, Dimension<$n>>>::map2(
                        a, b,
                        #[inline(always)] |x, y| x.wrapping_div(y)
                    )
                }
                // `Rem`, `BitAnd`, `BitOr`, `BitXor`, `Shl` and `Shr` are generated for vectors only.
                if_! { $n == 1 {
                    #[inline(always)]
                    fn rem(a: Self, b: Self) -> Self {
                        let mask = ArithOps::eq_(b.load(), ArithOps::ZERO_);
                        assert!(
                            !paste::paste!([<mask $bits x $m x $n _any>])(mask),
                            "attempt to calculate the remainder with a divisor of zero",
                        );
                        <Self as StorageOps<$t, Dimension<$m>, Dimension<$n>>>::map2(
                            a, b,
                            #[inline(always)] |x, y| x.wrapping_rem(y)
                        )
                    }
                }}
            }}
            if_! { $float == float {
                // `Vector::from_bits` and `Vector::to_bits` are the only callers, and
                // both name a one-column shape.
                if_! { $n == 1 {
                    #[inline(always)]
                    fn from_bits(a: ConstStorage<<$t as Float>::Bits, $m, $n>) -> Self {
                        unpack_array!([(a) $primitive::from_bits; $len])
                    }
                    #[allow(clippy::wrong_self_convention)]
                    #[inline(always)]
                    fn to_bits(a: Self) -> ConstStorage<<$t as Float>::Bits, $m, $n> {
                        unpack_array!([(a) $primitive::to_bits; $len])
                    }
                }}
                // `Rem` is generated for vectors only.
                if_! { $n == 1 {
                    #[inline(always)]
                    fn rem(a: Self, b: Self) -> Self {
                        // TODO(codegen-optimization): Vectorize `fmodf` only with exact special-value
                        // and error-bound tests; `std::simd::Simd<f32, N>` delegates to Windows UCRT
                        // scalar `fmodf` calls on x86-64, while libm provides a possible implementation.
                        <Self as StorageOps<$t, Dimension<$m>, Dimension<$n>>>::map2(a, b, core::ops::Rem::rem)
                    }
                }}
            }}
            // TODO: Revisit these swizzle methods and their dispatch traits; independent
            // element and dimension bounds may allow a substantially simpler implementation.
            if_! { $n == 1 and $m != 1 {
                #[inline(always)]
                fn swizzle2<const I0: usize, const I1: usize>(a: Self) -> ConstStorage<$t, 2, 1>
                where
                    Indices2<I0, I1>: SwizzleDispatchAny<2>,
                {
                    <Indices2<I0, I1> as SwizzleDispatch<$t, $m, 2>>::dispatch(a)
                }
                #[inline(always)]
                fn swizzle3<const I0: usize, const I1: usize, const I2: usize>(a: Self) -> ConstStorage<$t, 3, 1>
                where
                    Indices3<I0, I1, I2>: SwizzleDispatchAny<3>,
                {
                    <Indices3<I0, I1, I2> as SwizzleDispatch<$t, $m, 3>>::dispatch(a)
                }
                #[inline(always)]
                fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(a: Self) -> ConstStorage<$t, 4, 1>
                where
                    Indices4<I0, I1, I2, I3>: SwizzleDispatchAny<4>,
                {
                    <Indices4<I0, I1, I2, I3> as SwizzleDispatch<$t, $m, 4>>::dispatch(a)
                }
            }}

            if_! { $n == 1 {
                #[inline(always)]
                fn reduce_sum(a: Self) -> $t { kernels::reduce::sum::<Self, $m>(a) }
                if_! { $float == float {
                    #[inline(always)]
                    fn dot(a: Self, b: Self) -> $t {
                        paste::paste!(kernels::matmul::$t:: [<matmul1x $m x1>] (a.load(), b.load()).store())
                    }
                }}
            }}
            if_! { $m == 1 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: ConstMaskStorage<$t, $m, $n>) -> u64 {
                        u64::from(mask.into_inner() < 0)
                    }
                }}
            }}
            if_! { $m == 2 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: ConstMaskStorage<$t, $m, $n>) -> u64 {
                        // TODO(to-bitmask-lane-width): NEON and the 64-bit types hold two
                        // lanes outright, so masking is only needed elsewhere.
                        u64::from(mask.into_inner().load().to_bitmask() & 0b11)
                    }
                }}
                const POS_X: Self = $primitive::new([1 as _, 0 as _]);
                const POS_Y: Self = $primitive::new([0 as _, 1 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self = $primitive::new([-1 as _, 0 as _]);
                    const NEG_Y: Self = $primitive::new([0 as _, -1 as _]);
                }}
            }}
            if_! { $m == 3 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: ConstMaskStorage<$t, $m, $n>) -> u64 {
                        u64::from(mask.into_inner().to_bitmask() & 0b111)
                    }
                }}
                const POS_X: Self = $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]);
                const POS_Y: Self = $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]);
                const POS_Z: Self = $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self = $primitive::new([-1 as _, 0 as _, 0 as _, 0 as _]);
                    const NEG_Y: Self = $primitive::new([0 as _, -1 as _, 0 as _, 0 as _]);
                    const NEG_Z: Self = $primitive::new([0 as _, 0 as _, -1 as _, 0 as _]);
                }}
            }}
            if_! { $m == 4 and $n == 1 {
                if_! { $signed $int == signed int {
                    #[inline(always)]
                    fn to_bitmask(mask: ConstMaskStorage<$t, $m, $n>) -> u64 {
                        u64::from(mask.into_inner().to_bitmask())
                    }
                }}
                const POS_X: Self = $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]);
                const POS_Y: Self = $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]);
                const POS_Z: Self = $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]);
                const POS_W: Self = $primitive::new([0 as _, 0 as _, 0 as _, 1 as _]);
                if_! { $signed == signed {
                    const NEG_X: Self = $primitive::new([-1 as _, 0 as _, 0 as _, 0 as _]);
                    const NEG_Y: Self = $primitive::new([0 as _, -1 as _, 0 as _, 0 as _]);
                    const NEG_Z: Self = $primitive::new([0 as _, 0 as _, -1 as _, 0 as _]);
                    const NEG_W: Self = $primitive::new([0 as _, 0 as _, 0 as _, -1 as _]);
                }}
            }}
            if_! { $m == 1 and $n == 1 {
                const IDENTITY: Self = 1 as _;
                #[inline(always)]
                fn diagonal(a: Self) -> Self { a }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self) -> Self { kernels::inverse::$t::_1x1(a) }
                    #[inline(always)]
                    fn determinant(a: Self) -> $t { a }
                }}
            }}
            if_! { $m == 2 and $n == 2 {
                const IDENTITY: Self = $primitive::new([1 as _, 0 as _, 0 as _, 1 as _]);
                #[inline(always)]
                fn diagonal(a: Self) -> ConstStorage<$t, $m> {
                    kernels::diagonal::diagonal2x2(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self) -> Self { kernels::inverse::$t::_2x2(a) }
                    #[inline(always)]
                    fn determinant(a: Self) -> $t { kernels::determinant::$t::_2x2(a) }
                }}
            }}
            if_! { $m == 3 and $n == 3 {
                const IDENTITY: Self = [
                    $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]),
                ];
                #[inline(always)]
                fn diagonal(a: Self) -> ConstStorage<$t, $m> {
                    kernels::diagonal::diagonal3x3(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self) -> Self { kernels::inverse::$t::_3x3(a) }
                    #[inline(always)]
                    fn determinant(a: Self) -> $t { kernels::determinant::$t::_3x3(a) }
                }}
            }}
            if_! { $m == 4 and $n == 4 {
                const IDENTITY: Self = [
                    $primitive::new([1 as _, 0 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 1 as _, 0 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 1 as _, 0 as _]),
                    $primitive::new([0 as _, 0 as _, 0 as _, 1 as _]),
                ];
                #[inline(always)]
                fn diagonal(a: Self) -> ConstStorage<$t, $m> {
                    kernels::diagonal::diagonal4x4(a).store()
                }
                if_! { $float == float {
                    #[inline(always)]
                    fn inverse(a: Self) -> Self { kernels::inverse::$t::_4x4(a) }
                    #[inline(always)]
                    fn determinant(a: Self) -> $t { kernels::determinant::$t::_4x4(a) }
                }}
            }}

            $($item)*
        }

        // These methods depend only on the scalar type, so the 1x1 layout expansion emits
        // the sole `SealedSupportedElement` implementation for each scalar type.
        if_! { $m == 1 and $n == 1 {
            impl private::SealedSupportedElement for $t {
                #[inline(always)]
                fn vector_concat_1_1(
                    a: <Self as private::SealedElement<1, 1>>::Storage,
                    b: <Self as private::SealedElement<1, 1>>::Storage,
                ) -> <Self as private::SealedElement<2, 1>>::Storage {
                    let [[a]] = crate::api::vector::call!(<Self, 1>::to_array(a));
                    let [[b]] = crate::api::vector::call!(<Self, 1>::to_array(b));
                    // NEON can swizzle straight from a 64-bit (2-lane) width without first widening
                    // to 128-bit, but this path stays shared with SSE (which has no such shortcut)
                    // for implementation simplicity, leaving that codegen-level optimization to LLVM
                    // rather than hand-writing a NEON-specific 64-bit-first version: for these fixed
                    // concat patterns, LLVM already collapses the zero-padded 128-bit form down to
                    // the same instruction count as a hand-written 64-bit-first version would need.
                    let zero = <Self as crate::utils::ArithOps>::ZERO_;
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
                    let [[a]] = crate::api::vector::call!(<Self, 1>::to_array(a));
                    let zero = <Self as crate::utils::ArithOps>::ZERO_;
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
                    let [[b]] = crate::api::vector::call!(<Self, 1>::to_array(b));
                    let zero = <Self as crate::utils::ArithOps>::ZERO_;
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
            }
        }}
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
            fn substantiate_f32(a: Self) -> Self { a }
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
            fn substantiate_f64(a: Self) -> Self { a }
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
            fn substantiate_i32(a: Self) -> Self { a }

            if_! { $n == 1 {
                #[inline(always)]
                fn cast_i32(mask: ConstMaskStorage<i32, $m, $n>) -> ConstMaskStorage<i32, $m, $n> { mask }
                #[inline(always)]
                fn cast_i64(mask: ConstMaskStorage<i32, $m, $n>) -> ConstMaskStorage<i64, $m, $n> { mask.cast_i64() }
                #[inline(always)]
                fn mask_select_any<Mask: SupportedElement>(
                    mask: ConstMaskStorage<Mask, $m, $n>,
                    true_values: ConstMaskStorage<i32, $m, $n>,
                    false_values: ConstMaskStorage<i32, $m, $n>,
                ) -> ConstMaskStorage<i32, $m, $n> {
                    // TODO(mask-representation): this casts before loading, which for a 64-bit
                    // mask means narrowing to the two-lane storage type and immediately loading
                    // it back. Deferred: picking the cheaper order per lane width and target
                    // would mean exposing `Load` through `SealedElement`, which is not worth it
                    // for the one instruction it might save.
                    let mask = <Mask as private::SealedElement<$m, $n>>::Storage::cast_i32(mask)
                        .load_mask()
                        .select(true_values.load_mask(), false_values.load_mask());
                    CanonicalMask::store_mask(mask)
                }
            }}
            $($item)*
        });)*
    };
}
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
            fn substantiate_i64(a: Self) -> Self { a }

            if_! { $n == 1 {
                #[inline(always)]
                fn cast_i32(mask: ConstMaskStorage<i64, $m, $n>) -> ConstMaskStorage<i32, $m, $n> { mask.cast_i32() }
                #[inline(always)]
                fn cast_i64(mask: ConstMaskStorage<i64, $m, $n>) -> ConstMaskStorage<i64, $m, $n> { mask }
                #[inline(always)]
                fn mask_select_any<Mask: SupportedElement>(
                    mask: ConstMaskStorage<Mask, $m, $n>,
                    true_values: ConstMaskStorage<i64, $m, $n>,
                    false_values: ConstMaskStorage<i64, $m, $n>,
                ) -> ConstMaskStorage<i64, $m, $n> {
                    <Mask as private::SealedElement<$m, $n>>::Storage::cast_i64(mask)
                        .select(true_values, false_values)
                }
            }}
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
            fn substantiate_u32(a: Self) -> Self { a }
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
            fn substantiate_u64(a: Self) -> Self { a }
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

// 2x3 is the one shape whose storage depends on the element width, so its layouts are spelled out
// here rather than in `call_layouts!`.
//
// A 2x3 matrix holds three columns of two lanes. Packing the first two columns into one four-lane
// unit and leaving the third alone in a second unit is the right shape where a four-lane 64-bit
// value is one 256-bit register. Everywhere else that value is a pair of 128-bit registers, and
// three two-lane units fit better. So a 64-bit element type takes `$vec4 x 2` on AVX2 and
// `$vec2 x 3` without it, while a 32-bit one always takes `$vec4 x 2`.
//
// The gate is AVX2 even though AVX alone already holds a four-lane 64-bit value in one 256-bit
// register. It asks what pays rather than what exists: an AVX-only part may run a 256-bit operation
// as two 128-bit passes, and then the single register buys nothing while the wider unit still costs
// what two narrow ones cost. Every choice in the crate that turns on whether a four-lane 64-bit
// value counts as one register uses this gate; a choice about whether an instruction exists at all
// still gates on the feature that introduces it.
//
// Every kernel that touches the shape therefore comes in an `_in_vec4` and an `_in_vec2` form, and
// each element width re-exports the one it uses under the plain name. `RelayoutStorage` below
// converts between the two arrangements, which is what a cast between element widths needs.

trait RelayoutStorage<const M: usize, const N: usize, const BITS: usize> {
    type Output;
    fn relayout_storage(self) -> Self::Output;
}
macro_rules! impl_relayout_storage {
    ($m:literal, $n:literal) => {
        impl<T, const BITS: usize> RelayoutStorage<$m, $n, BITS> for T {
            type Output = T;
            #[inline(always)]
            fn relayout_storage(self) -> Self::Output { self }
        }
    };
}
impl_relayout_storage!(1, 1);
impl_relayout_storage!(1, 2);
impl_relayout_storage!(1, 3);
impl_relayout_storage!(1, 4);
impl_relayout_storage!(2, 1);
impl_relayout_storage!(2, 2);
impl_relayout_storage!(2, 4);
impl_relayout_storage!(3, 1);
impl_relayout_storage!(3, 2);
impl_relayout_storage!(3, 3);
impl_relayout_storage!(3, 4);
impl_relayout_storage!(4, 1);
impl_relayout_storage!(4, 2);
impl_relayout_storage!(4, 3);
impl_relayout_storage!(4, 4);

impl_layouts_f32!((2, 3; f32x4 x 2 valid [4, 2]) => {});
impl_layouts_i32!((2, 3; i32x4 x 2 valid [4, 2]) => {});
impl_layouts_u32!((2, 3; u32x4 x 2 valid [4, 2]) => {});

#[cfg(target_feature = "avx2")]
mod _2x3 {
    use super::*;
    impl_layouts_f64!((2, 3; f64x4 x 2 valid [4, 2]) => {});
    impl_layouts_i64!((2, 3; i64x4 x 2 valid [4, 2]) => {});
    impl_layouts_u64!((2, 3; u64x4 x 2 valid [4, 2]) => {});
    impl_relayout_storage!(2, 3);
}
#[cfg(not(target_feature = "avx2"))]
mod _2x3 {
    // Only this branch has to move lanes around; the other one relayouts nothing.
    use super::{utils::swizzle, *};

    impl_layouts_f64!((2, 3; f64x2 x 3 valid [2, 2, 2]) => {});
    impl_layouts_i64!((2, 3; i64x2 x 3 valid [2, 2, 2]) => {});
    impl_layouts_u64!((2, 3; u64x2 x 3 valid [2, 2, 2]) => {});

    // A 2x3 matrix is three columns of two lanes. The 32-bit layout packs the first two columns
    // into one four-lane unit and gives the third a unit of its own; the 64-bit layout gives each
    // column its own two-lane unit.
    macro_rules! impl_relayout_storage_2x3 {
        ($f32x2:ident, $f32x4:ident, $f64x2:ident, $f64x4:ident) => {
            impl RelayoutStorage<2, 3, 64> for [$f32x4; 2] {
                type Output = [$f32x2; 3];
                #[inline(always)]
                fn relayout_storage(self) -> Self::Output {
                    let [packed, third] = self;
                    [
                        swizzle!(packed, [0, 1]).store(),
                        swizzle!(packed, [2, 3]).store(),
                        swizzle!(third, [0, 1]).store(),
                    ]
                }
            }
            impl RelayoutStorage<2, 3, 32> for [$f64x2; 3] {
                type Output = [$f64x4; 2];
                #[inline(always)]
                fn relayout_storage(self) -> Self::Output {
                    let [first, second, third] = self;
                    [swizzle!(first, second, @concat), third.widen()]
                }
            }
            impl RelayoutStorage<2, 3, 32> for [$f32x4; 2] {
                type Output = [$f32x4; 2];
                #[inline(always)]
                fn relayout_storage(self) -> Self::Output { self }
            }
            impl RelayoutStorage<2, 3, 64> for [$f64x2; 3] {
                type Output = [$f64x2; 3];
                #[inline(always)]
                fn relayout_storage(self) -> Self::Output { self }
            }
        };
    }
    impl_relayout_storage_2x3!(f32x2, f32x4, f64x2, f64x4);
    impl_relayout_storage_2x3!(i32x2, i32x4, i64x2, i64x4);
    impl_relayout_storage_2x3!(u32x2, u32x4, u64x2, u64x4);
}
