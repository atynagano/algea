#![allow(missing_docs, clippy::missing_inline_in_public_items)]

use algea::{EachOrd, Mask, Select, Vector, column_major, row_major};
use paste::paste;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};

macro_rules! vector_binary {
    ($t:ty, $d:literal, $name:ident, $trait:ident, $method:ident) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _ $name _vector>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Vector<$t, $d> {
                $trait::$method(a, b)
            }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _ $name _scalar>](a: Vector<$t, $d>, b: $t) -> Vector<$t, $d> {
                $trait::$method(a, b)
            }
            #[unsafe(no_mangle)]
            pub fn [<scalar_ $t _vector_ $d _ $name>](a: $t, b: Vector<$t, $d>) -> Vector<$t, $d> {
                $trait::$method(a, b)
            }
        }
    };
}

macro_rules! vector_all {
    ($t:ty, $m:ty, $d:literal) => {
        vector_binary!($t, $d, add, Add, add);
        vector_binary!($t, $d, sub, Sub, sub);
        vector_binary!($t, $d, mul, Mul, mul);
        vector_binary!($t, $d, div, Div, div);
        vector_binary!($t, $d, rem, Rem, rem);
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _splat>](a: $t) -> Vector<$t, $d> { Vector::splat(a) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _from_array>](a: [$t; $d]) -> Vector<$t, $d> { Vector::from_array(a) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _to_array>](a: Vector<$t, $d>) -> [$t; $d] { a.to_array() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_f32>](a: Vector<$t, $d>) -> Vector<f32, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_f64>](a: Vector<$t, $d>) -> Vector<f64, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_i32>](a: Vector<$t, $d>) -> Vector<i32, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_i64>](a: Vector<$t, $d>) -> Vector<i64, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_u32>](a: Vector<$t, $d>) -> Vector<u32, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_u64>](a: Vector<$t, $d>) -> Vector<u64, $d> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _eq>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> bool { a == b }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _ne>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> bool { a != b }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_eq>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_eq(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_ne>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_ne(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_lt>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_lt(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_le>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_le(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_gt>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_gt(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_ge>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Mask<$m, $d> { a.each_ge(b) }
        }
    };
}

macro_rules! vector_float {
    ($t:ty, $bits:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _dot>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> $t { a.dot(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _norm>](a: Vector<$t, $d>) -> $t { a.norm() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _norm_squared>](a: Vector<$t, $d>) -> $t { a.norm_squared() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _distance>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> $t { a.distance(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _distance_squared>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> $t { a.distance_squared(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _normalize>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.normalize() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _floor>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.floor() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _ceil>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.ceil() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _round>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.round() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _round_ties_even>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.round_ties_even() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _trunc>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.trunc() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _fract>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.fract() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _sqrt>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.sqrt() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _recip>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.recip() }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _from_bits>](a: Vector<$bits, $d>) -> Vector<$t, $d> { Vector::from_bits(a) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _to_bits>](a: Vector<$t, $d>) -> Vector<$bits, $d> { a.to_bits() }
        }
    };
}

macro_rules! vector_signed {
    ($t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _neg>](a: Vector<$t, $d>) -> Vector<$t, $d> { -a }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _abs>](a: Vector<$t, $d>) -> Vector<$t, $d> { a.abs() }
        }
    };
}

macro_rules! vector_order {
    ($t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_max>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Vector<$t, $d> { a.each_max(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_min>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Vector<$t, $d> { a.each_min(b) }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _each_clamp>](a: Vector<$t, $d>, min: Vector<$t, $d>, max: Vector<$t, $d>) -> Vector<$t, $d> { a.each_clamp(min, max) }
        }
    };
}

macro_rules! vector_int {
    ($t:ty, $u:ty, $d:literal) => {
        vector_binary!($t, $d, bitand, BitAnd, bitand);
        vector_binary!($t, $d, bitor, BitOr, bitor);
        vector_binary!($t, $d, bitxor, BitXor, bitxor);
        vector_binary!($t, $d, shl, Shl, shl);
        vector_binary!($t, $d, shr, Shr, shr);
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _not>](a: Vector<$t, $d>) -> Vector<$t, $d> { !a }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _abs_diff>](a: Vector<$t, $d>, b: Vector<$t, $d>) -> Vector<$u, $d> { a.abs_diff(b) }
        }
    };
}

macro_rules! vector_signed_int {
    ($t:ty, $u:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_unsigned>](a: Vector<$t, $d>) -> Vector<$u, $d> { a.cast_unsigned() }
        }
    };
}

macro_rules! vector_unsigned_int {
    ($t:ty, $s:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _cast_signed>](a: Vector<$t, $d>) -> Vector<$s, $d> { a.cast_signed() }
        }
    };
}

macro_rules! mask_methods {
    ($m:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _from_array>](a: [bool; $d]) -> Mask<$m, $d> { a.into() }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _splat>](a: bool) -> Mask<$m, $d> { Mask::splat(a) }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _to_array>](a: Mask<$m, $d>) -> [bool; $d] { a.to_array() }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _to_vector>](a: Mask<$m, $d>) -> Vector<$m, $d> { a.to_vector() }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _all>](a: Mask<$m, $d>) -> bool { a.all() }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _any>](a: Mask<$m, $d>) -> bool { a.any() }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _not>](a: Mask<$m, $d>) -> Mask<$m, $d> { !a }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _bitand>](a: Mask<$m, $d>, b: Mask<$m, $d>) -> Mask<$m, $d> { a & b }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _bitor>](a: Mask<$m, $d>, b: Mask<$m, $d>) -> Mask<$m, $d> { a | b }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _bitxor>](a: Mask<$m, $d>, b: Mask<$m, $d>) -> Mask<$m, $d> { a ^ b }
        }
    };
}

macro_rules! mask_bitmask_methods {
    ($t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<mask_ $t _ $d _from_bitmask>](bitmask: u8) -> Mask<$t, $d> {
                Mask::from_bitmask(bitmask)
            }
            #[unsafe(no_mangle)]
            pub fn [<mask_ $t _ $d _to_bitmask>](mask: Mask<$t, $d>) -> u8 {
                mask.to_bitmask()
            }
            #[unsafe(no_mangle)]
            pub fn [<scalar_u8_mask_ $t _ $d _select>](
                bitmask: u8,
                true_values: Mask<$t, $d>,
                false_values: Mask<$t, $d>,
            ) -> Mask<$t, $d> {
                bitmask.select(true_values, false_values)
            }
            #[unsafe(no_mangle)]
            pub fn [<scalar_u8_vector_ $t _ $d _select>](
                bitmask: u8,
                true_values: Vector<$t, $d>,
                false_values: Vector<$t, $d>,
            ) -> Vector<$t, $d> {
                bitmask.select(true_values, false_values)
            }
        }
    };
}

macro_rules! mask_cast {
    ($source:ty, $target:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<mask_ $source _ $d _cast_ $target>](mask: Mask<$source, $d>) -> Mask<$target, $d> {
                mask.cast()
            }
        }
    };
}

macro_rules! mask_select {
    ($m:ty, $t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _select_vector_ $t>](mask: Mask<$m, $d>, a: Vector<$t, $d>, b: Vector<$t, $d>) -> Vector<$t, $d> {
                mask.select(a, b)
            }
        }
    };
}

macro_rules! mask_select_mask {
    ($m:ty, $u:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<mask_ $m _ $d _select_mask_ $u>](mask: Mask<$m, $d>, a: Mask<$u, $d>, b: Mask<$u, $d>) -> Mask<$u, $d> {
                mask.select(a, b)
            }
        }
    };
}

macro_rules! swizzle_d {
    ($t:ty, $d:literal, $a:ident, $b:ident, $c:ident; [$($e:ident),+]) => {
        $(paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _swizzle_ $a $b $c $e>](a: Vector<$t, $d>) -> Vector<$t, 4> {
                a.[<$a $b $c $e>]()
            }
        })+
    };
}
macro_rules! swizzle_c {
    ($t:ty, $d:literal, $a:ident, $b:ident; $letters:tt) => {
        swizzle_c!(@each $t, $d, $a, $b; $letters; $letters);
    };
    (@each $t:ty, $d:literal, $a:ident, $b:ident; [$($c:ident),+]; $letters:tt) => {
        $(paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _swizzle_ $a $b $c>](a: Vector<$t, $d>) -> Vector<$t, 3> {
                a.[<$a $b $c>]()
            }
        }
        swizzle_d!($t, $d, $a, $b, $c; $letters);)+
    };
}
macro_rules! swizzle_b {
    ($t:ty, $d:literal, $a:ident; $letters:tt) => {
        swizzle_b!(@each $t, $d, $a; $letters; $letters);
    };
    (@each $t:ty, $d:literal, $a:ident; [$($b:ident),+]; $letters:tt) => {
        $(paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _ $d _swizzle_ $a $b>](a: Vector<$t, $d>) -> Vector<$t, 2> {
                a.[<$a $b>]()
            }
        }
        swizzle_c!($t, $d, $a, $b; $letters);)+
    };
}
macro_rules! swizzles {
    ($t:ty, $d:literal, $letters:tt) => {
        swizzles!(@each $t, $d; $letters; $letters);
    };
    (@each $t:ty, $d:literal; [$($a:ident),+]; $letters:tt) => {
        $(swizzle_b!($t, $d, $a; $letters);)+
    };
}

macro_rules! dimension {
    ($d:literal, $letters:tt, [$($t:ty),+]) => {
        vector_all!(f32, i32, $d);
        vector_all!(f64, i64, $d);
        vector_all!(i32, i32, $d);
        vector_all!(i64, i64, $d);
        vector_all!(u32, i32, $d);
        vector_all!(u64, i64, $d);
        vector_float!(f32, u32, $d);
        vector_float!(f64, u64, $d);
        vector_signed!(f32, $d);
        vector_signed!(f64, $d);
        vector_signed!(i32, $d);
        vector_signed!(i64, $d);
        vector_order!(f32, $d);
        vector_order!(f64, $d);
        vector_order!(i32, $d);
        vector_order!(i64, $d);
        vector_order!(u32, $d);
        vector_order!(u64, $d);
        vector_int!(i32, u32, $d);
        vector_int!(i64, u64, $d);
        vector_int!(u32, u32, $d);
        vector_int!(u64, u64, $d);
        vector_signed_int!(i32, u32, $d);
        vector_signed_int!(i64, u64, $d);
        vector_unsigned_int!(u32, i32, $d);
        vector_unsigned_int!(u64, i64, $d);
        mask_methods!(i32, $d);
        mask_methods!(i64, $d);
        mask_cast!(i32, i32, $d);
        mask_cast!(i32, i64, $d);
        mask_cast!(i64, i32, $d);
        mask_cast!(i64, i64, $d);
        mask_bitmask_methods!(i32, $d);
        mask_bitmask_methods!(i64, $d);
        mask_select_mask!(i32, i32, $d);
        mask_select_mask!(i32, i64, $d);
        mask_select_mask!(i64, i32, $d);
        mask_select_mask!(i64, i64, $d);
        $(mask_select!(i32, $t, $d); mask_select!(i64, $t, $d);)*
        $(swizzles!($t, $d, $letters);)*
    };
}

dimension!(1, [x], [f32, f64, i32, i64, u32, u64]);
dimension!(2, [x, y], [f32, f64, i32, i64, u32, u64]);
dimension!(3, [x, y, z], [f32, f64, i32, i64, u32, u64]);
dimension!(4, [x, y, z, w], [f32, f64, i32, i64, u32, u64]);

macro_rules! vector_concat {
    ($t:ty) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_1>](a: $t, b: $t) -> Vector<$t, 2> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_1_1>](a: $t, b: $t, c: $t) -> Vector<$t, 3> { algea::vector![a, b, c] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_1_1_1>](a: $t, b: $t, c: $t, d: $t) -> Vector<$t, 4> { algea::vector![a, b, c, d] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_2>](a: $t, b: Vector<$t, 2>) -> Vector<$t, 3> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_2_1>](a: Vector<$t, 2>, b: $t) -> Vector<$t, 3> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_3>](a: $t, b: Vector<$t, 3>) -> Vector<$t, 4> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_3_1>](a: Vector<$t, 3>, b: $t) -> Vector<$t, 4> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_2_2>](a: Vector<$t, 2>, b: Vector<$t, 2>) -> Vector<$t, 4> { algea::vector![a, b] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_1_2>](a: $t, b: $t, c: Vector<$t, 2>) -> Vector<$t, 4> { algea::vector![a, b, c] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_1_2_1>](a: $t, b: Vector<$t, 2>, c: $t) -> Vector<$t, 4> { algea::vector![a, b, c] }
            #[unsafe(no_mangle)]
            pub fn [<vector_ $t _concat_2_1_1>](a: Vector<$t, 2>, b: $t, c: $t) -> Vector<$t, 4> { algea::vector![a, b, c] }
        }
    };
}

vector_concat!(f32);
vector_concat!(f64);
vector_concat!(i32);
vector_concat!(i64);
vector_concat!(u32);
vector_concat!(u64);

macro_rules! matrix_binary {
    ($layout:ident, $t:ty, $r:literal, $c:literal, $name:ident, $trait:ident, $method:ident) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _ $name _matrix>](a: $layout::Matrix<$t, $r, $c>, b: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<$t, $r, $c> {
                $trait::$method(a, b)
            }
        }
    };
}

macro_rules! matrix_scalar {
    ($layout:ident, $t:ty, $r:literal, $c:literal, $name:ident, $trait:ident, $method:ident) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _ $name _scalar>](a: $layout::Matrix<$t, $r, $c>, b: $t) -> $layout::Matrix<$t, $r, $c> {
                $trait::$method(a, b)
            }
            #[unsafe(no_mangle)]
            pub fn [<scalar_ $t _ $layout _matrix_ $r x $c _ $name>](a: $t, b: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<$t, $r, $c> {
                $trait::$method(a, b)
            }
        }
    };
}

macro_rules! matrix_all {
    ($layout:ident, $t:ty, $r:literal, $c:literal) => {
        matrix_binary!($layout, $t, $r, $c, add, Add, add);
        matrix_binary!($layout, $t, $r, $c, sub, Sub, sub);
        matrix_scalar!($layout, $t, $r, $c, add, Add, add);
        matrix_scalar!($layout, $t, $r, $c, sub, Sub, sub);
        matrix_scalar!($layout, $t, $r, $c, mul, Mul, mul);
        matrix_scalar!($layout, $t, $r, $c, div, Div, div);
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _filled>](a: $t) -> $layout::Matrix<$t, $r, $c> { $layout::Matrix::filled(a) }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_f32>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<f32, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_f64>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<f64, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_i32>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<i32, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_i64>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<i64, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_u32>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<u32, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _cast_u64>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<u64, $r, $c> { a.cast() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _eq>](a: $layout::Matrix<$t, $r, $c>, b: $layout::Matrix<$t, $r, $c>) -> bool { a == b }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _ne>](a: $layout::Matrix<$t, $r, $c>, b: $layout::Matrix<$t, $r, $c>) -> bool { a != b }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _transpose>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<$t, $c, $r> { a.transpose() }
        }
    };
}

macro_rules! row_methods {
    ($t:ty, $r:literal, $c:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x $c _from_rows>](a: [[$t; $c]; $r]) -> row_major::Matrix<$t, $r, $c> { row_major::Matrix::from_rows(a) }
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x $c _from_row_vecs>](a: [Vector<$t, $c>; $r]) -> row_major::Matrix<$t, $r, $c> { row_major::Matrix::from_row_vecs(a) }
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x $c _to_rows>](a: row_major::Matrix<$t, $r, $c>) -> [[$t; $c]; $r] { a.to_rows() }
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x $c _to_row_vecs>](a: row_major::Matrix<$t, $r, $c>) -> [Vector<$t, $c>; $r] { a.to_row_vecs() }
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x $c _to_column_major_transposed>](a: row_major::Matrix<$t, $r, $c>) -> column_major::Matrix<$t, $c, $r> { a.to_column_major_transposed() }
        }
    };
}

macro_rules! column_methods {
    ($t:ty, $r:literal, $c:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _from_columns>](a: [[$t; $r]; $c]) -> column_major::Matrix<$t, $r, $c> { column_major::Matrix::from_columns(a) }
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _from_column_vecs>](a: [Vector<$t, $r>; $c]) -> column_major::Matrix<$t, $r, $c> { column_major::Matrix::from_column_vecs(a) }
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _to_columns>](a: column_major::Matrix<$t, $r, $c>) -> [[$t; $r]; $c] { a.to_columns() }
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _to_column_vecs>](a: column_major::Matrix<$t, $r, $c>) -> [Vector<$t, $r>; $c] { a.to_column_vecs() }
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _to_row_major_transposed>](a: column_major::Matrix<$t, $r, $c>) -> row_major::Matrix<$t, $c, $r> { a.to_row_major_transposed() }
        }
    };
}

macro_rules! matrix_signed {
    ($layout:ident, $t:ty, $r:literal, $c:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $c _neg>](a: $layout::Matrix<$t, $r, $c>) -> $layout::Matrix<$t, $r, $c> { -a }
        }
    };
}

macro_rules! matrix_square {
    ($layout:ident, $t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $d x $d _diagonal>](a: $layout::Matrix<$t, $d, $d>) -> Vector<$t, $d> { a.diagonal() }
        }
    };
}

macro_rules! matrix_float_square {
    ($layout:ident, $t:ty, $d:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $d x $d _determinant>](a: $layout::Matrix<$t, $d, $d>) -> $t { a.determinant() }
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $d x $d _inverse>](a: $layout::Matrix<$t, $d, $d>) -> $layout::Matrix<$t, $d, $d> { a.inverse() }
        }
    };
}

macro_rules! matrix_shape {
    ($r:literal, $c:literal; [$($t:ty),+]; [$($signed:ty),+]) => {
        $(matrix_all!(row_major, $t, $r, $c); matrix_all!(column_major, $t, $r, $c);
          row_methods!($t, $r, $c); column_methods!($t, $r, $c);)*
        $(matrix_signed!(row_major, $signed, $r, $c); matrix_signed!(column_major, $signed, $r, $c);)*
    };
}

macro_rules! matrix_shape_all {
    ($r:literal, $c:literal) => {
        matrix_shape!($r, $c; [f32, f64, i32, i64, u32, u64]; [f32, f64, i32, i64]);
    };
}

macro_rules! matrix_product {
    ($layout:ident, $t:ty, $r:literal, $k:literal, $c:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<$layout _matrix_ $t _ $r x $k _mul_ $k x $c>](a: $layout::Matrix<$t, $r, $k>, b: $layout::Matrix<$t, $k, $c>) -> $layout::Matrix<$t, $r, $c> { a * b }
        }
    };
}

macro_rules! product_shape {
    ($t:ty, $r:literal, $c:literal) => {
        paste! {
            #[unsafe(no_mangle)]
            pub fn [<row_major_vector_ $t _ $r _mul_matrix_ $r x $c>](a: Vector<$t, $r>, b: row_major::Matrix<$t, $r, $c>) -> Vector<$t, $c> { a * b }
            #[unsafe(no_mangle)]
            pub fn [<column_major_matrix_ $t _ $r x $c _mul_vector_ $c>](a: column_major::Matrix<$t, $r, $c>, b: Vector<$t, $c>) -> Vector<$t, $r> { a * b }
            #[unsafe(no_mangle)]
            pub fn [<row_major_matrix_ $t _ $r x 1 _mul_vector_ $c>](a: row_major::Matrix<$t, $r, 1>, b: Vector<$t, $c>) -> row_major::Matrix<$t, $r, $c> { a * b }
            #[unsafe(no_mangle)]
            pub fn [<column_major_vector_ $t _ $r _mul_matrix_ 1 x $c>](a: Vector<$t, $r>, b: column_major::Matrix<$t, 1, $c>) -> column_major::Matrix<$t, $r, $c> { a * b }
        }
    };
}

macro_rules! product_k {
    ($t:ty, $r:literal, $c:literal; [$($k:literal),+]) => {
        $(matrix_product!(row_major, $t, $r, $k, $c);
          matrix_product!(column_major, $t, $r, $k, $c);)+
    };
}

macro_rules! all_shapes_for_row {
    ($r:literal) => {
        all_shapes_for_row!(@cols $r; [1, 2, 3, 4]);
    };
    (@cols $r:literal; [$($c:literal),+]) => {
        $(matrix_shape_all!($r, $c);
          product_shape!(f32, $r, $c);
          product_shape!(f64, $r, $c);
          product_k!(f32, $r, $c; [1, 2, 3, 4]);
          product_k!(f64, $r, $c; [1, 2, 3, 4]);)+
    };
}

macro_rules! square {
    ($d:literal; [$($t:ty),+]; [$($f:ty),+]) => {
        $(matrix_square!(row_major, $t, $d); matrix_square!(column_major, $t, $d);)*
        $(matrix_float_square!(row_major, $f, $d); matrix_float_square!(column_major, $f, $d);)*
    };
}

all_shapes_for_row!(1);
all_shapes_for_row!(2);
all_shapes_for_row!(3);
all_shapes_for_row!(4);
square!(1; [f32, f64, i32, i64, u32, u64]; [f32, f64]);
square!(2; [f32, f64, i32, i64, u32, u64]; [f32, f64]);
square!(3; [f32, f64, i32, i64, u32, u64]; [f32, f64]);
square!(4; [f32, f64, i32, i64, u32, u64]; [f32, f64]);
