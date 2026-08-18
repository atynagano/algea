//! The WebAssembly swizzle backend: one `swizzle4!` macro and one `Swizzle`/`SwizzleConcat` trait
//! pair covering both 32-bit and 64-bit lanes.
//!
//! Inputs are four-lane compute vectors; results are two or four lanes drawn from one or two
//! operands.
//!
//! # The two widths, and why they share one trait
//!
//! Both widths need constants that the macro must compute while the lane indices are still
//! literals, because stable Rust cannot pass a *computed* const-generic argument (that needs
//! `generic_const_exprs`), and a macro cannot branch on the type of its argument. So the macro
//! computes every constant either width could want and passes them all; each implementation reads
//! only the encoding it can use. A bare const-generic parameter *is* a legal const argument, so an
//! implementation can forward what it receives straight into an intrinsic's own const-generic
//! slot — it just cannot derive anything new on the way.
//!
//! * 32-bit lanes read `I0..I3` directly. `u32x4_shuffle` takes four lane indices over the
//!   concatenation of its two operands, which is exactly the index convention below, so the
//!   indices *are* the encoding.
//! * 64-bit lanes read `I0..I3` to pick which 128-bit half each output lane comes from, and
//!   `S0..S3` as the `u64x2_shuffle` selectors that build the two output halves.
//!
//! # One shuffle, so no instruction selection
//!
//! A 128-bit register holds exactly two 64-bit lanes, so any output half is `[x[.], y[.]]` for two
//! freely chosen halves `x` and `y`. `u64x2_shuffle` expresses all four combinations by itself,
//! numbering its first operand's lanes 0 and 1 and its second operand's 2 and 3, so unlike the SSE2
//! and NEON backends there is nothing to select between: the macro computes the two selectors and
//! the implementation forwards them. Each output lane's selector is `I & 1` for the operand that
//! comes first and `2 + (I & 1)` for the one that comes second.
//!
//! Because each output half is one shuffle over *any* two halves, the operand a lane reads is
//! carried entirely by its own index: `I >> 1` selects from `[a.lo, a.hi, b.lo, b.hi]`. There is no
//! grouping constraint of the kind `shufps` imposes on the SSE2 backend, so neither a family of
//! methods named for the source pattern nor an instruction-selection decoder is needed.
//!
//! # Checking
//!
//! The macro computes `S0..S3` from the same indices it passes as `I0..I3`, and the implementation
//! consumes both directly, so there is no second, independently stated encoding that could
//! disagree with the first — nothing corresponding to the SSE2 backend's symbolic-execution
//! assertions applies here. The tests at the bottom of this file cover every element type.

#![cfg(target_feature = "simd128")]

use crate::simd::utils::ComputeVector;
use core::arch::wasm32::{u32x4_shuffle, u64x2_shuffle, u64x2_splat, v128};
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

// ---------------------------------------------------------------------------
// The constants the macro computes
// ---------------------------------------------------------------------------

/// The `u64x2_shuffle` selector for a lane read from the operand passed first, whose lanes it
/// numbers 0 and 1.
pub(crate) const fn first(index: usize) -> usize { index & 1 }

/// The `u64x2_shuffle` selector for a lane read from the operand passed second, whose lanes it
/// numbers 2 and 3.
pub(crate) const fn second(index: usize) -> usize { 2 + (index & 1) }

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------
//
// `v128` is untyped, so every element type reaches it through `From`/`Into` alone and one
// implementation body serves all three at each width. The only reinterpretation that cannot go
// that way is splitting a four-lane 64-bit value into its halves, because `wide` keeps those
// fields private.

macro_rules! define_conversions {
    ($ty:ty, $reg:ident, $val:ident) => {
        #[inline(always)]
        fn $reg(v: $ty) -> v128 { v.into() }
        #[inline(always)]
        fn $val(v: v128) -> $ty { v.into() }
    };
}
define_conversions!(f32x4, f32_reg, f32_val);
define_conversions!(i32x4, i32_reg, i32_val);
define_conversions!(u32x4, u32_reg, u32_val);
define_conversions!(f64x2, f64_reg, f64_val);
define_conversions!(i64x2, i64_reg, i64_val);
define_conversions!(u64x2, u64_reg, u64_val);

macro_rules! define_split_join {
    ($x4:ty, $x2:ty, $split:ident, $join:ident, $reg:ident, $val:ident) => {
        /// The operand's two 128-bit halves, in lane order.
        #[inline(always)]
        fn $split(v: $x4) -> [v128; 2] {
            // SAFETY: without a 256-bit register, `wide`'s four-lane 64-bit types are
            // `#[repr(C)] { a: X2, b: X2 }`, which has the same layout as `[X2; 2]`; `transmute`
            // additionally checks that the two sizes agree.
            let [lo, hi] = unsafe { core::mem::transmute::<$x4, [$x2; 2]>(v) };
            [$reg(lo), $reg(hi)]
        }
        #[inline(always)]
        fn $join(lo: v128, hi: v128) -> $x4 {
            // SAFETY: see the split above.
            unsafe { core::mem::transmute::<[$x2; 2], $x4>([$val(lo), $val(hi)]) }
        }
    };
}
define_split_join!(f64x4, f64x2, f64_split, f64_join, f64_reg, f64_val);
define_split_join!(i64x4, i64x2, i64_split, i64_join, i64_reg, i64_val);
define_split_join!(u64x4, u64x2, u64_split, u64_join, u64_reg, u64_val);

// ---------------------------------------------------------------------------
// The traits
// ---------------------------------------------------------------------------

/// Rejects an index selecting the upper 128-bit half of an operand that has only one.
macro_rules! assert_lower_half {
    ($($i:ident),+) => {
        const {
            assert!(
                $($i & 2 == 0 &&)+ true,
                "a two-lane operand has no upper half; each lane index must select lane 0 or 1 of \
                 its own operand",
            )
        }
    };
}

#[rustfmt::skip]
pub(crate) trait Swizzle: ComputeVector {
    /// The low two lanes, which is what `Simd4Ext::xy` needs from a four-lane value.
    fn __xy(a: Self) -> Self::Vector2;
    /// The four-lane value whose low two lanes are `a`, which is what `Simd2Ext::widen` needs.
    fn __widen(a: Self) -> Self::Vector4;

    fn swizzle2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self) -> Self::Vector2;
    fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self) -> Self::Vector4;
}

/// Index convention: `i` selects lane `i` of `a` when `i < 4` and lane `i - 4` of `b` otherwise,
/// with both operands seen as four lanes. A two-lane operand's upper two lanes are padding, and are
/// rejected rather than read.
#[rustfmt::skip]
pub(crate) trait SwizzleConcat: Swizzle {
    /// `[a0, a1, b0, b1]`.
    fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self;

    fn swizzle_concat2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self, b: Self) -> Self::Vector2;
    fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self, b: Self) -> Self::Vector4;
}

// ---------------------------------------------------------------------------
// 64-bit lanes
// ---------------------------------------------------------------------------

/// Four-lane operands: the halves of both operands form `[a.lo, a.hi, b.lo, b.hi]`, which `I >> 1`
/// indexes directly.
macro_rules! impl_swizzle_concat_64bit {
    ($self:ty, $split:ident, $join:ident, $reg:ident, $val:ident) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { $val($split(a)[0]) }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }

            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self) -> Self::Vector2 {
                let h = $split(a);
                $val(u64x2_shuffle::<S0, S1>(h[I0 >> 1], h[I1 >> 1]))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self) -> Self::Vector4 {
                let h = $split(a);
                $join(
                    u64x2_shuffle::<S0, S1>(h[I0 >> 1], h[I1 >> 1]),
                    u64x2_shuffle::<S2, S3>(h[I2 >> 1], h[I3 >> 1]),
                )
            }
        }

        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self { $join($reg(a), $reg(b)) }

            #[inline(always)]
            fn swizzle_concat2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self, b: Self) -> Self::Vector2 {
                let (a, b) = ($split(a), $split(b));
                let h = [a[0], a[1], b[0], b[1]];
                $val(u64x2_shuffle::<S0, S1>(h[I0 >> 1], h[I1 >> 1]))
            }

            #[inline(always)]
            fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self, b: Self) -> Self::Vector4 {
                let (a, b) = ($split(a), $split(b));
                let h = [a[0], a[1], b[0], b[1]];
                $join(
                    u64x2_shuffle::<S0, S1>(h[I0 >> 1], h[I1 >> 1]),
                    u64x2_shuffle::<S2, S3>(h[I2 >> 1], h[I3 >> 1]),
                )
            }
        }
    };
}
impl_swizzle_concat_64bit!(f64x4, f64_split, f64_join, f64_reg, f64_val);
impl_swizzle_concat_64bit!(i64x4, i64_split, i64_join, i64_reg, i64_val);
impl_swizzle_concat_64bit!(u64x4, u64_split, u64_join, u64_reg, u64_val);

/// Two-lane operands: one 128-bit register each, so `[a, a, b, b]` keeps `I >> 1` indexing the same
/// way without materializing a padding register.
macro_rules! impl_swizzle_64bit {
    ($self:ty, $reg:ident, $val:ident, $join:ident) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { a }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { $join($reg(a), u64x2_splat(0)) }
            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self) -> Self::Vector2 {
                assert_lower_half!(I0, I1);
                let a = $reg(a);
                $val(u64x2_shuffle::<S0, S1>(a, a))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self) -> Self::Vector4 {
                assert_lower_half!(I0, I1, I2, I3);
                let a = $reg(a);
                $join(u64x2_shuffle::<S0, S1>(a, a), u64x2_shuffle::<S2, S3>(a, a))
            }
        }
    };
}
impl_swizzle_64bit!(f64x2, f64_reg, f64_val, f64_join);
impl_swizzle_64bit!(i64x2, i64_reg, i64_val, i64_join);
impl_swizzle_64bit!(u64x2, u64_reg, u64_val, u64_join);

// ---------------------------------------------------------------------------
// 32-bit lanes
// ---------------------------------------------------------------------------
//
// No compact two-lane register exists here, so the two-lane compute type is the four-lane type
// itself and a two-lane result repeats the requested pair into the padding lanes. `S0..S3` are
// carried through the signatures and ignored: `u32x4_shuffle` already takes the whole pattern.

macro_rules! impl_swizzle_32bit {
    ($self:ty, $reg:ident, $val:ident) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { a }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }
            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self) -> Self::Vector2 {
                $val(u32x4_shuffle::<I0, I1, I0, I1>($reg(a), $reg(a)))
            }
            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self) -> Self::Vector4 {
                $val(u32x4_shuffle::<I0, I1, I2, I3>($reg(a), $reg(a)))
            }
        }
        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self {
                $val(u32x4_shuffle::<0, 1, 4, 5>($reg(a), $reg(b)))
            }

            #[inline(always)]
            fn swizzle_concat2<const I0: usize, const I1: usize, const S0: usize, const S1: usize>(a: Self, b: Self) -> Self::Vector2 {
                $val(u32x4_shuffle::<I0, I1, I0, I1>($reg(a), $reg(b)))
            }
            #[inline(always)]
            fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const S0: usize, const S1: usize, const S2: usize, const S3: usize>(a: Self, b: Self) -> Self::Vector4 {
                $val(u32x4_shuffle::<I0, I1, I2, I3>($reg(a), $reg(b)))
            }
        }
    };
}
impl_swizzle_32bit!(f32x4, f32_reg, f32_val);
impl_swizzle_32bit!(i32x4, i32_reg, i32_val);
impl_swizzle_32bit!(u32x4, u32_reg, u32_val);

// ---------------------------------------------------------------------------
// The macro
// ---------------------------------------------------------------------------

/// A partly specified index list is completed by repeating a lane, never by routing to a different
/// shuffle: every pattern already costs one shuffle per output half here, so a special case has
/// nothing to save.
macro_rules! swizzle4 {
    ($a:expr, [$i0:tt]) => {
        compile_error!(
            "a swizzle produces at least two lanes; a single index selects a scalar, not a vector"
        )
    };
    ($a:expr, [$i0:tt, _, _, _]) => {
        compile_error!(
            "only the first lane is given; the other three cannot be inferred, so spell them out"
        )
    };
    ($a:expr, [$i0:tt, $i1:tt]) => {
        $crate::simd::swizzle_wasm::Swizzle::swizzle2::<
            { $crate::simd::utils::validate_lane4!($i0) },
            { $crate::simd::utils::validate_lane4!($i1) },
            { $crate::simd::swizzle_wasm::first($i0) },
            { $crate::simd::swizzle_wasm::second($i1) },
        >($a)
    };
    ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, [$i0, $i1, $i2, _])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_wasm::Swizzle::swizzle4::<
            { $crate::simd::utils::validate_lane4!($i0) },
            { $crate::simd::utils::validate_lane4!($i1) },
            { $crate::simd::utils::validate_lane4!($i2) },
            { $crate::simd::utils::validate_lane4!($i3) },
            { $crate::simd::swizzle_wasm::first($i0) },
            { $crate::simd::swizzle_wasm::second($i1) },
            { $crate::simd::swizzle_wasm::first($i2) },
            { $crate::simd::swizzle_wasm::second($i3) },
        >($a)
    };

    ($a:expr, $b:expr, @concat) => {
        $crate::simd::swizzle_wasm::SwizzleConcat::concat_4($a, $b)
    };
    ($a:expr, $b:expr, [$i0:tt]) => {
        compile_error!(
            "a swizzle produces at least two lanes; a single index selects a scalar, not a vector"
        )
    };
    ($a:expr, $b:expr, [$i0:tt, _, _, _]) => {
        compile_error!(
            "only the first lane is given; the other three cannot be inferred, so spell them out"
        )
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt]) => {
        $crate::simd::swizzle_wasm::SwizzleConcat::swizzle_concat2::<
            { $crate::simd::utils::validate_lane8!($i0) },
            { $crate::simd::utils::validate_lane8!($i1) },
            { $crate::simd::swizzle_wasm::first($i0) },
            { $crate::simd::swizzle_wasm::second($i1) },
        >($a, $b)
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, $b, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, $b, [$i0, $i1, $i2, _])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_wasm::swizzle4!($a, $b, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_wasm::SwizzleConcat::swizzle_concat4::<
            { $crate::simd::utils::validate_lane8!($i0) },
            { $crate::simd::utils::validate_lane8!($i1) },
            { $crate::simd::utils::validate_lane8!($i2) },
            { $crate::simd::utils::validate_lane8!($i3) },
            { $crate::simd::swizzle_wasm::first($i0) },
            { $crate::simd::swizzle_wasm::second($i1) },
            { $crate::simd::swizzle_wasm::first($i2) },
            { $crate::simd::swizzle_wasm::second($i3) },
        >($a, $b)
    };
}

pub(crate) use swizzle4;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Four-lane 64-bit operands, whose two-lane companion is a genuine two-lane type.
    macro_rules! check_4lane_64bit {
        ($name:ident, $s:ty, $new:expr, $read4:expr, $read2:expr) => {
            #[test]
            #[rustfmt::skip]
            fn $name() {
                let new = $new;
                let read4 = $read4;
                let read2 = $read2;
                let a = new([10 as $s, 11 as $s, 12 as $s, 13 as $s]);
                let b = new([20 as $s, 21 as $s, 22 as $s, 23 as $s]);

                // Both output halves read the one operand.
                assert_eq!(read4(swizzle4!(a, [3, 1, 2, 0])), [13 as $s, 11 as $s, 12 as $s, 10 as $s]);
                // Every output lane reads a different half.
                assert_eq!(read4(swizzle4!(a, b, [7, 0, 5, 2])), [23 as $s, 10 as $s, 21 as $s, 12 as $s]);
                // The interleaves the SSE2 backend special-cases as `unpack`.
                assert_eq!(read4(swizzle4!(a, b, [0, 4, 1, 5])), [10 as $s, 20 as $s, 11 as $s, 21 as $s]);
                assert_eq!(read4(swizzle4!(a, [0, 2, 1, 3])), [10 as $s, 12 as $s, 11 as $s, 13 as $s]);
                // Whole halves.
                assert_eq!(read4(swizzle4!(a, [2, 3, 0, 1])), [12 as $s, 13 as $s, 10 as $s, 11 as $s]);
                assert_eq!(read4(swizzle4!(a, b, [0, 1, 4, 5])), [10 as $s, 11 as $s, 20 as $s, 21 as $s]);
                // Broadcast.
                assert_eq!(read4(swizzle4!(a, [1, 1, 1, 1])), [11 as $s, 11 as $s, 11 as $s, 11 as $s]);

                // Concatenating the low halves of two two-lane values.
                assert_eq!(
                    read4(swizzle4!(swizzle4!(a, [0, 1]), swizzle4!(b, [0, 1]), @concat)),
                    [10 as $s, 11 as $s, 20 as $s, 21 as $s],
                );

                // Two-lane results.
                assert_eq!(read2(swizzle4!(a, [3, 1])), [13 as $s, 11 as $s]);
                assert_eq!(read2(swizzle4!(a, b, [4, 1])), [20 as $s, 11 as $s]);
                assert_eq!(read2(swizzle4!(a, b, [2, 6])), [12 as $s, 22 as $s]);
            }
        };
    }
    check_4lane_64bit!(f64x4_lanes, f64, f64x4::new, f64x4::to_array, f64x2::to_array);
    check_4lane_64bit!(i64x4_lanes, i64, i64x4::new, i64x4::to_array, i64x2::to_array);
    check_4lane_64bit!(u64x4_lanes, u64, u64x4::new, u64x4::to_array, u64x2::to_array);

    /// A two-lane 64-bit operand. Only lanes 0 and 1 exist, so every index here satisfies
    /// `i & 2 == 0`. There is no two-operand case to check: that needs `SwizzleConcat`, which only
    /// the four-lane types implement.
    macro_rules! check_2lane_64bit {
        ($name:ident, $s:ty, $new:expr, $read2:expr, $read4:expr) => {
            #[test]
            #[rustfmt::skip]
            fn $name() {
                let new = $new;
                let read2 = $read2;
                let read4 = $read4;
                let a = new([10 as $s, 11 as $s]);

                assert_eq!(read2(swizzle4!(a, [1, 0])), [11 as $s, 10 as $s]);
                assert_eq!(read2(swizzle4!(a, [0, 0])), [10 as $s, 10 as $s]);
                assert_eq!(read4(swizzle4!(a, [1, 0, 0, 1])), [11 as $s, 10 as $s, 10 as $s, 11 as $s]);
            }
        };
    }
    check_2lane_64bit!(f64x2_lanes, f64, f64x2::new, f64x2::to_array, f64x4::to_array);
    check_2lane_64bit!(i64x2_lanes, i64, i64x2::new, i64x2::to_array, i64x4::to_array);
    check_2lane_64bit!(u64x2_lanes, u64, u64x2::new, u64x2::to_array, u64x4::to_array);

    /// 32-bit operands. A two-lane result is the four-lane type with the requested pair repeated,
    /// so it is read back four lanes at a time.
    macro_rules! check_32bit {
        ($name:ident, $s:ty, $new:expr, $read:expr) => {
            #[test]
            #[rustfmt::skip]
            fn $name() {
                let new = $new;
                let read = $read;
                let a = new([10 as $s, 11 as $s, 12 as $s, 13 as $s]);
                let b = new([20 as $s, 21 as $s, 22 as $s, 23 as $s]);

                assert_eq!(read(swizzle4!(a, [3, 1, 2, 0])), [13 as $s, 11 as $s, 12 as $s, 10 as $s]);
                assert_eq!(read(swizzle4!(a, b, [7, 0, 5, 2])), [23 as $s, 10 as $s, 21 as $s, 12 as $s]);
                assert_eq!(read(swizzle4!(a, b, [0, 4, 1, 5])), [10 as $s, 20 as $s, 11 as $s, 21 as $s]);
                assert_eq!(read(swizzle4!(a, [0, 2, 1, 3])), [10 as $s, 12 as $s, 11 as $s, 13 as $s]);
                assert_eq!(read(swizzle4!(a, [2, 3, 0, 1])), [12 as $s, 13 as $s, 10 as $s, 11 as $s]);
                assert_eq!(read(swizzle4!(a, b, [0, 1, 4, 5])), [10 as $s, 11 as $s, 20 as $s, 21 as $s]);
                assert_eq!(read(swizzle4!(a, [1, 1, 1, 1])), [11 as $s, 11 as $s, 11 as $s, 11 as $s]);

                // Two-lane requests repeat the pair into the padding lanes.
                assert_eq!(read(swizzle4!(a, [3, 1])), [13 as $s, 11 as $s, 13 as $s, 11 as $s]);
                assert_eq!(read(swizzle4!(a, b, [4, 1])), [20 as $s, 11 as $s, 20 as $s, 11 as $s]);
            }
        };
    }
    check_32bit!(f32x4_lanes, f32, f32x4::new, f32x4::to_array);
    check_32bit!(i32x4_lanes, i32, i32x4::new, i32x4::to_array);
    check_32bit!(u32x4_lanes, u32, u32x4::new, u32x4::to_array);
}
