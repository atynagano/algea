//! The aarch64 swizzle backend: one `swizzle4!` macro and one `Swizzle`/`SwizzleConcat` trait pair
//! covering both 32-bit and 64-bit lanes.
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
//! only the encoding it can use.
//!
//! * 32-bit lanes read `I0..I3` directly. A byte-table lookup (`vqtbl1q_u8`/`vqtbl2q_u8`) expresses
//!   every index pattern as one instruction sequence, so the indices *are* the encoding.
//! * 64-bit lanes read `I0..I3` to pick which 128-bit half each output lane comes from, and
//!   `PD_LO`/`PD_HI` to pick the instruction that builds each output half.
//!
//! # `PD` selects an instruction, not an immediate
//!
//! A 128-bit register holds exactly two 64-bit lanes, so any output half is `[x[.], y[.]]` for two
//! freely chosen halves `x` and `y` — four combinations in total. SSE2 covers all four with
//! `_mm_shuffle_pd` and a two-bit immediate; aarch64 spreads them over four distinct instructions.
//! Encoding the choice as that same two-bit value and dispatching through `Shuffle<MASK>` keeps the
//! constant identical to the one the SSE2 backend computes for the same pair of lane indices:
//!
//! ```text
//! 0b00 -> [x0, y0]   zip1
//! 0b01 -> [x1, y0]   ext #8
//! 0b10 -> [x0, y1]   ins  (mov v.d[1], w.d[1])
//! 0b11 -> [x1, y1]   zip2
//! ```
//!
//! Because each output half is one instruction over *any* two halves, the operand a lane reads is
//! carried entirely by its own index: `I >> 1` selects from `[a.lo, a.hi, b.lo, b.hi]`. There is no
//! grouping constraint of the kind `shufps` imposes on the SSE2 backend, so neither a family of
//! methods named for the source pattern nor an instruction-selection decoder is needed.
//!
//! # Checking
//!
//! The macro computes `PD` from the same indices it passes as `I0..I3`, and the implementation
//! consumes both directly, so there is no second, independently stated encoding that could
//! disagree with the first — nothing corresponding to the SSE2 backend's symbolic-execution
//! assertions applies here. What compile-time checking cannot reach is the four-line `Shuffle`
//! table above: a wrong entry produces wrong lanes with no diagnostic, so the tests at the bottom
//! of this file cover every element type.

#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use super::utils::{ComputeVector, f32x2, i32x2, u32x2};
use core::arch::aarch64::*;
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};

// ---------------------------------------------------------------------------
// The constant the macro computes
// ---------------------------------------------------------------------------

/// Selects the instruction that builds one 128-bit output half: bit 0 is the lane taken from the
/// first operand, bit 1 the lane taken from the second.
pub(crate) const fn pd(l0: usize, l1: usize) -> i32 { ((l0 & 1) | (l1 & 1) << 1) as i32 }

// ---------------------------------------------------------------------------
// Mask to instruction
// ---------------------------------------------------------------------------

/// One 128-bit output half, `[a[MASK & 1], b[(MASK >> 1) & 1]]`, as a single instruction.
///
/// Implemented only for the four valid masks, so a mask the macro should never produce fails to
/// resolve instead of selecting a wrong instruction.
pub(crate) trait Shuffle<const MASK: i32> {
    fn shuffle(a: Self, b: Self) -> Self;
}

macro_rules! impl_shuffle {
    ($mask:literal => $call:ident $(::<$($n:literal),+>)?) => {
        impl Shuffle<$mask> for float64x2_t {
            #[inline(always)]
            fn shuffle(a: Self, b: Self) -> Self {
                // SAFETY: NEON is part of the aarch64 baseline, which this module is gated on.
                unsafe { $call$(::<$($n),+>)?(a, b) }
            }
        }
    };
}
impl_shuffle!(0b00 => vzip1q_f64);
impl_shuffle!(0b01 => vextq_f64::<1>);
impl_shuffle!(0b10 => vcopyq_laneq_f64::<1, 1>);
impl_shuffle!(0b11 => vzip2q_f64);

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------
//
// Each element type is carried to one common register type so that a single implementation body
// serves all three: `uint8x16_t` for the byte-table path, `float64x2_t` for the 64-bit path. The
// only reinterpretation that cannot go through `From`/`vreinterpretq_*` is splitting a four-lane
// 64-bit value into its halves, because `wide` keeps those fields private.

/// Stands in for a `vreinterpretq_*` where the element type already matches. Declared `unsafe` so
/// that one macro can call it exactly as it calls the real reinterpretations.
///
/// # Safety
///
/// Always safe; the marker exists only for uniformity.
#[inline(always)]
unsafe fn same_f64(v: float64x2_t) -> float64x2_t { v }

macro_rules! define_64bit_conversions {
    ($x4:ty, $x2:ty, $split:ident, $join:ident, $reg:ident, $val:ident, $to:ident, $from:ident) => {
        #[inline(always)]
        fn $reg(v: $x2) -> float64x2_t {
            // SAFETY: NEON is part of the aarch64 baseline; reinterpreting one 128-bit vector
            // register as another has no preconditions.
            unsafe { $to(v.into()) }
        }
        #[inline(always)]
        fn $val(v: float64x2_t) -> $x2 {
            // SAFETY: see the conversion above.
            unsafe { $from(v) }.into()
        }
        /// The operand's two 128-bit halves, in lane order.
        #[inline(always)]
        fn $split(v: $x4) -> [float64x2_t; 2] {
            // SAFETY: without a 256-bit register, `wide`'s four-lane 64-bit types are
            // `#[repr(C)] { a: X2, b: X2 }`, which has the same layout as `[X2; 2]`; `transmute`
            // additionally checks that the two sizes agree.
            let [lo, hi] = unsafe { core::mem::transmute::<$x4, [$x2; 2]>(v) };
            [$reg(lo), $reg(hi)]
        }
        #[inline(always)]
        fn $join(lo: float64x2_t, hi: float64x2_t) -> $x4 {
            // SAFETY: see the split above.
            unsafe { core::mem::transmute::<[$x2; 2], $x4>([$val(lo), $val(hi)]) }
        }
    };
}
define_64bit_conversions!(f64x4, f64x2, f64_split, f64_join, f64_reg, f64_val, same_f64, same_f64);
define_64bit_conversions!(
    i64x4,
    i64x2,
    i64_split,
    i64_join,
    i64_reg,
    i64_val,
    vreinterpretq_f64_s64,
    vreinterpretq_s64_f64
);
define_64bit_conversions!(
    u64x4,
    u64x2,
    u64_split,
    u64_join,
    u64_reg,
    u64_val,
    vreinterpretq_f64_u64,
    vreinterpretq_u64_f64
);

macro_rules! define_32bit_conversions {
    (
        $x4:ty, $x2:ty, $bytes4:ident, $value4:ident, $bytes2:ident, $value2:ident,
        $to4:ident, $from4:ident, $to2:ident, $from2:ident
    ) => {
        #[inline(always)]
        fn $bytes4(v: $x4) -> uint8x16_t {
            // SAFETY: NEON is part of the aarch64 baseline; reinterpreting one 128-bit vector
            // register as another has no preconditions.
            unsafe { $to4(v.into()) }
        }
        #[inline(always)]
        fn $value4(v: uint8x16_t) -> $x4 {
            // SAFETY: see the conversion above.
            unsafe { $from4(v) }.into()
        }
        /// Widens a two-lane operand into the low half of a 128-bit register, zero-filling the
        /// upper half. A valid index never selects that padding.
        #[inline(always)]
        fn $bytes2(v: $x2) -> uint8x16_t {
            // SAFETY: see the conversion above; `vcombine_u8` only concatenates two 64-bit halves.
            unsafe { vcombine_u8($to2(v.0), vdup_n_u8(0)) }
        }
        #[inline(always)]
        fn $value2(v: uint8x16_t) -> $x2 {
            // SAFETY: see the conversion above.
            unsafe { $from2(vget_low_u8(v)) }.into()
        }
    };
}
define_32bit_conversions!(
    f32x4,
    f32x2,
    f32_bytes4,
    f32_value4,
    f32_bytes2,
    f32_value2,
    vreinterpretq_u8_f32,
    vreinterpretq_f32_u8,
    vreinterpret_u8_f32,
    vreinterpret_f32_u8
);
define_32bit_conversions!(
    i32x4,
    i32x2,
    i32_bytes4,
    i32_value4,
    i32_bytes2,
    i32_value2,
    vreinterpretq_u8_s32,
    vreinterpretq_s32_u8,
    vreinterpret_u8_s32,
    vreinterpret_s32_u8
);
define_32bit_conversions!(
    u32x4,
    u32x2,
    u32_bytes4,
    u32_value4,
    u32_bytes2,
    u32_value2,
    vreinterpretq_u8_u32,
    vreinterpretq_u32_u8,
    vreinterpret_u8_u32,
    vreinterpret_u32_u8
);

// Narrowing to, and widening from, the compact two-lane register. These are the only places a
// 64-bit NEON register is formed directly; everything else works at 128 bits.
macro_rules! define_32bit_halves {
    ($x4:ty, $x2:ty, $narrow:ident, $widen:ident, $concat:ident, $low:ident, $combine:ident, $dup:ident, $zero:expr) => {
        /// The low two lanes.
        #[inline(always)]
        fn $narrow(v: $x4) -> $x2 {
            // SAFETY: NEON is part of the aarch64 baseline, which this module is gated on.
            unsafe { $low(v.into()) }.into()
        }
        /// The four-lane value whose low two lanes are `v`, with the upper two zero-filled.
        #[inline(always)]
        fn $widen(v: $x2) -> $x4 {
            // SAFETY: see the narrowing above.
            unsafe { $combine(v.0, $dup($zero)) }.into()
        }
        /// `[a0, a1, b0, b1]`.
        #[inline(always)]
        fn $concat(a: $x2, b: $x2) -> $x4 {
            // SAFETY: see the narrowing above.
            unsafe { $combine(a.0, b.0) }.into()
        }
    };
}
define_32bit_halves!(
    f32x4,
    f32x2,
    f32_narrow,
    f32_widen,
    f32_concat,
    vget_low_f32,
    vcombine_f32,
    vdup_n_f32,
    0.
);
define_32bit_halves!(
    i32x4,
    i32x2,
    i32_narrow,
    i32_widen,
    i32_concat,
    vget_low_s32,
    vcombine_s32,
    vdup_n_s32,
    0
);
define_32bit_halves!(
    u32x4,
    u32x2,
    u32_narrow,
    u32_widen,
    u32_concat,
    vget_low_u32,
    vcombine_u32,
    vdup_n_u32,
    0
);

// ---------------------------------------------------------------------------
// The byte-table primitives the 32-bit path uses
// ---------------------------------------------------------------------------
//
// Every 32-bit lane is four bytes wide whether the caller reads it as `f32`, `i32`, or `u32`, so a
// lane index `I` always selects bytes `[I*4, I*4+4)` and one implementation serves all three.
// Passing a compile-time-constant index table lets codegen pick the same specialized instruction
// (`zip1`, `ext` + `trn1`, ...) it would for a raw shuffle rather than emitting a literal table
// lookup.

#[inline(always)]
const fn byte_table<const I0: usize, const I1: usize, const I2: usize, const I3: usize>() -> [u8; 16]
{
    let lanes = [I0, I1, I2, I3];
    let mut table = [0u8; 16];
    let mut lane = 0;
    while lane < 4 {
        let mut byte = 0;
        while byte < 4 {
            table[lane * 4 + byte] = (lanes[lane] * 4 + byte) as u8;
            byte += 1;
        }
        lane += 1;
    }
    table
}

#[inline(always)]
fn tbl1_16<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: uint8x16_t,
) -> uint8x16_t {
    let idx = byte_table::<I0, I1, I2, I3>();
    // SAFETY: `idx` is a fully initialized 16-byte array and NEON permits unaligned loads; NEON
    // itself is part of the aarch64 baseline this module is gated on.
    unsafe { vqtbl1q_u8(a, vld1q_u8(idx.as_ptr())) }
}

#[inline(always)]
fn tbl2_16<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
    a: uint8x16_t,
    b: uint8x16_t,
) -> uint8x16_t {
    let idx = byte_table::<I0, I1, I2, I3>();
    // SAFETY: see `tbl1_16`.
    unsafe { vqtbl2q_u8(uint8x16x2_t(a, b), vld1q_u8(idx.as_ptr())) }
}

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

    fn swizzle2<const I0: usize, const I1: usize, const PD: i32>(a: Self) -> Self::Vector2
    where float64x2_t: Shuffle<PD>;

    fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self) -> Self::Vector4
    where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>;
}

/// Index convention: `i` selects lane `i` of `a` when `i < 4` and lane `i - 4` of `b` otherwise,
/// with both operands seen as four lanes. A two-lane operand's upper two lanes are padding, and are
/// rejected rather than read.
#[rustfmt::skip]
pub(crate) trait SwizzleConcat: Swizzle {
    /// `[a0, a1, b0, b1]`.
    fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self;

    fn swizzle_concat2<const I0: usize, const I1: usize, const PD: i32>(a: Self, b: Self) -> Self::Vector2
    where float64x2_t: Shuffle<PD>;

    fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self, b: Self) -> Self::Vector4
    where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>;
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
            fn swizzle2<const I0: usize, const I1: usize, const PD: i32>(a: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                let h = $split(a);
                $val(<float64x2_t as Shuffle<PD>>::shuffle(h[I0 >> 1], h[I1 >> 1]))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                let h = $split(a);
                $join(
                    <float64x2_t as Shuffle<PD_LO>>::shuffle(h[I0 >> 1], h[I1 >> 1]),
                    <float64x2_t as Shuffle<PD_HI>>::shuffle(h[I2 >> 1], h[I3 >> 1]),
                )
            }
        }

        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self { $join($reg(a), $reg(b)) }

            #[inline(always)]
            fn swizzle_concat2<const I0: usize, const I1: usize, const PD: i32>(a: Self, b: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                let (a, b) = ($split(a), $split(b));
                let h = [a[0], a[1], b[0], b[1]];
                $val(<float64x2_t as Shuffle<PD>>::shuffle(h[I0 >> 1], h[I1 >> 1]))
            }

            #[inline(always)]
            fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self, b: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                let (a, b) = ($split(a), $split(b));
                let h = [a[0], a[1], b[0], b[1]];
                $join(
                    <float64x2_t as Shuffle<PD_LO>>::shuffle(h[I0 >> 1], h[I1 >> 1]),
                    <float64x2_t as Shuffle<PD_HI>>::shuffle(h[I2 >> 1], h[I3 >> 1]),
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
            fn __widen(a: Self) -> Self::Vector4 {
                // SAFETY: NEON is part of the aarch64 baseline, which this module is gated on.
                let zero = unsafe { vdupq_n_f64(0.) };
                $join($reg(a), zero)
            }

            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const PD: i32>(a: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                assert_lower_half!(I0, I1);
                let a = $reg(a);
                $val(<float64x2_t as Shuffle<PD>>::shuffle(a, a))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                assert_lower_half!(I0, I1, I2, I3);
                let a = $reg(a);
                $join(
                    <float64x2_t as Shuffle<PD_LO>>::shuffle(a, a),
                    <float64x2_t as Shuffle<PD_HI>>::shuffle(a, a),
                )
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
// `PD`/`PD_LO`/`PD_HI` are carried through the signatures and ignored: the byte table already
// expresses the whole pattern.

macro_rules! impl_swizzle_concat_32bit {
    ($self:ty, $bytes:ident, $value4:ident, $value2:ident, $narrow:ident, $concat:ident) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { $narrow(a) }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }

            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const PD: i32>(a: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                $value2(tbl1_16::<I0, I1, 0, 0>($bytes(a)))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                $value4(tbl1_16::<I0, I1, I2, I3>($bytes(a)))
            }
        }

        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self { $concat(a, b) }

            #[inline(always)]
            fn swizzle_concat2<const I0: usize, const I1: usize, const PD: i32>(a: Self, b: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                $value2(tbl2_16::<I0, I1, 0, 0>($bytes(a), $bytes(b)))
            }

            #[inline(always)]
            fn swizzle_concat4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self, b: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                $value4(tbl2_16::<I0, I1, I2, I3>($bytes(a), $bytes(b)))
            }
        }
    };
}
impl_swizzle_concat_32bit!(f32x4, f32_bytes4, f32_value4, f32_value2, f32_narrow, f32_concat);
impl_swizzle_concat_32bit!(i32x4, i32_bytes4, i32_value4, i32_value2, i32_narrow, i32_concat);
impl_swizzle_concat_32bit!(u32x4, u32_bytes4, u32_value4, u32_value2, u32_narrow, u32_concat);

macro_rules! impl_swizzle_32bit {
    ($self:ty, $bytes:ident, $value4:ident, $value2:ident, $widen:ident) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { a }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { $widen(a) }

            #[inline(always)]
            fn swizzle2<const I0: usize, const I1: usize, const PD: i32>(a: Self) -> Self::Vector2
            where float64x2_t: Shuffle<PD>
            {
                $value2(tbl1_16::<I0, I1, 0, 0>($bytes(a)))
            }

            #[inline(always)]
            fn swizzle4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD_LO: i32, const PD_HI: i32>(a: Self) -> Self::Vector4
            where float64x2_t: Shuffle<PD_LO> + Shuffle<PD_HI>
            {
                $value4(tbl1_16::<I0, I1, I2, I3>($bytes(a)))
            }
        }
    };
}
impl_swizzle_32bit!(f32x2, f32_bytes2, f32_value4, f32_value2, f32_widen);
impl_swizzle_32bit!(i32x2, i32_bytes2, i32_value4, i32_value2, i32_widen);
impl_swizzle_32bit!(u32x2, u32_bytes2, u32_value4, u32_value2, u32_widen);

// ---------------------------------------------------------------------------
// The macro
// ---------------------------------------------------------------------------

/// A partly specified index list is completed by repeating a lane, never by routing to a different
/// instruction: every pattern already costs one instruction per output half, so a special case has
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
        $crate::simd::swizzle_arm::Swizzle::swizzle2::<
            { $crate::simd::utils::validate_lane4!($i0) },
            { $crate::simd::utils::validate_lane4!($i1) },
            { $crate::simd::swizzle_arm::pd($i0, $i1) },
        >($a)
    };
    ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, [$i0, $i1, $i2, _])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_arm::Swizzle::swizzle4::<
            { $crate::simd::utils::validate_lane4!($i0) },
            { $crate::simd::utils::validate_lane4!($i1) },
            { $crate::simd::utils::validate_lane4!($i2) },
            { $crate::simd::utils::validate_lane4!($i3) },
            { $crate::simd::swizzle_arm::pd($i0, $i1) },
            { $crate::simd::swizzle_arm::pd($i2, $i3) },
        >($a)
    };

    ($a:expr, $b:expr, @concat) => {
        $crate::simd::swizzle_arm::SwizzleConcat::concat_4($a, $b)
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
        $crate::simd::swizzle_arm::SwizzleConcat::swizzle_concat2::<
            { $crate::simd::utils::validate_lane8!($i0) },
            { $crate::simd::utils::validate_lane8!($i1) },
            { $crate::simd::swizzle_arm::pd($i0, $i1) },
        >($a, $b)
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, $b, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, $b, [$i0, $i1, $i2, _])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_arm::swizzle4!($a, $b, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_arm::SwizzleConcat::swizzle_concat4::<
            { $crate::simd::utils::validate_lane8!($i0) },
            { $crate::simd::utils::validate_lane8!($i1) },
            { $crate::simd::utils::validate_lane8!($i2) },
            { $crate::simd::utils::validate_lane8!($i3) },
            { $crate::simd::swizzle_arm::pd($i0, $i1) },
            { $crate::simd::swizzle_arm::pd($i2, $i3) },
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

    macro_rules! define_2lane_helpers {
        ($ty:ty, $scalar:ty, $new:ident, $read:ident, $load:ident, $store:ident) => {
            fn $new(v: [$scalar; 2]) -> $ty {
                // SAFETY: `v` holds two initialized lanes; NEON permits unaligned loads.
                unsafe { $load(v.as_ptr()) }.into()
            }
            fn $read(v: $ty) -> [$scalar; 2] {
                let mut out = [0 as $scalar; 2];
                // SAFETY: `out` has room for two lanes; NEON permits unaligned stores.
                unsafe { $store(out.as_mut_ptr(), v.0) };
                out
            }
        };
    }
    define_2lane_helpers!(f32x2, f32, f32x2_new, f32x2_read, vld1_f32, vst1_f32);
    define_2lane_helpers!(i32x2, i32, i32x2_new, i32x2_read, vld1_s32, vst1_s32);
    define_2lane_helpers!(u32x2, u32, u32x2_new, u32x2_read, vld1_u32, vst1_u32);

    /// Four-lane operands. The expected values are the same at both lane widths, so the byte-table
    /// path and the `Shuffle` path are checked against one statement of what each pattern means.
    macro_rules! check_4lane {
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
                // Whole halves, which cost no instruction at all.
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
    check_4lane!(f64x4_lanes, f64, f64x4::new, f64x4::to_array, f64x2::to_array);
    check_4lane!(i64x4_lanes, i64, i64x4::new, i64x4::to_array, i64x2::to_array);
    check_4lane!(u64x4_lanes, u64, u64x4::new, u64x4::to_array, u64x2::to_array);
    check_4lane!(f32x4_lanes, f32, f32x4::new, f32x4::to_array, f32x2_read);
    check_4lane!(i32x4_lanes, i32, i32x4::new, i32x4::to_array, i32x2_read);
    check_4lane!(u32x4_lanes, u32, u32x4::new, u32x4::to_array, u32x2_read);

    /// Two-lane operands. Only lanes 0 and 1 of each operand exist, so every index here satisfies
    /// `i & 2 == 0`.
    macro_rules! check_2lane {
        ($name:ident, $s:ty, $new:expr, $read2:expr, $read4:expr) => {
            #[test]
            #[rustfmt::skip]
            fn $name() {
                let new = $new;
                let read2 = $read2;
                let read4 = $read4;
                let a = new([10 as $s, 11 as $s]);
                let b = new([20 as $s, 21 as $s]);

                assert_eq!(read2(swizzle4!(a, [1, 0])), [11 as $s, 10 as $s]);
                assert_eq!(read2(swizzle4!(a, [0, 0])), [10 as $s, 10 as $s]);
                assert_eq!(read4(swizzle4!(a, [1, 0, 0, 1])), [11 as $s, 10 as $s, 10 as $s, 11 as $s]);
            }
        };
    }
    check_2lane!(f64x2_lanes, f64, f64x2::new, f64x2::to_array, f64x4::to_array);
    check_2lane!(i64x2_lanes, i64, i64x2::new, i64x2::to_array, i64x4::to_array);
    check_2lane!(u64x2_lanes, u64, u64x2::new, u64x2::to_array, u64x4::to_array);
    check_2lane!(f32x2_lanes, f32, f32x2_new, f32x2_read, f32x4::to_array);
    check_2lane!(i32x2_lanes, i32, i32x2_new, i32x2_read, i32x4::to_array);
    check_2lane!(u32x2_lanes, u32, u32x2_new, u32x2_read, u32x4::to_array);
}
