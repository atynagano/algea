//! The x86-64 swizzle backend: one `swizzle4!` macro and one `Swizzle`/`SwizzleConcat` trait pair
//! covering both 32-bit and 64-bit lanes.
//!
//! Inputs are four-lane compute vectors; results are two or four lanes drawn from one or two
//! operands. `simd/utils.rs` re-exports the macro as `swizzle!` and the bounds as
//! `ComputeVector2`/`ComputeVector4`, which is how the rest of the crate reaches this module.
//!
//! # Why the constants come from the macro
//!
//! `_mm_shuffle_ps` and `_mm_shuffle_pd` take their control byte as a const-generic argument, and
//! stable Rust cannot pass a *computed* const-generic argument (that needs `generic_const_exprs`).
//! So every control byte has to be computed before crossing into const-generic land — which means
//! in the macro, where the indices are still literals. The two widths need entirely different
//! control bytes: one `shufps` byte describing four 32-bit lanes, versus two `shufpd` bytes, one
//! per 128-bit half. And a macro cannot branch on the type of its argument.
//!
//! The resolution is to have the macro compute **every** constant either width could want and pass
//! them all: the 32-bit implementations read `PS*`, the 64-bit ones read `H0..H3`/`PD*`, and each
//! ignores the rest. Redundant, but the trait is internal.
//!
//! # Why the instruction choice is a `match`, not more macro arms
//!
//! Which sequence a pattern needs is decided by `match` and `if` over the const-generic indices
//! inside the trait methods, rather than by one macro arm per pattern. The indices are literals
//! after monomorphization, so the branches and `select_half`'s indexing fold away and the emitted
//! code is the same either way; what differs is that the compiler no longer has to expand, name-
//! resolve and type-check one function per pattern, which dominated this crate's build time.
//!
//! Two consequences to keep in mind:
//!
//! * **A missing case is no longer a compile error by construction.** The `match` on
//!   `source_pattern` covers all eight values, and the arms of `shuffle_ps1`/`shuffle_ps2` are
//!   evaluated in const position, so a gap there still fails to compile. The index range and the
//!   requirement that a two-operand swizzle read both operands are asserted in free `const` items
//!   at the call site, which are evaluated during type checking and so fire under `cargo check`.
//! * **A pattern that stops reaching its cheapest sequence is invisible to the tests.** The tests
//!   below check lane order, never instruction counts; the patterns that reach one instruction are
//!   marked as such so the intent is at least written down.
//!
//! # The source pattern
//!
//! `source_pattern` encodes which operand lanes 1 through 3 read relative to lane 0, after lane 0
//! has been normalized to the first operand by swapping the operands when it reads the second.
//! That leaves eight cases, and the pattern selects both the instruction sequence and the pair of
//! `shufps` control bytes that reproduces the requested order from `(x, y)` in that fixed order.

#![cfg(all(target_arch = "x86_64", target_feature = "sse2"))]

use crate::simd::utils::ComputeVector;
use core::arch::x86_64::*;
use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};
// ---------------------------------------------------------------------------
// Constants the macro computes
// ---------------------------------------------------------------------------

/// `shufps` control byte: two bits per output lane.
pub(crate) const fn ps(l0: i32, l1: i32, l2: i32, l3: i32) -> i32 {
    l0 | l1 << 2 | l2 << 4 | l3 << 6
}

/// `shufpd` control byte for one 128-bit output half: bit 0 picks the lane taken from the first
/// operand, bit 1 the lane taken from the second. Only the low bit of each lane index matters,
/// because its high bit already chose which 128-bit half that operand contributes.
pub(crate) const fn pd(l0: usize, l1: usize) -> i32 { ((l0 & 1) | (l1 & 1) << 1) as i32 }

/// Encodes which operand lanes 1 through 3 use relative to lane 0.
///
/// Lane 0 is always normalized to the first operand inside `shuffle_concat_4`, leaving exactly
/// eight source patterns. Bit 2 describes lane 1, bit 1 lane 2, and bit 0 lane 3.
pub(crate) const fn source_pattern(i0: usize, i1: usize, i2: usize, i3: usize) -> usize {
    let first = i0 >> 2;
    ((i1 >> 2 ^ first) << 2) | ((i2 >> 2 ^ first) << 1) | (i3 >> 2 ^ first)
}

pub(crate) const fn source_pattern2(i0: usize, i1: usize) -> usize { (i0 >> 2) ^ (i1 >> 2) }

/// First `shufps` control byte for the normalized source pattern.
pub(crate) const fn shuffle_ps1(i0: usize, i1: usize, i2: usize, i3: usize) -> i32 {
    let l0 = (i0 & 3) as i32;
    let l1 = (i1 & 3) as i32;
    let l2 = (i2 & 3) as i32;
    let l3 = (i3 & 3) as i32;
    match source_pattern(i0, i1, i2, i3) {
        0 | 3 => ps(l0, l1, l2, l3),
        1 => ps(l2, 0, l3, 0),
        2 => ps(l3, 0, l2, 0),
        4 | 7 => ps(l0, 0, l1, 0),
        5 => ps(l0, l2, l1, l3),
        6 => ps(l0, l3, l1, l2),
        _ => unreachable!(),
    }
}

/// Second `shufps` control byte for the normalized source pattern.
pub(crate) const fn shuffle_ps2(i0: usize, i1: usize, i2: usize, i3: usize) -> i32 {
    let l0 = (i0 & 3) as i32;
    let l1 = (i1 & 3) as i32;
    let l2 = (i2 & 3) as i32;
    let l3 = (i3 & 3) as i32;
    match source_pattern(i0, i1, i2, i3) {
        0 | 3 => 0,
        1 => ps(l0, l1, 0, 2),
        2 => ps(l0, l1, 2, 0),
        4 | 7 => ps(0, 2, l2, l3),
        5 => ps(0, 2, 1, 3),
        6 => ps(0, 2, 3, 1),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

#[rustfmt::skip]
pub(crate) trait Swizzle: ComputeVector {
    fn __xy(a: Self) -> Self::Vector2;
    fn __widen(a: Self) -> Self::Vector4;
    fn shuffle_aa<const H0: usize, const H1: usize, const PD: i32, const PS: i32>(a: Self) -> Self::Vector2;
    fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self::Vector4;
}

#[rustfmt::skip]
pub(crate) trait SwizzleConcat: Swizzle {
    /// `[a0, a1, b0, b1]`.
    fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self;
    fn shuffle_concat_2<const I0: usize, const I1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2;
    fn shuffle_concat_4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD1: i32, const PD2: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
}

// ---------------------------------------------------------------------------
// 32-bit lanes: reads PS*, ignores the halves and the shufpd bytes
// ---------------------------------------------------------------------------

#[inline(always)]
fn then_lo<const PS1: i32, const PS2: i32>(a: __m128, b: __m128, other: __m128) -> __m128 {
    // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
    unsafe {
        let temp = _mm_shuffle_ps::<PS1>(a, b);
        _mm_shuffle_ps::<PS2>(temp, other)
    }
}
#[inline(always)]
fn then_hi<const PS1: i32, const PS2: i32>(a: __m128, b: __m128, other: __m128) -> __m128 {
    // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
    unsafe {
        let temp = _mm_shuffle_ps::<PS1>(a, b);
        _mm_shuffle_ps::<PS2>(other, temp)
    }
}
#[inline(always)]
fn then_self<const PS1: i32, const PS2: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
    unsafe {
        let temp = _mm_shuffle_ps::<PS1>(a, b);
        _mm_shuffle_ps::<PS2>(temp, temp)
    }
}

macro_rules! impl_swizzle_32bit {
    ($self:ty, $from:expr, $into:expr) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { a }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }
            #[inline(always)]
            fn shuffle_aa<const H0: usize, const H1: usize, const PD: i32, const PS: i32>(a: Self) -> Self::Vector2 {
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<PS>($from(a), $from(a)) })
            }
            #[inline(always)]
            fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self::Vector4 {
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<PS>($from(a), $from(a)) })
            }
        }

        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self {
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<{ ps(0, 1, 0, 1) }>($from(a), $from(b)) })
            }

            #[inline(always)]
            fn shuffle_concat_2<const I0: usize, const I1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2 {
                let a = $from(a);
                let b = $from(b);

                // The low half of an unpack is already the requested pair. `shufps` needs two
                // instructions for it, because its low two lanes both come from its first operand.
                match [I0, I1] {
                    [0, 4] => return $into(unsafe { _mm_unpacklo_ps(a, b) }),
                    [2, 6] => return $into(unsafe { _mm_unpackhi_ps(a, b) }),
                    [4, 0] => return $into(unsafe { _mm_unpacklo_ps(b, a) }),
                    [6, 2] => return $into(unsafe { _mm_unpackhi_ps(b, a) }),
                    _ => {}
                }

                let (x, y) = if I0 < 4 { (a, b) } else { (b, a) };
                $into(then_self::<PS1, PS2>(x, y))
            }

            #[inline(always)]
            fn shuffle_concat_4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD1: i32, const PD2: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4 {
                let a = $from(a);
                let b = $from(b);

                match[I0, I1, I2, I3] {
                    [0, 4, 1, 5] => return $into(unsafe { _mm_unpacklo_ps(a, b) }),
                    [2, 6, 3, 7] => return $into(unsafe { _mm_unpackhi_ps(a, b) }),
                    [4, 0, 5, 1] => return $into(unsafe { _mm_unpacklo_ps(b, a) }),
                    [6, 2, 7, 3] => return $into(unsafe { _mm_unpackhi_ps(b, a) }),
                    _ => {}
                }

                let (x, y) = if I0 < 4 { (a, b) } else { (b, a) };
                let result = match source_pattern(I0, I1, I2, I3) {
                    0 | 3 => {
                        // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                        unsafe { _mm_shuffle_ps::<PS1>(x, if source_pattern(I0, I1, I2, I3) == 0 { x } else { y }) }
                    }
                    1 | 2 => then_hi::<PS1, PS2>(x, y, x),
                    4 => then_lo::<PS1, PS2>(x, y, x),
                    7 => then_lo::<PS1, PS2>(x, y, y),
                    5 | 6 => then_self::<PS1, PS2>(x, y),
                    _ => unreachable!(),
                };
                $into(result)
            }
        }
    };
}

#[inline(always)]
fn from_i32x4(v: i32x4) -> __m128 { unsafe { _mm_castsi128_ps(v.into()) } }
#[inline(always)]
fn from_u32x4(v: u32x4) -> __m128 { unsafe { _mm_castsi128_ps(v.into()) } }
#[inline(always)]
fn into_i32x4(v: __m128) -> i32x4 { unsafe { _mm_castps_si128(v).into() } }
#[inline(always)]
fn into_u32x4(v: __m128) -> u32x4 { unsafe { _mm_castps_si128(v).into() } }
impl_swizzle_32bit!(f32x4, __m128::from, __m128::into);
impl_swizzle_32bit!(i32x4, from_i32x4, into_i32x4);
impl_swizzle_32bit!(u32x4, from_u32x4, into_u32x4);

// ---------------------------------------------------------------------------
// 64-bit lane storage
// ---------------------------------------------------------------------------

// `From<f64x4> for __m256d` is gated on `target_arch` only, and `transmute` needs no target
// feature, so splitting into halves needs no cfg. Rejoining does: `_mm256_set_m128d` keeps the
// result one 256-bit value while a transmute lets LLVM leave it as two 128-bit halves. The
// difference is invisible once the value feeds 256-bit arithmetic, but the intrinsic is never
// worse.
#[inline(always)]
fn halves(v: __m256d) -> [__m128d; 2] {
    // SAFETY: a 256-bit vector is two 128-bit halves, low half first.
    unsafe { core::mem::transmute::<__m256d, [__m128d; 2]>(v) }
}

#[cfg(target_feature = "avx")]
#[inline(always)]
fn join(lo: __m128d, hi: __m128d) -> __m256d {
    // SAFETY: guarded by `target_feature = "avx"`.
    unsafe { _mm256_set_m128d(hi, lo) }
}

#[cfg(not(target_feature = "avx"))]
#[inline(always)]
fn join(lo: __m128d, hi: __m128d) -> __m256d {
    // SAFETY: see `halves`.
    unsafe { core::mem::transmute::<[__m128d; 2], __m256d>([lo, hi]) }
}

// ---------------------------------------------------------------------------
// 64-bit lanes: reads the halves and the shufpd bytes, ignores PS*
// ---------------------------------------------------------------------------

macro_rules! assert_single_half {
    ($($h:ident),+) => {
        const {
            assert!(
                $($h == 0 &&)+ true,
                "a two-lane operand has no upper half; lane indices must be 0 or 1",
            )
        }
    };
}

macro_rules! impl_swizzle_64bit {
    ($self:ty, $from:expr, $into2:expr, $into4:expr) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { a }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 {
                cfg_select! {
                    target_feature = "avx" => unsafe {
                        $into4(_mm256_castpd128_pd256($from(a)))
                    },
                    _ => unsafe {
                        $into4(join($from(a), _mm_setzero_pd()))
                    }
                }
            }
            #[inline(always)]
            fn shuffle_aa<const H0: usize, const H1: usize, const PD: i32, const PS: i32>(a: Self) -> Self::Vector2 {
                assert_single_half!(H0, H1);
                let a = $from(a);
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_shuffle_pd::<PD>(a, a)) }
            }
            #[inline(always)]
            fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self::Vector4 {
                assert_single_half!(H0, H1, H2, H3);
                let a = $from(a);
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe {
                    $into4(join(
                        _mm_shuffle_pd::<PD_LO>(a, a),
                        _mm_shuffle_pd::<PD_HI>(a, a),
                    ))
                }
            }
        }
    };
}

#[inline(always)]
fn select_half(a: [__m128d; 2], b: [__m128d; 2], index: usize) -> __m128d {
    if index < 4 { a[index >> 1] } else { b[(index - 4) >> 1] }
}

macro_rules! impl_swizzle_concat_64bit {
    ($self:ty, $from2:expr, $from4:expr, $into2:expr, $into4:expr) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { $into2(halves($from4(a))[0]) }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }
            #[inline(always)]
            fn shuffle_aa<const H0: usize, const H1: usize, const PD: i32, const PS: i32>(a: Self) -> Self::Vector2 {
                let a = halves($from4(a));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_shuffle_pd::<PD>(a[H0], a[H1])) }
            }
            #[inline(always)]
            fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self::Vector4 {
                let a = halves($from4(a));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe {
                    $into4(join(
                        _mm_shuffle_pd::<PD_LO>(a[H0], a[H1]),
                        _mm_shuffle_pd::<PD_HI>(a[H2], a[H3]),
                    ))
                }
            }
        }

        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn concat_4(a: Self::Vector2, b: Self::Vector2) -> Self {
                $into4(join($from2(a), $from2(b)))
            }

            #[inline(always)]
            fn shuffle_concat_2<const I0: usize, const I1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2 {
                let a = halves($from4(a));
                let b = halves($from4(b));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_shuffle_pd::<PD>(select_half(a, b, I0), select_half(a, b, I1))) }
            }

            #[inline(always)]
            fn shuffle_concat_4<const I0: usize, const I1: usize, const I2: usize, const I3: usize, const PD1: i32, const PD2: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4 {
                let a = halves($from4(a));
                let b = halves($from4(b));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe {
                    $into4(join(
                        _mm_shuffle_pd::<PD1>(select_half(a, b, I0), select_half(a, b, I1)),
                        _mm_shuffle_pd::<PD2>(select_half(a, b, I2), select_half(a, b, I3)),
                    ))
                }
            }
        }
    };
}

#[inline(always)]
fn from_i64x4(v: i64x4) -> __m256d { unsafe { core::mem::transmute::<__m256i, __m256d>(v.into()) } }
#[inline(always)]
fn from_u64x4(v: u64x4) -> __m256d { unsafe { core::mem::transmute::<__m256i, __m256d>(v.into()) } }
#[inline(always)]
fn into_i64x4(v: __m256d) -> i64x4 { unsafe { core::mem::transmute::<__m256d, __m256i>(v).into() } }
#[inline(always)]
fn into_u64x4(v: __m256d) -> u64x4 { unsafe { core::mem::transmute::<__m256d, __m256i>(v).into() } }
#[inline(always)]
fn from_i64x2(v: i64x2) -> __m128d { unsafe { _mm_castsi128_pd(v.into()) } }
#[inline(always)]
fn from_u64x2(v: u64x2) -> __m128d { unsafe { _mm_castsi128_pd(v.into()) } }
#[inline(always)]
fn into_i64x2(v: __m128d) -> i64x2 { unsafe { _mm_castpd_si128(v).into() } }
#[inline(always)]
fn into_u64x2(v: __m128d) -> u64x2 { unsafe { _mm_castpd_si128(v).into() } }

impl_swizzle_concat_64bit!(f64x4, __m128d::from, __m256d::from, __m128d::into, __m256d::into);
impl_swizzle_concat_64bit!(i64x4, from_i64x2, from_i64x4, into_i64x2, into_i64x4);
impl_swizzle_concat_64bit!(u64x4, from_u64x2, from_u64x4, into_u64x2, into_u64x4);
impl_swizzle_64bit!(f64x2, __m128d::from, __m128d::into, __m256d::into);
impl_swizzle_64bit!(i64x2, from_i64x2, into_i64x2, into_i64x4);
impl_swizzle_64bit!(u64x2, from_u64x2, into_u64x2, into_u64x4);

/// `swizzle4!(a, b, [i0, i1, i2, i3])` where each index selects a lane of `a ++ b`.
///
/// One input is expressed by passing the same value twice.
macro_rules! swizzle4 {
    // Complete partial unpack requests with the corresponding full unpack pattern.
    ($a:expr, $b:expr, [0, 4, 1]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [0, 4, 1, 5])
    };
    ($a:expr, $b:expr, [2, 6, 3]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [2, 6, 3, 7])
    };
    ($a:expr, $b:expr, [4, 0, 5]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [4, 0, 5, 1])
    };
    ($a:expr, $b:expr, [6, 2, 7]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [6, 2, 7, 3])
    };
    ($a:expr, $b:expr, [0, 4, _, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [0, 4, 1, 5])
    };
    ($a:expr, $b:expr, [2, 6, _, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [2, 6, 3, 7])
    };
    ($a:expr, $b:expr, [4, 0, _, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [4, 0, 5, 1])
    };
    ($a:expr, $b:expr, [6, 2, _, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [6, 2, 7, 3])
    };
    ($a:expr, $b:expr, [0, 4, 1, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [0, 4, 1, 5])
    };
    ($a:expr, $b:expr, [2, 6, 3, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [2, 6, 3, 7])
    };
    ($a:expr, $b:expr, [4, 0, 5, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [4, 0, 5, 1])
    };
    ($a:expr, $b:expr, [6, 2, 7, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [6, 2, 7, 3])
    };

    // Pass-through: the requested lanes are already in place. `__xy` and `__widen` say so
    // directly, instead of leaving the identity to be recognized as one further down.
    // `[0, 1, 2]` reaches these through the three-index arm below.
    //
    // `[0, 1, 2, _]` and `[0, 1, 2, 3]` name lane 2, so the operand has four lanes and `__widen` is
    // the identity. With a two-lane operand `[0, 1, _, _]` is a real widening instead, which fills
    // the padding lanes with zeros; a two-lane operand that only needs its own lanes back should
    // ask for `[0, 1]`.
    ($a:expr, [0, 1]) => {
        $crate::simd::swizzle_x86::Swizzle::__xy($a)
    };
    ($a:expr, [0, 1, _, _]) => {
        $crate::simd::swizzle_x86::Swizzle::__widen($a)
    };
    ($a:expr, [0, 1, 2, _]) => {
        $crate::simd::swizzle_x86::Swizzle::__widen($a)
    };
    ($a:expr, [0, 1, 2, 3]) => {
        $crate::simd::swizzle_x86::Swizzle::__widen($a)
    };
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
        $crate::simd::swizzle_x86::Swizzle::shuffle_aa::<
            { $i0 >> 1 },
            { $i1 >> 1 },
            { $crate::simd::swizzle_x86::pd($i0, $i1) },
            { $crate::simd::swizzle_x86::ps($i0, $i1, $i0, $i1) },
        >($a)
    };
    ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::swizzle_x86::swizzle4!($a, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, [$i0, $i1, $i2, _])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::utils::complete_swizzle4!([$crate::simd::swizzle_x86::swizzle4], ($a), [
            $i0,
            $i1,
            $i2,
            _
        ])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_x86::Swizzle::shuffle_aaaa::<
            { $i0 >> 1 },
            { $i1 >> 1 },
            { $i2 >> 1 },
            { $i3 >> 1 },
            { $crate::simd::swizzle_x86::pd($i0, $i1) },
            { $crate::simd::swizzle_x86::pd($i2, $i3) },
            { $crate::simd::swizzle_x86::ps($i0, $i1, $i2, $i3) },
        >($a)
    };

    ($a:expr, $b:expr, @concat) => {
        $crate::simd::swizzle_x86::SwizzleConcat::concat_4($a, $b)
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
    ($a:expr, $b:expr, [$i0:tt, $i1:tt]) => {{
        const _: () = assert!(
            $i0 < 8 && $i1 < 8 && $crate::simd::swizzle_x86::source_pattern2($i0, $i1) != 0,
            "a two-input swizzle must use both operands and indices in 0..8",
        );
        $crate::simd::swizzle_x86::SwizzleConcat::shuffle_concat_2::<
            $i0,
            $i1,
            { $crate::simd::swizzle_x86::pd($i0, $i1) },
            { $crate::simd::swizzle_x86::ps($i0 & 3, 0, $i1 & 3, 0) },
            { $crate::simd::swizzle_x86::ps(0, 2, 0, 2) },
        >($a, $b)
    }};
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [$i0, $i1, $i2, _])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::utils::complete_swizzle4!([$crate::simd::swizzle_x86::swizzle4], ($a, $b), [
            $i0,
            $i1,
            $i2,
            _
        ])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
        const _: () = assert!(
            $i0 < 8
                && $i1 < 8
                && $i2 < 8
                && $i3 < 8
                && $crate::simd::swizzle_x86::source_pattern($i0, $i1, $i2, $i3) != 0,
            "a two-input swizzle must use both operands and indices in 0..8",
        );
        $crate::simd::swizzle_x86::SwizzleConcat::shuffle_concat_4::<
            $i0,
            $i1,
            $i2,
            $i3,
            { $crate::simd::swizzle_x86::pd($i0, $i1) },
            { $crate::simd::swizzle_x86::pd($i2, $i3) },
            { $crate::simd::swizzle_x86::shuffle_ps1($i0, $i1, $i2, $i3) },
            { $crate::simd::swizzle_x86::shuffle_ps2($i0, $i1, $i2, $i3) },
        >($a, $b)
    }};
}

pub(crate) use swizzle4;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check {
        ([$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            let b = f32x4::new([4., 5., 6., 7.]);
            assert_eq!(swizzle4!(a, b, [$i0, $i1, $i2, $i3]).to_array(), [
                $i0 as f32, $i1 as f32, $i2 as f32, $i3 as f32
            ],);

            let a = f64x4::new([0., 1., 2., 3.]);
            let b = f64x4::new([4., 5., 6., 7.]);
            assert_eq!(swizzle4!(a, b, [$i0, $i1, $i2, $i3]).to_array(), [
                $i0 as f64, $i1 as f64, $i2 as f64, $i3 as f64
            ],);
        }};
    }

    macro_rules! check_two {
        ([$i0:tt, $i1:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            let b = f32x4::new([4., 5., 6., 7.]);
            let produced = swizzle4!(a, b, [$i0, $i1]).to_array();
            assert_eq!([produced[0], produced[1]], [$i0 as f32, $i1 as f32]);

            let a = f64x4::new([0., 1., 2., 3.]);
            let b = f64x4::new([4., 5., 6., 7.]);
            assert_eq!(swizzle4!(a, b, [$i0, $i1]).to_array(), [$i0 as f64, $i1 as f64,]);
        }};
    }

    // One operand, four lanes out. `a` holds [0, 1, 2, 3] and the indices are its lane numbers.
    macro_rules! check_one {
        ([$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            assert_eq!(swizzle4!(a, [$i0, $i1, $i2, $i3]).to_array(), [
                $i0 as f32, $i1 as f32, $i2 as f32, $i3 as f32
            ],);

            let a = f64x4::new([0., 1., 2., 3.]);
            assert_eq!(swizzle4!(a, [$i0, $i1, $i2, $i3]).to_array(), [
                $i0 as f64, $i1 as f64, $i2 as f64, $i3 as f64
            ],);
        }};
    }

    // One operand, two lanes out. For 32-bit lanes `Vector2` is the four-lane type, so only the
    // first two lanes are meaningful there.
    macro_rules! check_one_two {
        ([$i0:tt, $i1:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            let produced = swizzle4!(a, [$i0, $i1]).to_array();
            assert_eq!([produced[0], produced[1]], [$i0 as f32, $i1 as f32]);

            let a = f64x4::new([0., 1., 2., 3.]);
            assert_eq!(swizzle4!(a, [$i0, $i1]).to_array(), [$i0 as f64, $i1 as f64]);
        }};
    }

    #[test]
    fn concat_source_patterns() {
        // One instruction each.
        check!([0, 1, 4, 5]);
        check!([4, 5, 0, 1]);
        check!([0, 4, 1, 5]);
        check!([2, 6, 3, 7]);
        check!([4, 0, 5, 1]);
        check!([6, 2, 7, 3]);

        // Two instructions each: one `shufps` per output pair.

        check!([0, 1, 2, 4]);
        check!([0, 1, 4, 2]);
        check!([0, 4, 1, 2]);
        check!([1, 4, 3, 6]);
        check!([0, 4, 5, 1]);
        check!([0, 4, 5, 6]);
        check!([4, 0, 1, 2]);
        check!([4, 0, 1, 5]);
        check!([5, 0, 7, 2]);
        check!([4, 0, 5, 6]);
        check!([4, 5, 0, 6]);
        check!([4, 5, 6, 0]);
        check!([2, 2, 6, 6]);
        check!([5, 5, 1, 1]);
    }

    #[test]
    fn concat_two_lane_patterns() {
        // The four that reach one instruction, then three that need two.
        check_two!([0, 4]);
        check_two!([2, 6]);
        check_two!([4, 0]);
        check_two!([6, 2]);
        check_two!([1, 6]);
        check_two!([5, 3]);
        check_two!([7, 0]);
    }

    /// The index lists that ask for lanes already in place. A `_` tail leaves the padding lanes to
    /// the operand, so only the named lanes are compared.
    #[test]
    fn pass_through_patterns() {
        check_one!([0, 1, 2, 3]);
        check_one_two!([0, 1]);

        let a = f32x4::new([0., 1., 2., 3.]);
        assert_eq!(swizzle4!(a, [0, 1, 2, _]).to_array()[..3], [0., 1., 2.]);
        assert_eq!(swizzle4!(a, [0, 1, 2]).to_array()[..3], [0., 1., 2.]);
        assert_eq!(swizzle4!(a, [0, 1, _, _]).to_array()[..2], [0., 1.]);

        let a = f64x4::new([0., 1., 2., 3.]);
        assert_eq!(swizzle4!(a, [0, 1, 2, _]).to_array()[..3], [0., 1., 2.]);
        assert_eq!(swizzle4!(a, [0, 1, 2]).to_array()[..3], [0., 1., 2.]);
        assert_eq!(swizzle4!(a, [0, 1, _, _]).to_array()[..2], [0., 1.]);
    }

    #[test]
    fn one_operand_patterns() {
        check_one!([3, 1, 0, 2]);
        check_one!([3, 2, 1, 0]);
        check_one!([0, 0, 0, 0]);
        check_one!([3, 3, 2, 2]);
        check_one!([1, 1, 3, 3]);
        check_one!([2, 2, 3, 3]);
        check_one!([0, 1, 1, 0]);
    }

    #[test]
    fn one_operand_two_lane_patterns() {
        check_one_two!([0, 1]);
        check_one_two!([1, 0]);
        check_one_two!([3, 1]);
        check_one_two!([2, 2]);
        check_one_two!([2, 3]);
    }

    #[test]
    fn duplicated_lanes() {
        check!([2, 2, 6, 6]);
        check!([5, 5, 1, 1]);
        check_one!([1, 1, 1, 1]);
        check_one!([2, 2, 2, 2]);
    }
}
