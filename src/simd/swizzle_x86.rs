//! The x86-64 swizzle backend: one `swizzle4!` macro and one `Swizzle`/`SwizzleConcat` trait pair
//! covering both 32-bit and 64-bit lanes.
//!
//! Inputs are four-lane compute vectors; results are two or four lanes drawn from one or two
//! operands. `simd/utils.rs` re-exports the macro as `swizzle!` and the bounds as
//! `ComputeVector2`/`ComputeVector4`, which is how the rest of the crate reaches this module.
//!
//! # The problem this shape solves
//!
//! `_mm_shuffle_ps` and `_mm_shuffle_pd` take their control byte as a const-generic argument, and
//! stable Rust cannot pass a *computed* const-generic argument (that needs `generic_const_exprs`).
//! So every control byte has to be computed before crossing into const-generic land — which means
//! in the macro, where the indices are still literals. The two widths need entirely different
//! control bytes: one `shufps` byte describing four 32-bit lanes, versus two `shufpd` bytes, one
//! per 128-bit half. And a macro cannot branch on the type of its argument.
//!
//! The resolution is to have the macro compute **every** constant either width could want and pass
//! them all: `f32x4` reads `PS*`, `f64x4` reads `H0..H3`/`PD_LO`/`PD_HI`, and each ignores the
//! rest. Redundant, but the redundancy is checked (see below) and the trait is internal.
//!
//! # The method name carries the source pattern
//!
//! Each `shuffle_*` method is named for which operand every output lane reads: `shuffle_aabb`
//! produces `[a[.], a[.], b[.], b[.]]`. Only the eight names beginning with `a` exist; a pattern
//! that begins with `b` is served by the same method with the operands handed over the other way
//! round, so `[B,B,A,A]` becomes `shuffle_aabb(b, a)`.
//!
//! Two properties make that work:
//!
//! * **`H0..H3` are operand-local.** Each names the 128-bit half *within the operand that lane
//!   reads*, so swapping which value is passed first leaves them unchanged.
//! * **The first step of every two-step method is `shufps(a, b, PS1)`.** Some patterns would rather
//!   build the intermediate from `(b, a)`; choosing different control bytes reproduces the same
//!   result from `(a, b)`, because the intermediate's lanes merely land elsewhere and `PS2` reads
//!   them from the other positions. Keeping the order fixed means the method name is the *only*
//!   statement of how the operands are used, so there is no second statement to disagree with.
//!
//! # Checking
//!
//! `verify_*` below symbolically executes the emitted instruction sequence over lane indices and
//! compares against the requested pattern, which the macro states independently as the original
//! index list. The macro asserts this in a free `const` item at the call site, which is evaluated
//! during type checking — so it fires under `cargo check`, at every optimization level. (An inline
//! `const {}` inside the trait method would not: MIR inlining drops it at `opt-level >= 2`.)
//!
//! Each `swizzle4_call!` arm names one method and emits both its call and its checks, so an emit
//! arm cannot state the method one way and its constants another.

#![cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
// The trait and the macro cover every index pattern; the crate's current call sites reach only
// some of them, and the helpers behind the unreached ones are used solely by this file's tests.
#![allow(dead_code, unused_macros)]

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

/// Restates an index from `x ++ y` to `y ++ x`.
///
/// Only the assertions need this. The trait call does not: `H0..H3` are operand-local, so handing
/// the operands over in the other order leaves them alone. But the requested pattern has to be
/// stated in the frame the method actually sees, and stating it from the original index list is
/// what keeps the assertion independent of the emit arm's own decoding.
pub(crate) const fn swap(index: usize) -> usize { index ^ 4 }

// ---------------------------------------------------------------------------
// Symbolic execution used by the call-site assertions
// ---------------------------------------------------------------------------

/// Lanes of the first operand a method receives, and of the second.
pub(crate) const FIRST: [usize; 4] = [0, 1, 2, 3];
pub(crate) const SECOND: [usize; 4] = [4, 5, 6, 7];

const fn sim_shuffle(x: [usize; 4], y: [usize; 4], control: i32) -> [usize; 4] {
    [
        x[(control & 3) as usize],
        x[(control >> 2 & 3) as usize],
        y[(control >> 4 & 3) as usize],
        y[(control >> 6 & 3) as usize],
    ]
}

pub(crate) const fn same(produced: [usize; 4], wanted: [usize; 4]) -> bool {
    produced[0] == wanted[0]
        && produced[1] == wanted[1]
        && produced[2] == wanted[2]
        && produced[3] == wanted[3]
}

/// `shufps(a, a, PS)`.
pub(crate) const fn verify_aaaa(control: i32, wanted: [usize; 4]) -> bool {
    same(sim_shuffle(FIRST, FIRST, control), wanted)
}

/// `shufps(a, b, PS)`.
pub(crate) const fn verify_aabb(control: i32, wanted: [usize; 4]) -> bool {
    same(sim_shuffle(FIRST, SECOND, control), wanted)
}

/// `t = shufps(a, b, PS1)`, then `shufps(other, t, PS2)`.
pub(crate) const fn verify_then_hi(
    control1: i32,
    control2: i32,
    other: [usize; 4],
    wanted: [usize; 4],
) -> bool {
    let temp = sim_shuffle(FIRST, SECOND, control1);
    same(sim_shuffle(other, temp, control2), wanted)
}

/// `t = shufps(a, b, PS1)`, then `shufps(t, other, PS2)`.
pub(crate) const fn verify_then_lo(
    control1: i32,
    control2: i32,
    other: [usize; 4],
    wanted: [usize; 4],
) -> bool {
    let temp = sim_shuffle(FIRST, SECOND, control1);
    same(sim_shuffle(temp, other, control2), wanted)
}

/// `t = shufps(a, b, PS1)`, then `shufps(t, t, PS2)`.
pub(crate) const fn verify_then_self(control1: i32, control2: i32, wanted: [usize; 4]) -> bool {
    let temp = sim_shuffle(FIRST, SECOND, control1);
    same(sim_shuffle(temp, temp, control2), wanted)
}

/// The 64-bit encoding, for every method at once. `sources` says, per output lane, whether it reads
/// the method's first operand (`0`) or its second (`1`) — that is the method's name spelled out.
///
/// What this catches is a mismatch between the chosen method name and the requested pattern:
/// `halves`/`pd_lo`/`pd_hi` are derived from the same lane values as `wanted`, but `sources` comes
/// from the emit arm's choice of method.
pub(crate) const fn verify_pd(
    sources: [usize; 4],
    halves: [usize; 4],
    pd_lo: i32,
    pd_hi: i32,
    wanted: [usize; 4],
) -> bool {
    const fn lane(source: usize, half: usize, control: i32, position: u32) -> usize {
        source * 4 + half * 2 + (control >> position & 1) as usize
    }
    lane(sources[0], halves[0], pd_lo, 0) == wanted[0]
        && lane(sources[1], halves[1], pd_lo, 1) == wanted[1]
        && lane(sources[2], halves[2], pd_hi, 0) == wanted[2]
        && lane(sources[3], halves[3], pd_hi, 1) == wanted[3]
}

// The two-lane forms only ever produce the low half, so the checks below compare the first two
// lanes of the same symbolic execution.

/// `shufps(a, a, PS)`, low half.
pub(crate) const fn verify_aa(control: i32, wanted: [usize; 2]) -> bool {
    let produced = sim_shuffle(FIRST, FIRST, control);
    produced[0] == wanted[0] && produced[1] == wanted[1]
}

/// `t = shufps(a, b, PS1)`, then `shufps(t, t, PS2)`, low half.
///
/// One `shufps` cannot do this: it always takes both of its low output lanes from the same operand,
/// so a lane from `a` next to a lane from `b` needs the intermediate.
pub(crate) const fn verify_ab(control1: i32, control2: i32, wanted: [usize; 2]) -> bool {
    let temp = sim_shuffle(FIRST, SECOND, control1);
    let produced = sim_shuffle(temp, temp, control2);
    produced[0] == wanted[0] && produced[1] == wanted[1]
}

/// The 64-bit encoding for the two-lane forms; see [`verify_pd`].
pub(crate) const fn verify_pd2(
    sources: [usize; 2],
    halves: [usize; 2],
    control: i32,
    wanted: [usize; 2],
) -> bool {
    const fn lane(source: usize, half: usize, control: i32, position: u32) -> usize {
        source * 4 + half * 2 + (control >> position & 1) as usize
    }
    lane(sources[0], halves[0], control, 0) == wanted[0]
        && lane(sources[1], halves[1], control, 1) == wanted[1]
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

/// `H0..H3` name, per output lane, the 128-bit half of that lane's own operand. `PD_LO`/`PD_HI` are
/// the `shufpd` control bytes for the low and high output halves. `PS`/`PS1`/`PS2` are the `shufps`
/// control bytes for the one or two steps the 32-bit form takes. Each implementation reads only the
/// encoding it can use.
#[rustfmt::skip]
pub(crate) trait SwizzleConcat: Swizzle {
    /// `[a0, b0]`.
    fn unpack_lo_2(a: Self, b: Self) -> Self::Vector2;
    /// `[a2, b2]`.
    fn unpack_hi_2(a: Self, b: Self) -> Self::Vector2;
    /// `[a0, b0, a1, b1]`.
    fn unpack_lo_4(a: Self, b: Self) -> Self::Vector4;
    /// `[a2, b2, a3, b3]`.
    fn unpack_hi_4(a: Self, b: Self) -> Self::Vector4;

    fn shuffle_ab<const H0: usize, const H1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2;
    fn shuffle_aabb<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_aaab<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_aaba<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_abaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_abbb<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_abab<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
    fn shuffle_abba<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector4;
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

/// At this width `Vector2` and `Vector4` are both `Self`, so one conversion covers both; the
/// 64-bit macros need a separate `$into2`/`$into4`.
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
                // The two meaningful lanes are the low half; `PS` also fills the upper half, which
                // this width treats as padding.
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<PS>($from(a), $from(a)) })
            }
            #[inline(always)]
            fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self {
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<PS>($from(a), $from(a)) })
            }
        }
        #[rustfmt::skip]
        impl SwizzleConcat for $self {
            #[inline(always)]
            fn unpack_lo_2(a: Self, b: Self) -> Self {
                // SAFETY: `_mm_unpacklo_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_unpacklo_ps($from(a), $from(b)) })
            }
            #[inline(always)]
            fn unpack_hi_2(a: Self, b: Self) -> Self {
                // SAFETY: `_mm_unpackhi_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_unpackhi_ps($from(a), $from(b)) })
            }
            #[inline(always)]
            fn unpack_lo_4(a: Self, b: Self) -> Self {
                // SAFETY: `_mm_unpacklo_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_unpacklo_ps($from(a), $from(b)) })
            }
            #[inline(always)]
            fn unpack_hi_4(a: Self, b: Self) -> Self {
                // SAFETY: `_mm_unpackhi_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_unpackhi_ps($from(a), $from(b)) })
            }

            #[inline(always)]
            fn shuffle_ab<const H0: usize, const H1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2 {
                // Two steps, unlike `shuffle_aa`: one `shufps` takes both of its low output lanes
                // from the same operand, so a lane from `a` next to a lane from `b` has to go
                // through the intermediate.
                $into(then_self::<PS1, PS2>($from(a), $from(b)))
            }
            #[inline(always)]
            fn shuffle_aabb<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self, b: Self) -> Self {
                // SAFETY: `_mm_shuffle_ps` is SSE, implied by this module's `sse2` gate.
                $into(unsafe { _mm_shuffle_ps::<PS>($from(a), $from(b)) })
            }
            // `aaab` and `aaba` share a body on purpose: both take their first two output lanes from `a`
            // and put the intermediate in the second position. Only the control bytes differ, and those
            // come from the macro.
            #[inline(always)]
            fn shuffle_aaab<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_hi::<PS1, PS2>($from(a), $from(b), $from(a)))
            }
            #[inline(always)]
            fn shuffle_aaba<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_hi::<PS1, PS2>($from(a), $from(b), $from(a)))
            }
            #[inline(always)]
            fn shuffle_abaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_lo::<PS1, PS2>($from(a), $from(b), $from(a)))
            }
            #[inline(always)]
            fn shuffle_abbb<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_lo::<PS1, PS2>($from(a), $from(b), $from(b)))
            }
            #[inline(always)]
            fn shuffle_abab<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_self::<PS1, PS2>($from(a), $from(b)))
            }
            #[inline(always)]
            fn shuffle_abba<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self {
                $into(then_self::<PS1, PS2>($from(a), $from(b)))
            }
        }
    }
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
// feature, so splitting into halves needs no cfg. Rejoining does:
// `dev/swizzle-64bit-split-join-codegen.md` measured that `_mm256_set_m128d` keeps the result one
// 256-bit value while the transmute lets LLVM leave it as two 128-bit halves. The difference is
// invisible once the value feeds 256-bit arithmetic, but the intrinsic is never worse.
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

/// Two `shufpd`, whatever the pattern: `shufpd` chooses its two source registers independently, so
/// any output half is one instruction and there is no case analysis to do. `$v0..$v3` name the
/// operand each output lane reads, which is exactly the method's name.
/// A physically two-lane operand has one 128-bit half, so every `H` has to name it. Without this
/// the halves would simply be ignored and a padding index would quietly return lanes 0 and 1.
///
/// Unlike the `verify_*` assertions, this one cannot fire under `cargo check`: it depends on the
/// operand's type, so it has to live inside a monomorphized item, and MIR inlining drops those at
/// `opt-level >= 2`. It fires under `cargo test` and `cargo build`, which is where a wrong index
/// list in kernel code would be exercised.
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

/// Physically two-lane types: they can be a swizzle source but never a concat operand, so this
/// implements [`Swizzle`] only. See [`impl_swizzle_concat_64bit!`] for the four-lane types.
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

    }
}
macro_rules! impl_shuffle_64bit {
    (
        $f:ident: ($a:ident, $b:ident) => [$v0:ident, $v1:ident, $v2:ident, $v3:ident],
        $from:expr, $into4:expr
    ) => {
        #[rustfmt::skip]
        #[inline(always)]
        fn $f<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS1: i32, const PS2: i32>($a: Self, $b: Self) -> Self {
            let $a = halves($from($a));
            let $b = halves($from($b));
            // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
            unsafe {
                $into4(join(
                    _mm_shuffle_pd::<PD_LO>($v0[H0], $v1[H1]),
                    _mm_shuffle_pd::<PD_HI>($v2[H2], $v3[H3]),
                ))
            }
        }
    };
}
/// `$into2`/`$into4` name the conversion back to the two-lane and four-lane result types. Both
/// 64-bit macros take them in this order so the same name always means the same width.
macro_rules! impl_swizzle_concat_64bit {
    ($self:ty, $from:expr, $into2:expr, $into4:expr) => {
        #[rustfmt::skip]
        impl Swizzle for $self {
            #[inline(always)]
            fn __xy(a: Self) -> Self::Vector2 { $into2(halves($from(a))[0]) }
            #[inline(always)]
            fn __widen(a: Self) -> Self::Vector4 { a }
            #[inline(always)]
            fn shuffle_aa<const H0: usize, const H1: usize, const PD: i32, const PS: i32>(a: Self) -> Self::Vector2 {
                let a = halves($from(a));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_shuffle_pd::<PD>(a[H0], a[H1])) }
            }
            #[inline(always)]
            fn shuffle_aaaa<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self) -> Self {
                let a = halves($from(a));
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
            fn unpack_lo_2(a: Self, b: Self) -> Self::Vector2 {
                let a_lo = halves($from(a))[0];
                let b_lo = halves($from(b))[0];
                // SAFETY: `_mm_unpack*_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_unpacklo_pd(a_lo, b_lo)) }
            }
            #[inline(always)]
            fn unpack_hi_2(a: Self, b: Self) -> Self::Vector2 {
                let a_hi = halves($from(a))[1];
                let b_hi = halves($from(b))[1];
                // The wanted lanes are `a[2]` and `b[2]`, which are lane 0 of each *high* half, so
                // this is `unpacklo` applied to the high halves — not `unpackhi`.
                // SAFETY: see `unpack_lo_2`.
                unsafe { $into2(_mm_unpacklo_pd(a_hi, b_hi)) }
            }
            #[inline(always)]
            fn unpack_lo_4(a: Self, b: Self) -> Self {
                let a_lo = halves($from(a))[0];
                let b_lo = halves($from(b))[0];
                // SAFETY: `_mm_unpack*_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into4(join(_mm_unpacklo_pd(a_lo, b_lo), _mm_unpackhi_pd(a_lo, b_lo))) }
            }
            #[inline(always)]
            fn unpack_hi_4(a: Self, b: Self) -> Self {
                let a_hi = halves($from(a))[1];
                let b_hi = halves($from(b))[1];
                // SAFETY: see `unpack_lo`.
                unsafe { $into4(join(_mm_unpacklo_pd(a_hi, b_hi), _mm_unpackhi_pd(a_hi, b_hi))) }
            }

            #[inline(always)]
            fn shuffle_ab<const H0: usize, const H1: usize, const PD: i32, const PS1: i32, const PS2: i32>(a: Self, b: Self) -> Self::Vector2 {
                let a = halves($from(a));
                let b = halves($from(b));
                // SAFETY: `_mm_shuffle_pd` is SSE2, guaranteed by this module's gate.
                unsafe { $into2(_mm_shuffle_pd::<PD>(a[H0], b[H1])) }
            }
            #[inline(always)]
            fn shuffle_aabb<const H0: usize, const H1: usize, const H2: usize, const H3: usize, const PD_LO: i32, const PD_HI: i32, const PS: i32>(a: Self, b: Self) -> Self {
                let a = halves($from(a));
                let b = halves($from(b));
                // SAFETY: see `shuffle_aaaa`.
                unsafe {
                    $into4(join(
                        _mm_shuffle_pd::<PD_LO>(a[H0], a[H1]),
                        _mm_shuffle_pd::<PD_HI>(b[H2], b[H3]),
                    ))
                }
            }
            impl_shuffle_64bit!(shuffle_aaab: (a, b) => [a, a, a, b], $from, $into4);
            impl_shuffle_64bit!(shuffle_aaba: (a, b) => [a, a, b, a], $from, $into4);
            impl_shuffle_64bit!(shuffle_abaa: (a, b) => [a, b, a, a], $from, $into4);
            impl_shuffle_64bit!(shuffle_abbb: (a, b) => [a, b, b, b], $from, $into4);
            impl_shuffle_64bit!(shuffle_abab: (a, b) => [a, b, a, b], $from, $into4);
            impl_shuffle_64bit!(shuffle_abba: (a, b) => [a, b, b, a], $from, $into4);
        }
    }
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

impl_swizzle_concat_64bit!(f64x4, __m256d::from, __m128d::into, __m256d::into);
impl_swizzle_concat_64bit!(i64x4, from_i64x4, into_i64x2, into_i64x4);
impl_swizzle_concat_64bit!(u64x4, from_u64x4, into_u64x2, into_u64x4);
impl_swizzle_64bit!(f64x2, __m128d::from, __m128d::into, __m256d::into);
impl_swizzle_64bit!(i64x2, from_i64x2, into_i64x2, into_i64x4);
impl_swizzle_64bit!(u64x2, from_u64x2, into_u64x2, into_u64x4);

// ---------------------------------------------------------------------------
// The macro
// ---------------------------------------------------------------------------

/// One arm per trait method. Each emits the call and both checks, so the method name, the source
/// pattern the checks assume, and the control bytes cannot drift apart.
///
/// `$w0..$w3` are the requested lanes stated in the frame the method sees (the original indices,
/// restated with [`swap`] when the emit arm hands the operands over the other way round).
/// `$l0..$l3` are the lane indices inside each lane's own operand.
#[rustfmt::skip]
macro_rules! swizzle4_call {
    // The unpack forms take no control bytes: the method name is the whole pattern, so the check
    // is just that the request matches what that method does.
    (@unpack_lo_2 $x:expr, $y:expr; [$w0:expr, $w1:expr]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            $w0 == FIRST[0] && $w1 == SECOND[0],
            "unpack_lo_2 does not produce the requested lane pattern",
        );
        SwizzleConcat::unpack_lo_2($x, $y)
    }};
    (@unpack_hi_2 $x:expr, $y:expr; [$w0:expr, $w1:expr]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            $w0 == FIRST[2] && $w1 == SECOND[2],
            "unpack_hi_2 does not produce the requested lane pattern",
        );
        SwizzleConcat::unpack_hi_2($x, $y)
    }};
    (@unpack_lo_4 $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            same([$w0, $w1, $w2, $w3], [FIRST[0], SECOND[0], FIRST[1], SECOND[1]]),
            "unpack_lo_4 does not produce the requested lane pattern",
        );
        SwizzleConcat::unpack_lo_4($x, $y)
    }};
    (@unpack_hi_4 $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            same([$w0, $w1, $w2, $w3], [FIRST[2], SECOND[2], FIRST[3], SECOND[3]]),
            "unpack_hi_4 does not produce the requested lane pattern",
        );
        SwizzleConcat::unpack_hi_4($x, $y)
    }};

    // Two-lane output. `$w0`/`$w1` and `$l0`/`$l1` carry the same meaning as in the four-lane arms.
    (@aa $x:expr; [$w0:expr, $w1:expr]; [$l0:tt, $l1:tt]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_aa(ps($l0, $l1, $l0, $l1), [$w0, $w1]),
            "emitted shufps does not reproduce the requested lane pattern",
        );
        const _: () = assert!(
            verify_pd2([0, 0], [$l0 >> 1, $l1 >> 1], pd($l0, $l1), [$w0, $w1]),
            "emitted shufpd control byte does not reproduce the requested lane pattern",
        );
        Swizzle::shuffle_aa::<
            { $l0 >> 1 }, { $l1 >> 1 },
            { pd($l0, $l1) },
            { ps($l0, $l1, $l0, $l1) },
        >($x)
    }};
    (@ab $x:expr, $y:expr; [$w0:expr, $w1:expr]; [$l0:tt, $l1:tt]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_ab(ps($l0, 0, $l1, 0), ps(0, 2, 0, 2), [$w0, $w1]),
            "emitted shufps pair does not reproduce the requested lane pattern",
        );
        const _: () = assert!(
            verify_pd2([0, 1], [$l0 >> 1, $l1 >> 1], pd($l0, $l1), [$w0, $w1]),
            "emitted shufpd control byte does not reproduce the requested lane pattern",
        );
        SwizzleConcat::shuffle_ab::<
            { $l0 >> 1 }, { $l1 >> 1 },
            { pd($l0, $l1) },
            { ps($l0, 0, $l1, 0) }, { ps(0, 2, 0, 2) },
        >($x, $y)
    }};

    (@aaaa $x:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_aaaa(ps($l0, $l1, $l2, $l3), [$w0, $w1, $w2, $w3]),
            "emitted shufps does not reproduce the requested lane pattern",
        );
        swizzle4_call!(@check_pd [0, 0, 0, 0]; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3]);
        Swizzle::shuffle_aaaa::<
            { $l0 >> 1 }, { $l1 >> 1 }, { $l2 >> 1 }, { $l3 >> 1 },
            { pd($l0, $l1) }, { pd($l2, $l3) },
            { ps($l0, $l1, $l2, $l3) },
        >($x)
    }};

    (@aabb $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_aabb(ps($l0, $l1, $l2, $l3), [$w0, $w1, $w2, $w3]),
            "emitted shufps does not reproduce the requested lane pattern",
        );
        swizzle4_call!(@check_pd [0, 0, 1, 1]; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3]);
        SwizzleConcat::shuffle_aabb::<
            { $l0 >> 1 }, { $l1 >> 1 }, { $l2 >> 1 }, { $l3 >> 1 },
            { pd($l0, $l1) }, { pd($l2, $l3) },
            { ps($l0, $l1, $l2, $l3) },
        >($x, $y)
    }};

    // The two-step arms differ only in the method, the verifier, which operand the second step
    // reuses, the source pattern, and the two control bytes; `@emit2` holds everything else.
    (@aaab $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2 shuffle_aaab, verify_then_hi, FIRST, [0, 0, 0, 1];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l2, 0, $l3, 0); ps($l0, $l1, 0, 2))
    };
    (@aaba $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2 shuffle_aaba, verify_then_hi, FIRST, [0, 0, 1, 0];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l3, 0, $l2, 0); ps($l0, $l1, 2, 0))
    };
    (@abaa $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2 shuffle_abaa, verify_then_lo, FIRST, [0, 1, 0, 0];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l0, 0, $l1, 0); ps(0, 2, $l2, $l3))
    };
    (@abbb $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2 shuffle_abbb, verify_then_lo, SECOND, [0, 1, 1, 1];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l0, 0, $l1, 0); ps(0, 2, $l2, $l3))
    };
    (@abab $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2_self shuffle_abab, [0, 1, 0, 1];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l0, $l2, $l1, $l3); ps(0, 2, 1, 3))
    };
    (@abba $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@emit2_self shuffle_abba, [0, 1, 1, 0];
            $x, $y; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3];
            ps($l0, $l3, $l1, $l2); ps(0, 2, 3, 1))
    };

    (
        @emit2 $method:ident, $verify:ident, $other:expr, $sources:expr;
        $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr];
        [$l0:tt, $l1:tt, $l2:tt, $l3:tt]; $ps1:expr; $ps2:expr
    ) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            $verify($ps1, $ps2, $other, [$w0, $w1, $w2, $w3]),
            "emitted shufps pair does not reproduce the requested lane pattern",
        );
        swizzle4_call!(@check_pd $sources; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3]);
        SwizzleConcat::$method::<
            { $l0 >> 1 }, { $l1 >> 1 }, { $l2 >> 1 }, { $l3 >> 1 },
            { pd($l0, $l1) }, { pd($l2, $l3) },
            { $ps1 }, { $ps2 },
        >($x, $y)
    }};

    // `verify_then_self` reuses the intermediate for both operands, so it takes no `other`.
    (
        @emit2_self $method:ident, $sources:expr;
        $x:expr, $y:expr; [$w0:expr, $w1:expr, $w2:expr, $w3:expr];
        [$l0:tt, $l1:tt, $l2:tt, $l3:tt]; $ps1:expr; $ps2:expr
    ) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_then_self($ps1, $ps2, [$w0, $w1, $w2, $w3]),
            "emitted shufps pair does not reproduce the requested lane pattern",
        );
        swizzle4_call!(@check_pd $sources; [$w0, $w1, $w2, $w3]; [$l0, $l1, $l2, $l3]);
        SwizzleConcat::$method::<
            { $l0 >> 1 }, { $l1 >> 1 }, { $l2 >> 1 }, { $l3 >> 1 },
            { pd($l0, $l1) }, { pd($l2, $l3) },
            { $ps1 }, { $ps2 },
        >($x, $y)
    }};

    (
        @check_pd $sources:expr;
        [$w0:expr, $w1:expr, $w2:expr, $w3:expr]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]
    ) => {{
        use $crate::simd::swizzle_x86::*;
        const _: () = assert!(
            verify_pd(
                $sources,
                [$l0 >> 1, $l1 >> 1, $l2 >> 1, $l3 >> 1],
                pd($l0, $l1),
                pd($l2, $l3),
                [$w0, $w1, $w2, $w3],
            ),
            "emitted shufpd control bytes do not reproduce the requested lane pattern",
        );
    }};
}

/// Picks the trait method for one source pattern. `$s0..$s3` say which operand each output lane
/// comes from, `$l0..$l3` are the lane indices inside that operand, and `$i0..$i3` are the original
/// `a ++ b` indices.
///
/// Only the eight `a`-first names exist, so a `b`-first pattern hands the operands over the other
/// way round. `$l0..$l3` are unaffected by that (they are operand-local); the requested pattern
/// passed to the assertions is restated with [`swap`].
#[rustfmt::skip]
macro_rules! swizzle4_emit {
    // --- one `unpackXps` ---
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, A, B]; [0, 0, 1, 1]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_lo_4 $a, $b; [$i0, $i1, $i2, $i3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, A, B]; [2, 2, 3, 3]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_hi_4 $a, $b; [$i0, $i1, $i2, $i3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, B, A]; [0, 0, 1, 1]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_lo_4 $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, B, A]; [2, 2, 3, 3]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_hi_4 $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)])
    };

    // --- one `shufps` ---
    // A two-input request that only reads one operand is a mistake, not a shorthand: writing it
    // with two operands says the other one contributes, and silently ignoring it would hide a
    // wrong index list. The one-input form of `swizzle4!` is the way to say this.
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, A, A, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        compile_error!("every lane reads the first operand; drop the second and use `swizzle4!(a, [..])`")
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, B, B, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        compile_error!("every lane reads the second operand; drop the first and use `swizzle4!(b, [..])`")
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, A, B, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aabb $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, B, A, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aabb $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };

    // --- two `shufps` ---
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, A, A, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aaab $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, B, B, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aaab $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, A, B, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aaba $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, B, A, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aaba $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, A, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abaa $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, B, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abaa $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, B, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abbb $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, A, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abbb $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, A, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abab $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, B, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abab $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [A, B, B, A]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abba $a, $b; [$i0, $i1, $i2, $i3]; [$l0, $l1, $l2, $l3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt, $i2:tt, $i3:tt]; [B, A, A, B]; [$l0:tt, $l1:tt, $l2:tt, $l3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@abba $b, $a; [swap($i0), swap($i1), swap($i2), swap($i3)]; [$l0, $l1, $l2, $l3])
    };
}

/// Walks the index list, splitting each index into "which operand" and "which lane within it".
macro_rules! swizzle4_decode {
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 0) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, A]; [$l0, $l1, $l2, 0])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 1) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, A]; [$l0, $l1, $l2, 1])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 2) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, A]; [$l0, $l1, $l2, 2])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 3) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, A]; [$l0, $l1, $l2, 3])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 4) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, B]; [$l0, $l1, $l2, 0])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 5) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, B]; [$l0, $l1, $l2, 1])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 6) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, B]; [$l0, $l1, $l2, 2])
    };
    ($a:expr, $b:expr; [$($i:tt),*]; []; [$s0:ident, $s1:ident, $s2:ident]; [$l0:tt, $l1:tt, $l2:tt]; 7) => {
        $crate::simd::swizzle_x86::swizzle4_emit!($a, $b; [$($i),*]; [$s0, $s1, $s2, B]; [$l0, $l1, $l2, 3])
    };

    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 0) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* A]; [$($l,)* 0]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 1) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* A]; [$($l,)* 1]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 2) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* A]; [$($l,)* 2]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 3) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* A]; [$($l,)* 3]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 4) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* B]; [$($l,)* 0]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 5) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* B]; [$($l,)* 1]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 6) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* B]; [$($l,)* 2]; $next)
    };
    ($a:expr, $b:expr; [$($i:tt),*]; [$next:tt $(, $rest:tt)*]; [$($s:ident),*]; [$($l:tt),*]; 7) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$($i),*]; [$($rest),*]; [$($s,)* B]; [$($l,)* 3]; $next)
    };
}

/// Picks the trait method for a two-lane output. Mirrors [`swizzle4_emit!`], including the refusal
/// to serve a two-input request that only reads one operand.
#[rustfmt::skip]
macro_rules! swizzle2_emit {
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [A, A]; [$l0:tt, $l1:tt]) => {
        compile_error!("both lanes read the first operand; drop the second and use `swizzle4!(a, [..])`")
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [B, B]; [$l0:tt, $l1:tt]) => {
        compile_error!("both lanes read the second operand; drop the first and use `swizzle4!(b, [..])`")
    };
    // Both lanes at the same position within their operand is one `unpack` instruction; the general
    // path below needs two `shufps` at 32-bit width.
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [A, B]; [0, 0]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_lo_2 $a, $b; [$i0, $i1])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [A, B]; [2, 2]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_hi_2 $a, $b; [$i0, $i1])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [B, A]; [0, 0]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_lo_2 $b, $a; [swap($i0), swap($i1)])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [B, A]; [2, 2]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@unpack_hi_2 $b, $a; [swap($i0), swap($i1)])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [A, B]; [$l0:tt, $l1:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@ab $a, $b; [$i0, $i1]; [$l0, $l1])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [B, A]; [$l0:tt, $l1:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@ab $b, $a; [swap($i0), swap($i1)]; [$l0, $l1])
    };
}

/// Two-lane counterpart of [`swizzle4_decode!`]: splits each index into "which operand" and "which
/// lane within it".
#[rustfmt::skip]
macro_rules! swizzle2_decode {
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 0) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, A]; [$l0, 0])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 1) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, A]; [$l0, 1])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 2) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, A]; [$l0, 2])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 3) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, A]; [$l0, 3])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 4) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, B]; [$l0, 0])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 5) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, B]; [$l0, 1])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 6) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, B]; [$l0, 2])
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; [$s0:ident]; [$l0:tt]; 7) => {
        $crate::simd::swizzle_x86::swizzle2_emit!($a, $b; [$i0, $i1]; [$s0, B]; [$l0, 3])
    };

    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 0) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [A]; [0]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 1) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [A]; [1]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 2) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [A]; [2]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 3) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [A]; [3]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 4) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [B]; [0]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 5) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [B]; [1]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 6) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [B]; [2]; $i1)
    };
    ($a:expr, $b:expr; [$i0:tt, $i1:tt]; []; []; 7) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; [B]; [3]; $i1)
    };
}

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

    ($a:expr, [$i0:tt]) => {
        compile_error!("a swizzle produces at least two lanes; a single index selects a scalar, not a vector")
    };
    ($a:expr, [$i0:tt, _, _, _]) => {
        compile_error!("only the first lane is given; the other three cannot be inferred, so spell them out")
    };
    ($a:expr, [$i0:tt, $i1:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_call!(@aa $a; [$i0, $i1]; [$i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::swizzle_x86::swizzle4!($a, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, [$i0, $i1, $i2, _])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        // Every lane reads the one operand, so the indices are already the lane values.
        $crate::simd::swizzle_x86::swizzle4_call!(@aaaa $a; [$i0, $i1, $i2, $i3]; [$i0, $i1, $i2, $i3])
    };

    ($a:expr, $b:expr, [$i0:tt]) => {
        compile_error!("a swizzle produces at least two lanes; a single index selects a scalar, not a vector")
    };
    ($a:expr, $b:expr, [$i0:tt, _, _, _]) => {
        compile_error!("only the first lane is given; the other three cannot be inferred, so spell them out")
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt]) => {
        $crate::simd::swizzle_x86::swizzle2_decode!($a, $b; [$i0, $i1]; []; []; $i0)
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [$i0, $i1, $i2, _])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::swizzle_x86::swizzle4!($a, $b, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {
        $crate::simd::swizzle_x86::swizzle4_decode!($a, $b; [$i0, $i1, $i2, $i3]; [$i1, $i2, $i3]; []; []; $i0)
    };
}

pub(crate) use swizzle2_decode;
pub(crate) use swizzle2_emit;
pub(crate) use swizzle4;
pub(crate) use swizzle4_call;
pub(crate) use swizzle4_decode;
pub(crate) use swizzle4_emit;

#[cfg(test)]
mod tests {
    use super::*;

    // `a` holds [0, 1, 2, 3] and `b` holds [4, 5, 6, 7], so output lane `k` must equal `Ik`.
    macro_rules! check {
        ([$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            let b = f32x4::new([4., 5., 6., 7.]);
            assert_eq!(
                swizzle4!(a, b, [$i0, $i1, $i2, $i3]).to_array(),
                [$i0 as f32, $i1 as f32, $i2 as f32, $i3 as f32],
                concat!(
                    "f32x4 [",
                    stringify!($i0),
                    ",",
                    stringify!($i1),
                    ",",
                    stringify!($i2),
                    ",",
                    stringify!($i3),
                    "]",
                ),
            );

            let a = f64x4::new([0., 1., 2., 3.]);
            let b = f64x4::new([4., 5., 6., 7.]);
            assert_eq!(
                swizzle4!(a, b, [$i0, $i1, $i2, $i3]).to_array(),
                [$i0 as f64, $i1 as f64, $i2 as f64, $i3 as f64],
                concat!(
                    "f64x4 [",
                    stringify!($i0),
                    ",",
                    stringify!($i1),
                    ",",
                    stringify!($i2),
                    ",",
                    stringify!($i3),
                    "]",
                ),
            );
        }};
    }

    // One input, four lanes out. `a` holds [0, 1, 2, 3] and the indices are its lane numbers.
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

    // Two lanes out. Only the low half is meaningful, and for 32-bit lanes `Vector2` is the
    // four-lane type, so compare the first two lanes there.
    macro_rules! check_two {
        ([$i0:tt, $i1:tt]) => {{
            let a = f32x4::new([0., 1., 2., 3.]);
            let b = f32x4::new([4., 5., 6., 7.]);
            let produced = swizzle4!(a, b, [$i0, $i1]).to_array();
            assert_eq!([produced[0], produced[1]], [$i0 as f32, $i1 as f32]);

            let a = f64x4::new([0., 1., 2., 3.]);
            let b = f64x4::new([4., 5., 6., 7.]);
            assert_eq!(swizzle4!(a, b, [$i0, $i1]).to_array(), [$i0 as f64, $i1 as f64]);
        }};
    }

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
    fn one_instruction_patterns() {
        check!([0, 1, 4, 5]); // aabb
        check!([4, 5, 0, 1]); // aabb, operands swapped
        check!([0, 4, 1, 5]); // unpack_lo(a, b)
        check!([2, 6, 3, 7]); // unpack_hi(a, b)
        check!([4, 0, 5, 1]); // unpack_lo(b, a)
        check!([6, 2, 7, 3]); // unpack_hi(b, a)
    }

    #[test]
    fn one_input_patterns() {
        check_one!([3, 1, 0, 2]);
        check_one!([3, 2, 1, 0]);
        check_one!([0, 0, 0, 0]);
        check_one!([3, 3, 2, 2]);
        check_one!([1, 1, 3, 3]);
    }

    #[test]
    fn two_lane_patterns() {
        check_one_two!([0, 1]);
        check_one_two!([3, 1]);
        check_one_two!([2, 2]);
        check_two!([0, 4]); // unpack_lo_2(a, b)
        check_two!([2, 6]); // unpack_hi_2(a, b)
        check_two!([4, 0]); // unpack_lo_2(b, a)
        check_two!([6, 2]); // unpack_hi_2(b, a)
        check_two!([1, 6]);
        check_two!([5, 3]);
        check_two!([7, 0]);
    }

    #[test]
    fn two_instruction_patterns() {
        check!([0, 1, 2, 4]); // aaab
        check!([0, 1, 4, 2]); // aaba
        check!([0, 4, 1, 2]); // abaa
        check!([1, 4, 3, 6]); // abab
        check!([0, 4, 5, 1]); // abba
        check!([0, 4, 5, 6]); // abbb
        check!([4, 0, 1, 2]); // abbb, operands swapped
        check!([4, 0, 1, 5]); // abba, operands swapped
        check!([5, 0, 7, 2]); // abab, operands swapped
        check!([4, 0, 5, 6]); // abaa, operands swapped
        check!([4, 5, 0, 6]); // aaba, operands swapped
        check!([4, 5, 6, 0]); // aaab, operands swapped
    }

    #[test]
    fn duplicated_lanes() {
        check!([2, 2, 6, 6]);
        check!([5, 5, 1, 1]);
    }
}
