#[cfg(target_feature = "sse")]
use crate::arch::sse;
use crate::utils::Swizzle;
use wide::{f32x4, i32x4, u32x4};

impl Swizzle for f32x4 {
    #[inline(always)]
    fn swizzle_generic<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: Self,
        b: Self,
    ) -> Self {
        #[rustfmt::skip]
        cfg_select! {
            target_feature = "simd128" => unsafe {
                use core::arch::wasm32::{u32x4_shuffle, v128_load, v128_store};

                let a = a.to_array();
                let b = b.to_array();
                // SAFETY: Both arrays contain exactly 16 initialized bytes.
                // WebAssembly's v128 loads and stores permit unaligned addresses.
                let a = v128_load(a.as_ptr().cast());
                let b = v128_load(b.as_ptr().cast());
                let shuffled = u32x4_shuffle::<I0, I1, I2, I3>(a, b);
                let mut result = [0.; 4];
                v128_store(result.as_mut_ptr().cast(), shuffled);
                Self::new(result)
            }
            any(
                target_feature = "ssse3",
                all(target_feature = "neon", target_arch = "aarch64"),
            ) => {{
                use wide::i8x16;

                const fn byte_swizzle_indices(indices: [usize; 4], lane_offset: usize) -> [i8; 16] {
                    let mut result = [-1; 16];
                    let mut byte = 0;
                    while byte < result.len() {
                        let lane = indices[byte / 4];
                        if lane >= lane_offset && lane < lane_offset + 4 {
                            result[byte] = ((lane - lane_offset) * 4 + byte % 4) as i8;
                        }
                        byte += 1;
                    }
                    result
                }

                let indices = [I0, I1, I2, I3];
                let a: i8x16 = wide::bytemuck::cast(a);
                let b: i8x16 = wide::bytemuck::cast(b);
                let from_a = a.swizzle_relaxed(i8x16::new(byte_swizzle_indices(indices, 0)));
                let from_b = b.swizzle_relaxed(i8x16::new(byte_swizzle_indices(indices, 4)));
                wide::bytemuck::cast(from_a | from_b)
            }}
            _ => {
                let a = a.to_array();
                let b = b.to_array();
                Self::new([
                    if I0 < 4 { a[I0] } else { b[I0 - 4] },
                    if I1 < 4 { a[I1] } else { b[I1 - 4] },
                    if I2 < 4 { a[I2] } else { b[I2 - 4] },
                    if I3 < 4 { a[I3] } else { b[I3 - 4] },
                ])
            }
        }
    }

    // TODO(portable-simd-backend): Revisit the shuffle abstraction once a native NEON design can
    // be validated on an ARM host or CI runner.
    #[inline(always)]
    #[cfg_attr(not(target_feature = "sse"), allow(unused_variables))]
    fn shuffle<const M: i32>(a: Self, b: Self) -> Self {
        cfg_select! {
            target_feature = "sse" => unsafe {
                // Let LLVM select `movelh` when this shuffle pattern permits it.
                let a = a.into();
                let b = b.into();
                Self::from(sse::_mm_shuffle_ps::<M>(a, b))
            },
            _ => panic!(),
        }
    }
    #[inline(always)]
    #[cfg_attr(not(target_feature = "sse"), allow(unused_variables))]
    fn unpack_lo(a: Self, b: Self) -> Self {
        cfg_select! {
            target_feature = "sse" => unsafe {
                let a = a.into();
                let b = b.into();
                Self::from(sse::_mm_unpacklo_ps(a, b))
            },
            _ => panic!(),
        }
    }
    #[inline(always)]
    #[cfg_attr(not(target_feature = "sse"), allow(unused_variables))]
    fn unpack_hi(a: Self, b: Self) -> Self {
        cfg_select! {
            target_feature = "sse" => unsafe {
                let a = a.into();
                let b = b.into();
                Self::from(sse::_mm_unpackhi_ps(a, b))
            },
            _ => panic!(),
        }
    }
}

impl Swizzle for i32x4 {
    #[inline(always)]
    fn swizzle_generic<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: Self,
        b: Self,
    ) -> Self {
        f32x4::swizzle_generic::<I0, I1, I2, I3>(
            f32x4::from_bits(a.cast_unsigned()),
            f32x4::from_bits(b.cast_unsigned()),
        )
        .to_bits()
        .cast_signed()
    }

    #[inline(always)]
    fn shuffle<const M: i32>(a: Self, b: Self) -> Self {
        f32x4::shuffle::<M>(
            f32x4::from_bits(a.cast_unsigned()),
            f32x4::from_bits(b.cast_unsigned()),
        )
        .to_bits()
        .cast_signed()
    }
    #[inline(always)]
    fn unpack_lo(a: Self, b: Self) -> Self {
        f32x4::unpack_lo(f32x4::from_bits(a.cast_unsigned()), f32x4::from_bits(b.cast_unsigned()))
            .to_bits()
            .cast_signed()
    }
    #[inline(always)]
    fn unpack_hi(a: Self, b: Self) -> Self {
        f32x4::unpack_hi(f32x4::from_bits(a.cast_unsigned()), f32x4::from_bits(b.cast_unsigned()))
            .to_bits()
            .cast_signed()
    }
}

impl Swizzle for u32x4 {
    #[inline(always)]
    fn swizzle_generic<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
        a: Self,
        b: Self,
    ) -> Self {
        f32x4::swizzle_generic::<I0, I1, I2, I3>(f32x4::from_bits(a), f32x4::from_bits(b)).to_bits()
    }

    #[inline(always)]
    fn shuffle<const M: i32>(a: Self, b: Self) -> Self {
        f32x4::shuffle::<M>(f32x4::from_bits(a), f32x4::from_bits(b)).to_bits()
    }
    #[inline(always)]
    fn unpack_lo(a: Self, b: Self) -> Self {
        f32x4::unpack_lo(f32x4::from_bits(a), f32x4::from_bits(b)).to_bits()
    }
    #[inline(always)]
    fn unpack_hi(a: Self, b: Self) -> Self {
        f32x4::unpack_hi(f32x4::from_bits(a), f32x4::from_bits(b)).to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::{Swizzle, u32x4};

    #[test]
    fn swizzle_generic_selects_lanes_from_one_input() {
        let a = u32x4::new([10, 11, 12, 13]);

        let actual = u32x4::swizzle_generic::<3, 1, 0, 2>(a, a);

        assert_eq!(actual.to_array(), [13, 11, 10, 12]);
    }

    #[test]
    fn swizzle_generic_selects_lanes_from_both_inputs() {
        let a = u32x4::new([10, 11, 12, 13]);
        let b = u32x4::new([20, 21, 22, 23]);

        let actual = u32x4::swizzle_generic::<7, 0, 5, 2>(a, b);

        assert_eq!(actual.to_array(), [23, 10, 21, 12]);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) const fn sse_shuffle_mask(l0: i32, l1: i32, l2: i32, l3: i32) -> i32 {
    l0 | l1 << 2 | l2 << 4 | l3 << 6
}

#[cfg(target_feature = "sse")]
macro_rules! emit_sse_swizzle {
    ($a:expr, $b:expr; [A, B, A, B]; [0, 0, 1, 1]) => {
        $crate::utils::Swizzle::unpack_lo($a, $b)
    };
    ($a:expr, $b:expr; [A, B, A, B]; [2, 2, 3, 3]) => {
        $crate::utils::Swizzle::unpack_hi($a, $b)
    };
    ($a:expr, $b:expr; [B, A, B, A]; [0, 0, 1, 1]) => {
        $crate::utils::Swizzle::unpack_lo($b, $a)
    };
    ($a:expr, $b:expr; [B, A, B, A]; [2, 2, 3, 3]) => {
        $crate::utils::Swizzle::unpack_hi($b, $a)
    };
    // Use the general shuffle and let LLVM select `movelh` when applicable.
    ($a:expr, $b:expr; [A, A, A, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, $lane2, $lane3) }>($a, $a)
    }};
    ($a:expr, $b:expr; [A, A, A, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane2, 0, $lane3, 0) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, 0, 2) }>($a, temp)
    }};
    ($a:expr, $b:expr; [A, A, B, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane2, 0, $lane3, 0) }>($b, $a);
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, 0, 2) }>($a, temp)
    }};
    ($a:expr, $b:expr; [A, A, B, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, $lane2, $lane3) }>($a, $b)
    }};
    ($a:expr, $b:expr; [A, B, A, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, 0, $lane1, 0) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, $lane2, $lane3) }>(temp, $a)
    }};
    ($a:expr, $b:expr; [A, B, A, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane2, $lane1, $lane3) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, 1, 3) }>(temp, temp)
    }};
    ($a:expr, $b:expr; [A, B, B, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane3, $lane1, $lane2) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, 3, 1) }>(temp, temp)
    }};
    ($a:expr, $b:expr; [A, B, B, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, 0, $lane1, 0) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, $lane2, $lane3) }>(temp, $b)
    }};
    ($a:expr, $b:expr; [B, A, A, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, 0, $lane1, 0) }>($b, $a);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, $lane2, $lane3) }>(temp, $a)
    }};
    ($a:expr, $b:expr; [B, A, A, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane1, $lane2, $lane0, $lane3) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(2, 0, 1, 3) }>(temp, temp)
    }};
    ($a:expr, $b:expr; [B, A, B, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane1, $lane3, $lane0, $lane2) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask(2, 0, 3, 1) }>(temp, temp)
    }};
    ($a:expr, $b:expr; [B, A, B, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane0, 0, $lane1, 0) }>($b, $a);
        Swizzle::shuffle::<{ sse_shuffle_mask(0, 2, $lane2, $lane3) }>(temp, $b)
    }};
    ($a:expr, $b:expr; [B, B, A, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, $lane2, $lane3) }>($b, $a)
    }};
    ($a:expr, $b:expr; [B, B, A, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane2, 0, $lane3, 0) }>($a, $b);
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, 0, 2) }>($b, temp)
    }};
    ($a:expr, $b:expr; [B, B, B, A]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        let temp = Swizzle::shuffle::<{ sse_shuffle_mask($lane2, 0, $lane3, 0) }>($b, $a);
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, 0, 2) }>($b, temp)
    }};
    ($a:expr, $b:expr; [B, B, B, B]; [$lane0:tt, $lane1:tt, $lane2:tt, $lane3:tt]) => {{
        use crate::{simd::utils::sse_shuffle_mask, utils::Swizzle};
        Swizzle::shuffle::<{ sse_shuffle_mask($lane0, $lane1, $lane2, $lane3) }>($b, $b)
    }};
}

#[cfg(target_feature = "sse")]
macro_rules! decode_sse_swizzle_indices {
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 0) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, A]; [$lane0, $lane1, $lane2, 0])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 1) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, A]; [$lane0, $lane1, $lane2, 1])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 2) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, A]; [$lane0, $lane1, $lane2, 2])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 3) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, A]; [$lane0, $lane1, $lane2, 3])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 4) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, B]; [$lane0, $lane1, $lane2, 0])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 5) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, B]; [$lane0, $lane1, $lane2, 1])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 6) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, B]; [$lane0, $lane1, $lane2, 2])
    };
    ($a:expr, $b:expr; []; [$s0:ident, $s1:ident, $s2:ident]; [$lane0:tt, $lane1:tt, $lane2:tt]; 7) => {
        $crate::simd::utils::emit_sse_swizzle!($a, $b; [$s0, $s1, $s2, B]; [$lane0, $lane1, $lane2, 3])
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 0) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* A]; [$($lane,)* 0]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 1) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* A]; [$($lane,)* 1]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 2) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* A]; [$($lane,)* 2]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 3) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* A]; [$($lane,)* 3]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 4) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* B]; [$($lane,)* 0]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 5) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* B]; [$($lane,)* 1]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 6) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* B]; [$($lane,)* 2]; $next)
    };
    ($a:expr, $b:expr; [$next:tt $(, $i:tt)*]; [$($src:ident),*]; [$($lane:tt),*]; 7) => {
        $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$($i),*]; [$($src,)* B]; [$($lane,)* 3]; $next)
    };
}

#[rustfmt::skip]
macro_rules! validate_lane4 {
    (0) => { 0 };
    (1) => { 1 };
    (2) => { 2 };
    (3) => { 3 };
}

#[rustfmt::skip]
#[cfg(not(target_feature = "sse"))]
macro_rules! validate_lane8 {
    (0) => { 0 };
    (1) => { 1 };
    (2) => { 2 };
    (3) => { 3 };
    (4) => { 4 };
    (5) => { 5 };
    (6) => { 6 };
    (7) => { 7 };
}

macro_rules! swizzle {
    ($a:expr, [$i0:tt, _, _, _]) => {
        compile_error!()
    };
    ($a:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::utils::swizzle!($a, [$i0, $i1, $i0, $i1])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::utils::swizzle!($a, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
        cfg_select! {
            target_feature = "sse" => {
                $crate::utils::Swizzle::shuffle::<{
                    $crate::simd::utils::sse_shuffle_mask(
                        $crate::simd::utils::validate_lane4!($i0),
                        $crate::simd::utils::validate_lane4!($i1),
                        $crate::simd::utils::validate_lane4!($i2),
                        $crate::simd::utils::validate_lane4!($i3),
                    )
                }>($a, $a)
            },
            _ => {
                $crate::utils::Swizzle::swizzle_generic::<
                    { $crate::simd::utils::validate_lane4!($i0) },
                    { $crate::simd::utils::validate_lane4!($i1) },
                    { $crate::simd::utils::validate_lane4!($i2) },
                    { $crate::simd::utils::validate_lane4!($i3) },
                >($a, $a)
            }
        }
    }};

    ($a:expr, $b:expr, [$i0:tt, _, _, _]) => {
        compile_error!()
    };
    // Complete partial unpack requests with the corresponding full unpack pattern.
    ($a:expr, $b:expr, [0, 4, _, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [0, 4, 1, 5])
    };
    ($a:expr, $b:expr, [2, 6, _, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [2, 6, 3, 7])
    };
    ($a:expr, $b:expr, [4, 0, _, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [4, 0, 5, 1])
    };
    ($a:expr, $b:expr, [6, 2, _, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [6, 2, 7, 3])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, _, _]) => {
        // Let codegen select `movddup` when profitable.
        $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i0, $i1])
    };
    // Complete partial unpack requests with the corresponding full unpack pattern.
    ($a:expr, $b:expr, [0, 4, 1, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [0, 4, 1, 5])
    };
    ($a:expr, $b:expr, [2, 6, 3, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [2, 6, 3, 7])
    };
    ($a:expr, $b:expr, [4, 0, 5, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [4, 0, 5, 1])
    };
    ($a:expr, $b:expr, [6, 2, 7, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [6, 2, 7, 3])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, _]) => {
        $crate::simd::utils::swizzle!($a, $b, [$i0, $i1, $i2, $i2])
    };
    ($a:expr, $b:expr, [$i0:tt, $i1:tt, $i2:tt, $i3:tt]) => {{
        cfg_select! {
            target_feature = "sse" => {
                $crate::simd::utils::decode_sse_swizzle_indices!($a, $b; [$i1, $i2, $i3]; []; []; $i0)
            },
            _ => {
                $crate::utils::Swizzle::swizzle_generic::<
                    { $crate::simd::utils::validate_lane8!($i0) },
                    { $crate::simd::utils::validate_lane8!($i1) },
                    { $crate::simd::utils::validate_lane8!($i2) },
                    { $crate::simd::utils::validate_lane8!($i3) },
                >($a, $b)
            }
        }
    }};
}

macro_rules! sign {
    ($vector:expr, [+, +, +, -]) => {
        $vector ^ f32x4::new([0., 0., 0., -0.])
    };
    ($vector:expr, [+, +, -, +]) => {
        $vector ^ f32x4::new([0., 0., -0., 0.])
    };
    ($vector:expr, [+, +, -, -]) => {
        $vector ^ f32x4::new([0., 0., -0., -0.])
    };
    ($vector:expr, [+, -, +, +]) => {
        $vector ^ f32x4::new([0., -0., 0., 0.])
    };
    ($vector:expr, [+, -, +, -]) => {
        $vector ^ f32x4::new([0., -0., 0., -0.])
    };
    ($vector:expr, [+, -, -, +]) => {
        $vector ^ f32x4::new([0., -0., -0., 0.])
    };
    ($vector:expr, [+, -, -, -]) => {
        $vector ^ f32x4::new([0., -0., -0., -0.])
    };
    ($vector:expr, [-, +, +, +]) => {
        $vector ^ f32x4::new([-0., 0., 0., 0.])
    };
    ($vector:expr, [-, +, +, -]) => {
        $vector ^ f32x4::new([-0., 0., 0., -0.])
    };
    ($vector:expr, [-, +, -, +]) => {
        $vector ^ f32x4::new([-0., 0., -0., 0.])
    };
    ($vector:expr, [-, +, -, -]) => {
        $vector ^ f32x4::new([-0., 0., -0., -0.])
    };
    ($vector:expr, [-, -, +, +]) => {
        $vector ^ f32x4::new([-0., -0., 0., 0.])
    };
    ($vector:expr, [-, -, +, -]) => {
        $vector ^ f32x4::new([-0., -0., 0., -0.])
    };
    ($vector:expr, [-, -, -, +]) => {
        $vector ^ f32x4::new([-0., -0., -0., 0.])
    };
}

#[cfg(target_feature = "sse")]
pub(crate) use decode_sse_swizzle_indices;
#[cfg(target_feature = "sse")]
pub(crate) use emit_sse_swizzle;
pub(crate) use sign;
pub(crate) use swizzle;
pub(crate) use validate_lane4;
#[cfg(not(target_feature = "sse"))]
pub(crate) use validate_lane8;

// A future `std::simd::Simd` backend must preserve the current eight-byte two-lane storage layout.
// `std::simd` can represent LLVM `<2 x float>` directly, whereas `[f32; 2]` remains an aggregate;
// stable Rust currently offers no equally optimizable portable representation with this layout.
mod not_neon {
    use crate::utils::{Load, MaskStorage, Store};
    use wide::{f32x4, i32x4, u32x4};

    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct f32x2([f32; 2]);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct i32x2([i32; 2]);
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone)]
    #[repr(C, align(8))]
    pub(crate) struct u32x2([u32; 2]);

    impl f32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [f32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [f32; 2] { self.0 }
        #[inline(always)]
        pub(crate) fn from_bits(bits: u32x2) -> Self {
            Self::new(bits.to_array().map(f32::from_bits))
        }
        #[inline(always)]
        pub(crate) fn to_bits(self) -> u32x2 { u32x2::new(self.to_array().map(f32::to_bits)) }
    }
    impl i32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [i32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [i32; 2] { self.0 }
    }
    impl u32x2 {
        #[inline(always)]
        pub(crate) const fn new(a: [u32; 2]) -> Self { Self(a) }
        #[inline(always)]
        pub(crate) fn to_array(self) -> [u32; 2] { self.0 }
    }

    // SAFETY: the C layout contains exactly two integer lanes. Their total
    // size is eight bytes, equal to the explicit alignment, so there is no
    // padding, and every bit pattern is valid for the fields.
    const _: () = {
        // Confirm that the two-lane storage types contain no padding bytes.
        assert!(size_of::<f32x2>() == size_of::<[f32; 2]>());
        assert!(size_of::<i32x2>() == size_of::<[i32; 2]>());
        assert!(size_of::<u32x2>() == size_of::<[u32; 2]>());
    };
    unsafe impl wide::bytemuck::Zeroable for i32x2 {}
    unsafe impl wide::bytemuck::Zeroable for u32x2 {}
    unsafe impl wide::bytemuck::Pod for i32x2 {}
    unsafe impl wide::bytemuck::Pod for u32x2 {}

    unsafe impl crate::utils::MaskPrimitive for i32x2 {
        fn is_valid(self) -> bool {
            self.to_array().into_iter().all(crate::utils::MaskPrimitive::is_valid)
        }
        #[inline(always)]
        fn not(self) -> Self { unimplemented!() }
        #[inline(always)]
        fn bitand(self, _rhs: Self) -> Self { unimplemented!() }
        #[inline(always)]
        fn bitor(self, _rhs: Self) -> Self { unimplemented!() }
        #[inline(always)]
        fn bitxor(self, _rhs: Self) -> Self { unimplemented!() }
        #[inline(always)]
        fn select(self, _true_values: Self, _false_values: Self) -> Self { unimplemented!() }
    }
    // TODO(portable-simd-backend): Add native NEON two-lane operations without widening to 128
    // bits once an ARM host or CI runner can validate ABI, instructions, and behavior.
    // Most operations delegate through four-lane storage, but constants and array views must
    // preserve the explicit two-lane layout.
    impl crate::utils::ArithPrimitive for f32x2 {
        type Scalar = f32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0., 0.]);
        const ONE_: Self = Self::new([1., 1.]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }
    impl crate::utils::ArithPrimitive for i32x2 {
        type Scalar = i32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }
    impl crate::utils::ArithPrimitive for u32x2 {
        type Scalar = u32;
        type F32 = f32x2;
        type I32 = i32x2;
        type U32 = u32x2;
        type Mask = i32x2;
        const ZERO_: Self = Self::new([0, 0]);
        const ONE_: Self = Self::new([1, 1]);
        #[inline(always)]
        fn filled_(a: Self::Scalar) -> Self { Self::new([a; 2]) }
        #[inline(always)]
        fn as_array_(&self) -> &[Self::Scalar] { &self.0 }
        #[inline(always)]
        fn as_mut_array_(&mut self) -> &mut [Self::Scalar] { &mut self.0 }
    }

    impl Load for f32x2 {
        type Output = f32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { f32x4::new([self.0[0], self.0[1], 0.0, 0.0]) }
    }
    impl Load for i32x2 {
        type Output = i32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { i32x4::new([self.0[0], self.0[1], 0, 0]) }
    }
    impl Load for u32x2 {
        type Output = u32x4;
        #[inline(always)]
        fn load(self) -> Self::Output { u32x4::new([self.0[0], self.0[1], 0, 0]) }
    }
    impl Store<f32x2> for f32x4 {
        #[inline(always)]
        fn store(self) -> f32x2 {
            let [a, b, ..] = self.to_array();
            f32x2([a, b])
        }
    }
    impl Store<i32x2> for i32x4 {
        #[inline(always)]
        fn store(self) -> i32x2 {
            let [a, b, ..] = self.to_array();
            i32x2([a, b])
        }
    }
    impl Store<u32x2> for u32x4 {
        #[inline(always)]
        fn store(self) -> u32x2 {
            let [a, b, ..] = self.to_array();
            u32x2([a, b])
        }
    }
    impl Load for MaskStorage<i32x4> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl<const N: usize> Load for MaskStorage<[i32x4; N]> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl<const N: usize> Store<MaskStorage<[i32x4; N]>> for [MaskStorage<i32x4>; N] {
        #[inline(always)]
        fn store(self) -> MaskStorage<[i32x4; N]> { self.into() }
    }
    impl Load for MaskStorage<i32x2> {
        type Output = MaskStorage<i32x4>;
        #[inline(always)]
        fn load(self) -> Self::Output {
            let inner = self.into_inner().load();
            unsafe {
                // SAFETY: padding lanes are zeroed, so this is safe
                MaskStorage::new_unchecked(inner)
            }
        }
    }
    impl Store<MaskStorage<i32x2>> for MaskStorage<i32x4> {
        #[inline(always)]
        fn store(self) -> MaskStorage<i32x2> {
            let inner = self.into_inner().store();
            unsafe {
                // SAFETY: just dropping the padding lanes, so this is safe
                MaskStorage::new_unchecked(inner)
            }
        }
    }

    macro_rules! impl_from {
        ($($tx2:ty:$t:ty),*) => {
            $(
                impl From<[$t; 2]> for $tx2 {
                    #[inline]
                    fn from(value: [$t; 2]) -> Self { Self::new(value) }
                }
                impl From<$tx2> for [$t; 2] {
                    #[inline]
                    fn from(value: $tx2) -> Self { value.to_array() }
                }
            )*
        };
    }
    impl_from!(f32x2:f32, i32x2:i32, u32x2:u32);

    macro_rules! impl_load {
        ($($t:ty),*) => {
            $(
                impl Load for $t {
                    type Output = $t;
                    #[inline(always)]
                    fn load(self) -> Self::Output { self }
                }
            )*
        };
    }
    impl_load!(f32, i32, u32);
    impl_load!(f32x4, i32x4, u32x4);
    impl<T: Load<Output = T>, const N: usize> Load for [T; N] {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
    impl Load for MaskStorage<i32> {
        type Output = Self;
        #[inline(always)]
        fn load(self) -> Self::Output { self }
    }
}

pub(crate) use not_neon::{f32x2, i32x2, u32x2};

macro_rules! impl_swizzle2_for_i0 {
    ($i0:tt; $($i1:tt),*) => {$(
        impl crate::private::SwizzleDispatch for crate::private::Indices2<$i0, $i1> {
            #[inline(always)]
            fn dispatch<T: Copy + Swizzle>(v: T) -> T { swizzle!(v, [$i0, $i1, _, _]) }
        }
    )*};
}

macro_rules! impl_swizzle3_for_i0_i1 {
    ($i0:tt, $i1:tt; $($i2:tt),*) => {$(
        impl crate::private::SwizzleDispatch for crate::private::Indices3<$i0, $i1, $i2> {
            #[inline(always)]
            fn dispatch<T: Copy + Swizzle>(v: T) -> T { swizzle!(v, [$i0, $i1, $i2, _]) }
        }
    )*};
}
macro_rules! impl_swizzle3_for_i0 {
    ($i0:tt; $($i1:tt),*) => {
        $(impl_swizzle3_for_i0_i1!($i0, $i1; 0, 1, 2, 3);)*
    };
}

macro_rules! impl_swizzle4_for_i0_i1_i2 {
    ($i0:tt, $i1:tt, $i2:tt; $($i3:tt),*) => {$(
        impl crate::private::SwizzleDispatch
            for crate::private::Indices4<$i0, $i1, $i2, $i3>
        {
            #[inline(always)]
            fn dispatch<T: Copy + Swizzle>(v: T) -> T { swizzle!(v, [$i0, $i1, $i2, $i3]) }
        }
    )*};
}
macro_rules! impl_swizzle4_for_i0_i1 {
    ($i0:tt, $i1:tt; $($i2:tt),*) => {
        $(impl_swizzle4_for_i0_i1_i2!($i0, $i1, $i2; 0, 1, 2, 3);)*
    };
}
macro_rules! impl_swizzle4_for_i0 {
    ($i0:tt; $($i1:tt),*) => {
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
