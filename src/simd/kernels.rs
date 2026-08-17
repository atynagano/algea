#[cfg(test)]
mod tests;

pub(crate) mod reduce {
    use crate::utils::ArithPrimitive;
    use std::ops::Add;

    // TODO: product, min, max

    // TODO(reduce-sum-codegen): confirm which reduction shape wins per lane count and target.
    #[inline(always)]
    pub(crate) fn sum<T: ArithPrimitive<Scalar: Copy + Add<Output = T::Scalar>>, const N: usize>(
        v: T,
    ) -> T::Scalar {
        match N {
            1 => {
                let [x] = *v.as_array_().first_chunk::<1>().unwrap();
                x
            }
            2 => {
                let [x, y] = *v.as_array_().first_chunk::<2>().unwrap();
                x + y
            }
            3 => {
                let [x, y, z] = *v.as_array_().first_chunk::<3>().unwrap();
                x + y + z
            }
            4 => {
                // Preserve this addition tree to avoid the usual `hadd` latency and throughput cost;
                // revisit it only with representative benchmark or codegen evidence.
                let [x, y, z, w] = *v.as_array_().first_chunk::<4>().unwrap();
                (x + z) + (y + w)
            }
            _ => unimplemented!(),
        }
    }
}

pub(crate) mod mask {
    macro_rules! impl_matrix_conversion {
        ($scalar:ty, $vec2:ty, $vec4:ty) => {
            #[inline(always)]
            pub(crate) fn from_array_1x1([a]: [[bool; 1]; 1]) -> MaskStorage<$scalar> {
                from_array_1(a)
            }
            #[inline(always)]
            pub(crate) fn to_array_1x1(mask: MaskStorage<$scalar>) -> [[bool; 1]; 1] {
                [to_array_1(mask)]
            }
            #[inline(always)]
            pub(crate) fn from_array_2x1([a]: [[bool; 2]; 1]) -> MaskStorage<$vec2> {
                from_array_2(a)
            }
            #[inline(always)]
            pub(crate) fn to_array_2x1(mask: MaskStorage<$vec2>) -> [[bool; 2]; 1] {
                [to_array_2(mask)]
            }
            #[inline(always)]
            pub(crate) fn from_array_3x1([a]: [[bool; 3]; 1]) -> MaskStorage<$vec4> {
                from_array_3(a)
            }
            #[inline(always)]
            pub(crate) fn to_array_3x1(mask: MaskStorage<$vec4>) -> [[bool; 3]; 1] {
                [to_array_3(mask)]
            }
            #[inline(always)]
            pub(crate) fn from_array_4x1([a]: [[bool; 4]; 1]) -> MaskStorage<$vec4> {
                from_array_4(a)
            }
            #[inline(always)]
            pub(crate) fn to_array_4x1(mask: MaskStorage<$vec4>) -> [[bool; 4]; 1] {
                [to_array_4(mask)]
            }

            #[inline(always)]
            pub(crate) fn from_array_1x2([[a], [b]]: [[bool; 1]; 2]) -> MaskStorage<$vec2> {
                from_array_2([a, b])
            }
            #[inline(always)]
            pub(crate) fn to_array_1x2(mask: MaskStorage<$vec2>) -> [[bool; 1]; 2] {
                let [a, b] = to_array_2(mask);
                [[a], [b]]
            }
            #[inline(always)]
            pub(crate) fn from_array_2x2([a, b]: [[bool; 2]; 2]) -> MaskStorage<$vec4> {
                from_array_4([a[0], a[1], b[0], b[1]])
            }
            #[inline(always)]
            pub(crate) fn to_array_2x2(mask: MaskStorage<$vec4>) -> [[bool; 2]; 2] {
                let [a, b, c, d] = to_array_4(mask);
                [[a, b], [c, d]]
            }
            #[inline(always)]
            pub(crate) fn from_array_3x2([a, b]: [[bool; 3]; 2]) -> MaskStorage<[$vec4; 2]> {
                [from_array_3(a), from_array_3(b)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_3x2(mask: MaskStorage<[$vec4; 2]>) -> [[bool; 3]; 2] {
                let [a, b] = mask.unpack();
                [to_array_3(a), to_array_3(b)]
            }
            #[inline(always)]
            pub(crate) fn from_array_4x2([a, b]: [[bool; 4]; 2]) -> MaskStorage<[$vec4; 2]> {
                [from_array_4(a), from_array_4(b)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_4x2(mask: MaskStorage<[$vec4; 2]>) -> [[bool; 4]; 2] {
                let [a, b] = mask.unpack();
                [to_array_4(a), to_array_4(b)]
            }

            #[inline(always)]
            pub(crate) fn from_array_1x3([[a], [b], [c]]: [[bool; 1]; 3]) -> MaskStorage<$vec4> {
                from_array_3([a, b, c])
            }
            #[inline(always)]
            pub(crate) fn to_array_1x3(mask: MaskStorage<$vec4>) -> [[bool; 1]; 3] {
                let [a, b, c] = to_array_3(mask);
                [[a], [b], [c]]
            }
            #[inline(always)]
            pub(crate) fn from_array_2x3(array: [[bool; 2]; 3]) -> MaskStorage<[$vec4; 2]> {
                let [[a, b], [c, d], [e, f]] = array;
                [from_array_4([a, b, c, d]), from_array_2([e, f]).widen()].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_2x3(mask: MaskStorage<[$vec4; 2]>) -> [[bool; 2]; 3] {
                let [first, last] = mask.unpack();
                let [a, b, c, d] = to_array_4(first);
                let [e, f] = to_array_2(last.xy());
                [[a, b], [c, d], [e, f]]
            }
            #[inline(always)]
            pub(crate) fn from_array_3x3([a, b, c]: [[bool; 3]; 3]) -> MaskStorage<[$vec4; 3]> {
                [from_array_3(a), from_array_3(b), from_array_3(c)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_3x3(mask: MaskStorage<[$vec4; 3]>) -> [[bool; 3]; 3] {
                let [a, b, c] = mask.unpack();
                [to_array_3(a), to_array_3(b), to_array_3(c)]
            }
            #[inline(always)]
            pub(crate) fn from_array_4x3([a, b, c]: [[bool; 4]; 3]) -> MaskStorage<[$vec4; 3]> {
                [from_array_4(a), from_array_4(b), from_array_4(c)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_4x3(mask: MaskStorage<[$vec4; 3]>) -> [[bool; 4]; 3] {
                let [a, b, c] = mask.unpack();
                [to_array_4(a), to_array_4(b), to_array_4(c)]
            }

            #[inline(always)]
            pub(crate) fn from_array_1x4(
                [[a], [b], [c], [d]]: [[bool; 1]; 4],
            ) -> MaskStorage<$vec4> {
                from_array_4([a, b, c, d])
            }
            #[inline(always)]
            pub(crate) fn to_array_1x4(mask: MaskStorage<$vec4>) -> [[bool; 1]; 4] {
                let [a, b, c, d] = to_array_4(mask);
                [[a], [b], [c], [d]]
            }
            #[inline(always)]
            pub(crate) fn from_array_2x4(array: [[bool; 2]; 4]) -> MaskStorage<[$vec4; 2]> {
                let [[a, b], [c, d], [e, f], [g, h]] = array;
                [from_array_4([a, b, c, d]), from_array_4([e, f, g, h])].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_2x4(mask: MaskStorage<[$vec4; 2]>) -> [[bool; 2]; 4] {
                let [first, last] = mask.unpack();
                let [a, b, c, d] = to_array_4(first);
                let [e, f, g, h] = to_array_4(last);
                [[a, b], [c, d], [e, f], [g, h]]
            }
            #[inline(always)]
            pub(crate) fn from_array_3x4([a, b, c, d]: [[bool; 3]; 4]) -> MaskStorage<[$vec4; 4]> {
                [from_array_3(a), from_array_3(b), from_array_3(c), from_array_3(d)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_3x4(mask: MaskStorage<[$vec4; 4]>) -> [[bool; 3]; 4] {
                let [a, b, c, d] = mask.unpack();
                [to_array_3(a), to_array_3(b), to_array_3(c), to_array_3(d)]
            }
            #[inline(always)]
            pub(crate) fn from_array_4x4([a, b, c, d]: [[bool; 4]; 4]) -> MaskStorage<[$vec4; 4]> {
                [from_array_4(a), from_array_4(b), from_array_4(c), from_array_4(d)].into()
            }
            #[inline(always)]
            pub(crate) fn to_array_4x4(mask: MaskStorage<[$vec4; 4]>) -> [[bool; 4]; 4] {
                let [a, b, c, d] = mask.unpack();
                [to_array_4(a), to_array_4(b), to_array_4(c), to_array_4(d)]
            }
        };
    }

    pub(crate) mod i32 {
        use super::*;
        use crate::{
            simd::utils::{Simd2Ext, Simd4Ext, compute_i32x2},
            utils::MaskStorage,
        };
        use wide::i32x4;

        #[allow(unused_imports)]
        #[cfg(target_arch = "x86_64")]
        use crate::arch::{x86_64::__m128i, *};

        #[inline(always)]
        fn from_array<const N: usize>(array: [bool; 4]) -> MaskStorage<i32x4> {
            std::assert_matches!(N, 2..=4);
            #[rustfmt::skip]
            let inner = cfg_select! {
                all(target_feature = "avx512bw", target_feature = "avx512vl") => unsafe {
                    let bytes = sse2::_mm_cvtsi32_si128(i32::from_le_bytes(array.map(u8::from)));
                    let byte_mask =
                        avx512_bw_vl::_mm_movm_epi8(avx512_bw_vl::_mm_test_epi8_mask(bytes, bytes));
                    i32x4::from(sse41::_mm_cvtepi8_epi32(byte_mask))
                },
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                    use core::arch::aarch64::*;
                    assert_ne!(N, 2);
                    let v: [i32; 4] = [array[0] as i32, array[1] as i32, array[2] as i32, array[3] as i32];
                    let v = vld1q_s32(v.as_ptr());
                    vreinterpretq_s32_u32(vtstq_s32(v, v)).into()
                },
                _ => {{
                    use wide::u8x16;
                    let packed = i32::from_le_bytes(array.map(u8::from));
                    let bytes: u8x16 = wide::bytemuck::cast(i32x4::new([packed, 0, 0, 0]));
                    let words = u8x16::unpack_low(bytes, bytes);
                    let dwords = u8x16::unpack_low(words, words);
                    wide::bytemuck::cast(dwords.simd_eq(u8x16::splat(0)) ^ u8x16::splat(u8::MAX))
                }}
            };
            unsafe {
                // SAFETY: Every branch converts each physical lane to a canonical mask value.
                // The AVX-512 path expands each nonzero bool byte to `0xff` and then sign-extends
                // it to `-1`; the NEON path tests each `0`/`1` lane against itself (`TST`), which is
                // all-zero or all-one depending on whether the lane is nonzero, and reinterprets
                // those bits directly; the SSE2 path duplicates each `0` or `1` byte across an i32
                // lane and maps it to `0` or all-one bits; and the scalar path negates `0` or `1`.
                // Consequently every lane, including lanes supplied as padding, is `0` or `-1`.
                MaskStorage::new_unchecked(inner)
            }
        }

        #[inline(always)]
        fn from_array_1([a]: [bool; 1]) -> MaskStorage<i32> { MaskStorage::<i32>::new(a) }
        #[inline(always)]
        fn from_array_2(array: [bool; 2]) -> MaskStorage<compute_i32x2> {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                    use core::arch::aarch64::*;
                    let v: [i32; 2] = [array[0] as i32, array[1] as i32];
                    let v = vld1_s32(v.as_ptr());
                    let result = vreinterpret_s32_u32(vtst_s32(v, v));
                    // SAFETY: `vtst_s32(v, v)` is all-zero or all-one per lane depending on whether
                    // that `0`/`1` lane is nonzero, and `vreinterpret_s32_u32` preserves those bits.
                    MaskStorage::new_unchecked(result.into())
                },
                _ => from_array::<2>([array[0], array[1], false, false]),
            }
        }
        #[inline(always)]
        fn from_array_3(array: [bool; 3]) -> MaskStorage<i32x4> {
            from_array::<3>([array[0], array[1], array[2], false])
        }
        #[inline(always)]
        fn from_array_4(array: [bool; 4]) -> MaskStorage<i32x4> { from_array::<4>(array) }

        #[inline(always)]
        fn to_array<const N: usize>(mask: MaskStorage<i32x4>) -> [bool; N] {
            std::assert_matches!(N, 2..=4);

            #[rustfmt::skip]
            let bools_u32 = cfg_select! {
                target_feature = "ssse3" => {{
                    use wide::u32x4;
                    // `u8_from_i32` lowers through SSSE3 `_mm_shuffle_epi8` and extracts each lane's
                    // least-significant bit rather than testing whether the signed lane is negative.
                    let bytes = super::cast::u8_from_i32(mask.into_inner());
                    wide::bytemuck::cast::<_, u32x4>(bytes).to_array()[0]
                }},
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {{
                    use core::arch::aarch64::*;
                    assert_ne!(N, 2);
                    let bits = vmovn_s32(mask.into_inner().into());
                    let bits_u8 = vreinterpret_u8_s16(bits);
                    let packed = vuzp1_u8(bits_u8, bits_u8);
                    let packed_u32 = vreinterpret_u32_u8(packed);
                    vget_lane_u32::<0>(packed_u32)
                }},
                _ => {{
                    // `to_bitmask` extracts the most-significant bit of each lane.
                    let bits = mask.into_inner().to_bitmask();
                    // Expand bits 0 through 3 into the least-significant bit of separate bytes.
                    bits.wrapping_mul(0x0020_4081)
                }}
            } & 0x0101_0101;
            // SAFETY: each byte is either 0 or 1 here, so transmuting to bool is safe
            let bools =
                unsafe { core::mem::transmute::<[u8; 4], [bool; 4]>(bools_u32.to_le_bytes()) };
            core::array::from_fn(
                #[inline(always)]
                |i| bools[i],
            )
        }
        #[inline(always)]
        fn to_array_1(mask: MaskStorage<i32>) -> [bool; 1] { [mask.into_inner() < 0] }
        #[inline(always)]
        fn to_array_2(mask: MaskStorage<compute_i32x2>) -> [bool; 2] {
            cfg_select! {
                all(target_feature = "neon", target_arch = "aarch64") => unsafe {
                    use core::arch::aarch64::*;
                    let ones = vdup_n_s32(1);
                    let bits = vand_s32(mask.into_inner().into(), ones);
                    let bits_u16 = vreinterpret_u16_s32(bits);
                    let step1 = vuzp1_u16(bits_u16, bits_u16);
                    let step1_u8 = vreinterpret_u8_u16(step1);
                    let packed = vuzp1_u8(step1_u8, step1_u8);
                    let packed_u16 = vreinterpret_u16_u8(packed);
                    core::mem::transmute::<u16, [bool; 2]>(vget_lane_u16::<0>(packed_u16))
                },
                _ => to_array(mask),
            }
        }
        #[inline(always)]
        fn to_array_3(mask: MaskStorage<i32x4>) -> [bool; 3] { to_array(mask) }
        #[inline(always)]
        fn to_array_4(mask: MaskStorage<i32x4>) -> [bool; 4] { to_array(mask) }

        impl_matrix_conversion!(i32, compute_i32x2, i32x4);
    }

    pub(crate) mod i64 {
        use super::*;
        use crate::{
            simd::utils::{Simd2Ext, Simd4Ext},
            utils::MaskStorage,
        };
        use wide::{i64x2, i64x4};

        #[inline(always)]
        fn from_array_1([a]: [bool; 1]) -> MaskStorage<i64> { MaskStorage::<i64>::new(a) }
        #[inline(always)]
        fn from_array_2(array: [bool; 2]) -> MaskStorage<i64x2> { todo!() }
        #[inline(always)]
        fn from_array_3(array: [bool; 3]) -> MaskStorage<i64x4> { todo!() }
        #[inline(always)]
        fn from_array_4(array: [bool; 4]) -> MaskStorage<i64x4> { todo!() }

        #[inline(always)]
        fn to_array_1(mask: MaskStorage<i64>) -> [bool; 1] { [mask.into_inner() < 0] }
        #[inline(always)]
        fn to_array_2(mask: MaskStorage<i64x2>) -> [bool; 2] { todo!() }
        #[inline(always)]
        fn to_array_3(mask: MaskStorage<i64x4>) -> [bool; 3] { todo!() }
        #[inline(always)]
        fn to_array_4(mask: MaskStorage<i64x4>) -> [bool; 4] { todo!() }

        impl_matrix_conversion!(i64, i64x2, i64x4);
    }
}

pub(crate) mod diagonal {
    use crate::simd::utils::{ComputeVector4, swizzle};

    #[inline(always)]
    pub(crate) fn diagonal2x2<Tx4: ComputeVector4>(a: Tx4) -> Tx4::Vector2 { swizzle!(a, [0, 3]) }
    #[inline(always)]
    pub(crate) fn diagonal3x3<Tx4: ComputeVector4>(a: [Tx4; 3]) -> Tx4 {
        let temp = swizzle!(a[0], a[1], [0, 4, 1, 5]);
        swizzle!(temp, a[2], [0, 3, 6, _])
    }
    #[inline(always)]
    pub(crate) fn diagonal4x4<Tx4: ComputeVector4>(a: [Tx4; 4]) -> Tx4 {
        let xy = swizzle!(a[0], a[1], [0, 4, 1, 5]);
        let zw = swizzle!(a[2], a[3], [2, 6, 3, 7]);
        swizzle!(xy, zw, [0, 3, 4, 7])
    }
}

pub(crate) mod transpose {
    use crate::simd::utils::{ComputeVector4, swizzle};

    #[inline(always)]
    pub(crate) fn transpose1x1<T>(a: T) -> T { a }
    #[inline(always)]
    pub(crate) fn transpose2x1<Tx2>(a: Tx2) -> Tx2 { a }
    #[inline(always)]
    pub(crate) fn transpose3x1<Tx4>(a: Tx4) -> Tx4 { a }
    #[inline(always)]
    pub(crate) fn transpose4x1<Tx4>(a: Tx4) -> Tx4 { a }
    #[inline(always)]
    pub(crate) fn transpose1x2<Tx2>(a: Tx2) -> Tx2 { a }
    #[inline(always)]
    pub(crate) fn transpose2x2<Tx4: ComputeVector4>(a: Tx4) -> Tx4 { swizzle!(a, [0, 2, 1, 3]) }
    #[inline(always)]
    pub(crate) fn transpose3x2<Tx4: ComputeVector4>(a: [Tx4; 2]) -> [Tx4; 2] { transpose4x2(a) }
    #[inline(always)]
    pub(crate) fn transpose4x2<Tx4: ComputeVector4>(a: [Tx4; 2]) -> [Tx4; 2] {
        [swizzle!(a[0], a[1], [0, 4, 1, 5]), swizzle!(a[0], a[1], [2, 6, 3, 7])]
    }
    #[inline(always)]
    pub(crate) fn transpose1x3<Tx4>(a: Tx4) -> Tx4 { a }
    #[inline(always)]
    pub(crate) fn transpose2x3<Tx4: ComputeVector4>(a: [Tx4; 2]) -> [Tx4; 2] { transpose2x4(a) }

    #[inline(always)]
    pub(crate) fn transpose3x3<Tx4: ComputeVector4>(a: [Tx4; 3]) -> [Tx4; 3] {
        let ab_lo = swizzle!(a[0], a[1], [0, 4, 1, 5]);
        let ab_hi = swizzle!(a[0], a[1], [2, 6, 3, 7]);
        [
            swizzle!(ab_lo, a[2], [0, 1, 4, _]),
            swizzle!(ab_lo, a[2], [2, 3, 5, _]),
            swizzle!(ab_hi, a[2], [0, 1, 6, _]),
        ]
    }
    #[inline(always)]
    pub(crate) fn transpose4x3<Tx4: ComputeVector4>(a: [Tx4; 3]) -> [Tx4; 4] {
        let ab_lo = swizzle!(a[0], a[1], [0, 4, 1, 5]);
        let ab_hi = swizzle!(a[0], a[1], [2, 6, 3, 7]);
        [
            swizzle!(ab_lo, a[2], [0, 1, 4, _]),
            swizzle!(ab_lo, a[2], [2, 3, 5, _]),
            swizzle!(ab_hi, a[2], [0, 1, 6, _]),
            swizzle!(ab_hi, a[2], [2, 3, 7, _]),
        ]
    }
    #[inline(always)]
    pub(crate) fn transpose1x4<Tx4>(a: Tx4) -> Tx4 { a }
    #[inline(always)]
    pub(crate) fn transpose2x4<Tx4: ComputeVector4>(a: [Tx4; 2]) -> [Tx4; 2] {
        [swizzle!(a[0], a[1], [0, 2, 4, 6]), swizzle!(a[0], a[1], [1, 3, 5, 7])]
    }
    #[inline(always)]
    pub(crate) fn transpose3x4<Tx4: ComputeVector4>(a: [Tx4; 4]) -> [Tx4; 3] {
        let ab_lo = swizzle!(a[0], a[1], [0, 4, 1, 5]);
        let ab_hi = swizzle!(a[0], a[1], [2, 6, 3, 7]);
        let cd_lo = swizzle!(a[2], a[3], [0, 4, 1, 5]);
        let cd_hi = swizzle!(a[2], a[3], [2, 6, 3, 7]);
        [
            swizzle!(ab_lo, cd_lo, [0, 1, 4, 5]),
            swizzle!(ab_lo, cd_lo, [2, 3, 6, 7]),
            swizzle!(ab_hi, cd_hi, [0, 1, 4, 5]),
        ]
    }
    #[inline(always)]
    pub(crate) fn transpose4x4<Tx4: ComputeVector4>(a: [Tx4; 4]) -> [Tx4; 4] {
        let [col0, col1, col2, col3] = a;

        let cols01_lo = swizzle!(col0, col1, [0, 4, 1, 5]);
        let cols23_lo = swizzle!(col2, col3, [0, 4, 1, 5]);
        let cols01_hi = swizzle!(col0, col1, [2, 6, 3, 7]);
        let cols23_hi = swizzle!(col2, col3, [2, 6, 3, 7]);
        [
            swizzle!(cols01_lo, cols23_lo, [0, 1, 4, 5]),
            swizzle!(cols01_lo, cols23_lo, [2, 3, 6, 7]),
            swizzle!(cols01_hi, cols23_hi, [0, 1, 4, 5]),
            swizzle!(cols01_hi, cols23_hi, [2, 3, 6, 7]),
        ]
    }
}

pub(crate) mod inverse {
    #![allow(unused_parens)]

    use super::{determinant, transpose::transpose3x3};
    use crate::{
        simd::utils::{sign, swizzle},
        utils::arith,
    };

    macro_rules! impl_inverse {
        ($scalar:ident, $vec4:ident) => {
            #[inline(always)]
            pub(crate) fn _1x1(a: $scalar) -> $scalar { 1. / a }

            #[inline(always)]
            pub(crate) fn _2x2(a: $vec4) -> $vec4 {
                // [a, b, c, d] -> [d, -b, -c, a] / (ad - bc)
                let adj = sign!(swizzle!(a, [3, 1, 2, 0]), [+, -, -, +]);
                adj / determinant::$scalar::_2x2(a)
            }

            #[inline(always)]
            pub(crate) fn _3x3(a: [$vec4; 3]) -> [$vec4; 3] {
                let [r0, r1, r2] = a;
                let r0_yzx = swizzle!(r0, [1, 2, 0, _]);
                let r1_yzx = swizzle!(r1, [1, 2, 0, _]);
                let r2_yzx = swizzle!(r2, [1, 2, 0, _]);
                // The zxy-ordered cross product forms a column of the adjugate.
                let c0 = arith!(r1 * r2_yzx - r1_yzx * r2);
                let c1 = arith!(r2 * r0_yzx - r2_yzx * r0);
                let c2 = arith!(r0 * r1_yzx - r0_yzx * r1);

                // det = dot(r0, cross(r1, r2))
                let det = matmul1x3x1(c0, swizzle!(r0, [2, 0, 1, _]));
                // TODO(f64-scalar-splat): without AVX an `f64x4` is two registers, so divide
                // as a scalar and splat instead.
                let r_det = $vec4::splat(1.) / det;

                let [r2, r0, r1] = transpose3x3([c0 * r_det, c1 * r_det, c2 * r_det]);
                [r0, r1, r2]
            }

            #[inline(always)]
            pub(crate) fn _4x4(a: [$vec4; 4]) -> [$vec4; 4] {
                #[inline(always)]
                fn mat2_adj_mul(a: $vec4, b: $vec4) -> $vec4 {
                    arith!(
                        (swizzle!(a, [3, 3, 0, 0])) * b
                            - (swizzle!(a, [1, 1, 2, 2])) * (swizzle!(b, [2, 3, 0, 1]))
                    )
                }

                #[inline(always)]
                fn mat2_mul_adj(a: $vec4, b: $vec4) -> $vec4 {
                    arith!(
                        a * (swizzle!(b, [3, 0, 3, 0]))
                            - (swizzle!(a, [1, 0, 3, 2])) * (swizzle!(b, [2, 1, 2, 1]))
                    )
                }

                let a00 = swizzle!(a[0], a[1], [0, 1, 4, 5]);
                let b00 = swizzle!(a[0], a[1], [2, 3, 6, 7]);
                let c00 = swizzle!(a[2], a[3], [0, 1, 4, 5]);
                let d00 = swizzle!(a[2], a[3], [2, 3, 6, 7]);

                let det_sub = arith!(
                    (swizzle!(a[0], a[2], [0, 2, 4, 6])) * (swizzle!(a[1], a[3], [1, 3, 5, 7]))
                        - (swizzle!(a[0], a[2], [1, 3, 5, 7])) * (swizzle!(a[1], a[3], [0, 2, 4, 6]))
                );

                // TODO(f64-scalar-splat): measure splatting before against after, per width.
                let det_a = swizzle!(det_sub, [0, 0, 0, 0]);
                let det_b = swizzle!(det_sub, [1, 1, 1, 1]);
                let det_c = swizzle!(det_sub, [2, 2, 2, 2]);
                let det_d = swizzle!(det_sub, [3, 3, 3, 3]);

                let d_adj_c = mat2_adj_mul(d00, c00);
                let a_adj_b = mat2_adj_mul(a00, b00);

                // Keep X/Z and W/Y adjacent so each pair's shared inputs can die early,
                // avoiding temporary stack spills in SSE2 codegen.
                let x_ = arith!(det_d * a00 - (matmul2x2x2(d_adj_c, b00)));
                let z_ = arith!(det_c * b00 - (mat2_mul_adj(a00, d_adj_c)));
                let w_ = arith!(det_a * d00 - (matmul2x2x2(a_adj_b, c00)));
                let y_ = arith!(det_b * c00 - (mat2_mul_adj(d00, a_adj_b)));

                let tr_terms = a_adj_b * swizzle!(d_adj_c, [0, 2, 1, 3]);
                let tr_pair = tr_terms + swizzle!(tr_terms, [1, 0, 3, 2]);
                let tr = tr_pair + swizzle!(tr_pair, [2, 3, 0, 1]);

                let det_m = arith!(det_a * det_d + (arith!(det_b * det_c - tr)));

                let r_det = $vec4::new([1., -1., -1., 1.]) / det_m;

                let x_ = x_ * r_det;
                let y_ = y_ * r_det;
                let z_ = z_ * r_det;
                let w_ = w_ * r_det;

                [
                    swizzle!(x_, y_, [3, 1, 7, 5]),
                    swizzle!(x_, y_, [2, 0, 6, 4]),
                    swizzle!(z_, w_, [3, 1, 7, 5]),
                    swizzle!(z_, w_, [2, 0, 6, 4]),
                ]
            }
        };
    }

    pub(crate) mod f32 {
        use super::{
            super::matmul::f32::{matmul1x3x1, matmul2x2x2},
            *,
        };
        use wide::f32x4;
        impl_inverse!(f32, f32x4);
    }
    pub(crate) mod f64 {
        use super::{
            super::matmul::f64::{matmul1x3x1, matmul2x2x2},
            *,
        };
        use wide::f64x4;
        impl_inverse!(f64, f64x4);
    }
}

pub(crate) mod determinant {
    #![allow(unused_parens)]

    use super::*;
    use crate::{simd::utils::swizzle, utils::arith};

    macro_rules! impl_determinant {
        ($scalar:ident, $vec4:ident) => {
            #[inline(always)]
            pub(crate) fn _2x2(a: $vec4) -> $scalar {
                // TODO(codegen-optimization): review determinant codegen when FMA is unavailable.
                let [a, b, c, d] = a.to_array();
                // TODO(det2x2-fma): the two-lane dot product came out faster without FMA,
                // because gathering the operands into one lane cost more than the multiply.
                // This is a difference rather than a sum, so re-measure before assuming it.
                arith!(a * d - b * c)
            }

            #[inline(always)]
            pub(crate) fn _3x3(a: [$vec4; 3]) -> $scalar {
                let [r0, r1, r2] = a;
                let r1_yzx = swizzle!(r1, [1, 2, 0, _]);
                let r2_yzx = swizzle!(r2, [1, 2, 0, _]);
                // zxy order cross product, matching the determinant path in inverse::_3x3.
                let c0 = arith!(r1 * r2_yzx - r1_yzx * r2);

                // det = dot(r0, cross(r1, r2))
                matmul::$scalar::matmul1x3x1(c0, swizzle!(r0, [2, 0, 1, _]))
            }

            #[inline(always)]
            pub(crate) fn _4x4(a: [$vec4; 4]) -> $scalar {
                // Leave lane/scalar optimization to LLVM. Preserve this operation and addition order as-is
                // so the determinant remains numerically compatible with the path in inverse::_4x4.
                #[inline(always)]
                fn mat2_adj_mul(a: $vec4, b: $vec4) -> $vec4 {
                    arith!(
                        (swizzle!(a, [3, 3, 0, 0])) * b
                            - (swizzle!(a, [1, 1, 2, 2])) * (swizzle!(b, [2, 3, 0, 1]))
                    )
                }

                let a00 = swizzle!(a[0], a[1], [0, 1, 4, 5]);
                let b00 = swizzle!(a[0], a[1], [2, 3, 6, 7]);
                let c00 = swizzle!(a[2], a[3], [0, 1, 4, 5]);
                let d00 = swizzle!(a[2], a[3], [2, 3, 6, 7]);

                let det_sub = arith!(
                    (swizzle!(a[0], a[2], [0, 2, 4, 6])) * (swizzle!(a[1], a[3], [1, 3, 5, 7]))
                        - (swizzle!(a[0], a[2], [1, 3, 5, 7]))
                            * (swizzle!(a[1], a[3], [0, 2, 4, 6]))
                );

                let d_adj_c = mat2_adj_mul(d00, c00);
                let a_adj_b = mat2_adj_mul(a00, b00);
                // TODO(f64-scalar-splat): as above.
                let det_a = swizzle!(det_sub, [0, 0, 0, 0]);
                let det_b = swizzle!(det_sub, [1, 1, 1, 1]);
                let det_c = swizzle!(det_sub, [2, 2, 2, 2]);
                let det_d = swizzle!(det_sub, [3, 3, 3, 3]);

                let tr_terms = a_adj_b * swizzle!(d_adj_c, [0, 2, 1, 3]);
                let tr_pair = tr_terms + swizzle!(tr_terms, [1, 0, 3, 2]);
                let tr = tr_pair + swizzle!(tr_pair, [2, 3, 0, 1]);

                let det_m = arith!(det_a * det_d + (arith!(det_b * det_c - tr)));
                det_m.to_array()[0]
            }
        };
    }

    pub(crate) mod f32 {
        use super::*;
        use wide::f32x4;
        impl_determinant!(f32, f32x4);
    }
    pub(crate) mod f64 {
        use super::*;
        use wide::f64x4;
        impl_determinant!(f64, f64x4);
    }
}

pub(crate) mod index {
    use crate::utils::ArithPrimitive;

    macro_rules! impl_index {
        ($get:ident, $as_array:ident $(, $mut:tt)?) => {
            #[inline(always)]
            pub(crate) fn _1x1<T: Copy>(a: & $($mut)? T, (i, j): (usize, usize)) -> Option<& $($mut)? T> {
                if i == 0 && j == 0 { Some(a) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _2x1<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j == 0 && i < 2 { a.$as_array().$get(i) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _3x1<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j == 0 && i < 3 { a.$as_array().$get(i) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _4x1<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j == 0 { a.$as_array().$get(i) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _1x2<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j < 2 && i == 0 { a.$as_array().$get(j) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _2x2<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if i < 2 && j < 2 { a.$as_array().$get(2 * j + i) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _3x2<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 2],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) if i < 3 => column.$as_array().$get(i),
                    _ => None,
                }
            }
            #[inline(always)]
            pub(crate) fn _4x2<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 2],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) => column.$as_array().$get(i),
                    _ => None,
                }
            }
            #[inline(always)]
            pub(crate) fn _1x3<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j < 3 && i == 0 { a.$as_array().$get(j) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _2x3<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 2],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j < 3 && i < 2 {
                    if let Some(chunk) = a.$get(j / 2) {
                        // TODO(codegen-optimization): Verify that this packed index calculation
                        // lowers without division on representative x86-64 targets.
                        return chunk.$as_array().$get((2 * j + i) % 4);
                    }
                }
                None
            }
            #[inline(always)]
            pub(crate) fn _3x3<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 3],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) if i < 3 => column.$as_array().$get(i),
                    _ => None,
                }
            }
            #[inline(always)]
            pub(crate) fn _4x3<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 3],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) => column.$as_array().$get(i),
                    _ => None,
                }
            }
            #[inline(always)]
            pub(crate) fn _1x4<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? Tx4,
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if i == 0 { a.$as_array().$get(j) } else { None }
            }
            #[inline(always)]
            pub(crate) fn _2x4<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 2],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                if j < 4 && i < 2 {
                    if let Some(chunk) = a.$get(j / 2) {
                        // TODO(codegen-optimization): Verify that this packed index calculation
                        // lowers without division on representative x86-64 targets.
                        return chunk.$as_array().$get((2 * j + i) % 4);
                    }
                }
                None
            }
            #[inline(always)]
            pub(crate) fn _3x4<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 4],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) if i < 3 => column.$as_array().$get(i),
                    _ => None,
                }
            }
            #[inline(always)]
            pub(crate) fn _4x4<T, Tx4: ArithPrimitive<Scalar = T>>(
                a: & $($mut)? [Tx4; 4],
                (i, j): (usize, usize),
            ) -> Option<& $($mut)? T> {
                match a.$get(j) {
                    Some(column) => column.$as_array().$get(i),
                    _ => None,
                }
            }
        };
    }

    pub(super) use impl_index;
    impl_index!(get, as_array_);
}

pub(crate) mod index_mut {
    use crate::utils::ArithPrimitive;
    super::index::impl_index!(get_mut, as_mut_array_, mut);
}

pub(crate) mod to_array {
    #[inline(always)]
    pub fn __3x1<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [T; 3] {
        let [a, b, c, _] = a.into();
        [a, b, c]
    }

    #[inline(always)]
    pub(crate) fn _1x1<T: Copy>(a: T) -> [[T; 1]; 1] { [[a]] }
    #[inline(always)]
    pub(crate) fn _2x1<T, Tx2: Copy + Into<[T; 2]>>(a: Tx2) -> [[T; 2]; 1] {
        let [a, b] = a.into();
        [[a, b]]
    }
    #[inline(always)]
    pub(crate) fn _3x1<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [[T; 3]; 1] { [__3x1(a)] }
    #[inline(always)]
    pub(crate) fn _4x1<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [[T; 4]; 1] { [a.into()] }
    #[inline(always)]
    pub(crate) fn _1x2<T, Tx2: Copy + Into<[T; 2]>>(a: Tx2) -> [[T; 1]; 2] {
        let [a, b] = a.into();
        [[a], [b]]
    }
    #[inline(always)]
    pub(crate) fn _2x2<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [[T; 2]; 2] {
        let [a, b, c, d] = a.into();
        [[a, b], [c, d]]
    }
    #[inline(always)]
    pub(crate) fn _3x2<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 2]) -> [[T; 3]; 2] {
        [__3x1(a[0]), __3x1(a[1])]
    }
    #[inline(always)]
    pub(crate) fn _4x2<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 2]) -> [[T; 4]; 2] {
        [a[0].into(), a[1].into()]
    }
    #[inline(always)]
    pub(crate) fn _1x3<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [[T; 1]; 3] {
        let [a, b, c, _] = a.into();
        [[a], [b], [c]]
    }
    #[inline(always)]
    pub(crate) fn _2x3<T, Tx4: Copy + Into<[T; 4]>>(x: [Tx4; 2]) -> [[T; 2]; 3] {
        let [a, b, c, d] = x[0].into();
        let [e, f, _, _] = x[1].into();
        [[a, b], [c, d], [e, f]]
    }
    #[inline(always)]
    pub(crate) fn _3x3<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 3]) -> [[T; 3]; 3] {
        [__3x1(a[0]), __3x1(a[1]), __3x1(a[2])]
    }
    #[inline(always)]
    pub(crate) fn _4x3<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 3]) -> [[T; 4]; 3] {
        [a[0].into(), a[1].into(), a[2].into()]
    }
    #[inline(always)]
    pub(crate) fn _1x4<T, Tx4: Copy + Into<[T; 4]>>(a: Tx4) -> [[T; 1]; 4] {
        let [a, b, c, d] = a.into();
        [[a], [b], [c], [d]]
    }
    #[inline(always)]
    pub(crate) fn _2x4<T, Tx4: Copy + Into<[T; 4]>>(x: [Tx4; 2]) -> [[T; 2]; 4] {
        let [a, b, c, d] = x[0].into();
        let [e, f, g, h] = x[1].into();
        [[a, b], [c, d], [e, f], [g, h]]
    }
    #[inline(always)]
    pub(crate) fn _3x4<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 4]) -> [[T; 3]; 4] {
        [__3x1(a[0]), __3x1(a[1]), __3x1(a[2]), __3x1(a[3])]
    }
    #[inline(always)]
    pub(crate) fn _4x4<T, Tx4: Copy + Into<[T; 4]>>(a: [Tx4; 4]) -> [[T; 4]; 4] {
        [a[0].into(), a[1].into(), a[2].into(), a[3].into()]
    }
}

pub(crate) mod from_array {
    // Use `new` instead of `From<[T; 4]>` so these constructors remain `const fn`.
    macro_rules! impl_fns {
        ($f32:ty, $x2:ty, $x4:ty, $zero:literal) => {
            #[inline(always)]
            pub(crate) const fn _1x1([a]: [[$f32; 1]; 1]) -> $f32 { a[0] }
            #[inline(always)]
            pub(crate) const fn _2x1([a]: [[$f32; 2]; 1]) -> $x2 { <$x2>::new([a[0], a[1]]) }
            #[inline(always)]
            pub(crate) const fn _3x1([a]: [[$f32; 3]; 1]) -> $x4 {
                <$x4>::new([a[0], a[1], a[2], $zero])
            }
            #[inline(always)]
            pub(crate) const fn _4x1([a]: [[$f32; 4]; 1]) -> $x4 { <$x4>::new(a) }
            #[inline(always)]
            pub(crate) const fn _1x2([[a], [b]]: [[$f32; 1]; 2]) -> $x2 { <$x2>::new([a, b]) }
            #[inline(always)]
            pub(crate) const fn _2x2([a, b]: [[$f32; 2]; 2]) -> $x4 {
                <$x4>::new([a[0], a[1], b[0], b[1]])
            }
            #[inline(always)]
            pub(crate) const fn _3x2([a, b]: [[$f32; 3]; 2]) -> [$x4; 2] { [_3x1([a]), _3x1([b])] }
            #[inline(always)]
            pub(crate) const fn _4x2(a: [[$f32; 4]; 2]) -> [$x4; 2] {
                [<$x4>::new(a[0]), <$x4>::new(a[1])]
            }
            #[inline(always)]
            pub(crate) const fn _1x3(a: [[$f32; 1]; 3]) -> $x4 {
                <$x4>::new([a[0][0], a[1][0], a[2][0], $zero])
            }
            #[inline(always)]
            pub(crate) const fn _2x3(x: [[$f32; 2]; 3]) -> [$x4; 2] {
                [
                    <$x4>::new([x[0][0], x[0][1], x[1][0], x[1][1]]),
                    <$x4>::new([x[2][0], x[2][1], $zero, $zero]),
                ]
            }
            #[inline(always)]
            pub(crate) const fn _3x3(a: [[$f32; 3]; 3]) -> [$x4; 3] {
                [_3x1([a[0]]), _3x1([a[1]]), _3x1([a[2]])]
            }
            #[inline(always)]
            pub(crate) const fn _4x3(a: [[$f32; 4]; 3]) -> [$x4; 3] {
                [<$x4>::new(a[0]), <$x4>::new(a[1]), <$x4>::new(a[2])]
            }
            #[inline(always)]
            pub(crate) const fn _1x4(a: [[$f32; 1]; 4]) -> $x4 {
                <$x4>::new([a[0][0], a[1][0], a[2][0], a[3][0]])
            }
            #[inline(always)]
            pub(crate) const fn _2x4(x: [[$f32; 2]; 4]) -> [$x4; 2] {
                [
                    <$x4>::new([x[0][0], x[0][1], x[1][0], x[1][1]]),
                    <$x4>::new([x[2][0], x[2][1], x[3][0], x[3][1]]),
                ]
            }
            #[inline(always)]
            pub(crate) const fn _3x4(a: [[$f32; 3]; 4]) -> [$x4; 4] {
                [_3x1([a[0]]), _3x1([a[1]]), _3x1([a[2]]), _3x1([a[3]])]
            }
            #[inline(always)]
            pub(crate) const fn _4x4(a: [[$f32; 4]; 4]) -> [$x4; 4] {
                [<$x4>::new(a[0]), <$x4>::new(a[1]), <$x4>::new(a[2]), <$x4>::new(a[3])]
            }
        };
    }

    pub(crate) mod f32 {
        use crate::simd::utils::f32x2;
        use wide::f32x4;
        impl_fns!(f32, f32x2, f32x4, 0.);
    }
    pub(crate) mod f64 {
        use wide::{f64x2, f64x4};
        impl_fns!(f64, f64x2, f64x4, 0.);
    }
    pub(crate) mod i32 {
        use crate::simd::utils::i32x2;
        use wide::i32x4;
        impl_fns!(i32, i32x2, i32x4, 0);
    }
    pub(crate) mod i64 {
        use wide::{i64x2, i64x4};
        impl_fns!(i64, i64x2, i64x4, 0);
    }
    pub(crate) mod u32 {
        use crate::simd::utils::u32x2;
        use wide::u32x4;
        impl_fns!(u32, u32x2, u32x4, 0);
    }
    pub(crate) mod u64 {
        use wide::{u64x2, u64x4};
        impl_fns!(u64, u64x2, u64x4, 0);
    }
}

pub(crate) mod from_vecs {
    // These functions need no macro because they are not `const`.
    macro_rules! impl_fns {
        ($f32:ty, $x2:ty, $x4:ty, $zero:literal) => {
            #[inline(always)]
            pub(crate) fn _1x1([a]: [Vector<$f32, 1>; 1]) -> $f32 { a.storage }
            #[inline(always)]
            pub(crate) fn _2x1([a]: [Vector<$f32, 2>; 1]) -> $x2 { a.storage }
            #[inline(always)]
            pub(crate) fn _3x1([a]: [Vector<$f32, 3>; 1]) -> $x4 { a.storage }
            #[inline(always)]
            pub(crate) fn _4x1([a]: [Vector<$f32, 4>; 1]) -> $x4 { a.storage }
            #[inline(always)]
            pub(crate) fn _1x2([a, b]: [Vector<$f32, 1>; 2]) -> $x2 {
                <$x2>::new([a.storage, b.storage])
            }
            #[inline(always)]
            pub(crate) fn _2x2([a, b]: [Vector<$f32, 2>; 2]) -> $x4 {
                swizzle!(a.storage.load(), b.storage.load(), @concat)
            }
            #[inline(always)]
            pub(crate) fn _3x2([a, b]: [Vector<$f32, 3>; 2]) -> [$x4; 2] { [a.storage, b.storage] }
            #[inline(always)]
            pub(crate) fn _4x2([a, b]: [Vector<$f32, 4>; 2]) -> [$x4; 2] { [a.storage, b.storage] }
            #[inline(always)]
            pub(crate) fn _1x3([a, b, c]: [Vector<$f32, 1>; 3]) -> $x4 {
                <$x4>::new([a.storage, b.storage, c.storage, $zero])
            }
            #[inline(always)]
            pub(crate) fn _2x3([a, b, c]: [Vector<$f32, 2>; 3]) -> [$x4; 2] {
                [_2x2([a, b]), c.storage.load().widen()]
            }
            #[inline(always)]
            pub(crate) fn _3x3([a, b, c]: [Vector<$f32, 3>; 3]) -> [$x4; 3] {
                [a.storage, b.storage, c.storage]
            }
            #[inline(always)]
            pub(crate) fn _4x3([a, b, c]: [Vector<$f32, 4>; 3]) -> [$x4; 3] {
                [a.storage, b.storage, c.storage]
            }
            #[inline(always)]
            pub(crate) fn _1x4([a, b, c, d]: [Vector<$f32, 1>; 4]) -> $x4 {
                <$x4>::new([a.storage, b.storage, c.storage, d.storage])
            }
            #[inline(always)]
            pub(crate) fn _2x4([a, b, c, d]: [Vector<$f32, 2>; 4]) -> [$x4; 2] {
                [_2x2([a, b]), _2x2([c, d])]
            }
            #[inline(always)]
            pub(crate) fn _3x4([a, b, c, d]: [Vector<$f32, 3>; 4]) -> [$x4; 4] {
                [a.storage, b.storage, c.storage, d.storage]
            }
            #[inline(always)]
            pub(crate) fn _4x4([a, b, c, d]: [Vector<$f32, 4>; 4]) -> [$x4; 4] {
                [a.storage, b.storage, c.storage, d.storage]
            }
        };
    }

    use crate::{
        Vector,
        simd::utils::{Simd2Ext, f32x2, i32x2, swizzle, u32x2},
        utils::Load,
    };
    use wide::{f32x4, f64x2, f64x4, i32x4, i64x2, i64x4, u32x4, u64x2, u64x4};
    pub(crate) mod f32 {
        use super::*;
        impl_fns!(f32, f32x2, f32x4, 0.);
    }
    pub(crate) mod f64 {
        use super::*;
        impl_fns!(f64, f64x2, f64x4, 0.);
    }
    pub(crate) mod i32 {
        use super::*;
        impl_fns!(i32, i32x2, i32x4, 0);
    }
    pub(crate) mod i64 {
        use super::*;
        impl_fns!(i64, i64x2, i64x4, 0);
    }
    pub(crate) mod u32 {
        use super::*;
        impl_fns!(u32, u32x2, u32x4, 0);
    }
    pub(crate) mod u64 {
        use super::*;
        impl_fns!(u64, u64x2, u64x4, 0);
    }
}

pub(crate) mod cast {
    #[allow(unused_imports)]
    use crate::arch::*;
    use crate::simd::utils::{f32x2, i32x2, u32x2};
    #[allow(unused_imports)]
    use wide::{bytemuck::cast, f32x4, i16x8, i32x4, u8x16, u32x4};
    use wide::{f64x2, f64x4, i64x2, i64x4, u64x2, u64x4};

    pub(crate) use core::convert::{
        identity as f32x4_from_f32,
        identity as i32x4_from_i32,
        identity as u32x4_from_u32,
        identity as f64x4_from_f64,
        identity as i64x4_from_i64,
        identity as u64x4_from_u64,
        identity as f64x2_from_f64,
        identity as i64x2_from_i64,
        identity as u64x2_from_u64,
    };

    #[inline(always)]
    pub(crate) fn f32x4_from_f64(v: f64x4) -> f32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f32x4_from_i32(v: i32x4) -> f32x4 { f32x4::from_i32x4(v) }
    #[inline(always)]
    pub(crate) fn f32x4_from_i64(v: i64x4) -> f32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f32x4_from_u32(v: u32x4) -> f32x4 {
        cfg_select! {
            target_feature = "avx512f" => unsafe {
                let input: x86_64::__m128i = v.into();
                let extended = avx512_f::_mm512_zextsi128_si512(input);
                let converted = avx512_f::_mm512_cvtepu32_ps(extended);
                let low = avx512_f::_mm512_castps512_ps128(converted);
                f32x4::from(low)
            },
            _ => {
                const MASK: i16x8 = i16x8::new([-1, 0, -1, 0, -1, 0, -1, 0]);
                let low =
                    MASK.select(cast::<_, i16x8>(v), cast::<_, i16x8>(u32x4::splat(0x4b00_0000)));
                let high = MASK
                    .select(cast::<_, i16x8>(v >> 16), cast::<_, i16x8>(u32x4::splat(0x5300_0000)));
                cast::<_, f32x4>(low) + (cast::<_, f32x4>(high) - 549_764_200_000.0_f32)
            }
        }
    }
    #[inline(always)]
    pub(crate) fn f32x4_from_u64(v: u64x4) -> f32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i32x4_from_f32(v: f32x4) -> i32x4 { f32x4::trunc_int(v) }
    #[inline(always)]
    pub(crate) fn i32x4_from_f64(v: f64x4) -> i32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i32x4_from_i64(v: i64x4) -> i32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i32x4_from_u32(v: u32x4) -> i32x4 { v.cast_signed() }
    #[inline(always)]
    pub(crate) fn i32x4_from_u64(v: u64x4) -> i32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u32x4_from_f32(v: f32x4) -> u32x4 {
        cfg_select! {
            target_feature = "avx512vl" => unsafe {
                use core::arch::x86_64::{_CMP_GE_OQ, _CMP_GT_OQ};
                let input: x86_64::__m128 = input.into();
                let converted = avx512_vl::_mm_cvttps_epu32(v);
                let overflow = avx512_vl::_mm_cmp_ps_mask::<_CMP_GT_OQ>(
                    v,
                    sse::_mm_set1_ps(f32::from_bits(0x4f7f_ffff)),
                );
                let nonnegative =
                    avx512_vl::_mm_cmp_ps_mask::<_CMP_GE_OQ>(v, sse::_mm_setzero_ps());
                let saturated =
                    avx512_vl::_mm_mask_mov_epi32(converted, overflow, sse2::_mm_set1_epi32(-1));
                u32x4::from(avx512_vl::_mm_maskz_mov_epi32(nonnegative, saturated))
            },
            _ => {
                let nonnegative = v & v.simd_ge(f32x4::splat(0.));
                let two31 = f32x4::splat(2_147_483_648.);
                let high = nonnegative.simd_ge(two31);
                let adjusted = nonnegative - (high & two31);
                let converted = adjusted.fast_trunc_int()
                    ^ (high.to_bits().cast_signed() & i32x4::splat(i32::MIN));
                let overflow = nonnegative.simd_ge(f32x4::splat(4_294_967_296.));
                converted.cast_unsigned() | overflow.to_bits()
            }
        }
    }
    #[inline(always)]
    pub(crate) fn u32x4_from_f64(v: f64x4) -> u32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u32x4_from_i32(v: i32x4) -> u32x4 { v.cast_unsigned() }
    #[inline(always)]
    pub(crate) fn u32x4_from_i64(v: i64x4) -> u32x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u32x4_from_u64(v: u64x4) -> u32x4 { todo!() }

    #[inline(always)]
    pub(crate) fn f64x4_from_f32(v: f32x4) -> f64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x2_from_f32(v: f32x2) -> f64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x4_from_i32(v: i32x4) -> f64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x2_from_i32(v: i32x2) -> f64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x4_from_i64(v: i64x4) -> f64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x2_from_i64(v: i64x2) -> f64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x4_from_u32(v: u32x4) -> f64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x2_from_u32(v: u32x2) -> f64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x4_from_u64(v: u64x4) -> f64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn f64x2_from_u64(v: u64x2) -> f64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x4_from_f32(v: f32x4) -> i64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x2_from_f32(v: f32x2) -> i64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x4_from_f64(v: f64x4) -> i64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x2_from_f64(v: f64x2) -> i64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x4_from_i32(v: i32x4) -> i64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x2_from_i32(v: i32x2) -> i64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x4_from_u32(v: u32x4) -> i64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x2_from_u32(v: u32x2) -> i64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x4_from_u64(v: u64x4) -> i64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn i64x2_from_u64(v: u64x2) -> i64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x4_from_f32(v: f32x4) -> u64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x2_from_f32(v: f32x2) -> u64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x4_from_f64(v: f64x4) -> u64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x2_from_f64(v: f64x2) -> u64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x4_from_i32(v: i32x4) -> u64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x2_from_i32(v: i32x2) -> u64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x4_from_i64(v: i64x4) -> u64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x2_from_i64(v: i64x2) -> u64x2 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x4_from_u32(v: u32x4) -> u64x4 { todo!() }
    #[inline(always)]
    pub(crate) fn u64x2_from_u32(v: u32x2) -> u64x2 { todo!() }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn u8_from_i32(mask: i32x4) -> u8x16 {
        let low_bytes = cast::<_, i16x8>(mask & i32x4::splat(0xff));
        let words = cast::<_, i16x8>(u8x16::narrow_i16x8(low_bytes, low_bytes));
        u8x16::narrow_i16x8(words, words)
    }
}

pub(crate) mod round {
    use wide::{f32x4, f64x2, f64x4};

    pub(crate) fn f64x4_round_ties_even(v: f64x4) -> f64x4 { todo!() }
    pub(crate) fn f64x2_round_ties_even(v: f64x2) -> f64x2 { todo!() }

    // Rust semantics do not fix the NaN payload or quiet bit.
    #[inline(always)]
    pub(crate) fn f32x4_round_ties_even(x: f32x4) -> f32x4 {
        let rounded = x.round_ties_even();
        cfg_select! {
            target_feature = "sse2" => {
                // Restore the sign of negative zero lost by the SSE2 integer round trip.
                let sign = x & f32x4::splat(f32::from_bits(0x8000_0000));
                rounded | sign
            }
            _ => rounded,
        }
    }
}

// Kept but disabled: porting this to aarch64's `i32x2` is nontrivial, and `u64::select` is
// currently private and unreferenced.
#[cfg(false)]
pub(crate) mod select {
    use wide::{f32x4, i32x4, u32x4};

    #[expect(dead_code)]
    #[inline(always)]
    pub fn i32x4_f32x4(mask: i32x4, true_values: f32x4, false_values: f32x4) -> f32x4 {
        f32x4::from_bits(mask.cast_unsigned()).select(true_values, false_values)
    }
    #[expect(dead_code)]
    #[inline(always)]
    pub fn i32x4_i32x4(mask: i32x4, true_values: i32x4, false_values: i32x4) -> i32x4 {
        mask.select(true_values, false_values)
    }
    #[expect(dead_code)]
    #[inline(always)]
    pub fn i32x4_u32x4(mask: i32x4, true_values: u32x4, false_values: u32x4) -> u32x4 {
        mask.cast_unsigned().select(true_values, false_values)
    }
    #[inline(always)]
    pub fn u64_f32x4(mask: u64, true_values: f32x4, false_values: f32x4) -> f32x4 {
        f32x4::from_bits(u64_u32x4(mask, true_values.to_bits(), false_values.to_bits()))
    }
    #[inline(always)]
    pub fn u64_i32x4(mask: u64, true_values: i32x4, false_values: i32x4) -> i32x4 {
        const LANE_MASK: i32x4 = i32x4::new([0b0001, 0b0010, 0b0100, 0b1000]);
        let mask = (i32x4::splat(mask as i32) & LANE_MASK).simd_eq(LANE_MASK);
        mask.select(true_values, false_values)
    }
    #[inline(always)]
    pub fn u64_u32x4(mask: u64, true_values: u32x4, false_values: u32x4) -> u32x4 {
        const LANE_MASK: u32x4 = u32x4::new([0b0001, 0b0010, 0b0100, 0b1000]);
        let mask = (u32x4::splat(mask as u32) & LANE_MASK).simd_eq(LANE_MASK);
        mask.select(true_values, false_values)
    }
}

pub(crate) mod matmul {
    #![allow(unused_parens)]

    use super::{reduce, transpose};
    use crate::{
        simd::utils::{Simd2Ext, Simd4Ext, swizzle},
        utils::arith,
    };

    macro_rules! impl_matmul {
        ($scalar:ident, $vec2:ident, $vec4:ident) => {
            // Kernels in this module use the column-major storage contract.

            // TODO(codegen-optimization): Benchmark horizontal reductions on representative targets
            // and use `hadd` only where its latency and throughput improve the complete kernel.

            // ============================================================
            // matmul1xBxC
            // ============================================================

            #[inline(always)]
            pub(crate) fn matmul1x1x1(a: $scalar, b: $scalar) -> $scalar { a * b }

            #[inline(always)]
            pub(crate) fn matmul2x1x1(a: $vec2, b: $scalar) -> $vec2 { a * $vec2::splat(b) }

            #[inline(always)]
            pub(crate) fn matmul3x1x1(a: $vec4, b: $scalar) -> $vec4 { a * b }

            #[inline(always)]
            pub(crate) fn matmul4x1x1(a: $vec4, b: $scalar) -> $vec4 { a * b }

            #[inline(always)]
            pub(crate) fn matmul1x2x1(a: $vec2, b: $vec2) -> $scalar {
                reduce::sum::<$vec2, 2>(a * b)
            }

            #[inline(always)]
            pub(crate) fn matmul2x2x1(a: $vec4, b: $vec2) -> $vec2 {
                // b = [b0, b1, *, *], a (2x2 column-major packed) = [a00, a10, a01, a11]
                let xxyy = swizzle!(b, [0, 0, 1, 1]);
                let products = a * xxyy;
                let upper_products = swizzle!(products, [2, 3]);
                products.xy() + upper_products
            }

            #[inline(always)]
            pub(crate) fn matmul3x2x1(a: [$vec4; 2], b: $vec2) -> $vec4 { matmul4x2x1(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x2x1(a: [$vec4; 2], b: $vec2) -> $vec4 {
                let xxxx = swizzle!(b, [0, 0, 0, 0]);
                let yyyy = swizzle!(b, [1, 1, 1, 1]);
                arith!((a[0]) * xxxx + (a[1]) * yyyy)
            }

            #[inline(always)]
            pub(crate) fn matmul1x3x1(a: $vec4, b: $vec4) -> $scalar {
                // TODO(codegen-optimization): Compare two FMAs with two multiplies plus additions,
                // including lane-shuffle cost and targets without FMA; retain this path unless the
                // fused version wins consistently.
                reduce::sum::<$vec4, 3>(a * b)
            }

            #[inline(always)]
            pub(crate) fn matmul2x3x1(a: [$vec4; 2], b: $vec4) -> $vec2 {
                let xxyy = swizzle!(b, [0, 0, 1, 1]);
                let zz = swizzle!(b, [2, 2]);
                let products01 = a[0] * xxyy;
                let upper_products01 = swizzle!(products01, [2, 3]);
                arith!((products01.xy() + upper_products01) + (a[1].xy()) * zz)
            }

            #[inline(always)]
            pub(crate) fn matmul3x3x1(a: [$vec4; 3], b: $vec4) -> $vec4 { matmul4x3x1(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x3x1(a: [$vec4; 3], b: $vec4) -> $vec4 {
                let xxxx = swizzle!(b, [0, 0, 0, 0]);
                let yyyy = swizzle!(b, [1, 1, 1, 1]);
                let zzzz = swizzle!(b, [2, 2, 2, 2]);
                arith!((a[0]) * xxxx + (a[1]) * yyyy + (a[2]) * zzzz)
            }

            #[inline(always)]
            pub(crate) fn matmul1x4x1(a: $vec4, b: $vec4) -> $scalar {
                reduce::sum::<$vec4, 4>(a * b)
            }

            #[inline(always)]
            pub(crate) fn matmul2x4x1(a: [$vec4; 2], b: $vec4) -> $vec2 {
                let xxyy = swizzle!(b, [0, 0, 1, 1]);
                let zzww = swizzle!(b, [2, 2, 3, 3]);
                let pair_sums = arith!((a[0]) * xxyy + (a[1]) * zzww);
                let upper_pair_sums = swizzle!(pair_sums, [2, 3]);
                pair_sums.xy() + upper_pair_sums
            }

            #[inline(always)]
            pub(crate) fn matmul3x4x1(a: [$vec4; 4], b: $vec4) -> $vec4 { matmul4x4x1(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x4x1(a: [$vec4; 4], b: $vec4) -> $vec4 {
                let xxxx = swizzle!(b, [0, 0, 0, 0]);
                let yyyy = swizzle!(b, [1, 1, 1, 1]);
                let zzzz = swizzle!(b, [2, 2, 2, 2]);
                let wwww = swizzle!(b, [3, 3, 3, 3]);
                arith!((a[0]) * xxxx + (a[1]) * yyyy + (a[2]) * zzzz + (a[3]) * wwww)
            }

            // ============================================================
            // matmul2xBxC
            // ============================================================

            #[inline(always)]
            pub(crate) fn matmul1x1x2(a: $scalar, b: $vec2) -> $vec2 { $vec2::splat(a) * b }

            #[inline(always)]
            pub(crate) fn matmul2x1x2(a: $vec2, b: $vec2) -> $vec4 {
                // Outer product in 2x2 packed column-major order:
                // `[a0*b0, a1*b0, a0*b1, a1*b1]`.
                swizzle!(a, [0, 1, 0, 1]) * swizzle!(b, [0, 0, 1, 1])
            }

            #[inline(always)]
            pub(crate) fn matmul3x1x2(a: $vec4, b: $vec2) -> [$vec4; 2] { matmul4x1x2(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x1x2(a: $vec4, b: $vec2) -> [$vec4; 2] {
                let col0 = a * swizzle!(b, [0, 0, 0, 0]);
                let col1 = a * swizzle!(b, [1, 1, 1, 1]);
                [col0, col1]
            }

            #[inline(always)]
            pub(crate) fn matmul1x2x2(a: $vec2, b: $vec4) -> $vec2 {
                let products = swizzle!(a, [0, 1, 0, 1]) * b;
                swizzle!(products, [0, 2]) + swizzle!(products, [1, 3])
            }

            #[inline(always)]
            pub(crate) fn matmul2x2x2(a: $vec4, b: $vec4) -> $vec4 {
                arith!(
                    (swizzle!(a, [0, 3, 0, 3])) * b
                        + (swizzle!(a, [2, 1, 2, 1])) * (swizzle!(b, [1, 0, 3, 2]))
                )
            }

            #[inline(always)]
            pub(crate) fn matmul3x2x2(a: [$vec4; 2], b: $vec4) -> [$vec4; 2] { matmul4x2x2(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x2x2(a: [$vec4; 2], b: $vec4) -> [$vec4; 2] {
                let xxxx = swizzle!(b, [0, 0, 0, 0]);
                let yyyy = swizzle!(b, [1, 1, 1, 1]);
                let zzzz = swizzle!(b, [2, 2, 2, 2]);
                let wwww = swizzle!(b, [3, 3, 3, 3]);
                let col0 = arith!((a[0]) * xxxx + (a[1]) * yyyy);
                let col1 = arith!((a[0]) * zzzz + (a[1]) * wwww);
                [col0, col1]
            }

            #[inline(always)]
            pub(crate) fn matmul1x3x2(a: $vec4, b: [$vec4; 2]) -> $vec2 {
                // TODO(codegen-optimization): Compare this path with a transpose-and-FMA chain on
                // representative FMA and non-FMA targets before changing the kernel.
                let cols01_lo = swizzle!(b[0], b[1], [0, 4, 1, 5]);
                let cols01_hi = swizzle!(b[0], b[1], [2, 6]);
                let products01 = cols01_lo * swizzle!(a, [0, 0, 1, 1]);
                let sums01 = products01.xy() + swizzle!(products01, [2, 3]);
                arith!(sums01 + (swizzle!(a, [2, 2])) * cols01_hi)
            }

            #[inline(always)]
            pub(crate) fn matmul2x3x2(a: [$vec4; 2], b: [$vec4; 2]) -> $vec4 {
                // Packed 2x2 column-major output: `[r0c0, r1c0, r0c1, r1c1]`.
                let col0 = matmul2x3x1(a, b[0]);
                let col1 = matmul2x3x1(a, b[1]);
                swizzle!(col0, col1, @concat)
            }

            #[inline(always)]
            pub(crate) fn matmul3x3x2(a: [$vec4; 3], b: [$vec4; 2]) -> [$vec4; 2] {
                // TODO(codegen-optimization): Specialize this shape only if assembly or benchmarks
                // outperform delegation to the wider kernel on representative targets.
                matmul4x3x2(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x3x2(a: [$vec4; 3], b: [$vec4; 2]) -> [$vec4; 2] {
                let col0 = matmul4x3x1(a, b[0]);
                let col1 = matmul4x3x1(a, b[1]);
                [col0, col1]
            }

            #[inline(always)]
            pub(crate) fn matmul1x4x2(a: $vec4, b: [$vec4; 2]) -> $vec2 {
                let scaled_col0 = a * b[0];
                let scaled_col1 = a * b[1];
                let pair_sums = swizzle!(scaled_col0, scaled_col1, [0, 4, 1, 5])
                    + swizzle!(scaled_col0, scaled_col1, [2, 6, 3, 7]);
                pair_sums.xy() + swizzle!(pair_sums, [2, 3])
            }

            #[inline(always)]
            pub(crate) fn matmul2x4x2(a: [$vec4; 2], b: [$vec4; 2]) -> $vec4 {
                let col0 = matmul2x4x1(a, b[0]);
                let col1 = matmul2x4x1(a, b[1]);
                swizzle!(col0, col1, @concat)
            }

            #[inline(always)]
            pub(crate) fn matmul3x4x2(a: [$vec4; 4], b: [$vec4; 2]) -> [$vec4; 2] {
                matmul4x4x2(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x4x2(a: [$vec4; 4], b: [$vec4; 2]) -> [$vec4; 2] {
                let col0 = matmul4x4x1(a, b[0]);
                let col1 = matmul4x4x1(a, b[1]);
                [col0, col1]
            }

            // ============================================================
            // matmul3xBxC
            // ============================================================

            #[inline(always)]
            pub(crate) fn matmul1x1x3(a: $scalar, b: $vec4) -> $vec4 { $vec4::splat(a) * b }

            #[inline(always)]
            pub(crate) fn matmul2x1x3(a: $vec2, b: $vec4) -> [$vec4; 2] {
                // a = [a0, a1, *, *], b = [b0, b1, b2, *]
                // Packed output: `cols01 = [a0*b0, a1*b0, a0*b1, a1*b1]` and
                // `col2 = [a0*b2, a1*b2, *, *]`.
                let xyxy = swizzle!(a, [0, 1, 0, 1]);
                let xy__ = swizzle!(a, [0, 1, _, _]);
                let xxyy = swizzle!(b, [0, 0, 1, 1]);
                let zz__ = swizzle!(b, [2, 2, _, _]);
                [xyxy * xxyy, xy__ * zz__]
            }

            #[inline(always)]
            pub(crate) fn matmul3x1x3(a: $vec4, b: $vec4) -> [$vec4; 3] { matmul4x1x3(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x1x3(a: $vec4, b: $vec4) -> [$vec4; 3] {
                let col0 = a * swizzle!(b, [0, 0, 0, 0]);
                let col1 = a * swizzle!(b, [1, 1, 1, 1]);
                let col2 = a * swizzle!(b, [2, 2, 2, 2]);
                [col0, col1, col2]
            }

            #[inline(always)]
            pub(crate) fn matmul1x2x3(a: $vec2, b: [$vec4; 2]) -> $vec4 {
                // TODO(codegen-optimization): Compare this path with a single-horizontal-add packed
                // formulation, and adopt it only when complete-kernel benchmarks improve.

                let col0 = swizzle!(b[0], b[1], [0, 2, 4, _]);
                let col1 = swizzle!(b[0], b[1], [1, 3, 5, _]);
                let xxx_ = swizzle!(a, [0, 0, 0, _]);
                let yyy_ = swizzle!(a, [1, 1, 1, _]);
                arith!(xxx_ * col0 + yyy_ * col1)
            }

            #[inline(always)]
            pub(crate) fn matmul2x2x3(a: $vec4, b: [$vec4; 2]) -> [$vec4; 2] {
                // b[0] = [b00, b10, b01, b11], b[1] = [b02, b12, *, *]
                // `cols01` has the same first-two-column layout as `matmul2x2x2`; `col2` stores only
                // the final column.
                let xyxy = swizzle!(a, [0, 1, 0, 1]);
                let zwzw = swizzle!(a, [2, 3, 2, 3]);

                let xxzz0 = swizzle!(b[0], [0, 0, 2, 2]);
                let yyww0 = swizzle!(b[0], [1, 1, 3, 3]);
                let cols01 = arith!(xyxy * xxzz0 + zwzw * yyww0);

                let b1_xx = swizzle!(b[1], [0, 0, _, _]);
                let b1_yy = swizzle!(b[1], [1, 1, _, _]);
                let xy__ = swizzle!(a, [0, 1, _, _]);
                let zw__ = swizzle!(a, [2, 3, _, _]);
                let col2 = arith!(xy__ * b1_xx + zw__ * b1_yy);

                [cols01, col2]
            }

            #[inline(always)]
            pub(crate) fn matmul3x2x3(a: [$vec4; 2], b: [$vec4; 2]) -> [$vec4; 3] {
                // b[0] = [b00, b10, b01, b11] (columns 0,1 packed)
                // b[1] = [b02, b12, *, *] (column 2)
                let b0_xxxx = swizzle!(b[0], [0, 0, 0, 0]); // b00
                let b0_yyyy = swizzle!(b[0], [1, 1, 1, 1]); // b10
                let b0_zzzz = swizzle!(b[0], [2, 2, 2, 2]); // b01
                let b0_wwww = swizzle!(b[0], [3, 3, 3, 3]); // b11
                let col0 = arith!((a[0]) * b0_xxxx + (a[1]) * b0_yyyy);
                let col1 = arith!((a[0]) * b0_zzzz + (a[1]) * b0_wwww);

                let b1_xxxx = swizzle!(b[1], [0, 0, 0, 0]); // b02
                let b1_yyyy = swizzle!(b[1], [1, 1, 1, 1]); // b12
                let col2 = arith!((a[0]) * b1_xxxx + (a[1]) * b1_yyyy);

                [col0, col1, col2]
            }

            #[inline(always)]
            pub(crate) fn matmul4x2x3(a: [$vec4; 2], b: [$vec4; 2]) -> [$vec4; 3] {
                let b0_xxxx = swizzle!(b[0], [0, 0, 0, 0]);
                let b0_yyyy = swizzle!(b[0], [1, 1, 1, 1]);
                let b0_zzzz = swizzle!(b[0], [2, 2, 2, 2]);
                let b0_wwww = swizzle!(b[0], [3, 3, 3, 3]);
                let col0 = arith!((a[0]) * b0_xxxx + (a[1]) * b0_yyyy);
                let col1 = arith!((a[0]) * b0_zzzz + (a[1]) * b0_wwww);

                let b1_xxxx = swizzle!(b[1], [0, 0, 0, 0]);
                let b1_yyyy = swizzle!(b[1], [1, 1, 1, 1]);
                let col2 = arith!((a[0]) * b1_xxxx + (a[1]) * b1_yyyy);

                [col0, col1, col2]
            }

            #[inline(always)]
            pub(crate) fn matmul1x3x3(a: $vec4, b: [$vec4; 3]) -> $vec4 {
                // TODO(codegen-optimization): Compare this path with transposed columns and a
                // lane-splat FMA chain on representative targets before changing the kernel.
                let [coeff_x, coeff_y, coeff_z] = transpose::transpose3x3(b);
                let xxx_ = swizzle!(a, [0, 0, 0, _]);
                let yyy_ = swizzle!(a, [1, 1, 1, _]);
                let zzz_ = swizzle!(a, [2, 2, 2, _]);

                arith!(xxx_ * coeff_x + yyy_ * coeff_y + zzz_ * coeff_z)
            }

            #[inline(always)]
            pub(crate) fn matmul2x3x3(a: [$vec4; 2], b: [$vec4; 3]) -> [$vec4; 2] {
                let col0 = matmul2x3x1(a, b[0]);
                let col1 = matmul2x3x1(a, b[1]);
                let col2 = matmul2x3x1(a, b[2]);
                [swizzle!(col0, col1, @concat), col2.widen()]
            }

            #[inline(always)]
            pub(crate) fn matmul3x3x3(a: [$vec4; 3], b: [$vec4; 3]) -> [$vec4; 3] {
                // TODO(codegen-optimization): Specialize this shape only if assembly or benchmarks
                // outperform delegation to the wider kernel on representative targets.
                matmul4x3x3(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x3x3(a: [$vec4; 3], b: [$vec4; 3]) -> [$vec4; 3] {
                let col0 = matmul4x3x1(a, b[0]);
                let col1 = matmul4x3x1(a, b[1]);
                let col2 = matmul4x3x1(a, b[2]);
                [col0, col1, col2]
            }

            #[inline(always)]
            pub(crate) fn matmul1x4x3(a: $vec4, b: [$vec4; 3]) -> $vec4 {
                // TODO(codegen-optimization): Compare this path with an explicit transpose and
                // column-splat FMA chain, and adopt it only when complete-kernel benchmarks improve.

                let transposed = transpose::transpose4x3(b);
                matmul3x4x1(transposed, a)
            }

            #[inline(always)]
            pub(crate) fn matmul2x4x3(a: [$vec4; 2], b: [$vec4; 3]) -> [$vec4; 2] {
                let col0 = matmul2x4x1(a, b[0]);
                let col1 = matmul2x4x1(a, b[1]);
                let col2 = matmul2x4x1(a, b[2]);
                [swizzle!(col0, col1, @concat), col2.widen()]
            }

            #[inline(always)]
            pub(crate) fn matmul3x4x3(a: [$vec4; 4], b: [$vec4; 3]) -> [$vec4; 3] {
                matmul4x4x3(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x4x3(a: [$vec4; 4], b: [$vec4; 3]) -> [$vec4; 3] {
                let col0 = matmul4x4x1(a, b[0]);
                let col1 = matmul4x4x1(a, b[1]);
                let col2 = matmul4x4x1(a, b[2]);
                [col0, col1, col2]
            }

            // ============================================================
            // matmul4xBxC
            // ============================================================

            #[inline(always)]
            pub(crate) fn matmul1x1x4(a: $scalar, b: $vec4) -> $vec4 { $vec4::splat(a) * b }

            #[inline(always)]
            pub(crate) fn matmul2x1x4(a: $vec2, b: $vec4) -> [$vec4; 2] {
                // a = [a0, a1, *, *], b = [b0, b1, b2, b3]
                let xyxy = swizzle!(a, [0, 1, 0, 1]);
                let xxyy = swizzle!(b, [0, 0, 1, 1]);
                let zzww = swizzle!(b, [2, 2, 3, 3]);
                [xyxy * xxyy, xyxy * zzww]
            }

            #[inline(always)]
            pub(crate) fn matmul3x1x4(a: $vec4, b: $vec4) -> [$vec4; 4] { matmul4x1x4(a, b) }

            #[inline(always)]
            pub(crate) fn matmul4x1x4(a: $vec4, b: $vec4) -> [$vec4; 4] {
                let col0 = a * swizzle!(b, [0, 0, 0, 0]);
                let col1 = a * swizzle!(b, [1, 1, 1, 1]);
                let col2 = a * swizzle!(b, [2, 2, 2, 2]);
                let col3 = a * swizzle!(b, [3, 3, 3, 3]);
                [col0, col1, col2, col3]
            }

            #[inline(always)]
            pub(crate) fn matmul1x2x4(a: $vec2, b: [$vec4; 2]) -> $vec4 {
                let xyxy = swizzle!(a, [0, 1, 0, 1]);
                let scaled_cols01 = b[0] * xyxy;
                let scaled_cols23 = b[1] * xyxy;
                let even_products = swizzle!(scaled_cols01, scaled_cols23, [0, 2, 4, 6]);
                let odd_products = swizzle!(scaled_cols01, scaled_cols23, [1, 3, 5, 7]);
                even_products + odd_products
            }

            #[inline(always)]
            pub(crate) fn matmul2x2x4(a: $vec4, b: [$vec4; 2]) -> [$vec4; 2] {
                // Each of `b[0]` and `b[1]` stores two columns in the packed 2x2 layout. Apply the
                // `matmul2x2x2` pattern independently to each value.
                let xyxy = swizzle!(a, [0, 1, 0, 1]);
                let zwzw = swizzle!(a, [2, 3, 2, 3]);

                let xxzz0 = swizzle!(b[0], [0, 0, 2, 2]);
                let yyww0 = swizzle!(b[0], [1, 1, 3, 3]);
                let cols01 = arith!(xyxy * xxzz0 + zwzw * yyww0);

                let xxzz1 = swizzle!(b[1], [0, 0, 2, 2]);
                let yyww1 = swizzle!(b[1], [1, 1, 3, 3]);
                let cols23 = arith!(xyxy * xxzz1 + zwzw * yyww1);

                [cols01, cols23]
            }

            #[inline(always)]
            pub(crate) fn matmul3x2x4(a: [$vec4; 2], b: [$vec4; 2]) -> [$vec4; 4] {
                // b[0] = [b00, b10, b01, b11], b[1] = [b02, b12, b03, b13]
                let b0_xxxx = swizzle!(b[0], [0, 0, 0, 0]);
                let b0_yyyy = swizzle!(b[0], [1, 1, 1, 1]);
                let b0_zzzz = swizzle!(b[0], [2, 2, 2, 2]);
                let b0_wwww = swizzle!(b[0], [3, 3, 3, 3]);
                let col0 = arith!((a[0]) * b0_xxxx + (a[1]) * b0_yyyy);
                let col1 = arith!((a[0]) * b0_zzzz + (a[1]) * b0_wwww);

                let b1_xxxx = swizzle!(b[1], [0, 0, 0, 0]);
                let b1_yyyy = swizzle!(b[1], [1, 1, 1, 1]);
                let b1_zzzz = swizzle!(b[1], [2, 2, 2, 2]);
                let b1_wwww = swizzle!(b[1], [3, 3, 3, 3]);
                let col2 = arith!((a[0]) * b1_xxxx + (a[1]) * b1_yyyy);
                let col3 = arith!((a[0]) * b1_zzzz + (a[1]) * b1_wwww);

                [col0, col1, col2, col3]
            }

            #[inline(always)]
            pub(crate) fn matmul4x2x4(a: [$vec4; 2], b: [$vec4; 2]) -> [$vec4; 4] {
                let b0_xxxx = swizzle!(b[0], [0, 0, 0, 0]);
                let b0_yyyy = swizzle!(b[0], [1, 1, 1, 1]);
                let b0_zzzz = swizzle!(b[0], [2, 2, 2, 2]);
                let b0_wwww = swizzle!(b[0], [3, 3, 3, 3]);
                let col0 = arith!((a[0]) * b0_xxxx + (a[1]) * b0_yyyy);
                let col1 = arith!((a[0]) * b0_zzzz + (a[1]) * b0_wwww);

                let b1_xxxx = swizzle!(b[1], [0, 0, 0, 0]);
                let b1_yyyy = swizzle!(b[1], [1, 1, 1, 1]);
                let b1_zzzz = swizzle!(b[1], [2, 2, 2, 2]);
                let b1_wwww = swizzle!(b[1], [3, 3, 3, 3]);
                let col2 = arith!((a[0]) * b1_xxxx + (a[1]) * b1_yyyy);
                let col3 = arith!((a[0]) * b1_zzzz + (a[1]) * b1_wwww);

                [col0, col1, col2, col3]
            }

            #[inline(always)]
            pub(crate) fn matmul1x3x4(a: $vec4, b: [$vec4; 4]) -> $vec4 {
                // TODO(codegen-optimization): Compare LLVM's current structure-of-arrays lowering with
                // an explicit `$vec4` transpose, and change it only with assembly or benchmark evidence.

                let [coeff_x, coeff_y, coeff_z] = transpose::transpose3x4(b);
                let xxx_ = swizzle!(a, [0, 0, 0, _]);
                let yyy_ = swizzle!(a, [1, 1, 1, _]);
                let zzz_ = swizzle!(a, [2, 2, 2, _]);
                arith!(coeff_x * xxx_ + coeff_y * yyy_ + coeff_z * zzz_)
            }

            #[inline(always)]
            pub(crate) fn matmul2x3x4(a: [$vec4; 2], b: [$vec4; 4]) -> [$vec4; 2] {
                let col0 = matmul2x3x1(a, b[0]);
                let col1 = matmul2x3x1(a, b[1]);
                let col2 = matmul2x3x1(a, b[2]);
                let col3 = matmul2x3x1(a, b[3]);
                [
                    swizzle!(col0, col1, @concat),
                    swizzle!(col2, col3, @concat),
                ]
            }

            #[inline(always)]
            pub(crate) fn matmul3x3x4(a: [$vec4; 3], b: [$vec4; 4]) -> [$vec4; 4] {
                // TODO(codegen-optimization): Specialize this shape only if assembly or benchmarks
                // outperform delegation to the wider kernel on representative targets.
                matmul4x3x4(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x3x4(a: [$vec4; 3], b: [$vec4; 4]) -> [$vec4; 4] {
                let col0 = matmul4x3x1(a, b[0]);
                let col1 = matmul4x3x1(a, b[1]);
                let col2 = matmul4x3x1(a, b[2]);
                let col3 = matmul4x3x1(a, b[3]);
                [col0, col1, col2, col3]
            }

            #[inline(always)]
            pub(crate) fn matmul1x4x4(a: $vec4, b: [$vec4; 4]) -> $vec4 {
                let transposed = transpose::transpose4x4(b);
                matmul4x4x1(transposed, a)
            }

            #[inline(always)]
            pub(crate) fn matmul2x4x4(a: [$vec4; 2], b: [$vec4; 4]) -> [$vec4; 2] {
                let col0 = matmul2x4x1(a, b[0]);
                let col1 = matmul2x4x1(a, b[1]);
                let col2 = matmul2x4x1(a, b[2]);
                let col3 = matmul2x4x1(a, b[3]);
                [
                    swizzle!(col0, col1, @concat),
                    swizzle!(col2, col3, @concat),
                ]
            }

            #[inline(always)]
            pub(crate) fn matmul3x4x4(a: [$vec4; 4], b: [$vec4; 4]) -> [$vec4; 4] {
                matmul4x4x4(a, b)
            }

            #[inline(always)]
            pub(crate) fn matmul4x4x4(a: [$vec4; 4], b: [$vec4; 4]) -> [$vec4; 4] {
                let col0 = matmul4x4x1(a, b[0]);
                let col1 = matmul4x4x1(a, b[1]);
                let col2 = matmul4x4x1(a, b[2]);
                let col3 = matmul4x4x1(a, b[3]);
                [col0, col1, col2, col3]
            }
        };
    }

    pub(crate) mod f32 {
        use super::{super::super::utils::compute_f32x2, *};
        use wide::f32x4;
        impl_matmul!(f32, compute_f32x2, f32x4);
    }
    pub(crate) mod f64 {
        use super::*;
        use wide::{f64x2, f64x4};
        impl_matmul!(f64, f64x2, f64x4);
    }
}
