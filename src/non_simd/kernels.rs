pub(crate) trait Float:
    crate::marker::Float + crate::utils::ArithPrimitive<Scalar = Self>
{
}

impl Float for f32 {}
impl Float for f64 {}

pub(crate) mod reduce {

    #[inline(always)]
    pub(crate) fn sum<T: Copy + core::ops::Add<Output = T>, const N: usize>(v: [T; N]) -> T {
        match N {
            1 => v[0],
            2 => v[0] + v[1],
            3 => v[0] + v[1] + v[2],
            // Preserve this addition tree to avoid the usual `hadd` latency and throughput cost;
            // revisit it only with representative benchmark or codegen evidence.
            4 => (v[0] + v[2]) + (v[1] + v[3]),
            _ => unimplemented!(),
        }
    }
}

#[inline(always)]
pub(crate) fn diagonal<T: Copy, const N: usize>(a: [[T; N]; N]) -> [[T; N]; 1] {
    [core::array::from_fn(
        #[inline(always)]
        |i| a[i][i],
    )]
}

#[inline(always)]
pub(crate) fn transpose<T: Copy, const M: usize, const N: usize>(a: [[T; M]; N]) -> [[T; N]; M] {
    core::array::from_fn(
        #[inline(always)]
        |i| {
            core::array::from_fn(
                #[inline(always)]
                |j| a[j][i],
            )
        },
    )
}

#[inline(always)]
fn permute<T: Copy>(a: [T; 4], [i0, i1, i2, i3]: [usize; 4]) -> [T; 4] {
    [a[i0], a[i1], a[i2], a[i3]]
}

#[inline(always)]
fn permute2<T: Copy>(a: [T; 4], b: [T; 4], indices: [usize; 4]) -> [T; 4] {
    let ab = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
    [ab[indices[0]], ab[indices[1]], ab[indices[2]], ab[indices[3]]]
}

#[inline(always)]
fn add<T: Float, const N: usize>(a: [T; N], b: [T; N]) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| a[i] + b[i],
    )
}

#[inline(always)]
fn mul<T: Float, const N: usize>(a: [T; N], b: [T; N]) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| a[i] * b[i],
    )
}

#[inline(always)]
fn div<T: Float, const N: usize>(a: [T; N], b: [T; N]) -> [T; N] {
    core::array::from_fn(
        #[inline(always)]
        |i| a[i] / b[i],
    )
}

pub(crate) mod inverse {
    #![allow(unused_parens)]

    use super::{Float, add, div, mul, permute, permute2};
    use crate::utils::arith;

    #[inline(always)]
    fn matmul2x2x2<T: Float>(a: [T; 4], b: [T; 4]) -> [T; 4] {
        arith!(
            (permute(a, [0, 3, 0, 3])) * b
                + (permute(a, [2, 1, 2, 1])) * (permute(b, [1, 0, 3, 2]))
        )
    }

    #[inline(always)]
    pub(crate) fn _1x1<T: Float>([[a]]: [[T; 1]; 1]) -> [[T; 1]; 1] { [[T::ONE_ / a]] }

    #[inline(always)]
    pub(crate) fn _2x2<T: Float>(a: [[T; 2]; 2]) -> [[T; 2]; 2] {
        // [a, b, c, d] -> [d, -b, -c, a] / (ad - bc)
        let det = super::determinant::_2x2(a);
        let [[a, b], [c, d]] = a;
        [[d / det, -b / det], [-c / det, a / det]]
    }

    #[inline(always)]
    pub(crate) fn _3x3<T: Float>(a: [[T; 3]; 3]) -> [[T; 3]; 3] {
        let [r0, r1, r2] = a;
        let r0_yzx = [r0[1], r0[2], r0[0]];
        let r1_yzx = [r1[1], r1[2], r1[0]];
        let r2_yzx = [r2[1], r2[2], r2[0]];
        // The zxy-ordered cross product forms a column of the adjugate.
        let c0 = arith!(r1 * r2_yzx - r1_yzx * r2);
        let c1 = arith!(r2 * r0_yzx - r2_yzx * r0);
        let c2 = arith!(r0 * r1_yzx - r0_yzx * r1);

        // det = dot(r0, cross(r1, r2))
        let det = (c0[0] * r0[2] + c0[1] * r0[0]) + c0[2] * r0[1];
        let r_det = T::ONE_ / det;
        let c0 = mul(c0, [r_det; 3]);
        let c1 = mul(c1, [r_det; 3]);
        let c2 = mul(c2, [r_det; 3]);

        [[c0[1], c1[1], c2[1]], [c0[2], c1[2], c2[2]], [c0[0], c1[0], c2[0]]]
    }

    #[inline(always)]
    pub(crate) fn _4x4<T: Float>(a: [[T; 4]; 4]) -> [[T; 4]; 4] {
        #[inline(always)]
        fn mat2_adj_mul<T: Float>(a: [T; 4], b: [T; 4]) -> [T; 4] {
            arith!(
                (permute(a, [3, 3, 0, 0])) * b
                    - (permute(a, [1, 1, 2, 2])) * (permute(b, [2, 3, 0, 1]))
            )
        }

        #[inline(always)]
        fn mat2_mul_adj<T: Float>(a: [T; 4], b: [T; 4]) -> [T; 4] {
            arith!(
                a * (permute(b, [3, 0, 3, 0]))
                    - (permute(a, [1, 0, 3, 2])) * (permute(b, [2, 1, 2, 1]))
            )
        }

        let a00 = permute2(a[0], a[1], [0, 1, 4, 5]);
        let b00 = permute2(a[0], a[1], [2, 3, 6, 7]);
        let c00 = permute2(a[2], a[3], [0, 1, 4, 5]);
        let d00 = permute2(a[2], a[3], [2, 3, 6, 7]);

        let det_sub = arith!(
            (permute2(a[0], a[2], [0, 2, 4, 6])) * (permute2(a[1], a[3], [1, 3, 5, 7]))
                - (permute2(a[0], a[2], [1, 3, 5, 7])) * (permute2(a[1], a[3], [0, 2, 4, 6]))
        );

        let det_a = permute(det_sub, [0, 0, 0, 0]);
        let det_b = permute(det_sub, [1, 1, 1, 1]);
        let det_c = permute(det_sub, [2, 2, 2, 2]);
        let det_d = permute(det_sub, [3, 3, 3, 3]);

        let d_adj_c = mat2_adj_mul(d00, c00);
        let a_adj_b = mat2_adj_mul(a00, b00);

        let x_ = arith!(det_d * a00 - (matmul2x2x2(d_adj_c, b00)));
        let w_ = arith!(det_a * d00 - (matmul2x2x2(a_adj_b, c00)));
        let y_ = arith!(det_b * c00 - (mat2_mul_adj(d00, a_adj_b)));
        let z_ = arith!(det_c * b00 - (mat2_mul_adj(a00, d_adj_c)));

        let tr_terms = mul(a_adj_b, permute(d_adj_c, [0, 2, 1, 3]));
        let tr_pair = add(tr_terms, permute(tr_terms, [1, 0, 3, 2]));
        let tr = add(tr_pair, permute(tr_pair, [2, 3, 0, 1]));

        let det_m = arith!(det_a * det_d + (arith!(det_b * det_c - tr)));

        let one = T::ONE_;
        let r_det = div([one, -one, -one, one], det_m);

        let x_ = mul(x_, r_det);
        let y_ = mul(y_, r_det);
        let z_ = mul(z_, r_det);
        let w_ = mul(w_, r_det);

        [
            permute2(x_, y_, [3, 1, 7, 5]),
            permute2(x_, y_, [2, 0, 6, 4]),
            permute2(z_, w_, [3, 1, 7, 5]),
            permute2(z_, w_, [2, 0, 6, 4]),
        ]
    }
}

pub(crate) mod determinant {
    #![allow(unused_parens)]

    use super::{Float, add, mul, permute, permute2};
    use crate::utils::arith;

    #[inline(always)]
    pub(crate) fn _2x2<T: Float>([a, b]: [[T; 2]; 2]) -> T {
        // TODO(codegen-optimization): review determinant codegen when FMA is unavailable.
        arith!((a[0]) * (b[1]) - (a[1]) * (b[0]))
    }

    #[inline(always)]
    pub(crate) fn _3x3<T: Float>(a: [[T; 3]; 3]) -> T {
        let [r0, r1, r2] = a;
        let r1_yzx = [r1[1], r1[2], r1[0]];
        let r2_yzx = [r2[1], r2[2], r2[0]];
        // zxy order cross product, matching the determinant path in inverse::_3x3.
        let c0 = arith!(r1 * r2_yzx - r1_yzx * r2);

        // det = dot(r0, cross(r1, r2))
        (c0[0] * r0[2] + c0[1] * r0[0]) + c0[2] * r0[1]
    }

    #[inline(always)]
    pub(crate) fn _4x4<T: Float>(a: [[T; 4]; 4]) -> T {
        // Leave lane/scalar optimization to LLVM. Preserve this operation and addition order as-is
        // so the determinant remains numerically compatible with the path in inverse::_4x4.
        #[inline(always)]
        fn mat2_adj_mul<T: Float>(a: [T; 4], b: [T; 4]) -> [T; 4] {
            arith!(
                (permute(a, [3, 3, 0, 0])) * b
                    - (permute(a, [1, 1, 2, 2])) * (permute(b, [2, 3, 0, 1]))
            )
        }

        let a00 = permute2(a[0], a[1], [0, 1, 4, 5]);
        let b00 = permute2(a[0], a[1], [2, 3, 6, 7]);
        let c00 = permute2(a[2], a[3], [0, 1, 4, 5]);
        let d00 = permute2(a[2], a[3], [2, 3, 6, 7]);

        let det_sub = arith!(
            (permute2(a[0], a[2], [0, 2, 4, 6])) * (permute2(a[1], a[3], [1, 3, 5, 7]))
                - (permute2(a[0], a[2], [1, 3, 5, 7])) * (permute2(a[1], a[3], [0, 2, 4, 6]))
        );

        let d_adj_c = mat2_adj_mul(d00, c00);
        let a_adj_b = mat2_adj_mul(a00, b00);
        let det_a = permute(det_sub, [0, 0, 0, 0]);
        let det_b = permute(det_sub, [1, 1, 1, 1]);
        let det_c = permute(det_sub, [2, 2, 2, 2]);
        let det_d = permute(det_sub, [3, 3, 3, 3]);

        let tr_terms = mul(a_adj_b, permute(d_adj_c, [0, 2, 1, 3]));
        let tr_pair = add(tr_terms, permute(tr_terms, [1, 0, 3, 2]));
        let tr = add(tr_pair, permute(tr_pair, [2, 3, 0, 1]));

        let det_m = arith!(det_a * det_d + (arith!(det_b * det_c - tr)));
        det_m[0]
    }
}

pub(crate) mod from_array {
    // Directly called from Vector::from_array.
    macro_rules! impl_fns {
        ($($mod:ident),+) => {
            $(
                pub(crate) mod $mod {
                    pub(crate) use core::convert::{
                        identity as _1x1,
                        identity as _1x2,
                        identity as _1x3,
                        identity as _1x4,
                        identity as _2x1,
                        identity as _2x2,
                        identity as _2x3,
                        identity as _2x4,
                        identity as _3x1,
                        identity as _3x2,
                        identity as _3x3,
                        identity as _3x4,
                        identity as _4x1,
                        identity as _4x2,
                        identity as _4x3,
                        identity as _4x4,
                    };
                }
            )+
        };
    }

    impl_fns!(f32, f64, i32, i64, u32, u64);
}

pub(crate) mod select {
    #[expect(dead_code)]
    #[inline(always)]
    pub fn select_u64<T: Copy, const N: usize>(
        mask: u64,
        true_values: [[T; N]; 1],
        false_values: [[T; N]; 1],
    ) -> [[T; N]; 1] {
        [core::array::from_fn(
            #[inline(always)]
            |i| {
                if (mask & (1 << i)) != 0 { true_values[0][i] } else { false_values[0][i] }
            },
        )]
    }
}

pub(crate) mod matmul {
    #![allow(unused_parens)]

    use super::Float;
    use crate::utils::arith;

    #[inline(always)]
    pub(crate) fn matmul<T: Float, const R: usize, const K: usize, const C: usize>(
        a: [[T; R]; K],
        b: [[T; K]; C],
    ) -> [[T; R]; C] {
        // Match the SIMD addition order where doing so has no performance cost, because a
        // different order can amplify numerical differences. Bit-identical results across
        // platforms are not guaranteed; FMA contraction and its rounding may still differ.
        match (R, K, C) {
            // x, x + y, (x + y) + z
            (_, 1..=3, _) => core::array::from_fn(
                #[inline(always)]
                |j| {
                    core::array::from_fn(
                        #[inline(always)]
                        |i| {
                            (1..K).fold(
                                a[0][i] * b[j][0],
                                #[inline(always)]
                                |acc, k| acc + a[k][i] * b[j][k],
                            )
                        },
                    )
                },
            ),
            // (x + z) + (y + w)
            (1, 4, 1..=4) => core::array::from_fn(
                #[inline(always)]
                |j| {
                    core::array::from_fn(
                        #[inline(always)]
                        |i| {
                            (a[0][i] * b[j][0] + a[2][i] * b[j][2])
                                + (a[1][i] * b[j][1] + a[3][i] * b[j][3])
                        },
                    )
                },
            ),
            (2, 4, 1..=4) => core::array::from_fn(
                #[inline(always)]
                |j| {
                    core::array::from_fn(
                        #[inline(always)]
                        |i| {
                            arith!((a[0][i]) * (b[j][0]) + (a[2][i]) * (b[j][2]))
                                + arith!((a[1][i]) * (b[j][1]) + (a[3][i]) * (b[j][3]))
                        },
                    )
                },
            ),
            (3 | 4, 4, 1..=4) => core::array::from_fn(
                #[inline(always)]
                |j| {
                    core::array::from_fn(
                        #[inline(always)]
                        |i| {
                            arith!(
                                (a[0][i]) * (b[j][0])
                                    + (a[1][i]) * (b[j][1])
                                    + (a[2][i]) * (b[j][2])
                                    + (a[3][i]) * (b[j][3])
                            )
                        },
                    )
                },
            ),
            _ => unimplemented!(),
        }
    }

    // These aliases are used by row_major and column_major modules
    macro_rules! impl_mat_mul_mat {
        ([$($a:literal),*]; $b:tt; $c:tt) => {
            $(impl_mat_mul_mat!(@a $a; $b; $c);)*
        };
        (@a $a:literal; [$($b:literal),*]; $c:tt) => {
            $(impl_mat_mul_mat!(@ab $a; $b; $c);)*
        };
        (@ab $a:literal; $b:literal; [$($c:literal),*]) => {
            $(paste::paste!(pub(crate) use super::matmul as [<matmul $a x $b x $c>];);)*
        };
    }

    pub(crate) mod f32 {
        impl_mat_mul_mat!([1, 2, 3, 4]; [1, 2, 3, 4]; [1, 2, 3, 4]);
    }

    pub(crate) mod f64 {
        impl_mat_mul_mat!([1, 2, 3, 4]; [1, 2, 3, 4]; [1, 2, 3, 4]);
    }
}
