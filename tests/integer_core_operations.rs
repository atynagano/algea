//! Tests for core integer vector operations.

use algea::{Vector, row_major::Matrix};

fn map<T: Copy, const N: usize>(a: [T; N], f: impl Fn(T) -> T) -> [T; N] {
    core::array::from_fn(|i| f(a[i]))
}

fn map_with<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> T) -> [T; N] {
    core::array::from_fn(|i| f(a[i], b[i]))
}

fn map_rows<T: Copy, const R: usize, const C: usize>(
    a: [[T; C]; R],
    f: impl Fn(T) -> T,
) -> [[T; C]; R] {
    core::array::from_fn(|r| core::array::from_fn(|c| f(a[r][c])))
}

fn map_rows_with<T: Copy, const R: usize, const C: usize>(
    a: [[T; C]; R],
    b: [[T; C]; R],
    f: impl Fn(T, T) -> T,
) -> [[T; C]; R] {
    core::array::from_fn(|r| core::array::from_fn(|c| f(a[r][c], b[r][c])))
}

macro_rules! vector_ops {
    ($name:ident, $t:ty, $d:literal, $a:expr, $b:expr, $s:expr) => {
        #[test]
        fn $name() {
            let aa: [$t; $d] = $a;
            let bb: [$t; $d] = $b;
            let s: $t = $s;
            let a = Vector::<$t, $d>::from(aa);
            let b = Vector::<$t, $d>::from(bb);

            assert_eq!(<[$t; $d]>::from(a + b), map_with(aa, bb, <$t>::wrapping_add));
            assert_eq!(<[$t; $d]>::from(a - b), map_with(aa, bb, <$t>::wrapping_sub));
            assert_eq!(<[$t; $d]>::from(a * b), map_with(aa, bb, <$t>::wrapping_mul));
            assert_eq!(<[$t; $d]>::from(a / b), map_with(aa, bb, <$t>::wrapping_div));

            assert_eq!(<[$t; $d]>::from(a + s), map(aa, |x| x.wrapping_add(s)));
            assert_eq!(<[$t; $d]>::from(a - s), map(aa, |x| x.wrapping_sub(s)));
            assert_eq!(<[$t; $d]>::from(a * s), map(aa, |x| x.wrapping_mul(s)));
            assert_eq!(<[$t; $d]>::from(a / s), map(aa, |x| x.wrapping_div(s)));
            assert_eq!(<[$t; $d]>::from(s + a), map(aa, |x| s.wrapping_add(x)));
            assert_eq!(<[$t; $d]>::from(s - a), map(aa, |x| s.wrapping_sub(x)));
            assert_eq!(<[$t; $d]>::from(s * a), map(aa, |x| s.wrapping_mul(x)));
            assert_eq!(<[$t; $d]>::from(s / a), map(aa, |x| s.wrapping_div(x)));
        }
    };
}

vector_ops!(i32_vector_1, i32, 1, [i32::MIN], [-1], 3);
vector_ops!(i32_vector_2, i32, 2, [i32::MAX, i32::MIN], [2, -1], -3);
vector_ops!(i32_vector_3, i32, 3, [7, -8, i32::MIN], [2, -3, -1], 3);
vector_ops!(i32_vector_4, i32, 4, [7, -8, i32::MAX, i32::MIN], [2, -3, 2, -1], 3);
vector_ops!(u32_vector_1, u32, 1, [u32::MAX], [2], 3);
vector_ops!(u32_vector_2, u32, 2, [u32::MAX, 1], [2, 3], 3);
vector_ops!(u32_vector_3, u32, 3, [7, 1, u32::MAX], [2, 3, 2], 3);
vector_ops!(u32_vector_4, u32, 4, [7, 1, u32::MAX, 16], [2, 3, 2, 4], 3);

macro_rules! abs_diff {
    ($name:ident, $t:ty, $d:literal, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            let aa: [$t; $d] = $a;
            let bb: [$t; $d] = $b;
            let expected = core::array::from_fn(|i| aa[i].abs_diff(bb[i]));
            let a = Vector::<$t, $d>::from(aa);
            let b = Vector::<$t, $d>::from(bb);

            assert_eq!(<[u32; $d]>::from(a.abs_diff(b)), expected);
            assert_eq!(<[u32; $d]>::from(b.abs_diff(a)), expected);
        }
    };
}

abs_diff!(i32_abs_diff_1, i32, 1, [i32::MIN], [i32::MAX]);
abs_diff!(i32_abs_diff_2, i32, 2, [i32::MIN, -1], [i32::MAX, 1]);
abs_diff!(i32_abs_diff_3, i32, 3, [-7, 0, i32::MAX], [3, 0, i32::MIN]);
abs_diff!(i32_abs_diff_4, i32, 4, [i32::MIN, -8, 7, i32::MAX], [i32::MAX, 3, -2, 0]);
abs_diff!(u32_abs_diff_1, u32, 1, [0], [u32::MAX]);
abs_diff!(u32_abs_diff_2, u32, 2, [0, u32::MAX], [u32::MAX, 0]);
abs_diff!(u32_abs_diff_3, u32, 3, [0, 7, u32::MAX], [0, 2, 1]);
abs_diff!(u32_abs_diff_4, u32, 4, [0, 8, 7, u32::MAX], [u32::MAX, 3, 12, 0]);

macro_rules! matrix_ops {
    ($name:ident, $t:ty, $r:literal, $c:literal) => {
        #[test]
        fn $name() {
            let a: [[$t; $c]; $r] =
                core::array::from_fn(|r| core::array::from_fn(|c| (r * $c + c + 7) as $t));
            let b: [[$t; $c]; $r] =
                core::array::from_fn(|r| core::array::from_fn(|c| (r * $c + c + 2) as $t));
            let s: $t = 3;
            let ma = Matrix::<$t, $r, $c>::from(a);
            let mb = Matrix::<$t, $r, $c>::from(b);

            assert_eq!(<[[$t; $c]; $r]>::from(ma + mb), map_rows_with(a, b, <$t>::wrapping_add));
            assert_eq!(<[[$t; $c]; $r]>::from(ma - mb), map_rows_with(a, b, <$t>::wrapping_sub));
            assert_eq!(<[[$t; $c]; $r]>::from(ma + s), map_rows(a, |x| x.wrapping_add(s)));
            assert_eq!(<[[$t; $c]; $r]>::from(ma - s), map_rows(a, |x| x.wrapping_sub(s)));
            assert_eq!(<[[$t; $c]; $r]>::from(ma * s), map_rows(a, |x| x.wrapping_mul(s)));
            assert_eq!(<[[$t; $c]; $r]>::from(ma / s), map_rows(a, |x| x.wrapping_div(s)));
            assert_eq!(<[[$t; $c]; $r]>::from(s + ma), map_rows(a, |x| s.wrapping_add(x)));
            assert_eq!(<[[$t; $c]; $r]>::from(s - ma), map_rows(a, |x| s.wrapping_sub(x)));
            assert_eq!(<[[$t; $c]; $r]>::from(s * ma), map_rows(a, |x| s.wrapping_mul(x)));
            assert_eq!(<[[$t; $c]; $r]>::from(s / ma), map_rows(a, |x| s.wrapping_div(x)));
        }
    };
}

macro_rules! all_matrix_shapes {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            matrix_ops!([<$prefix _1x1>], $t, 1, 1); matrix_ops!([<$prefix _1x2>], $t, 1, 2);
            matrix_ops!([<$prefix _1x3>], $t, 1, 3); matrix_ops!([<$prefix _1x4>], $t, 1, 4);
            matrix_ops!([<$prefix _2x1>], $t, 2, 1); matrix_ops!([<$prefix _2x2>], $t, 2, 2);
            matrix_ops!([<$prefix _2x3>], $t, 2, 3); matrix_ops!([<$prefix _2x4>], $t, 2, 4);
            matrix_ops!([<$prefix _3x1>], $t, 3, 1); matrix_ops!([<$prefix _3x2>], $t, 3, 2);
            matrix_ops!([<$prefix _3x3>], $t, 3, 3); matrix_ops!([<$prefix _3x4>], $t, 3, 4);
            matrix_ops!([<$prefix _4x1>], $t, 4, 1); matrix_ops!([<$prefix _4x2>], $t, 4, 2);
            matrix_ops!([<$prefix _4x3>], $t, 4, 3); matrix_ops!([<$prefix _4x4>], $t, 4, 4);
        }
    };
}

all_matrix_shapes!(i32, i32_matrix);
all_matrix_shapes!(u32, u32_matrix);

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn integer_division_panics_for_zero_in_an_active_lane() {
    for d in 1..=4 {
        let i32_result = std::panic::catch_unwind(|| match d {
            1 => {
                let _ = Vector::<i32, 1>::ONE / Vector::from([0]);
            }
            2 => {
                let _ = Vector::<i32, 2>::ONE / Vector::from([1, 0]);
            }
            3 => {
                let _ = Vector::<i32, 3>::ONE / Vector::from([1, 0, 1]);
            }
            4 => {
                let _ = Vector::<i32, 4>::ONE / Vector::from([1, 1, 0, 1]);
            }
            _ => unreachable!(),
        });
        assert!(i32_result.is_err(), "i32 D={d} did not panic");

        let u32_result = std::panic::catch_unwind(|| match d {
            1 => {
                let _ = Vector::<u32, 1>::ONE / Vector::from([0]);
            }
            2 => {
                let _ = Vector::<u32, 2>::ONE / Vector::from([1, 0]);
            }
            3 => {
                let _ = Vector::<u32, 3>::ONE / Vector::from([1, 0, 1]);
            }
            4 => {
                let _ = Vector::<u32, 4>::ONE / Vector::from([1, 1, 0, 1]);
            }
            _ => unreachable!(),
        });
        assert!(u32_result.is_err(), "u32 D={d} did not panic");
    }
}

#[test]
fn integer_division_ignores_padding_and_wraps_signed_overflow() {
    assert_eq!(<[i32; 1]>::from(Vector::from([i32::MIN]) / Vector::from([-1])), [i32::MIN]);
    assert_eq!(<[i32; 2]>::from(Vector::from([i32::MIN, 6]) / Vector::from([-1, 2])), [
        i32::MIN,
        3
    ]);
    assert_eq!(<[i32; 3]>::from(Vector::from([i32::MIN, 6, 8]) / Vector::from([-1, 2, 4])), [
        i32::MIN,
        3,
        2
    ]);
}

#[test]
fn integer_unary_and_iterator_operations_use_wrapping_identities() {
    let i = Vector::<i32, 3>::from([i32::MIN, 1, -2]);
    assert_eq!(<[i32; 3]>::from(-i), [i32::MIN, -1, 2]);
    assert_eq!(<[i32; 3]>::from(!i), [!i32::MIN, !1, !-2]);
    let u = Vector::<u32, 3>::from([0, 1, u32::MAX]);
    assert_eq!(<[u32; 3]>::from(!u), [!0, !1, !u32::MAX]);

    let values = [Vector::<i32, 2>::from([i32::MAX, 3]), Vector::from([1, 4])];
    assert_eq!(<[i32; 2]>::from(values.into_iter().sum::<Vector<i32, 2>>()), [i32::MIN, 7]);
    assert_eq!(<[i32; 2]>::from(values.into_iter().product::<Vector<i32, 2>>()), [i32::MAX, 12]);
    assert_eq!(<[i32; 2]>::from(core::iter::empty().sum::<Vector<i32, 2>>()), [0, 0]);
    assert_eq!(<[i32; 2]>::from(core::iter::empty().product::<Vector<i32, 2>>()), [1, 1]);

    let unsigned = [Vector::<u32, 1>::from([u32::MAX]), Vector::from([2])];
    assert_eq!(<[u32; 1]>::from(unsigned.into_iter().sum::<Vector<u32, 1>>()), [1]);
    assert_eq!(<[u32; 1]>::from(unsigned.into_iter().product::<Vector<u32, 1>>()), [u32::MAX - 1]);
    assert_eq!(<[u32; 3]>::from(core::iter::empty().sum::<Vector<u32, 3>>()), [0; 3]);
    assert_eq!(<[u32; 3]>::from(core::iter::empty().product::<Vector<u32, 3>>()), [1; 3]);

    let matrix_values = [Matrix::<i32, 1, 1>::from([[i32::MAX]]), Matrix::<i32, 1, 1>::from([[1]])];
    assert_eq!(<[[i32; 1]; 1]>::from(-matrix_values[1]), [[-1]]);
    assert_eq!(<[[i32; 1]; 1]>::from(matrix_values.into_iter().sum::<Matrix<i32, 1, 1>>()), [[
        i32::MIN
    ]],);
    assert_eq!(<[[u32; 2]; 3]>::from(core::iter::empty().sum::<Matrix<u32, 3, 2>>()), [[0; 2]; 3],);
}
