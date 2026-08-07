//! Tests for integer vector bitwise operations.

use algea::Vector;

fn map<T: Copy, const N: usize>(a: [T; N], f: impl Fn(T) -> T) -> [T; N] {
    core::array::from_fn(|i| f(a[i]))
}

fn map_with<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> T) -> [T; N] {
    core::array::from_fn(|i| f(a[i], b[i]))
}

macro_rules! integer_vector_ops {
    ($name:ident, $t:ty, $d:literal, $a:expr, $b:expr, $scalar:expr, $shifts:expr) => {
        #[test]
        fn $name() {
            let aa: [$t; $d] = $a;
            let bb: [$t; $d] = $b;
            let shifts: [$t; $d] = $shifts;
            let scalar: $t = $scalar;
            let a = Vector::<$t, $d>::from(aa);
            let b = Vector::<$t, $d>::from(bb);
            let shift_vector = Vector::<$t, $d>::from(shifts);

            assert_eq!((a % b).to_array(), map_with(aa, bb, <$t>::wrapping_rem));
            assert_eq!((a & b).to_array(), map_with(aa, bb, |a, b| a & b));
            assert_eq!((a | b).to_array(), map_with(aa, bb, |a, b| a | b));
            assert_eq!((a ^ b).to_array(), map_with(aa, bb, |a, b| a ^ b));

            assert_eq!((a % scalar).to_array(), map(aa, |a| a.wrapping_rem(scalar)));
            assert_eq!((a & scalar).to_array(), map(aa, |a| a & scalar));
            assert_eq!((a | scalar).to_array(), map(aa, |a| a | scalar));
            assert_eq!((a ^ scalar).to_array(), map(aa, |a| a ^ scalar));

            assert_eq!((scalar % b).to_array(), map(bb, |b| scalar.wrapping_rem(b)));
            assert_eq!((scalar & a).to_array(), map(aa, |a| scalar & a));
            assert_eq!((scalar | a).to_array(), map(aa, |a| scalar | a));
            assert_eq!((scalar ^ a).to_array(), map(aa, |a| scalar ^ a));

            assert_eq!(
                (a << shift_vector).to_array(),
                map_with(aa, shifts, |a, shift| a.wrapping_shl(shift as u32)),
            );
            assert_eq!(
                (a >> shift_vector).to_array(),
                map_with(aa, shifts, |a, shift| a.wrapping_shr(shift as u32)),
            );
            assert_eq!((a << scalar).to_array(), map(aa, |a| a.wrapping_shl(scalar as u32)));
            assert_eq!((a >> scalar).to_array(), map(aa, |a| a.wrapping_shr(scalar as u32)));
            assert_eq!(
                (scalar << shift_vector).to_array(),
                map(shifts, |shift| scalar.wrapping_shl(shift as u32)),
            );
            assert_eq!(
                (scalar >> shift_vector).to_array(),
                map(shifts, |shift| scalar.wrapping_shr(shift as u32)),
            );

            let mut actual = a;
            actual %= b;
            assert_eq!(actual.to_array(), map_with(aa, bb, <$t>::wrapping_rem));
            actual = a;
            actual &= b;
            assert_eq!(actual.to_array(), map_with(aa, bb, |a, b| a & b));
            actual = a;
            actual |= b;
            assert_eq!(actual.to_array(), map_with(aa, bb, |a, b| a | b));
            actual = a;
            actual ^= b;
            assert_eq!(actual.to_array(), map_with(aa, bb, |a, b| a ^ b));
            actual = a;
            actual <<= shift_vector;
            assert_eq!(
                actual.to_array(),
                map_with(aa, shifts, |a, shift| a.wrapping_shl(shift as u32)),
            );
            actual = a;
            actual >>= shift_vector;
            assert_eq!(
                actual.to_array(),
                map_with(aa, shifts, |a, shift| a.wrapping_shr(shift as u32)),
            );

            actual = a;
            actual %= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a.wrapping_rem(scalar)));
            actual = a;
            actual &= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a & scalar));
            actual = a;
            actual |= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a | scalar));
            actual = a;
            actual ^= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a ^ scalar));
            actual = a;
            actual <<= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a.wrapping_shl(scalar as u32)));
            actual = a;
            actual >>= scalar;
            assert_eq!(actual.to_array(), map(aa, |a| a.wrapping_shr(scalar as u32)));
        }
    };
}

integer_vector_ops!(i32_vector_1, i32, 1, [i32::MIN], [-1], 3, [-1]);
integer_vector_ops!(i32_vector_2, i32, 2, [-17, i32::MIN], [5, -1], 3, [0, 32]);
integer_vector_ops!(i32_vector_3, i32, 3, [-17, 18, i32::MIN], [5, -7, -1], 3, [1, 31, 33]);
integer_vector_ops!(i32_vector_4, i32, 4, [-17, 18, i32::MIN, i32::MAX], [5, -7, -1, 11], 3, [
    0, 1, 32, -1
]);

integer_vector_ops!(u32_vector_1, u32, 1, [u32::MAX], [7], 3, [33]);
integer_vector_ops!(u32_vector_2, u32, 2, [17, u32::MAX], [5, 7], 3, [0, 32]);
integer_vector_ops!(u32_vector_3, u32, 3, [17, 18, u32::MAX], [5, 7, 11], 3, [1, 31, 33]);
integer_vector_ops!(u32_vector_4, u32, 4, [17, 18, u32::MAX, 0x8000_0000], [5, 7, 11, 13], 3, [
    0, 1, 32, 63
]);

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn remainder_panics_for_zero_in_an_active_lane() {
    let signed = std::panic::catch_unwind(|| {
        let _ = Vector::<i32, 3>::from([1, 2, 3]) % Vector::from([1, 0, 1]);
    });
    assert!(signed.is_err());

    let unsigned = std::panic::catch_unwind(|| {
        let _ = Vector::<u32, 3>::from([1, 2, 3]) % 0;
    });
    assert!(unsigned.is_err());
}

#[test]
fn signed_remainder_wraps_min_over_negative_one() {
    assert_eq!(
        (Vector::<i32, 3>::from([i32::MIN, -7, 7]) % Vector::from([-1, 3, -3])).to_array(),
        [0, -1, 1]
    );
}
