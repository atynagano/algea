//! Tests for component-wise matrix operators that do not have a matrix-algebra meaning.

use algea::{column_major, row_major};

fn map<T: Copy, U, const M: usize, const N: usize>(
    a: [[T; N]; M],
    mut f: impl FnMut(T) -> U,
) -> [[U; N]; M] {
    a.map(|major| major.map(&mut f))
}

fn map2<T: Copy, U: Copy, V, const M: usize, const N: usize>(
    a: [[T; N]; M],
    b: [[U; N]; M],
    mut f: impl FnMut(T, U) -> V,
) -> [[V; N]; M] {
    core::array::from_fn(|major| core::array::from_fn(|minor| f(a[major][minor], b[major][minor])))
}

#[test]
fn column_major_integer_operators_are_component_wise() {
    let aa = [[-17i32, 18], [i32::MIN, 9], [31, -32]];
    let bb = [[5i32, -7], [-1, 4], [6, 5]];
    let shifts = [[1i32, 31], [33, 0], [-1, 32]];
    let scalar = 3i32;
    let a = column_major::Matrix::<i32, 2, 3>::from_columns(aa);
    let b = column_major::Matrix::from_columns(bb);
    let shift_matrix = column_major::Matrix::from_columns(shifts);

    assert_eq!((a % b).to_columns(), map2(aa, bb, i32::wrapping_rem));
    assert_eq!((a % scalar).to_columns(), map(aa, |x| x.wrapping_rem(scalar)));
    assert_eq!((scalar % b).to_columns(), map(bb, |x| scalar.wrapping_rem(x)));
    assert_eq!((!a).to_columns(), map(aa, |x| !x));
    assert_eq!((a & b).to_columns(), map2(aa, bb, |x, y| x & y));
    assert_eq!((a | scalar).to_columns(), map(aa, |x| x | scalar));
    assert_eq!((scalar ^ a).to_columns(), map(aa, |x| scalar ^ x));
    assert_eq!(
        (a << shift_matrix).to_columns(),
        map2(aa, shifts, |x, shift| x.wrapping_shl(shift as u32)),
    );
    assert_eq!(
        (scalar >> shift_matrix).to_columns(),
        map(shifts, |shift| scalar.wrapping_shr(shift as u32)),
    );

    let mut assigned = a;
    assigned %= b;
    assert_eq!(assigned.to_columns(), map2(aa, bb, i32::wrapping_rem));
    assigned = a;
    assigned &= scalar;
    assert_eq!(assigned.to_columns(), map(aa, |x| x & scalar));
    assigned = a;
    assigned |= b;
    assert_eq!(assigned.to_columns(), map2(aa, bb, |x, y| x | y));
    assigned = a;
    assigned ^= scalar;
    assert_eq!(assigned.to_columns(), map(aa, |x| x ^ scalar));
    assigned = a;
    assigned <<= shift_matrix;
    assert_eq!(assigned.to_columns(), map2(aa, shifts, |x, shift| x.wrapping_shl(shift as u32)),);
    assigned = a;
    assigned >>= scalar;
    assert_eq!(assigned.to_columns(), map(aa, |x| x.wrapping_shr(scalar as u32)));
}

#[test]
fn row_major_integer_operators_are_component_wise() {
    let aa = [[17u64, u64::MAX, 0x8000_0000_0000_0000], [18, 31, 64]];
    let bb = [[5u64, 7, 11], [4, 6, 9]];
    let shifts = [[0u64, 63, 65], [1, 64, 127]];
    let a = row_major::Matrix::<u64, 2, 3>::from_rows(aa);
    let b = row_major::Matrix::from_rows(bb);
    let shift_matrix = row_major::Matrix::from_rows(shifts);

    assert_eq!((a % b).to_rows(), map2(aa, bb, u64::wrapping_rem));
    assert_eq!((!a).to_rows(), map(aa, |x| !x));
    assert_eq!((a & b).to_rows(), map2(aa, bb, |x, y| x & y));
    assert_eq!(
        (a >> shift_matrix).to_rows(),
        map2(aa, shifts, |x, shift| x.wrapping_shr(shift as u32)),
    );
}

#[test]
fn floating_remainder_is_component_wise() {
    let aa = [[-0.0f32, 5.5], [f32::INFINITY, -5.5], [1.0, f32::NAN]];
    let bb = [[1.0f32, 2.0], [2.0, -2.0], [f32::from_bits(1), 3.0]];
    let a = column_major::Matrix::<f32, 2, 3>::from_columns(aa);
    let b = column_major::Matrix::from_columns(bb);

    let actual = (a % b).to_columns();
    let expected = map2(aa, bb, |x, y| x % y);
    for (actual, expected) in actual.into_iter().flatten().zip(expected.into_iter().flatten()) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual:?}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    let mut assigned = a;
    assigned %= 2.0;
    let actual = assigned.to_columns();
    let expected = map(aa, |x| x % 2.0);
    for (actual, expected) in actual.into_iter().flatten().zip(expected.into_iter().flatten()) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual:?}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn integer_remainder_panics_for_zero_in_an_active_element() {
    let result = std::panic::catch_unwind(|| {
        let lhs = column_major::Matrix::<i32, 2, 3>::from_columns([[1, 2], [3, 4], [5, 6]]);
        let rhs = column_major::Matrix::from_columns([[1, 1], [1, 0], [1, 1]]);
        let _ = lhs % rhs;
    });
    assert!(result.is_err());
}
