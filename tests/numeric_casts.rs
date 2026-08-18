//! Tests for numeric vector and matrix casts.
//!
//! Every cast must produce what Rust's `as` produces on each lane. Where the destination is a
//! floating-point type the comparison is on bits, not on values: `-0.0 == 0.0` would hide a lost
//! sign. NaN payloads are exempt, so two NaNs count as equal.

use algea::{Vector, column_major, row_major};

/// Bit-exact equality, with all NaNs treated as one value.
trait SameAs {
    fn same_as(self, other: Self) -> bool;
}

macro_rules! impl_same_as {
    (float $($t:ty),+) => {
        $(impl SameAs for $t {
            fn same_as(self, other: Self) -> bool {
                (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
            }
        })+
    };
    (int $($t:ty),+) => {
        $(impl SameAs for $t {
            fn same_as(self, other: Self) -> bool { self == other }
        })+
    };
}
impl_same_as!(float f32, f64);
impl_same_as!(int i32, u32, i64, u64);

macro_rules! assert_same_lanes {
    ($actual:expr, $expected:expr, $src:ty => $dst:ty, $what:expr) => {{
        let actual = $actual;
        let expected = $expected;
        for (lane, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                got.same_as(want),
                "{} -> {}, {}, lane {}: got {:?}, want {:?}",
                stringify!($src),
                stringify!($dst),
                $what,
                lane,
                got,
                want,
            );
        }
    }};
}

/// Checks one vector cast lane by lane, bit-exactly.
macro_rules! assert_vector_cast_bits {
    ($src:ty => $dst:ty, $d:literal, $values:expr) => {{
        let source: [$src; $d] = $values;
        let expected: [$dst; $d] = source.map(|value| value as $dst);
        let actual: [$dst; $d] = Vector::<$src, $d>::from(source).cast::<$dst>().into();
        assert_same_lanes!(actual, expected, $src => $dst, format!("D={}", $d));
    }};
}

/// The same for a column-major matrix. `2x3` matters on its own: the second of its two four-lane
/// units holds only two live lanes, so it is the one shape that exercises the two-lane path of a
/// four-lane kernel.
macro_rules! assert_column_major_cast_bits {
    ($src:ty => $dst:ty, $r:literal, $c:literal, $values:expr) => {{
        let source: [[$src; $r]; $c] = $values;
        let expected: [[$dst; $r]; $c] = source.map(|column| column.map(|value| value as $dst));
        let actual: [[$dst; $r]; $c] =
            column_major::Matrix::<$src, $r, $c>::from(source).cast::<$dst>().into();
        for (column, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_same_lanes!(got, want, $src => $dst, format!("{}x{} column {}", $r, $c, column));
        }
    }};
}

macro_rules! assert_vector_cast {
    ($src:ty => $dst:ty, $d:literal, $values:expr) => {{
        let source: [$src; $d] = $values;
        let expected: [$dst; $d] = source.map(|value| value as $dst);
        let actual: [$dst; $d] = Vector::<$src, $d>::from(source).cast::<$dst>().into();
        assert_eq!(actual, expected, "{} -> {}, D={}", stringify!($src), stringify!($dst), $d);
    }};
}

macro_rules! assert_all_vector_casts {
    ($d:literal, $floats:expr, $signed:expr, $unsigned:expr) => {{
        assert_vector_cast!(f32 => f32, $d, $floats);
        assert_vector_cast!(f32 => i32, $d, $floats);
        assert_vector_cast!(f32 => u32, $d, $floats);
        assert_vector_cast!(i32 => f32, $d, $signed);
        assert_vector_cast!(i32 => i32, $d, $signed);
        assert_vector_cast!(i32 => u32, $d, $signed);
        assert_vector_cast!(u32 => f32, $d, $unsigned);
        assert_vector_cast!(u32 => i32, $d, $unsigned);
        assert_vector_cast!(u32 => u32, $d, $unsigned);
    }};
}

#[test]
fn vector_casts_match_scalar_as_for_all_types_and_dimensions() {
    assert_all_vector_casts!(1, [-1.75], [i32::MIN], [u32::MAX]);
    assert_all_vector_casts!(2, [-0.0, 1.75], [i32::MIN, i32::MAX], [0, u32::MAX]);
    assert_all_vector_casts!(3, [-12.75, 0.0, 16_777_217.0], [-1, 0, 16_777_217], [
        0,
        16_777_217,
        u32::MAX
    ]);
    assert_all_vector_casts!(
        4,
        [-1.5, 0.0, 1.5, 4_294_967_296.0],
        [i32::MIN, -16_777_217, 16_777_217, i32::MAX],
        [0, 1, 16_777_217, u32::MAX]
    );
}

#[test]
fn float_to_integer_casts_match_scalar_as_at_special_values_and_boundaries() {
    let inputs = [
        [f32::NAN, f32::NEG_INFINITY, -1.9, -0.0],
        [0.0, 1.9, 2_147_483_648.0, f32::INFINITY],
        [f32::MIN, f32::MAX, 4_294_967_296.0, -2_147_483_648.0],
    ];

    for input in inputs {
        assert_vector_cast!(f32 => i32, 4, input);
        assert_vector_cast!(f32 => u32, 4, input);
    }
}

#[test]
fn identity_float_cast_preserves_bits() {
    let source = [f32::from_bits(0x7fc1_2345), f32::from_bits(0xffc5_4321), -0.0, f32::INFINITY];
    let actual: [f32; 4] = Vector::<f32, 4>::from(source).cast::<f32>().into();

    assert_eq!(actual.map(f32::to_bits), source.map(f32::to_bits));
}

macro_rules! assert_row_major_cast {
    ($src:ty => $dst:ty, $r:literal, $c:literal, $values:expr) => {{
        let source: [[$src; $c]; $r] = $values;
        let expected: [[$dst; $c]; $r] = source.map(|row| row.map(|value| value as $dst));
        let actual: [[$dst; $c]; $r] =
            row_major::Matrix::<$src, $r, $c>::from(source).cast::<$dst>().into();
        assert_eq!(
            actual,
            expected,
            "row-major {} -> {}, {}x{}",
            stringify!($src),
            stringify!($dst),
            $r,
            $c
        );
    }};
}

macro_rules! assert_column_major_cast {
    ($src:ty => $dst:ty, $r:literal, $c:literal, $values:expr) => {{
        let source: [[$src; $r]; $c] = $values;
        let expected: [[$dst; $r]; $c] = source.map(|column| column.map(|value| value as $dst));
        let actual: [[$dst; $r]; $c] =
            column_major::Matrix::<$src, $r, $c>::from(source).cast::<$dst>().into();
        assert_eq!(
            actual,
            expected,
            "column-major {} -> {}, {}x{}",
            stringify!($src),
            stringify!($dst),
            $r,
            $c
        );
    }};
}

macro_rules! assert_all_matrix_casts {
    ($r:literal, $c:literal) => {{
        let float_rows: [[f32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column) as f32 - 3.5));
        let signed_rows: [[i32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column) as i32 - 5));
        let unsigned_rows: [[u32; $c]; $r] =
            core::array::from_fn(|row| core::array::from_fn(|column| (row * $c + column + 1) as u32));

        assert_row_major_cast!(f32 => f32, $r, $c, float_rows);
        assert_row_major_cast!(f32 => i32, $r, $c, float_rows);
        assert_row_major_cast!(f32 => u32, $r, $c, float_rows);
        assert_row_major_cast!(i32 => f32, $r, $c, signed_rows);
        assert_row_major_cast!(i32 => i32, $r, $c, signed_rows);
        assert_row_major_cast!(i32 => u32, $r, $c, signed_rows);
        assert_row_major_cast!(u32 => f32, $r, $c, unsigned_rows);
        assert_row_major_cast!(u32 => i32, $r, $c, unsigned_rows);
        assert_row_major_cast!(u32 => u32, $r, $c, unsigned_rows);

        let float_columns: [[f32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column) as f32 - 3.5));
        let signed_columns: [[i32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column) as i32 - 5));
        let unsigned_columns: [[u32; $r]; $c] =
            core::array::from_fn(|column| core::array::from_fn(|row| (row * $c + column + 1) as u32));

        assert_column_major_cast!(f32 => f32, $r, $c, float_columns);
        assert_column_major_cast!(f32 => i32, $r, $c, float_columns);
        assert_column_major_cast!(f32 => u32, $r, $c, float_columns);
        assert_column_major_cast!(i32 => f32, $r, $c, signed_columns);
        assert_column_major_cast!(i32 => i32, $r, $c, signed_columns);
        assert_column_major_cast!(i32 => u32, $r, $c, signed_columns);
        assert_column_major_cast!(u32 => f32, $r, $c, unsigned_columns);
        assert_column_major_cast!(u32 => i32, $r, $c, unsigned_columns);
        assert_column_major_cast!(u32 => u32, $r, $c, unsigned_columns);
    }};
}

#[test]
fn matrix_casts_match_scalar_as_for_all_types_shapes_and_layouts() {
    assert_all_matrix_casts!(1, 1);
    assert_all_matrix_casts!(1, 2);
    assert_all_matrix_casts!(1, 3);
    assert_all_matrix_casts!(1, 4);
    assert_all_matrix_casts!(2, 1);
    assert_all_matrix_casts!(2, 2);
    assert_all_matrix_casts!(2, 3);
    assert_all_matrix_casts!(2, 4);
    assert_all_matrix_casts!(3, 1);
    assert_all_matrix_casts!(3, 2);
    assert_all_matrix_casts!(3, 3);
    assert_all_matrix_casts!(3, 4);
    assert_all_matrix_casts!(4, 1);
    assert_all_matrix_casts!(4, 2);
    assert_all_matrix_casts!(4, 3);
    assert_all_matrix_casts!(4, 4);
}

// Values whose `f64 as f32` result is worth pinning down: a signed zero, both halfway cases that
// round-to-nearest-even resolves in opposite directions, overflow to infinity, and underflow
// through the subnormal range to zero.
const F64_TO_F32_CASES: [f64; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    // Exactly halfway between 1.0f32 and its successor: ties-even keeps 1.0.
    1.0 + 5.960_464_477_539_063e-8,
    // Three quarters of the way: rounds up.
    1.0 + 8.940_696_716_308_594e-8,
    // The same two, negated.
    -(1.0 + 5.960_464_477_539_063e-8),
    -(1.0 + 8.940_696_716_308_594e-8),
    f64::MAX,
    f64::MIN,
    f32::MAX as f64,
    f32::MIN as f64,
    // Just past f32::MAX, so it overflows to infinity.
    3.402_823_669e38,
    // Below the smallest f32 subnormal, so it underflows to zero.
    1e-46,
    f64::INFINITY,
    f64::NAN,
];

const F32_CASES: [f32; 8] =
    [0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX, f32::NEG_INFINITY, f32::NAN];

/// Runs one cast over every dimension, rotating the case list so each value lands in each lane.
macro_rules! assert_rotated_casts {
    ($src:ty => $dst:ty, $cases:expr) => {{
        let cases = $cases;
        assert!(cases.len() >= 6, "need six cases to fill a 2x3 matrix");
        for offset in 0..cases.len() {
            let pick = |i: usize| cases[(offset + i) % cases.len()];
            assert_vector_cast_bits!($src => $dst, 1, [pick(0)]);
            assert_vector_cast_bits!($src => $dst, 2, [pick(0), pick(1)]);
            assert_vector_cast_bits!($src => $dst, 3, [pick(0), pick(1), pick(2)]);
            assert_vector_cast_bits!($src => $dst, 4, [pick(0), pick(1), pick(2), pick(3)]);
            // Reaches the two-lane path of a four-lane kernel.
            assert_column_major_cast_bits!($src => $dst, 2, 3, [
                [pick(0), pick(1)],
                [pick(2), pick(3)],
                [pick(4), pick(5)],
            ]);
        }
    }};
}

#[test]
fn float_width_casts_match_scalar_as() {
    assert_rotated_casts!(f64 => f32, F64_TO_F32_CASES);
    assert_rotated_casts!(f32 => f64, F32_CASES);
    assert_rotated_casts!(f64 => f64, F64_TO_F32_CASES);
    assert_rotated_casts!(f32 => f32, F32_CASES);
}

// Sign and zero extension, and the two reinterpretations between 64-bit integers. The values are
// chosen so the high bit, the sign bit and the 2^24 / 2^53 rounding boundaries all appear.
const I32_CASES: [i32; 8] = [0, 1, -1, i32::MIN, i32::MAX, -16_777_217, 16_777_217, -2];
const U32_CASES: [u32; 8] = [0, 1, u32::MAX, 1 << 31, 16_777_217, (1 << 31) + 1, 0xffff_0000, 2];
const I64_CASES: [i64; 8] =
    [0, 1, -1, i64::MIN, i64::MAX, -9_007_199_254_740_993, 9_007_199_254_740_993, -2];
const U64_CASES: [u64; 8] =
    [0, 1, u64::MAX, 1 << 63, 9_007_199_254_740_993, (1 << 63) + 1, 0xffff_ffff_0000_0000, 2];

#[test]
fn widening_casts_match_scalar_as() {
    assert_rotated_casts!(i32 => f64, I32_CASES);
    assert_rotated_casts!(u32 => f64, U32_CASES);
    assert_rotated_casts!(i32 => i64, I32_CASES);
    assert_rotated_casts!(i32 => u64, I32_CASES);
    assert_rotated_casts!(u32 => i64, U32_CASES);
    assert_rotated_casts!(u32 => u64, U32_CASES);
}

#[test]
fn same_width_64_bit_casts_match_scalar_as() {
    assert_rotated_casts!(i64 => u64, I64_CASES);
    assert_rotated_casts!(u64 => i64, U64_CASES);
    assert_rotated_casts!(i64 => i64, I64_CASES);
    assert_rotated_casts!(u64 => u64, U64_CASES);
}

// `f64` values whose narrowing to a 32-bit integer exercises every branch of `as`: both signed and
// unsigned overflow, the exact boundaries, truncation toward zero on either side of zero, NaN and
// both infinities.
const F64_TO_INT_CASES: [f64; 20] = [
    0.0,
    -0.0,
    1.9,
    -1.9,
    0.5,
    -0.5,
    2_147_483_647.0,
    2_147_483_648.0,
    -2_147_483_648.0,
    -2_147_483_649.0,
    4_294_967_295.0,
    4_294_967_296.0,
    9_007_199_254_740_993.0,
    -9_007_199_254_740_993.0,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -f64::NAN,
];

#[test]
fn narrowing_casts_match_scalar_as() {
    assert_rotated_casts!(f64 => i32, F64_TO_INT_CASES);
    assert_rotated_casts!(f64 => u32, F64_TO_INT_CASES);
    assert_rotated_casts!(i64 => i32, I64_CASES);
    assert_rotated_casts!(i64 => u32, I64_CASES);
    assert_rotated_casts!(u64 => i32, U64_CASES);
    assert_rotated_casts!(u64 => u32, U64_CASES);
}

// `f32` values that pin down the 64-bit narrowings: saturation at both ends, truncation toward
// zero, NaN and the infinities. Every `f32` is exactly representable as `f64`, so widening first is
// lossless -- see the note on the other direction below.
const F32_TO_INT_CASES: [f32; 16] = [
    0.0,
    -0.0,
    1.9,
    -1.9,
    0.5,
    -0.5,
    9_223_372_036_854_775_808.0,
    -9_223_372_036_854_775_808.0,
    18_446_744_073_709_551_616.0,
    16_777_217.0,
    -16_777_217.0,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

// 64-bit integers whose conversion to `f32` must round exactly once. `2^53 + 2^29 + 1` is the
// witness that going through `f64` is wrong: rounding to `f64` first and then to `f32` yields
// `2^53`, while `as` yields `2^53 + 2^30`.
const I64_TO_FLOAT_CASES: [i64; 12] = [
    0,
    1,
    -1,
    i64::MIN,
    i64::MAX,
    9_007_199_791_611_905,
    -9_007_199_791_611_905,
    9_007_199_254_740_993,
    -9_007_199_254_740_993,
    16_777_217,
    -16_777_217,
    1 << 62,
];

const U64_TO_FLOAT_CASES: [u64; 12] = [
    0,
    1,
    u64::MAX,
    u64::MAX - 1,
    1 << 63,
    (1 << 63) + 1,
    9_007_199_791_611_905,
    9_007_199_254_740_993,
    16_777_217,
    0xffff_ffff_ffff_f800,
    0x0020_0000_2000_0001,
    1 << 62,
];

#[test]
fn float_to_64_bit_integer_casts_match_scalar_as() {
    assert_rotated_casts!(f64 => i64, F64_TO_INT_CASES);
    assert_rotated_casts!(f64 => u64, F64_TO_INT_CASES);
    assert_rotated_casts!(f32 => i64, F32_TO_INT_CASES);
    assert_rotated_casts!(f32 => u64, F32_TO_INT_CASES);
}

#[test]
fn integer_64_bit_to_float_casts_round_once() {
    assert_rotated_casts!(i64 => f64, I64_TO_FLOAT_CASES);
    assert_rotated_casts!(u64 => f64, U64_TO_FLOAT_CASES);
    assert_rotated_casts!(i64 => f32, I64_TO_FLOAT_CASES);
    assert_rotated_casts!(u64 => f32, U64_TO_FLOAT_CASES);
}

/// The witness for the double-rounding hazard, checked on its own so a regression names itself.
#[test]
fn u64_to_f32_rounds_once_not_twice() {
    const X: u64 = 0x0020_0000_2000_0001; // 2^53 + 2^29 + 1
    let single = X as f32;
    let double = (X as f64) as f32;
    assert_ne!(single.to_bits(), double.to_bits(), "the witness stopped witnessing");

    let actual: [f32; 4] = Vector::<u64, 4>::from([X; 4]).cast::<f32>().into();
    for lane in actual {
        assert_eq!(lane.to_bits(), single.to_bits(), "lane took the two-step route");
    }

    let signed = X as i64;
    let actual: [f32; 4] = Vector::<i64, 4>::from([signed; 4]).cast::<f32>().into();
    for lane in actual {
        assert_eq!(lane.to_bits(), (signed as f32).to_bits(), "lane took the two-step route");
    }
}
