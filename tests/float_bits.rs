//! Tests for floating-point bit conversions.

use algea::Vector;

macro_rules! assert_float_bits {
    ($t:ty => $u:ty, $d:literal, $bits:expr) => {{
        let bits: [$u; $d] = $bits;
        let float = Vector::<$t, $d>::from_bits(bits.into());

        assert_eq!(<[$t; $d]>::from(float).map(<$t>::to_bits), bits);
        assert_eq!(<[$u; $d]>::from(float.to_bits()), bits);
    }};
}

// Signed zero, one, both infinities, the smallest subnormal, and quiet and signalling NaN patterns.
#[test]
fn float_bit_casts_preserve_every_lane_for_all_dimensions() {
    assert_float_bits!(f32 => u32, 1, [0x0000_0000]);
    assert_float_bits!(f32 => u32, 2, [0x8000_0000, 0x3f80_0000]);
    assert_float_bits!(f32 => u32, 3, [0x7f80_0000, 0xff80_0000, 0x0000_0001]);
    assert_float_bits!(f32 => u32, 4, [0x7fc1_2345, 0xffc5_4321, 0x7f81_2345, 0xff81_2345]);

    assert_float_bits!(f64 => u64, 1, [0x0000_0000_0000_0000]);
    assert_float_bits!(f64 => u64, 2, [0x8000_0000_0000_0000, 0x3ff0_0000_0000_0000]);
    assert_float_bits!(f64 => u64, 3, [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x0000_0000_0000_0001
    ]);
    assert_float_bits!(f64 => u64, 4, [
        0x7ff8_1234_5678_9abc,
        0xfff8_1234_5678_9abc,
        0x7ff0_1234_5678_9abc,
        0xfff0_1234_5678_9abc
    ]);
}
