//! Tests for floating-point bit conversions.

use algea::Vector;

macro_rules! assert_float_bits {
    ($d:literal, $bits:expr) => {{
        let bits: [u32; $d] = $bits;
        let float = Vector::<f32, $d>::from_bits(bits.into());

        assert_eq!(<[f32; $d]>::from(float).map(f32::to_bits), bits);
        assert_eq!(<[u32; $d]>::from(float.to_bits()), bits);
    }};
}

#[test]
fn float_bit_casts_preserve_every_lane_for_all_dimensions() {
    assert_float_bits!(1, [0x0000_0000]);
    assert_float_bits!(2, [0x8000_0000, 0x3f80_0000]);
    assert_float_bits!(3, [0x7f80_0000, 0xff80_0000, 0x0000_0001]);
    assert_float_bits!(4, [0x7fc1_2345, 0xffc5_4321, 0x7f81_2345, 0xff81_2345]);
}
