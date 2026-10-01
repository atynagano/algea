//! Tests for mask bitmask conversions, casts, and selection.

use algea::{Mask, Select, Vector};

macro_rules! bitmask_tests {
    ($module:ident, $type:ty, $other:ty, $d:literal) => {
        mod $module {
            use super::*;

            const ACTIVE_BITS: u8 = (1 << $d) - 1;

            #[test]
            fn from_bitmask_uses_low_active_bits() {
                for bitmask in u8::MIN..=u8::MAX {
                    let mask = Mask::<$type, $d>::from_bitmask(bitmask);
                    let expected = core::array::from_fn(|lane| bitmask & (1 << lane) != 0);

                    assert_eq!(mask.to_array(), expected);
                    assert_eq!(mask.to_bitmask(), bitmask & ACTIVE_BITS);
                }
            }

            #[test]
            fn cast_preserves_lanes() {
                for bitmask in u8::MIN..=u8::MAX {
                    let mask = Mask::<$type, $d>::from_bitmask(bitmask);
                    let expected = mask.to_array();

                    assert_eq!(mask.cast::<$type>().to_array(), expected);
                    assert_eq!(mask.cast::<$other>().to_array(), expected);
                    assert_eq!(Mask::<$other, $d>::from(mask).to_array(), expected);
                    assert_eq!(mask.cast::<$other>().cast::<$type>().to_array(), expected);
                }
            }

            #[test]
            fn bitmask_selects_mask_lanes() {
                let true_values =
                    Mask::<$type, $d>::from(core::array::from_fn(|lane| lane % 2 == 0));
                let false_values =
                    Mask::<$type, $d>::from(core::array::from_fn(|lane| lane % 3 == 0));

                for bitmask in u8::MIN..=u8::MAX {
                    let actual: [bool; $d] = bitmask.select(true_values, false_values).into();
                    let expected = core::array::from_fn(|lane| {
                        if bitmask & (1 << lane) != 0 {
                            true_values.to_array()[lane]
                        } else {
                            false_values.to_array()[lane]
                        }
                    });
                    assert_eq!(actual, expected);
                }
            }

            #[test]
            fn bitmask_selects_vector_lanes() {
                let true_values =
                    Vector::<$type, $d>::from(core::array::from_fn(|lane| lane as $type));
                let false_values =
                    Vector::<$type, $d>::from(core::array::from_fn(|lane| -(lane as $type) - 1));

                for bitmask in u8::MIN..=u8::MAX {
                    let actual: [$type; $d] = bitmask.select(true_values, false_values).into();
                    let expected = core::array::from_fn(|lane| {
                        if bitmask & (1 << lane) != 0 {
                            true_values.to_array()[lane]
                        } else {
                            false_values.to_array()[lane]
                        }
                    });
                    assert_eq!(actual, expected);
                }
            }
        }
    };
}

bitmask_tests!(i32_dim1, i32, i64, 1);
bitmask_tests!(i32_dim2, i32, i64, 2);
bitmask_tests!(i32_dim3, i32, i64, 3);
bitmask_tests!(i32_dim4, i32, i64, 4);
bitmask_tests!(i64_dim1, i64, i32, 1);
bitmask_tests!(i64_dim2, i64, i32, 2);
bitmask_tests!(i64_dim3, i64, i32, 3);
bitmask_tests!(i64_dim4, i64, i32, 4);
