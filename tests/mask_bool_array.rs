//! Tests for mask and boolean-array conversions.

use algea::{Mask, MaskElement};

fn assert_all_bool_arrays_round_trip<const D: usize>()
where
    i32: MaskElement<D>,
{
    for bits in 0..(1_u32 << D) {
        let expected = core::array::from_fn(|lane| bits & (1 << lane) != 0);
        let mask = Mask::<i32, D>::from(expected);

        assert_eq!(mask.to_array(), expected, "bit pattern: {bits:#0width$b}", width = D + 2);
        assert_eq!(mask.all(), bits == (1 << D) - 1);
        assert_eq!(mask.any(), bits != 0);
    }
}

#[test]
fn bool_array_round_trips_cover_all_mask_dimensions_and_bit_patterns() {
    assert_all_bool_arrays_round_trip::<1>();
    assert_all_bool_arrays_round_trip::<2>();
    assert_all_bool_arrays_round_trip::<3>();
    assert_all_bool_arrays_round_trip::<4>();
}

fn assert_bitwise_operations<const D: usize>()
where
    i32: MaskElement<D>,
{
    for lhs_bits in 0..(1_u32 << D) {
        for rhs_bits in 0..(1_u32 << D) {
            let lhs_array = core::array::from_fn(|lane| lhs_bits & (1 << lane) != 0);
            let rhs_array = core::array::from_fn(|lane| rhs_bits & (1 << lane) != 0);
            let expected_and = core::array::from_fn(|lane| lhs_array[lane] && rhs_array[lane]);
            let expected_or = core::array::from_fn(|lane| lhs_array[lane] || rhs_array[lane]);
            let expected_xor = core::array::from_fn(|lane| lhs_array[lane] != rhs_array[lane]);
            let lhs = Mask::<i32, D>::from(lhs_array);
            let rhs = Mask::<i32, D>::from(rhs_array);

            assert_eq!((lhs & rhs).to_array(), expected_and);
            assert_eq!((lhs | rhs).to_array(), expected_or);
            assert_eq!((lhs ^ rhs).to_array(), expected_xor);

            let mut assigned = lhs;
            assigned &= rhs;
            assert_eq!(assigned.to_array(), expected_and);
            assigned = lhs;
            assigned |= rhs;
            assert_eq!(assigned.to_array(), expected_or);
            assigned = lhs;
            assigned ^= rhs;
            assert_eq!(assigned.to_array(), expected_xor);

            for (mask, expected) in
                [(lhs & rhs, expected_and), (lhs | rhs, expected_or), (lhs ^ rhs, expected_xor)]
            {
                assert_eq!(mask.all(), expected.into_iter().all(|lane| lane));
                assert_eq!(mask.any(), expected.into_iter().any(|lane| lane));
            }
        }
    }
}

#[test]
fn bitwise_operations_cover_all_mask_dimensions_and_bit_patterns() {
    assert_bitwise_operations::<1>();
    assert_bitwise_operations::<2>();
    assert_bitwise_operations::<3>();
    assert_bitwise_operations::<4>();
}
