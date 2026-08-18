//! Tests for indexing, arrays, and component access.

use algea::{Vector, row_major::Matrix};
#[cfg(not(target_arch = "wasm32"))]
use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(not(target_arch = "wasm32"))]
fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("operation did not panic");
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        panic!("panic payload was not a string")
    }
}

macro_rules! vector_access {
    ($name:ident, $t:ty, $d:literal) => {
        #[test]
        fn $name() {
            let source: [$t; $d] = core::array::from_fn(|i| (i + 1) as $t);
            let replacement: [$t; $d] = core::array::from_fn(|i| (i + 11) as $t);
            let mut vector = Vector::<$t, $d>::from(source);

            assert_eq!(vector.as_array(), &source);
            for i in 0..$d {
                assert_eq!(vector[i], source[i]);
                assert!(core::ptr::eq(&vector[i], &vector.as_array()[i]));
            }

            for i in 0..$d {
                vector[i] = replacement[i];
            }
            assert_eq!(vector.as_array(), &replacement);

            let final_values: [$t; $d] = core::array::from_fn(|i| (i + 21) as $t);
            *vector.as_mut_array() = final_values;
            assert_eq!(vector.as_array(), &final_values);
            assert_eq!(<[$t; $d]>::from(vector), final_values);
        }
    };
}

macro_rules! vector_access_all_types {
    ($d:literal) => {
        paste::paste! {
            vector_access!([<f32_vector_ $d>], f32, $d);
            vector_access!([<i32_vector_ $d>], i32, $d);
            vector_access!([<u32_vector_ $d>], u32, $d);
            vector_access!([<f64_vector_ $d>], f64, $d);
            vector_access!([<i64_vector_ $d>], i64, $d);
            vector_access!([<u64_vector_ $d>], u64, $d);
        }
    };
}

vector_access_all_types!(1);
vector_access_all_types!(2);
vector_access_all_types!(3);
vector_access_all_types!(4);

macro_rules! matrix_access {
    ($name:ident, $t:ty, $r:literal, $c:literal) => {
        #[test]
        fn $name() {
            let source: [[$t; $c]; $r] =
                core::array::from_fn(|r| core::array::from_fn(|c| (r * $c + c + 1) as $t));
            let expected: [[$t; $c]; $r] =
                core::array::from_fn(|r| core::array::from_fn(|c| (r * $c + c + 31) as $t));
            let mut matrix = Matrix::<$t, $r, $c>::from(source);

            for r in 0..$r {
                for c in 0..$c {
                    assert_eq!(matrix[(r, c)], source[r][c]);
                    matrix[(r, c)] = expected[r][c];
                }
            }
            assert_eq!(<[[$t; $c]; $r]>::from(matrix), expected);
        }
    };
}

macro_rules! matrix_access_all_shapes {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            matrix_access!([<$prefix _1x1>], $t, 1, 1); matrix_access!([<$prefix _1x2>], $t, 1, 2);
            matrix_access!([<$prefix _1x3>], $t, 1, 3); matrix_access!([<$prefix _1x4>], $t, 1, 4);
            matrix_access!([<$prefix _2x1>], $t, 2, 1); matrix_access!([<$prefix _2x2>], $t, 2, 2);
            matrix_access!([<$prefix _2x3>], $t, 2, 3); matrix_access!([<$prefix _2x4>], $t, 2, 4);
            matrix_access!([<$prefix _3x1>], $t, 3, 1); matrix_access!([<$prefix _3x2>], $t, 3, 2);
            matrix_access!([<$prefix _3x3>], $t, 3, 3); matrix_access!([<$prefix _3x4>], $t, 3, 4);
            matrix_access!([<$prefix _4x1>], $t, 4, 1); matrix_access!([<$prefix _4x2>], $t, 4, 2);
            matrix_access!([<$prefix _4x3>], $t, 4, 3); matrix_access!([<$prefix _4x4>], $t, 4, 4);
        }
    };
}

matrix_access_all_shapes!(f32, f32_matrix);
matrix_access_all_shapes!(i32, i32_matrix);
matrix_access_all_shapes!(u32, u32_matrix);
matrix_access_all_shapes!(f64, f64_matrix);
matrix_access_all_shapes!(i64, i64_matrix);
matrix_access_all_shapes!(u64, u64_matrix);

macro_rules! component_access {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let mut v2 = Vector::<$t, 2>::from([1 as $t, 2 as $t]);
            assert_eq!((v2.x, v2.y), (1 as $t, 2 as $t));
            v2.x = 11 as $t;
            v2.y = 12 as $t;
            assert_eq!(v2.as_array(), &[11 as $t, 12 as $t]);

            let mut v3 = Vector::<$t, 3>::from([1 as $t, 2 as $t, 3 as $t]);
            assert_eq!((v3.x, v3.y, v3.z), (1 as $t, 2 as $t, 3 as $t));
            *v3.as_mut_array() = [11 as $t, 12 as $t, 13 as $t];
            assert_eq!((v3.x, v3.y, v3.z), (11 as $t, 12 as $t, 13 as $t));
            v3.z = 23 as $t;
            assert_eq!(v3[2], 23 as $t);

            let mut v4 = Vector::<$t, 4>::from([1 as $t, 2 as $t, 3 as $t, 4 as $t]);
            assert_eq!((v4.x, v4.y, v4.z, v4.w), (1 as $t, 2 as $t, 3 as $t, 4 as $t));
            v4.x = 21 as $t;
            v4.y = 22 as $t;
            v4.z = 23 as $t;
            v4.w = 24 as $t;
            assert_eq!(<[$t; 4]>::from(v4), [21 as $t, 22 as $t, 23 as $t, 24 as $t]);
        }
    };
}

component_access!(f32_components, f32);
component_access!(i32_components, i32);
component_access!(u32_components, u32);
component_access!(f64_components, f64);
component_access!(i64_components, i64);
component_access!(u64_components, u64);

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn vector_out_of_bounds_messages_match_slices() {
    macro_rules! check {
        ($d:literal) => {{
            let vector = Vector::<i32, $d>::ZERO;
            let immutable = panic_message(|| {
                let _ = vector[$d];
            });
            assert_eq!(
                immutable,
                concat!(
                    "index out of bounds: the len is ",
                    stringify!($d),
                    " but the index is ",
                    stringify!($d)
                )
            );

            let mut vector = Vector::<i32, $d>::ZERO;
            let mutable = panic_message(|| vector[$d] = 1);
            assert_eq!(mutable, immutable);
        }};
    }
    check!(1);
    check!(2);
    check!(3);
    check!(4);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn matrix_out_of_bounds_messages_identify_row_and_column() {
    let matrix = Matrix::<i32, 2, 3>::ZERO;
    for (index, expected) in [
        ((2, 1), "matrix index out of bounds: the dimensions are 2x3 but the index is (2, 1)"),
        ((1, 3), "matrix index out of bounds: the dimensions are 2x3 but the index is (1, 3)"),
        ((2, 3), "matrix index out of bounds: the dimensions are 2x3 but the index is (2, 3)"),
    ] {
        assert_eq!(
            panic_message(|| {
                let _ = matrix[index];
            }),
            expected
        );
        let mut matrix = matrix;
        assert_eq!(panic_message(|| matrix[index] = 1), expected);
    }
}
