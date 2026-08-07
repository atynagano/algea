//! Tests for the integer vector and matrix API.

use algea::{Vector, row_major::Matrix};

macro_rules! check_vector_foundation {
    ($name:ident, $t:ty, $d:literal, $values:expr) => {
        #[test]
        fn $name() {
            let values: [$t; $d] = $values;
            let from_array = Vector::<$t, $d>::from_array(values);

            assert_eq!(<[$t; $d]>::from(from_array), values);
            assert_eq!(<[$t; $d]>::from(Vector::<$t, $d>::ZERO), [0; $d]);
            assert_eq!(<[$t; $d]>::from(Vector::<$t, $d>::ONE), [1; $d]);
            assert_eq!(<[$t; $d]>::from(Vector::<$t, $d>::splat(7)), [7; $d]);
        }
    };
}

check_vector_foundation!(i32_vector_1, i32, 1, [-11]);
check_vector_foundation!(i32_vector_2, i32, 2, [-11, 22]);
check_vector_foundation!(i32_vector_3, i32, 3, [-11, 22, -33]);
check_vector_foundation!(i32_vector_4, i32, 4, [-11, 22, -33, 44]);
check_vector_foundation!(u32_vector_1, u32, 1, [11]);
check_vector_foundation!(u32_vector_2, u32, 2, [11, 22]);
check_vector_foundation!(u32_vector_3, u32, 3, [11, 22, 33]);
check_vector_foundation!(u32_vector_4, u32, 4, [11, 22, 33, 44]);

macro_rules! check_f32_vector_constants {
    ($name:ident, $d:literal) => {
        #[test]
        fn $name() {
            assert_eq!(<[f32; $d]>::from(Vector::<f32, $d>::ZERO), [0.0; $d]);
            assert_eq!(<[f32; $d]>::from(Vector::<f32, $d>::ONE), [1.0; $d]);
            assert_eq!(<[f32; $d]>::from(Vector::<f32, $d>::splat(7.5)), [7.5; $d]);
        }
    };
}

check_f32_vector_constants!(f32_vector_constants_1, 1);
check_f32_vector_constants!(f32_vector_constants_2, 2);
check_f32_vector_constants!(f32_vector_constants_3, 3);
check_f32_vector_constants!(f32_vector_constants_4, 4);

#[test]
fn integer_vector_from_array_preserves_lane_order() {
    const I2: Vector<i32, 2> = Vector::from_array([-1, 2]);
    const I3: Vector<i32, 3> = Vector::from_array([-1, 2, -3]);
    const I4: Vector<i32, 4> = Vector::from_array([-1, 2, -3, 4]);
    const U2: Vector<u32, 2> = Vector::from_array([1, 2]);
    const U3: Vector<u32, 3> = Vector::from_array([1, 2, 3]);
    const U4: Vector<u32, 4> = Vector::from_array([1, 2, 3, 4]);

    assert_eq!(<[i32; 2]>::from(I2), [-1, 2]);
    assert_eq!(<[i32; 3]>::from(I3), [-1, 2, -3]);
    assert_eq!(<[i32; 4]>::from(I4), [-1, 2, -3, 4]);
    assert_eq!(<[u32; 2]>::from(U2), [1, 2]);
    assert_eq!(<[u32; 3]>::from(U3), [1, 2, 3]);
    assert_eq!(<[u32; 4]>::from(U4), [1, 2, 3, 4]);
}

macro_rules! check_matrix_foundation {
    ($name:ident, $t:ty, $r:literal, $c:literal) => {
        #[test]
        fn $name() {
            let mut next: $t = 1;
            let rows = core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    let value = next;
                    next += 1;
                    value
                })
            });

            let matrix = Matrix::<$t, $r, $c>::from_rows(rows);
            assert_eq!(<[[$t; $c]; $r]>::from(matrix), rows);
            assert_eq!(<[[$t; $c]; $r]>::from(Matrix::<$t, $r, $c>::ZERO), [[0; $c]; $r]);
            assert_eq!(<[[$t; $c]; $r]>::from(Matrix::<$t, $r, $c>::ONE), [[1; $c]; $r]);
        }
    };
}

macro_rules! check_all_matrix_shapes {
    ($t:ty, $prefix:ident) => {
        paste::paste! {
            check_matrix_foundation!([<$prefix _1x1>], $t, 1, 1);
            check_matrix_foundation!([<$prefix _1x2>], $t, 1, 2);
            check_matrix_foundation!([<$prefix _1x3>], $t, 1, 3);
            check_matrix_foundation!([<$prefix _1x4>], $t, 1, 4);
            check_matrix_foundation!([<$prefix _2x1>], $t, 2, 1);
            check_matrix_foundation!([<$prefix _2x2>], $t, 2, 2);
            check_matrix_foundation!([<$prefix _2x3>], $t, 2, 3);
            check_matrix_foundation!([<$prefix _2x4>], $t, 2, 4);
            check_matrix_foundation!([<$prefix _3x1>], $t, 3, 1);
            check_matrix_foundation!([<$prefix _3x2>], $t, 3, 2);
            check_matrix_foundation!([<$prefix _3x3>], $t, 3, 3);
            check_matrix_foundation!([<$prefix _3x4>], $t, 3, 4);
            check_matrix_foundation!([<$prefix _4x1>], $t, 4, 1);
            check_matrix_foundation!([<$prefix _4x2>], $t, 4, 2);
            check_matrix_foundation!([<$prefix _4x3>], $t, 4, 3);
            check_matrix_foundation!([<$prefix _4x4>], $t, 4, 4);
        }
    };
}

check_all_matrix_shapes!(i32, i32_matrix);
check_all_matrix_shapes!(u32, u32_matrix);

#[test]
fn integer_direction_constants_preserve_lane_order() {
    assert_eq!(<[i32; 2]>::from(Vector::<i32, 2>::POS_X), [1, 0]);
    assert_eq!(<[i32; 2]>::from(Vector::<i32, 2>::NEG_Y), [0, -1]);
    assert_eq!(<[i32; 3]>::from(Vector::<i32, 3>::POS_Z), [0, 0, 1]);
    assert_eq!(<[i32; 3]>::from(Vector::<i32, 3>::NEG_X), [-1, 0, 0]);
    assert_eq!(<[i32; 4]>::from(Vector::<i32, 4>::POS_W), [0, 0, 0, 1]);
    assert_eq!(<[i32; 4]>::from(Vector::<i32, 4>::NEG_Z), [0, 0, -1, 0]);

    assert_eq!(<[u32; 2]>::from(Vector::<u32, 2>::POS_X), [1, 0]);
    assert_eq!(<[u32; 2]>::from(Vector::<u32, 2>::POS_Y), [0, 1]);
    assert_eq!(<[u32; 3]>::from(Vector::<u32, 3>::POS_Z), [0, 0, 1]);
    assert_eq!(<[u32; 4]>::from(Vector::<u32, 4>::POS_W), [0, 0, 0, 1]);
}

#[test]
fn integer_array_round_trips_do_not_expose_backend_storage() {
    let i32_vector = Vector::<i32, 3>::from_array([-1, 2, -3]);
    let u32_vector = Vector::<u32, 2>::splat(9);
    let i32_matrix =
        Matrix::<i32, 4, 3>::from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    let u32_matrix = Matrix::<u32, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(<[i32; 3]>::from(i32_vector), [-1, 2, -3]);
    assert_eq!(<[u32; 2]>::from(u32_vector), [9, 9]);
    assert_eq!(<[[i32; 3]; 4]>::from(i32_matrix), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],]);
    assert_eq!(<[[u32; 3]; 2]>::from(u32_matrix), [[1, 2, 3], [4, 5, 6]]);
}
