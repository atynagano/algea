#![allow(missing_docs)]

use algea::{Vector, vector};

#[test]
fn vector_macro_supports_scalar_and_repeat_forms() {
    let scalar: Vector<i32, 1> = vector![1];
    assert_eq!(scalar.to_array(), [1]);

    let repeated: Vector<i32, 3> = vector![2; 3];
    assert_eq!(repeated.to_array(), [2, 2, 2]);

    let inferred_float: Vector<f32, 2> = vector![1.0, 2.0];
    assert_eq!(inferred_float.to_array(), [1.0, 2.0]);

    let vector2 = Vector::from_array([3, 4]);
    assert_eq!(vector![vector2].to_array(), [3, 4]);
}

#[test]
fn vector_macro_supports_all_binary_concat_specializations() {
    let vector1 = Vector::from_array([1]);
    let vector2 = Vector::from_array([2, 3]);
    let vector3 = Vector::from_array([2, 3, 4]);

    assert_eq!(vector![vector1, vector1].to_array(), [1, 1]);
    assert_eq!(vector![vector1, vector2].to_array(), [1, 2, 3]);
    assert_eq!(vector![vector1, vector3].to_array(), [1, 2, 3, 4]);
    assert_eq!(vector![vector2, vector1].to_array(), [2, 3, 1]);
    assert_eq!(vector![vector2, vector2].to_array(), [2, 3, 2, 3]);
    assert_eq!(vector![vector3, vector1].to_array(), [2, 3, 4, 1]);
}

#[test]
fn vector_macro_supports_all_ternary_concat_specializations() {
    let vector1 = Vector::from_array([1]);
    let vector2 = Vector::from_array([2, 3]);

    assert_eq!(vector![vector1, vector1, vector1].to_array(), [1, 1, 1]);
    assert_eq!(vector![vector1, vector1, vector2].to_array(), [1, 1, 2, 3]);
    assert_eq!(vector![vector1, vector2, vector1].to_array(), [1, 2, 3, 1]);
    assert_eq!(vector![vector2, vector1, vector1].to_array(), [2, 3, 1, 1]);
}

#[test]
fn vector_macro_supports_quaternary_concat_specialization() {
    let vector1 = Vector::from_array([1]);

    assert_eq!(vector![vector1, vector1, vector1, vector1].to_array(), [1, 1, 1, 1]);
}
