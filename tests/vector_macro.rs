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

/// Concatenation goes through the swizzle backend, whose 64-bit path is separate from its 32-bit
/// one, so every arity is worth repeating at the wider width.
#[test]
fn vector_macro_covers_the_64_bit_widths() {
    let scalar: Vector<i64, 1> = vector![1];
    assert_eq!(scalar.to_array(), [1]);

    let repeated: Vector<u64, 3> = vector![2; 3];
    assert_eq!(repeated.to_array(), [2, 2, 2]);

    let inferred_float: Vector<f64, 2> = vector![1.0, 2.0];
    assert_eq!(inferred_float.to_array(), [1.0, 2.0]);

    let one = Vector::from_array([1_i64]);
    let two = Vector::from_array([2_i64, 3]);
    let three = Vector::from_array([2_i64, 3, 4]);

    assert_eq!(vector![one, one].to_array(), [1, 1]);
    assert_eq!(vector![one, two].to_array(), [1, 2, 3]);
    assert_eq!(vector![one, three].to_array(), [1, 2, 3, 4]);
    assert_eq!(vector![two, one].to_array(), [2, 3, 1]);
    assert_eq!(vector![two, two].to_array(), [2, 3, 2, 3]);
    assert_eq!(vector![three, one].to_array(), [2, 3, 4, 1]);

    assert_eq!(vector![one, one, one].to_array(), [1, 1, 1]);
    assert_eq!(vector![one, one, two].to_array(), [1, 1, 2, 3]);
    assert_eq!(vector![one, two, one].to_array(), [1, 2, 3, 1]);
    assert_eq!(vector![two, one, one].to_array(), [2, 3, 1, 1]);
    assert_eq!(vector![one, one, one, one].to_array(), [1, 1, 1, 1]);
}
