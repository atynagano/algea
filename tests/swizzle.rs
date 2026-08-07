#![allow(missing_docs)]

use algea::Vector;

#[test]
fn representative_swizzles_preserve_requested_lane_order() {
    assert_eq!(Vector::from_array([7_i32]).xxxx().to_array(), [7; 4]);

    let vector2 = Vector::from_array([1_i32, 2]);
    assert_eq!(vector2.yx().to_array(), [2, 1]);

    let vector3 = Vector::from_array([1_u32, 2, 3]);
    assert_eq!(vector3.zxy().to_array(), [3, 1, 2]);

    let vector4 = Vector::from_array([1.0_f32, 2.0, 3.0, 4.0]);
    assert_eq!(vector4.wzyx().to_array(), [4.0, 3.0, 2.0, 1.0]);
}
