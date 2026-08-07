//! Tests for column-major matrix operations.

use algea::{Vector, column_major::Matrix, row_major};

#[test]
fn array_and_vector_views_preserve_logical_rows_and_columns() {
    let rows = [[1_i32, 2, 3], [4, 5, 6]];
    let columns = [[1_i32, 4], [2, 5], [3, 6]];

    let row_major = row_major::Matrix::<i32, 2, 3>::from_rows(rows);
    assert_eq!(row_major.to_rows(), rows);
    assert_eq!(row_major.to_row_vecs().map(Vector::to_array), rows);

    let column_major = Matrix::<i32, 2, 3>::from_columns(columns);
    assert_eq!(column_major.to_columns(), columns);
    assert_eq!(column_major.to_column_vecs().map(Vector::to_array), columns);

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(row_major[(i, j)], column_major[(i, j)]);
        }
    }
}

#[test]
fn columns_round_trip_and_index_by_logical_coordinates() {
    let columns = [[1_i32, 2, 3], [4, 5, 6]];
    let mut matrix = Matrix::<i32, 3, 2>::from_columns(columns);

    assert_eq!(matrix[(0, 0)], 1);
    assert_eq!(matrix[(2, 0)], 3);
    assert_eq!(matrix[(0, 1)], 4);
    assert_eq!(matrix[(2, 1)], 6);

    matrix[(1, 1)] = 50;
    assert_eq!(<[[i32; 3]; 2]>::from(matrix), [[1, 2, 3], [4, 50, 6]]);
}

#[test]
fn matrix_products_follow_column_major_logical_shapes() {
    let lhs = Matrix::<f32, 2, 3>::from_columns([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let rhs = Matrix::<f32, 3, 2>::from_columns([[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]);

    let product: [[f32; 2]; 2] = (lhs * rhs).into();
    assert_eq!(product, [[58.0, 139.0], [64.0, 154.0]]);

    let matrix_vector: [f32; 2] = (lhs * Vector::from([1.0, 2.0, 3.0])).into();
    assert_eq!(matrix_vector, [14.0, 32.0]);

    let outer: [[f32; 2]; 2] =
        (Vector::from([2.0, 3.0]) * Matrix::<f32, 1, 2>::from_columns([[4.0], [5.0]])).into();
    assert_eq!(outer, [[8.0, 12.0], [10.0, 15.0]]);
}

#[test]
fn elementwise_and_scalar_operations_use_column_storage() {
    let a = Matrix::<i32, 2, 3>::from_columns([[1, 2], [3, 4], [5, 6]]);
    let b = Matrix::<i32, 2, 3>::from_columns([[10, 20], [30, 40], [50, 60]]);

    assert_eq!(<[[i32; 2]; 3]>::from(a + b), [[11, 22], [33, 44], [55, 66]]);
    assert_eq!(<[[i32; 2]; 3]>::from(100 - a), [[99, 98], [97, 96], [95, 94]]);
    assert_eq!(a, a.clone());
}
