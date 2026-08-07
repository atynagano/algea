//! Multiplies non-square matrices in both supported storage orientations.

use algea::{column_major, row_major};

type RowMatrix2x3 = row_major::Matrix<f32, 2, 3>;
type RowMatrix3x2 = row_major::Matrix<f32, 3, 2>;
type ColumnMatrix2x3 = column_major::Matrix<f32, 2, 3>;
type ColumnMatrix3x2 = column_major::Matrix<f32, 3, 2>;

fn main() {
    let row_lhs = RowMatrix2x3::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let row_rhs = RowMatrix3x2::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let row_product = row_lhs * row_rhs;

    assert_eq!(row_product.to_rows(), [[58.0, 64.0], [139.0, 154.0]]);

    let column_lhs = ColumnMatrix2x3::from_columns([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    let column_rhs = ColumnMatrix3x2::from_columns([[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]);
    let column_product = column_lhs * column_rhs;

    assert_eq!(column_product.to_columns(), [[58.0, 139.0], [64.0, 154.0]],);

    println!("row-major product: {:?}", row_product.to_rows());
    println!("column-major product: {:?}", column_product.to_columns());
}
