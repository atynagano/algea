//! Tests for column-major storage layout.

use algea::{column_major::Matrix as ColumnMatrix, row_major::Matrix as RowMatrix};

macro_rules! assert_same_layout {
    ($column:ty, $row_transpose:ty) => {
        assert_eq!(
            size_of::<$column>(),
            size_of::<$row_transpose>(),
            "size: {}",
            stringify!($column)
        );
        assert_eq!(
            align_of::<$column>(),
            align_of::<$row_transpose>(),
            "align: {}",
            stringify!($column)
        );
    };
}

macro_rules! assert_all_layouts {
    ($t:ty) => {
        assert_same_layout!(ColumnMatrix<$t, 1, 1>, RowMatrix<$t, 1, 1>);
        assert_same_layout!(ColumnMatrix<$t, 1, 2>, RowMatrix<$t, 2, 1>);
        assert_same_layout!(ColumnMatrix<$t, 1, 3>, RowMatrix<$t, 3, 1>);
        assert_same_layout!(ColumnMatrix<$t, 1, 4>, RowMatrix<$t, 4, 1>);
        assert_same_layout!(ColumnMatrix<$t, 2, 1>, RowMatrix<$t, 1, 2>);
        assert_same_layout!(ColumnMatrix<$t, 2, 2>, RowMatrix<$t, 2, 2>);
        assert_same_layout!(ColumnMatrix<$t, 2, 3>, RowMatrix<$t, 3, 2>);
        assert_same_layout!(ColumnMatrix<$t, 2, 4>, RowMatrix<$t, 4, 2>);
        assert_same_layout!(ColumnMatrix<$t, 3, 1>, RowMatrix<$t, 1, 3>);
        assert_same_layout!(ColumnMatrix<$t, 3, 2>, RowMatrix<$t, 2, 3>);
        assert_same_layout!(ColumnMatrix<$t, 3, 3>, RowMatrix<$t, 3, 3>);
        assert_same_layout!(ColumnMatrix<$t, 3, 4>, RowMatrix<$t, 4, 3>);
        assert_same_layout!(ColumnMatrix<$t, 4, 1>, RowMatrix<$t, 1, 4>);
        assert_same_layout!(ColumnMatrix<$t, 4, 2>, RowMatrix<$t, 2, 4>);
        assert_same_layout!(ColumnMatrix<$t, 4, 3>, RowMatrix<$t, 3, 4>);
        assert_same_layout!(ColumnMatrix<$t, 4, 4>, RowMatrix<$t, 4, 4>);
    };
}

#[test]
fn all_column_major_layouts_match_transposed_row_major_layouts() {
    assert_all_layouts!(f32);
    assert_all_layouts!(i32);
    assert_all_layouts!(u32);
    assert_all_layouts!(f64);
    assert_all_layouts!(i64);
    assert_all_layouts!(u64);
}

#[test]
fn representative_column_storage_round_trips_are_stable() {
    let matrix_3x4 =
        ColumnMatrix::<i32, 3, 4>::from_columns([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
    let matrix_4x3 =
        ColumnMatrix::<i32, 4, 3>::from_columns([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    assert_eq!(<[[i32; 3]; 4]>::from(matrix_3x4), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],);
    assert_eq!(<[[i32; 4]; 3]>::from(matrix_4x3), [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],);
}
