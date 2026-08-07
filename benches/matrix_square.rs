use std::hint::black_box;

use algea::column_major::Matrix as ColumnMatrix;
use criterion::Criterion;

use crate::common::assert_close;

const MAT2_ROWS: [[f32; 2]; 2] = [[4.0, 7.0], [2.0, 6.0]];
const MAT3_ROWS: [[f32; 3]; 3] = [[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]];
const MAT4_ROWS: [[f32; 4]; 4] =
    [[5.0, 7.0, 9.0, 10.0], [2.0, 3.0, 3.0, 8.0], [8.0, 10.0, 2.0, 3.0], [3.0, 3.0, 4.0, 8.0]];

fn columns<const N: usize>(rows: [[f32; N]; N]) -> [[f32; N]; N] {
    core::array::from_fn(|column| core::array::from_fn(|row| rows[row][column]))
}

fn flatten_columns<const N: usize>(columns: [[f32; N]; N]) -> Vec<f32> {
    columns.into_iter().flatten().collect()
}

#[inline(never)]
fn inverse_column_4x4(matrix: &ColumnMatrix<f32, 4, 4>) -> ColumnMatrix<f32, 4, 4> {
    (*matrix).inverse()
}

macro_rules! register_size {
    ($criterion:expr, $size:literal, $rows:expr, $glam:ident, $column_inverse:expr, $glam_inverse:expr) => {{
        let rows = $rows;
        let algebra_columns = ColumnMatrix::<f32, $size, $size>::from_columns(columns(rows));
        let glam = glam::$glam::from_cols_array_2d(&columns(rows));

        let expected_inverse = glam.inverse().to_cols_array();
        assert_close(
            concat!("algea column-major inverse ", stringify!($size), "x", stringify!($size)),
            &flatten_columns(algebra_columns.inverse().to_columns()),
            &expected_inverse,
        );
        assert_close(
            concat!("algea column-major determinant ", stringify!($size), "x", stringify!($size)),
            &[algebra_columns.determinant()],
            &[glam.determinant()],
        );

        let mut group = $criterion.benchmark_group(concat!(
            "matrix/inverse/",
            stringify!($size),
            "x",
            stringify!($size)
        ));
        group.bench_function("algea/column_major", |bencher| {
            bencher.iter(|| {
                let matrix = black_box(&algebra_columns);
                let result = ($column_inverse)(matrix);
                black_box(&result);
            })
        });
        group.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let matrix = black_box(&glam);
                let result = ($glam_inverse)(matrix);
                black_box(&result);
            })
        });
        group.finish();

        let mut group = $criterion.benchmark_group(concat!(
            "matrix/determinant/",
            stringify!($size),
            "x",
            stringify!($size)
        ));
        group.bench_function("algea/column_major", |bencher| {
            bencher.iter(|| {
                let matrix = *black_box(&algebra_columns);
                black_box(matrix.determinant());
            })
        });
        group.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let matrix = *black_box(&glam);
                black_box(matrix.determinant());
            })
        });
        group.finish();
    }};
}

pub fn register(criterion: &mut Criterion) {
    register_size!(
        criterion,
        2,
        MAT2_ROWS,
        Mat2,
        |matrix: &ColumnMatrix<f32, 2, 2>| (*matrix).inverse(),
        |matrix: &glam::Mat2| matrix.inverse()
    );
    register_size!(
        criterion,
        3,
        MAT3_ROWS,
        Mat3,
        |matrix: &ColumnMatrix<f32, 3, 3>| (*matrix).inverse(),
        |matrix: &glam::Mat3| matrix.inverse()
    );
    register_size!(criterion, 4, MAT4_ROWS, Mat4, inverse_column_4x4, |matrix: &glam::Mat4| matrix
        .inverse());
}
