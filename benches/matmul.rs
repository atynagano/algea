use std::hint::black_box;

use algea::column_major::Matrix as ColumnMatrix;
use criterion::Criterion;

use crate::common::assert_close;

const LHS_ROWS: [[f32; 4]; 4] =
    [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, 3.0], [4.0, -1.0, 2.0, 0.25], [0.75, 1.25, -3.0, 2.5]];
const RHS_ROWS: [[f32; 4]; 4] =
    [[0.5, -1.0, 2.0, 3.0], [2.5, 1.5, -0.5, 4.0], [-2.0, 0.25, 1.0, -1.5], [3.5, -2.5, 0.75, 2.0]];

fn columns(rows: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    core::array::from_fn(|column| core::array::from_fn(|row| rows[row][column]))
}

fn flatten(rows: [[f32; 4]; 4]) -> [f32; 16] {
    core::array::from_fn(|index| rows[index / 4][index % 4])
}

fn flatten_columns(columns: [[f32; 4]; 4]) -> [f32; 16] {
    core::array::from_fn(|index| columns[index / 4][index % 4])
}

fn multiply_rows(lhs: [[f32; 4]; 4], rhs: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| (0..4).map(|k| lhs[row][k] * rhs[k][column]).sum())
    })
}

pub fn register(criterion: &mut Criterion) {
    let lhs_columns = ColumnMatrix::<f32, 4, 4>::from_columns(columns(LHS_ROWS));
    let rhs_columns = ColumnMatrix::<f32, 4, 4>::from_columns(columns(RHS_ROWS));
    let lhs_nalgebra = nalgebra::Matrix4::from_row_slice(&flatten(LHS_ROWS));
    let rhs_nalgebra = nalgebra::Matrix4::from_row_slice(&flatten(RHS_ROWS));
    let lhs_glam = glam::Mat4::from_cols_array_2d(&columns(LHS_ROWS));
    let rhs_glam = glam::Mat4::from_cols_array_2d(&columns(RHS_ROWS));

    let expected_rows = multiply_rows(LHS_ROWS, RHS_ROWS);
    let expected_columns = flatten_columns(columns(expected_rows));
    assert_close(
        "algea column-major mat4 multiply",
        &flatten_columns((lhs_columns * rhs_columns).to_columns()),
        &expected_columns,
    );
    assert_close(
        "nalgebra mat4 multiply",
        (lhs_nalgebra * rhs_nalgebra).as_slice(),
        &expected_columns,
    );
    assert_close("glam mat4 multiply", &(lhs_glam * rhs_glam).to_cols_array(), &expected_columns);

    let mut group = criterion.benchmark_group("matrix/mul/4x4");
    group.bench_function("algea/column_major", |bencher| {
        bencher.iter(|| {
            let lhs = *black_box(&lhs_columns);
            let rhs = *black_box(&rhs_columns);
            let result = lhs * rhs;
            black_box(&result);
        })
    });
    group.bench_function("nalgebra", |bencher| {
        bencher.iter(|| {
            let lhs = *black_box(&lhs_nalgebra);
            let rhs = *black_box(&rhs_nalgebra);
            let result = lhs * rhs;
            black_box(&result);
        })
    });
    group.bench_function("glam", |bencher| {
        bencher.iter(|| {
            let lhs = *black_box(&lhs_glam);
            let rhs = *black_box(&rhs_glam);
            let result = lhs * rhs;
            black_box(&result);
        })
    });
    group.finish();
}
