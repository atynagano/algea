//! Smoke tests for the public API.

use algea::{EachOrd, Mask, Select, Vector, row_major::Matrix};

fn consume<T>(_: T) {}

#[test]
fn released_vector_and_mask_api_is_callable() {
    let mut f = Vector::from_array([1.25_f32, 2.5, 4.0]);
    let g = Vector::<f32, 3>::splat(2.0);
    consume(Vector::<f32, 3>::ZERO);
    consume(Vector::<f32, 3>::ONE);
    consume(f + g);
    consume(f - g);
    consume(f * g);
    consume(f / g);
    consume(f + 2.0);
    consume(2.0 - f);
    consume(f * 2.0);
    consume(8.0 / f);
    consume(f.dot(g));
    consume(f.norm());
    consume(f.norm_squared());
    consume(f.distance(g));
    consume(f.distance_squared(g));
    consume(f.normalize());
    consume(f.floor());
    consume(f.ceil());
    consume(f.round());
    consume(f.round_ties_even());
    consume(f.trunc());
    consume(f.fract());
    consume(f.sqrt());
    consume(f.recip());
    consume(f.abs());

    consume(f == g);
    consume(f != g);
    consume(f.each_eq(g));
    consume(f.each_ne(g));
    consume(f.each_lt(g));
    consume(f.each_le(g));
    consume(f.each_gt(g));
    consume(f.each_ge(g));
    consume(f.each_max(g));
    consume(f.each_min(g));
    consume(f.each_clamp(Vector::ZERO, Vector::splat(8.0)));

    consume(f[0]);
    f[0] = 3.0;
    consume(f.as_array());
    f.as_mut_array()[1] = 5.0;
    consume((f.x, f.y, f.z));
    f.z = 6.0;
    consume(<[f32; 3]>::from(f));

    let i = Vector::<i32, 3>::from([-4, 6, 8]);
    let j = Vector::<i32, 3>::from([2, 3, 4]);
    consume(i + j);
    consume(i - j);
    consume(i * j);
    consume(i / j);
    consume(-i);
    consume(!i);
    consume(i.abs());
    consume(i.abs_diff(j));
    consume(i.cast_unsigned());
    consume(i.each_max(j));
    consume(i.each_min(j));
    consume(i.each_clamp(Vector::splat(-8), Vector::splat(8)));
    consume([i, j].into_iter().sum::<Vector<i32, 3>>());
    consume([i, j].into_iter().product::<Vector<i32, 3>>());

    let u = Vector::<u32, 3>::from([4, 6, 8]);
    let v = Vector::<u32, 3>::from([2, 3, 4]);
    consume(u + v);
    consume(u - v);
    consume(u * v);
    consume(u / v);
    consume(!u);
    consume(u.cast_signed());
    consume(u.each_max(v));
    consume(u.each_min(v));
    consume(u.each_clamp(Vector::splat(1), Vector::splat(8)));

    let mask: Mask<i32, 3> = i.each_lt(j);
    consume(mask.all());
    consume(mask.any());
    consume(mask.select(i, j));
    consume(!mask);
}

#[test]
fn released_matrix_api_is_callable() {
    let rows = [[1.0_f32, 2.0], [3.0, 5.0]];
    let mut a = Matrix::<f32, 2, 2>::from_rows(rows);
    let b =
        Matrix::<f32, 2, 2>::from_row_vecs([Vector::from([2.0, 1.0]), Vector::from([1.0, 2.0])]);
    consume(Matrix::<f32, 2, 2>::ZERO);
    consume(Matrix::<f32, 2, 2>::ONE);
    consume(Matrix::<f32, 2, 2>::IDENTITY);
    consume(Matrix::<f32, 2, 2>::filled(3.0));
    consume(a + b);
    consume(a - b);
    consume(a + 2.0);
    consume(2.0 - a);
    consume(a * 2.0);
    consume(2.0 / a);
    consume(a * b);
    consume(a.transpose());
    consume(a.diagonal());
    consume(a.inverse());
    consume(a.determinant());
    consume(a == b);
    consume(a[(0, 1)]);
    a[(1, 0)] = 7.0;
    consume(<[[f32; 2]; 2]>::from(a));
    consume([a, b].into_iter().sum::<Matrix<f32, 2, 2>>());
    consume([a, b].into_iter().product::<Matrix<f32, 2, 2>>());

    let row = Vector::<f32, 2>::from([1.0, 2.0]);
    consume(row * a);
    consume(Matrix::<f32, 2, 1>::from_rows([[1.0], [2.0]]) * row);

    let integer = Matrix::<i32, 2, 3>::from_rows([[-1, 2, -3], [4, -5, 6]]);
    let unsigned = Matrix::<u32, 2, 3>::from_rows([[1, 2, 3], [4, 5, 6]]);
    consume(integer + integer);
    consume(integer - integer);
    consume(integer * 2);
    consume(2 / integer);
    consume(integer.transpose());
    consume(unsigned + unsigned);
    consume(unsigned - unsigned);
    consume(unsigned * 2);
    consume(12 / unsigned);
    consume(unsigned.transpose());
}
