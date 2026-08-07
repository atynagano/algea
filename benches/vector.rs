use std::hint::black_box;

use algea::Vector;
use criterion::{BenchmarkId, Criterion, Throughput};

use crate::common::assert_close;

macro_rules! register_dimension {
    (
        $criterion:expr,
        $dimension:literal,
        $array:expr,
        $rhs_array:expr,
        $algebra_type:ty,
        $algebra_lhs:expr,
        $algebra_rhs:expr,
        $nalgebra_type:ty,
        $nalgebra_lhs:expr,
        $nalgebra_rhs:expr,
        $glam_type:ty,
        $glam_lhs:expr,
        $glam_rhs:expr,
        $glam_to_array:expr
    ) => {{
        let algebra_lhs: $algebra_type = $algebra_lhs;
        let algebra_rhs: $algebra_type = $algebra_rhs;
        let nalgebra_lhs: $nalgebra_type = $nalgebra_lhs;
        let nalgebra_rhs: $nalgebra_type = $nalgebra_rhs;
        let glam_lhs: $glam_type = $glam_lhs;
        let glam_rhs: $glam_type = $glam_rhs;

        let expected_add = algebra_lhs + algebra_rhs;
        assert_close(
            concat!("nalgebra vector", $dimension, " add"),
            (nalgebra_lhs + nalgebra_rhs).as_slice(),
            &expected_add.to_array(),
        );
        assert_close(
            concat!("glam vector", $dimension, " add"),
            &($glam_to_array)(glam_lhs + glam_rhs),
            &expected_add.to_array(),
        );

        let mut return_self =
            $criterion.benchmark_group(concat!("vector/return_self/", $dimension));
        return_self.bench_function("algea", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&algebra_lhs);
                black_box(&value);
            })
        });
        return_self.bench_function("nalgebra", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&nalgebra_lhs);
                black_box(&value);
            })
        });
        return_self.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&glam_lhs);
                black_box(&value);
            })
        });
        return_self.finish();

        let mut conversion = $criterion.benchmark_group(concat!("vector/conversion/", $dimension));
        conversion.bench_function("algea/from_array", |bencher| {
            bencher.iter(|| <$algebra_type>::from(black_box($array)))
        });
        conversion.bench_function("algea/to_array", |bencher| {
            bencher.iter(|| black_box(algebra_lhs).to_array())
        });
        conversion.finish();

        let mut add = $criterion.benchmark_group(concat!("vector/add/", $dimension));
        add.bench_function("algea", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&algebra_lhs);
                let rhs = *black_box(&algebra_rhs);
                let result = lhs + rhs;
                black_box(&result);
            })
        });
        add.bench_function("nalgebra", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&nalgebra_lhs);
                let rhs = *black_box(&nalgebra_rhs);
                let result = lhs + rhs;
                black_box(&result);
            })
        });
        add.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&glam_lhs);
                let rhs = *black_box(&glam_rhs);
                let result = lhs + rhs;
                black_box(&result);
            })
        });
        add.finish();

        let expected_max = algebra_lhs.each_max(algebra_rhs);
        assert_close(
            concat!("glam vector", $dimension, " max"),
            &($glam_to_array)(glam_lhs.max(glam_rhs)),
            &expected_max.to_array(),
        );

        let mut max = $criterion.benchmark_group(concat!("vector/max/", $dimension));
        max.bench_function("algea", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&algebra_lhs);
                let rhs = *black_box(&algebra_rhs);
                let result = lhs.each_max(rhs);
                black_box(&result);
            })
        });
        max.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&glam_lhs);
                let rhs = *black_box(&glam_rhs);
                let result = lhs.max(rhs);
                black_box(&result);
            })
        });
        max.finish();

        let expected_dot = algebra_lhs.dot(algebra_rhs);
        assert_close(
            concat!("nalgebra vector", $dimension, " dot"),
            &[nalgebra_lhs.dot(&nalgebra_rhs)],
            &[expected_dot],
        );
        assert_close(concat!("glam vector", $dimension, " dot"), &[glam_lhs.dot(glam_rhs)], &[
            expected_dot,
        ]);

        let mut dot = $criterion.benchmark_group(concat!("vector/dot/", $dimension));
        dot.bench_function("algea", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&algebra_lhs);
                let rhs = *black_box(&algebra_rhs);
                black_box(lhs.dot(rhs));
            })
        });
        dot.bench_function("nalgebra", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&nalgebra_lhs);
                let rhs = black_box(&nalgebra_rhs);
                black_box(lhs.dot(rhs));
            })
        });
        dot.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let lhs = *black_box(&glam_lhs);
                let rhs = *black_box(&glam_rhs);
                black_box(lhs.dot(rhs));
            })
        });
        dot.finish();

        assert_close(
            concat!("nalgebra vector", $dimension, " normalize"),
            nalgebra_lhs.normalize().as_slice(),
            &algebra_lhs.normalize().to_array(),
        );
        assert_close(
            concat!("glam vector", $dimension, " normalize"),
            &($glam_to_array)(glam_lhs.normalize()),
            &algebra_lhs.normalize().to_array(),
        );

        let mut normalize = $criterion.benchmark_group(concat!("vector/normalize/", $dimension));
        normalize.bench_function("algea", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&algebra_lhs);
                let result = value.normalize();
                black_box(&result);
            })
        });
        normalize.bench_function("nalgebra", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&nalgebra_lhs);
                let result = value.normalize();
                black_box(&result);
            })
        });
        normalize.bench_function("glam", |bencher| {
            bencher.iter(|| {
                let value = *black_box(&glam_lhs);
                let result = value.normalize();
                black_box(&result);
            })
        });
        normalize.finish();

        for element_count in [1_usize, 16, 256, 4096] {
            let algebra_lhs = vec![algebra_lhs; element_count];
            let algebra_rhs = vec![algebra_rhs; element_count];
            let nalgebra_lhs = vec![nalgebra_lhs; element_count];
            let nalgebra_rhs = vec![nalgebra_rhs; element_count];
            let glam_lhs = vec![glam_lhs; element_count];
            let glam_rhs = vec![glam_rhs; element_count];

            let mut throughput =
                $criterion.benchmark_group(concat!("vector/add_throughput/", $dimension));
            throughput.throughput(Throughput::Elements(element_count as u64));
            throughput.bench_with_input(
                BenchmarkId::new("algea", element_count),
                &element_count,
                |bencher, &element_count| {
                    let mut output = vec![<$algebra_type>::from($array); element_count];
                    bencher.iter(|| {
                        for index in 0..element_count {
                            output[index] =
                                *black_box(&algebra_lhs[index]) + *black_box(&algebra_rhs[index]);
                        }
                        black_box(&output);
                    });
                },
            );
            throughput.bench_with_input(
                BenchmarkId::new("nalgebra", element_count),
                &element_count,
                |bencher, &element_count| {
                    let mut output = vec![nalgebra_lhs[0]; element_count];
                    bencher.iter(|| {
                        for index in 0..element_count {
                            output[index] =
                                *black_box(&nalgebra_lhs[index]) + *black_box(&nalgebra_rhs[index]);
                        }
                        black_box(&output);
                    });
                },
            );
            throughput.bench_with_input(
                BenchmarkId::new("glam", element_count),
                &element_count,
                |bencher, &element_count| {
                    let mut output = vec![glam_lhs[0]; element_count];
                    bencher.iter(|| {
                        for index in 0..element_count {
                            output[index] =
                                *black_box(&glam_lhs[index]) + *black_box(&glam_rhs[index]);
                        }
                        black_box(&output);
                    });
                },
            );
            throughput.finish();
        }

        black_box($rhs_array);
    }};
}

pub fn register(criterion: &mut Criterion) {
    register_dimension!(
        criterion,
        "2",
        [1.25_f32, -2.5],
        [0.75_f32, 4.0],
        Vector<f32, 2>,
        Vector::from([1.25, -2.5]),
        Vector::from([0.75, 4.0]),
        nalgebra::Vector2<f32>,
        nalgebra::Vector2::new(1.25, -2.5),
        nalgebra::Vector2::new(0.75, 4.0),
        glam::Vec2,
        glam::Vec2::new(1.25, -2.5),
        glam::Vec2::new(0.75, 4.0),
        |value: glam::Vec2| value.to_array()
    );
    register_dimension!(
        criterion,
        "3",
        [1.25_f32, -2.5, 3.75],
        [0.75_f32, 4.0, -1.5],
        Vector<f32, 3>,
        Vector::from([1.25, -2.5, 3.75]),
        Vector::from([0.75, 4.0, -1.5]),
        nalgebra::Vector3<f32>,
        nalgebra::Vector3::new(1.25, -2.5, 3.75),
        nalgebra::Vector3::new(0.75, 4.0, -1.5),
        glam::Vec3A,
        glam::Vec3A::new(1.25, -2.5, 3.75),
        glam::Vec3A::new(0.75, 4.0, -1.5),
        |value: glam::Vec3A| value.to_array()
    );
    register_dimension!(
        criterion,
        "4",
        [1.25_f32, -2.5, 3.75, -4.5],
        [0.75_f32, 4.0, -1.5, 2.25],
        Vector<f32, 4>,
        Vector::from([1.25, -2.5, 3.75, -4.5]),
        Vector::from([0.75, 4.0, -1.5, 2.25]),
        nalgebra::Vector4<f32>,
        nalgebra::Vector4::new(1.25, -2.5, 3.75, -4.5),
        nalgebra::Vector4::new(0.75, 4.0, -1.5, 2.25),
        glam::Vec4,
        glam::Vec4::new(1.25, -2.5, 3.75, -4.5),
        glam::Vec4::new(0.75, 4.0, -1.5, 2.25),
        |value: glam::Vec4| value.to_array()
    );
}
