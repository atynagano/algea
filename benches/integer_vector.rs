use std::hint::black_box;

use algea::{EachOrd, Vector};
use criterion::{BenchmarkId, Criterion, Throughput};

macro_rules! register_type {
    (
        $criterion:expr,
        $name:literal,
        $scalar:ty,
        $array:expr,
        $rhs_array:expr,
        $glam_type:ty,
        $glam_lhs:expr,
        $glam_rhs:expr
    ) => {{
        let algebra_lhs = Vector::<$scalar, 2>::from($array);
        let algebra_rhs = Vector::<$scalar, 2>::from($rhs_array);
        let glam_lhs: $glam_type = $glam_lhs;
        let glam_rhs: $glam_type = $glam_rhs;

        assert_eq!((algebra_lhs + algebra_rhs).to_array(), (glam_lhs + glam_rhs).to_array());
        assert_eq!(algebra_lhs.each_max(algebra_rhs).to_array(), glam_lhs.max(glam_rhs).to_array());

        let mut max = $criterion.benchmark_group(concat!("integer_vector/", $name, "/max/2"));
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

        let element_count = 1_usize;
        let algebra_lhs = vec![algebra_lhs; element_count];
        let algebra_rhs = vec![algebra_rhs; element_count];
        let glam_lhs = vec![glam_lhs; element_count];
        let glam_rhs = vec![glam_rhs; element_count];

        let mut add =
            $criterion.benchmark_group(concat!("integer_vector/", $name, "/add_throughput/2"));
        add.throughput(Throughput::Elements(element_count as u64));
        add.bench_with_input(
            BenchmarkId::new("algea", element_count),
            &element_count,
            |bencher, &element_count| {
                let mut output = vec![Vector::<$scalar, 2>::from($array); element_count];
                bencher.iter(|| {
                    for index in 0..element_count {
                        output[index] =
                            *black_box(&algebra_lhs[index]) + *black_box(&algebra_rhs[index]);
                    }
                    black_box(&output);
                });
            },
        );
        add.bench_with_input(
            BenchmarkId::new("glam", element_count),
            &element_count,
            |bencher, &element_count| {
                let mut output = vec![glam_lhs[0]; element_count];
                bencher.iter(|| {
                    for index in 0..element_count {
                        output[index] = *black_box(&glam_lhs[index]) + *black_box(&glam_rhs[index]);
                    }
                    black_box(&output);
                });
            },
        );
        add.finish();
    }};
}

pub fn register(criterion: &mut Criterion) {
    register_type!(
        criterion,
        "i32",
        i32,
        [17_i32, -29],
        [-11_i32, 41],
        glam::IVec2,
        glam::IVec2::new(17, -29),
        glam::IVec2::new(-11, 41)
    );
    register_type!(
        criterion,
        "u32",
        u32,
        [17_u32, 29],
        [11_u32, 41],
        glam::UVec2,
        glam::UVec2::new(17, 29),
        glam::UVec2::new(11, 41)
    );
}
