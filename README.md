# algea

Portable, SIMD-accelerated vectors and matrices.

`algea` provides small fixed-size algebra types whose storage and operations
are selected for the target. It currently supports `f32`, `i32`, and `u32`
elements, with vector dimensions and matrix row and column counts from one
through four. Both [`row_major`] and [`column_major`] matrix storage are
supported.

Implementations pursue target-specific performance, so floating-point results
are not guaranteed to be bit-for-bit identical across targets or target-feature
sets. Other element widths, including `f64`, `i64`, and `u64`, are not currently
supported but are planned for the future.

## Stability

Documented public APIs follow Semantic Versioning.

The `__internal` module is public solely to support implementation details that
must remain reachable across crate boundaries, including bounds on public APIs.
It is not a supported public API and may be changed, moved, or removed in any
release. Downstream crates must not refer to it directly.

## Examples

Vectors provide lane-wise arithmetic and common geometric operations:

```rust
use algea::vector;

let a = vector![1.0_f32, 2.0, 3.0];
let b = vector![4.0_f32, 5.0, 6.0];

assert_eq!(a + b, vector![5.0, 7.0, 9.0]);
assert_eq!(a.dot(b), 32.0);
```

Row-major matrices are constructed from rows:

```rust
use algea::row_major::Matrix;

let a = Matrix::<f32, 2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
let b = Matrix::<f32, 2, 2>::from_rows([[5.0, 6.0], [7.0, 8.0]]);

assert_eq!((a * b).to_rows(), [[19.0, 22.0], [43.0, 50.0]]);
```

Column-major matrices represent the same operations but are constructed from
columns:

```rust
use algea::column_major::Matrix;

let a = Matrix::<f32, 2, 2>::from_columns([[1.0, 3.0], [2.0, 4.0]]);
let b = Matrix::<f32, 2, 2>::from_columns([[5.0, 7.0], [6.0, 8.0]]);

assert_eq!((a * b).to_columns(), [[19.0, 43.0], [22.0, 50.0]]);
```

## Generic code

The [`Element`] family of traits expresses which element types and dimensions
are available. Generic code can state that requirement in either of the
following equivalent forms:

```rust
use algea::{Element, Vector};

fn by_parameter<T: Element<D>, const D: usize>(value: Vector<T, D>) {
    let _ = value;
}

fn by_where_clause<const D: usize>(value: Vector<f32, D>)
where
    f32: Element<D>,
{
    let _ = value;
}
```

## Arithmetic semantics

Integer addition, subtraction, multiplication, and negation wrap in both debug
and release builds. Integer division follows `std::simd::Simd`: division by zero
panics, while `i32::MIN / -1` wraps to `i32::MIN`. SIMD padding lanes do not
participate in public operations or panic checks.

## Comparison and alternatives

`algea` focuses on portable, target-selected SIMD implementations of small
fixed-size vectors and matrices, with explicit row-major and column-major
storage. Depending on the application, the following crates may be a better
fit:

- [`glam`](https://docs.rs/glam) provides concrete vector and matrix types aimed
  at games and graphics, including SIMD-backed types.
- [`nalgebra`](https://www.nalgebra.rs) provides broader linear algebra support,
  including statically and dynamically sized matrices and decompositions.
- [`cgmath`](https://docs.rs/cgmath) provides generic mathematics for computer
  graphics.
- [`euclid`](https://docs.rs/euclid) provides strongly typed geometry with unit
  markers, particularly for 2D graphics and layout.
- [`pathfinder_geometry`](https://docs.rs/pathfinder_geometry) provides geometry
  types used by the Pathfinder rendering ecosystem.
- [`ultraviolet`](https://docs.rs/ultraviolet) provides graphics-oriented linear
  algebra with scalar and SIMD-wide types.
- [`vek`](https://docs.rs/vek) provides generic vectors, matrices, and geometric
  transforms.

## Layout

> **Note:** The layouts documented below describe the current implementation
> only. This crate does not guarantee that memory layout, size, alignment,
> padding, or element placement remains unchanged across crate versions,
> platforms, or selected backends. These details may change; do not rely on them
> for serialization, persistent data, FFI, or reinterpreting memory.

When the SIMD backend is selected, vectors have the following layouts. Sizes and
alignments are in bytes. In both tables, `T` is `f32`, `i32`, or `u32`.

| Type | Size | Alignment |
|---|---:|---:|
| `Vector<T, 1>` | 4 | 4 |
| `Vector<T, 2>` | 8 | 8 |
| `Vector<T, 3>` | 16 | 16 |
| `Vector<T, 4>` | 16 | 16 |

SIMD-backed matrices have the following layouts. Each column-major matrix and
the transposed row-major shape shown beside it use the same storage layout.

| Type | Size | Alignment |
|---|---:|---:|
| `column_major::Matrix<T, 1, 1>` / `row_major::Matrix<T, 1, 1>` | 4 | 4 |
| `column_major::Matrix<T, 2, 1>` / `row_major::Matrix<T, 1, 2>` | 8 | 8 |
| `column_major::Matrix<T, 3, 1>` / `row_major::Matrix<T, 1, 3>` | 16 | 16 |
| `column_major::Matrix<T, 4, 1>` / `row_major::Matrix<T, 1, 4>` | 16 | 16 |
| `column_major::Matrix<T, 1, 2>` / `row_major::Matrix<T, 2, 1>` | 8 | 8 |
| `column_major::Matrix<T, 2, 2>` / `row_major::Matrix<T, 2, 2>` | 16 | 16 |
| `column_major::Matrix<T, 3, 2>` / `row_major::Matrix<T, 2, 3>` | 32 | 16 |
| `column_major::Matrix<T, 4, 2>` / `row_major::Matrix<T, 2, 4>` | 32 | 16 |
| `column_major::Matrix<T, 1, 3>` / `row_major::Matrix<T, 3, 1>` | 16 | 16 |
| `column_major::Matrix<T, 2, 3>` / `row_major::Matrix<T, 3, 2>` | 32 | 16 |
| `column_major::Matrix<T, 3, 3>` / `row_major::Matrix<T, 3, 3>` | 48 | 16 |
| `column_major::Matrix<T, 4, 3>` / `row_major::Matrix<T, 3, 4>` | 48 | 16 |
| `column_major::Matrix<T, 1, 4>` / `row_major::Matrix<T, 4, 1>` | 16 | 16 |
| `column_major::Matrix<T, 2, 4>` / `row_major::Matrix<T, 4, 2>` | 32 | 16 |
| `column_major::Matrix<T, 3, 4>` / `row_major::Matrix<T, 4, 3>` | 64 | 16 |
| `column_major::Matrix<T, 4, 4>` / `row_major::Matrix<T, 4, 4>` | 64 | 16 |

When the non-SIMD backend is selected, storage uses scalar arrays instead. A
`Vector<T, D>` then has size `4 * D` and alignment 4, and a matrix has size
`4 * R * C` and alignment 4. The SIMD tables above therefore do not describe
non-SIMD layouts.

## Platform requirements

The current prototype depends on the `wide` SIMD backend and the Rust standard
library. It is not a `no_std` crate.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your
option.
