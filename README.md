# algea

[![Crates.io](https://img.shields.io/crates/v/algea.svg)](https://crates.io/crates/algea)
[![Documentation](https://img.shields.io/docsrs/algea)](https://docs.rs/algea)
[![CI](https://github.com/atynagano/algea/actions/workflows/ci.yml/badge.svg)](https://github.com/atynagano/algea/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/atynagano/algea/blob/main/LICENSE-MIT)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/atynagano/algea/blob/main/LICENSE-APACHE)

Portable, SIMD-accelerated vectors and matrices.

`algea` provides small fixed-size algebra types whose storage and operations
are selected for the target. It supports `f32`, `f64`, `i32`, `i64`, `u32`, and
`u64` elements, with vector dimensions and matrix row and column counts from one
through four. Both [`row_major`] and [`column_major`] matrix storage are
supported.

Implementations pursue target-specific performance, so floating-point results
are not guaranteed to be bit-for-bit identical across targets or target-feature
sets.

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

Elements convert between types with `cast`:

```rust
use algea::vector;

let v = vector![1.5_f64, -2.5, 3.5];

assert_eq!(v.cast::<i32>(), vector![1_i32, -2, 3]);
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

Operations follow `std::simd::Simd`, not the scalar operators, wherever the two
differ. Integer addition, subtraction, multiplication, and negation therefore
wrap in both debug and release builds; integer division by zero panics; and
dividing the most negative value by `-1` wraps back to that value. SIMD padding
lanes do not participate in public operations or panic checks.

Casting an element follows Rust's `as` conversion, as `std::simd::Simd::cast`
does, for every pair of supported types. That includes saturation at the
destination's limits and `NaN` becoming zero when a floating-point value becomes
an integer, and the sign of zero when a floating-point value changes width. The
bit pattern of a `NaN` result is not specified.

## Comparison and alternatives

`algea` focuses on portable, target-selected SIMD implementations of small
fixed-size vectors and matrices, with explicit row-major and column-major
storage. Depending on the application, another crate may be a better fit.

✅ supported, ⚠️ conditional or partial, see the footnote, ❌ not supported.

| Crate | x86 SIMD | ARM NEON | Wasm SIMD | Generic element | Generic size | Integer vectors | Lane-wise vector `*` | Non-square matrices | Column-major | Row-major | Same results across targets | Three-lane vector | 4x4 matrix |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | --- | --- |
| `algea` | ✅ | ✅ | ✅ | ✅ | ⚠️ [^four] | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ [^algeabits] | `Vector<f32, 3>` | `Matrix<f32, 4, 4>` |
| [`glam` 0.33](https://docs.rs/glam) | ✅ [^glam] | ✅ [^glam] | ✅ [^glam] | ❌ [^concrete] | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ [^glambits] | `Vec3`, `Vec3A` [^glam] | `Mat4` |
| [`nalgebra` 0.35](https://www.nalgebra.rs) | ⚠️ [^simba] | ⚠️ [^simba] | ⚠️ [^simba] | ✅ | ✅ [^dynamic] | ✅ | ❌ [^nalgmul] | ✅ | ✅ | ❌ | ✅ [^scalar] | `Matrix<f32, U3, U1, ArrayStorage<f32, 3, 1>>` [^alias] | `Matrix<f32, U4, U4, ArrayStorage<f32, 4, 4>>` [^alias] |
| [`cgmath` 0.18](https://docs.rs/cgmath) | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ [^cgmul] | ❌ | ✅ | ❌ | ✅ [^scalar] | `Vector3<f32>` | `Matrix4<f32>` |
| [`euclid` 0.22](https://docs.rs/euclid) | ❌ | ❌ | ❌ | ✅ [^units] | ❌ | ✅ | ❌ [^euclidmul] | ❌ | ✅ | ❌ | ✅ [^scalar] | `Vector3D<f32, U>` | `Transform3D<f32, S, D>` |
| [`ultraviolet` 0.9](https://docs.rs/ultraviolet) | ⚠️ [^wide] | ⚠️ [^wide] | ⚠️ [^wide] | ❌ [^concrete] | ❌ | ⚠️ [^uvint] | ✅ | ❌ | ✅ | ❌ | ⚠️ [^wide] | `Vec3` | `Mat4` |
| [`vek` 0.17](https://docs.rs/vek) | ⚠️ [^nightly] | ⚠️ [^nightly] | ⚠️ [^nightly] | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ [^vek] | ✅ [^scalar] | `Vec3<f32>` | `Mat4<f32>` [^vek] |
| [`pathfinder_geometry` 0.5](https://docs.rs/pathfinder_geometry) | ✅ | ⚠️ [^pf] | ❌ | ❌ [^concrete] | ❌ | ⚠️ [^pfint] | ✅ | ❌ | ✅ | ❌ | ❌ | `Vector3F` | `Transform4F` |

[^nalgmul]: `*` is the matrix product, and a vector is a one-column matrix, so
    `v * w` does not compile. The element-wise product is the `component_mul`
    method.

[^cgmul]: `*` multiplies by a scalar. The element-wise product is
    `ElementWise::mul_element_wise`.

[^euclidmul]: `*` multiplies by a scalar or applies a transform. The
    element-wise product is the `component_mul` method.

[^four]: Vector dimensions and matrix row and column counts are generic
    parameters, but only one through four are implemented.

[^algeabits]: `algea` picks the order of additions and whether to use fused
    multiply-add per target, so a floating-point result can differ between
    targets, between target-feature sets of one target, and between versions of
    the crate. Whether the other crates hold their arithmetic fixed from one
    version to the next is not stated here either way.

[^glam]: On by default and used for `Vec3A`, `Mat4` and the other 16-byte-aligned
    types; the `scalar-math` feature turns it off and `core-simd` swaps it for
    `std::simd`.

[^glambits]: Documented as the default, with the `fast-math` feature as the
    opt-out that allows platform-specific optimizations.

[^scalar]: Follows from having no target-specific code paths, rather than from a
    documented guarantee.

[^concrete]: Separate concrete types per element type rather than one generic
    type.

[^simba]: SIMD comes from `simba`'s SIMD element types, which put several
    matrices in the lanes of one value rather than the lanes of one matrix. The
    crate itself has no target-specific code paths.

[^dynamic]: Also `SMatrix<f32, R, C>` for any static size, dimensions beyond
    four, dynamically sized matrices, and decompositions.

[^alias]: Usually written through the aliases `Vector3<f32>` and `Matrix4<f32>`.

[^units]: Also generic over a unit marker, so lengths in different spaces do not
    mix.

[^wide]: In the wide types, `Vec3x4` and `Mat4x4` for example. Each of those
    holds four or eight separate vectors or matrices, one per lane, and follows
    `wide`'s per-target paths; a lone `Vec3` or `Mat4` is plain scalar code.

[^uvint]: Needs the `int` feature, which is not enabled by default.

[^nightly]: The `repr_simd` feature unlocks SIMD variants of the types, and needs
    a nightly compiler.

[^vek]: `Mat4<f32>` is the column-major type; the row-major one is
    `vek::mat::repr_c::row_major::Mat4<f32>`.

[^pf]: Only on a nightly compiler.

[^pfint]: `Vector2I` only.

## Layout

> **Note:** The layouts documented below describe the current implementation
> only. This crate does not guarantee that memory layout, size, alignment,
> padding, or element placement remains unchanged across crate versions,
> platforms, or selected backends. These details may change; do not rely on them
> for serialization, persistent data, FFI, or reinterpreting memory.

When the SIMD backend is selected, vectors have the following layouts. Sizes and
alignments are in bytes. In both tables, a 32-bit `T` is `f32`, `i32`, or `u32`,
and a 64-bit `T` is `f64`, `i64`, or `u64`.

| Type | Size (32-bit) | Alignment (32-bit) | Size (64-bit) | Alignment (64-bit) |
|---|---:|---:|---:|---:|
| `Vector<T, 1>` | 4 | 4 | 8 | 8 |
| `Vector<T, 2>` | 8 | 8 | 16 | 16 |
| `Vector<T, 3>` | 16 | 16 | 32 | 32 |
| `Vector<T, 4>` | 16 | 16 | 32 | 32 |

SIMD-backed matrices have the following layouts. Each column-major matrix and
the transposed row-major shape shown beside it use the same storage layout.

| Type | Size (32-bit) | Alignment (32-bit) | Size (64-bit) | Alignment (64-bit) |
|---|---:|---:|---:|---:|
| `column_major::Matrix<T, 1, 1>` / `row_major::Matrix<T, 1, 1>` | 4 | 4 | 8 | 8 |
| `column_major::Matrix<T, 2, 1>` / `row_major::Matrix<T, 1, 2>` | 8 | 8 | 16 | 16 |
| `column_major::Matrix<T, 3, 1>` / `row_major::Matrix<T, 1, 3>` | 16 | 16 | 32 | 32 |
| `column_major::Matrix<T, 4, 1>` / `row_major::Matrix<T, 1, 4>` | 16 | 16 | 32 | 32 |
| `column_major::Matrix<T, 1, 2>` / `row_major::Matrix<T, 2, 1>` | 8 | 8 | 16 | 16 |
| `column_major::Matrix<T, 2, 2>` / `row_major::Matrix<T, 2, 2>` | 16 | 16 | 32 | 32 |
| `column_major::Matrix<T, 3, 2>` / `row_major::Matrix<T, 2, 3>` | 32 | 16 | 64 | 32 |
| `column_major::Matrix<T, 4, 2>` / `row_major::Matrix<T, 2, 4>` | 32 | 16 | 64 | 32 |
| `column_major::Matrix<T, 1, 3>` / `row_major::Matrix<T, 3, 1>` | 16 | 16 | 32 | 32 |
| `column_major::Matrix<T, 2, 3>` / `row_major::Matrix<T, 3, 2>` | 32 | 16 | 48 or 64 | 16 or 32 |
| `column_major::Matrix<T, 3, 3>` / `row_major::Matrix<T, 3, 3>` | 48 | 16 | 96 | 32 |
| `column_major::Matrix<T, 4, 3>` / `row_major::Matrix<T, 3, 4>` | 48 | 16 | 96 | 32 |
| `column_major::Matrix<T, 1, 4>` / `row_major::Matrix<T, 4, 1>` | 16 | 16 | 32 | 32 |
| `column_major::Matrix<T, 2, 4>` / `row_major::Matrix<T, 4, 2>` | 32 | 16 | 64 | 32 |
| `column_major::Matrix<T, 3, 4>` / `row_major::Matrix<T, 4, 3>` | 64 | 16 | 128 | 32 |
| `column_major::Matrix<T, 4, 4>` / `row_major::Matrix<T, 4, 4>` | 64 | 16 | 128 | 32 |

The 2x3 shape is the one whose 64-bit storage depends on the target: three
two-lane units, 48 bytes with alignment 16, unless a four-lane 64-bit value
occupies one register, where it is two four-lane units, 64 bytes with
alignment 32.

When the non-SIMD backend is selected, storage uses scalar arrays instead. A
`Vector<T, D>` then has the size of `D` elements and the alignment of one, and a
matrix has the size of `R * C` elements with the same alignment. The SIMD tables
above therefore do not describe non-SIMD layouts.

## Platform requirements

The current prototype depends on the `wide` SIMD backend and the Rust standard
library. It is not a `no_std` crate.

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your
option.
