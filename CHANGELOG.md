# Changelog

Notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-08-23

### Changed

- The crate compiles substantially faster and produces a smaller debug artifact.
  `cargo build --lib` spends 2.45 s in rustc rather than 8.43 s, and the debug
  rlib is 14,300 KiB rather than 48,117 KiB.
- Nothing in the public API or in the documented behaviour, and next to nothing
  in the generated code. Release assembly was compared for 1,614 public entry
  points -- every swizzle accessor, the matrix kernels, the lane-wise and
  comparison operations, masks and selection, integer and floating-point
  division, the casts and the array accessors -- on x86-64 SSE2, AVX2 and
  AVX-512. Of the 4,842 resulting functions, 4,776 are byte-identical to 0.2.0,
  60 are the same instructions in a different schedule, and 6 are one
  instruction shorter. None is longer.

## [0.2.0] - 2026-08-20

### Added

- The `f64`, `i64`, and `u64` element types. Each supports what the 32-bit
  element of the same kind supports: vectors of one through four lanes, row-major
  and column-major matrices of every shape from 1x1 through 4x4, masks and
  comparisons, casts to and from every other element type, and, for `f64`, the
  floating-point operations including matrix products, determinants and inverses.
  Every backend implements them, from x86-64 SSE2 through AVX-512, aarch64 NEON,
  WebAssembly SIMD128, and the non-SIMD fallback.
- `Default` for `Mask`, which returns a mask with every lane false.

### Changed

- The documented contract on floating-point results is stronger, and stricter for
  callers: a result may differ between targets, between target-feature sets of one
  target, between WebAssembly runtimes when `relaxed-simd` is enabled, and between
  versions of this crate. 0.1.0 said only that results were not guaranteed
  identical across targets or target-feature sets. Nothing may depend on a
  floating-point result being reproducible across any of those. See "Floating-point
  reproducibility" in the README.

## [0.1.0] - 2026-08-08

### Added

- Initial release. Vectors and matrices with `f32`, `i32`, and `u32` elements,
  dimensions and matrix row and column counts from one through four, row-major
  and column-major storage, and a backend selected for the target.

[Unreleased]: https://github.com/atynagano/algea/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/atynagano/algea/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/atynagano/algea/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/atynagano/algea/releases/tag/v0.1.0
