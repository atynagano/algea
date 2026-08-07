#![allow(missing_docs)]

use std::{env, fmt::Write as _, fs, io, path::PathBuf};

const LANES: [(char, usize); 4] = [('x', 0), ('y', 1), ('z', 2), ('w', 3)];

fn emit_sequences(
    output: &mut String,
    lanes: &[(char, usize)],
    required_index: usize,
    remaining: usize,
    name: &mut String,
    indices: &mut Vec<usize>,
) -> usize {
    if remaining == 0 {
        if !indices.contains(&required_index) {
            return 0;
        }

        write!(output, "    {name} => [").unwrap();
        for (position, index) in indices.iter().enumerate() {
            if position != 0 {
                output.push_str(", ");
            }
            write!(output, "{index}").unwrap();
        }
        output.push_str("],\n");
        return 1;
    }

    let mut generated = 0;
    for &(lane, index) in lanes {
        name.push(lane);
        indices.push(index);
        generated += emit_sequences(output, lanes, required_index, remaining - 1, name, indices);
        indices.pop();
        name.pop();
    }
    generated
}

fn emit_impl(output: &mut String, dimension: usize) -> usize {
    writeln!(output, "impl_swizzles!({dimension}; {{").unwrap();

    let mut generated = 0;
    let mut name = String::new();
    let mut indices = Vec::with_capacity(4);
    // TODO(api-cleanup): Decide whether x(), y(), z(), and w() should return T instead of
    // Vector<T, 1> before generating one-lane swizzles.
    for length in 2..=4 {
        generated += emit_sequences(
            output,
            &LANES[..dimension],
            dimension - 1,
            length,
            &mut name,
            &mut indices,
        );
    }

    output.push_str("});\n\n");
    generated
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let mut output = String::new();
    let generated_per_dimension =
        [emit_impl(&mut output, 2), emit_impl(&mut output, 3), emit_impl(&mut output, 4)];

    assert_eq!(generated_per_dimension, [25, 89, 219]);
    assert_eq!(generated_per_dimension.iter().sum::<usize>(), 333);

    let output_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set")).join("swizzle_impls.rs");
    fs::write(output_path, output)
}
