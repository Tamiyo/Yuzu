//! Substrait's standard function catalogue, from the YAML the `substrait`
//! crate ships, as a table the translator searches at run time.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

use substrait::extensions::EXTENSIONS;
use substrait::text::simple_extensions::{Arguments, ArgumentsItem, Type};

/// One implementation: the function's name, its argument types, whether its
/// last argument repeats, and the URN of the extension that declares it.
type Row = (String, Vec<String>, bool, String);

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let mut rows: Vec<Row> = Vec::new();
    for extension in EXTENSIONS.values() {
        for function in &extension.scalar_functions {
            for item in &function.impls {
                rows.extend(row(
                    &function.name,
                    item.args.as_ref(),
                    item.variadic.is_some(),
                    &extension.urn,
                ));
            }
        }
        for function in &extension.aggregate_functions {
            for item in &function.impls {
                rows.extend(row(
                    &function.name,
                    item.args.as_ref(),
                    item.variadic.is_some(),
                    &extension.urn,
                ));
            }
        }
    }
    rows.sort();
    rows.dedup();

    let mut table = String::from("static CATALOGUE: &[(&str, &[&str], bool, &str)] = &[\n");
    for (name, params, variadic, urn) in &rows {
        writeln!(table, "    ({name:?}, &{params:?}, {variadic}, {urn:?}),")
            .expect("writing to a string cannot fail");
    }
    table.push_str("];\n");

    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets it"));
    fs::write(out.join("catalogue.rs"), table).expect("OUT_DIR is writable");
}

/// The implementation's row. `None` when it takes an argument that is not a
/// value: a call never passes an option or a type.
fn row(name: &str, args: Option<&Arguments>, variadic: bool, urn: &str) -> Option<Row> {
    let params = match args {
        Some(args) => args
            .iter()
            .map(|arg| match arg {
                ArgumentsItem::ValueArg(value) => match &value.value {
                    Type::String(ty) => Some(type_code(ty)),
                    Type::Object(_) => None,
                },
                ArgumentsItem::EnumerationArg(_) | ArgumentsItem::TypeArg(_) => None,
            })
            .collect::<Option<Vec<_>>>()?,
        None => Vec::new(),
    };
    Some((name.to_owned(), params, variadic, urn.to_owned()))
}

/// A catalogue type, as the translator's `type_code` spells an argument's
/// type. A generic type is `any`. A type the translator has no code for
/// keeps its base name, so it matches no argument.
fn type_code(ty: &str) -> String {
    let base = ty
        .trim_end_matches('?')
        .split('<')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    match base.as_str() {
        "boolean" => "bool".to_owned(),
        generic if generic.starts_with("any") => "any".to_owned(),
        _ => base,
    }
}
