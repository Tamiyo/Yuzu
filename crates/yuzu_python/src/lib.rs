use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass]
pub struct CompileOptions {
    #[pyo3(get)]
    pub debug_tokens: bool,

    #[pyo3(get)]
    pub debug_ast: bool,

    #[pyo3(get)]
    pub debug_hir: bool,

    #[pyo3(get)]
    pub debug_anf: bool,

    #[pyo3(get)]
    pub debug_reduce: bool,

    #[pyo3(get)]
    pub debug_substrait: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl CompileOptions {
    #[new]
    #[pyo3(signature = (
        debug_tokens = false,
        debug_ast = false,
        debug_hir = false,
        debug_anf = false,
        debug_reduce = false,
        debug_substrait = false,
    ))]
    fn new(
        debug_tokens: bool,
        debug_ast: bool,
        debug_hir: bool,
        debug_anf: bool,
        debug_reduce: bool,
        debug_substrait: bool,
    ) -> Self {
        Self {
            debug_tokens,
            debug_ast,
            debug_hir,
            debug_anf,
            debug_reduce,
            debug_substrait,
        }
    }
}

impl From<&CompileOptions> for yuzu_driver::CompileOptions {
    fn from(options: &CompileOptions) -> Self {
        Self {
            debug_tokens: options.debug_tokens,
            debug_ast: options.debug_ast,
            debug_hir: options.debug_hir,
            debug_anf: options.debug_anf,
            debug_reduce: options.debug_reduce,
            debug_plan: false,
            debug_substrait: options.debug_substrait,
            time_phases: false,
            target: None,
        }
    }
}

/// Compile Yuzu source to a Substrait plan, returned as protobuf bytes.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (source, options = None))]
fn compile<'py>(
    py: Python<'py>,
    source: &str,
    options: Option<&CompileOptions>,
) -> PyResult<Bound<'py, PyBytes>> {
    let options = options
        .map(yuzu_driver::CompileOptions::from)
        .unwrap_or_default();
    let resolver = yuzu_driver::modules::MapResolver(HashMap::new());
    match yuzu_driver::compile_to_substrait_mlir("<python>", source, &options, &resolver) {
        Ok(plan) => Ok(PyBytes::new(py, &plan)),
        Err(message) => Err(PyValueError::new_err(message)),
    }
}

#[pymodule]
fn yuzu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompileOptions>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
