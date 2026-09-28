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
    pub debug_yzl: bool,

    #[pyo3(get)]
    pub debug_yzr: bool,

    #[pyo3(get)]
    pub debug_substrait: bool,

    /// The engine to compile for; `DataFusion` when it is `None`.
    #[pyo3(get)]
    pub target: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CompileOptions {
    #[new]
    #[pyo3(signature = (
        debug_yzl = false,
        debug_yzr = false,
        debug_substrait = false,
        target = None,
    ))]
    fn new(
        debug_yzl: bool,
        debug_yzr: bool,
        debug_substrait: bool,
        target: Option<String>,
    ) -> Self {
        Self {
            debug_yzl,
            debug_yzr,
            debug_substrait,
            target,
        }
    }
}

impl From<&CompileOptions> for yuzu_driver::CompileOptions {
    fn from(options: &CompileOptions) -> Self {
        Self {
            debug_yzl: options.debug_yzl,
            debug_yzr: options.debug_yzr,
            debug_substrait: options.debug_substrait,
            target: options.target.clone(),
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
    match yuzu_driver::compile_to_substrait("<python>", source, &options, &resolver) {
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
