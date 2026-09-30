use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use yuzu_driver::modules::Origin;
use yuzu_driver::stdlib::Engine;

pyo3_stub_gen::create_exception!(
    yuzu,
    CompileError,
    PyValueError,
    "A compile that failed: the program, or the target it names. The message says why."
);

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug)]
pub struct CompileOptions {
    /// The engine to compile for. `None` means `datafusion`.
    #[pyo3(get)]
    pub target: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CompileOptions {
    #[new]
    #[pyo3(signature = (target = None))]
    fn new(target: Option<String>) -> Self {
        Self { target }
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
    let engine = match options.and_then(|options| options.target.as_deref()) {
        Some(name) => name
            .parse::<Engine>()
            .map_err(|unknown| CompileError::new_err(unknown.to_string()))?,
        None => Engine::default(),
    };
    let options = yuzu_driver::CompileOptions {
        engine,
        ..yuzu_driver::CompileOptions::default()
    };
    let resolver = yuzu_driver::modules::MapResolver(HashMap::new());
    // The compiler keeps its state on the thread, so other Python threads
    // run while it works.
    let plan = py.detach(|| {
        yuzu_driver::compile(
            &Origin::Named("<python>".to_owned()),
            source,
            &options,
            &resolver,
        )
        .into_plan()
        .map(|plan| plan.to_protobuf())
    });
    match plan {
        Ok(plan) => Ok(PyBytes::new(py, &plan)),
        Err(error) => Err(CompileError::new_err(error.to_string())),
    }
}

#[pymodule]
fn yuzu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompileOptions>()?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
