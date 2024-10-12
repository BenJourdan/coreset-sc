// src/lib.rs

mod rust;

// Import necessary dependencies from PyO3
#[cfg(feature = "bindings")]
use pyo3::prelude::*;


// Python bindings: conditionally included if the "bindings" feature is enabled
#[cfg(feature = "bindings")]
#[allow(non_snake_case)]
#[pymodule]
fn coreset_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "sum_as_string")]
    fn sum_as_string_py(a: usize, b: usize) -> PyResult<String>{
        Ok((a + b).to_string())
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust::rust_sum_as_string;
    // Pure Rust test
    #[test]
    fn test_rust_sum_as_string() {
        assert_eq!(rust_sum_as_string(1, 2), "3");
    }

}
