use ndarray::Array;
use pyo3::prelude::*;

use pyo3::{exceptions::PyValueError, types::PyBytes};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::tokenizer::PyTokenizer;
use numpy::ndarray::ArrayD;
use numpy::IntoPyArray;

#[pyfunction]
fn encode<'py>(
    py: Python<'py>,
    tokenizer: &PyTokenizer,
    tokens: &TokenByteBuffer,
) -> PyResult<Bound<'py, PyAny>> {
    let core = Arc::clone(&tokens.core);
    let tokenizer_clone = tokenizer.tokenizer.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let results = tokio::task::spawn_blocking(move || {
            let binding = core.read().unwrap();
            match tokenizer_clone.encode(binding.as_cow_strs(), false) {
                Ok(encoding) => {
                    let ids = encoding.get_ids();
                    let array: ArrayD<i64> = Array::from_shape_vec(
                        vec![ids.len()],
                        ids.iter().map(|id| *id as i64).collect(),
                    )
                    .expect("Failed to create ndarray from ids");

                    Ok(array)
                }
                Err(e) => Err(format!("Failed to encode text: {}", e)),
            }
        })
        .await;

        Python::with_gil(|py| match results {
            Ok(Ok(array)) => Ok(array.into_pyarray(py).unbind()),
            Ok(Err(e)) => Err(PyValueError::new_err(e)),
            Err(e) => Err(PyValueError::new_err(format!("Task failed: {}", e))),
        })
    })
}

#[pymodule]
pub fn vllm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_class::<TokenByteBuffer>()?;
    Ok(())
}

pub struct TokenBufferCore {
    buffer: Vec<u8>,
    lengths: Vec<u16>,
    offsets: Vec<u32>,
    token_count: usize,
}

impl TokenBufferCore {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            lengths: Vec::new(),
            offsets: Vec::new(),
            token_count: 0,
        }
    }

    fn pack_tokens(&mut self, tokens: &[String], max_token_len: usize) -> Result<(), String> {
        self.buffer.clear();
        self.lengths.clear();
        self.offsets.clear();

        for token in tokens.iter() {
            let token_bytes = token.as_bytes();

            if token_bytes.len() > max_token_len {
                return Err(format!(
                    "Token exceeds max length: {} > {}",
                    token_bytes.len(),
                    max_token_len
                ));
            }

            if token_bytes.len() > u16::MAX as usize {
                return Err("Token too long for u16 length".to_string());
            }

            self.offsets.push(self.buffer.len() as u32);
            self.lengths.push(token_bytes.len() as u16);
            self.buffer.extend_from_slice(token_bytes);
        }

        self.token_count = self.lengths.len();
        Ok(())
    }

    fn get_token_bytes(&self, index: usize) -> Option<&[u8]> {
        if index >= self.token_count {
            return None;
        }

        let offset = self.offsets[index] as usize;
        let length = self.lengths[index] as usize;
        Some(&self.buffer[offset..offset + length])
    }

    /// Convert the stored tokens to a vector of `Cow<str>` for zero-copy string access
    fn as_cow_strs(&self) -> Vec<std::borrow::Cow<str>> {
        let mut tokens = Vec::with_capacity(self.token_count);
        for i in 0..self.token_count {
            let offset = self.offsets[i] as usize;
            let length = self.lengths[i] as usize;
            let token_bytes = &self.buffer[offset..offset + length];
            if let Ok(token_str) = std::str::from_utf8(token_bytes) {
                tokens.push(std::borrow::Cow::Borrowed(token_str));
            } else {
                tokens.push(std::borrow::Cow::Owned("Invalid UTF-8".to_string()));
            }
        }
        tokens
    }
}

#[pyclass]
pub struct TokenByteBuffer {
    core: Arc<RwLock<TokenBufferCore>>,
}

#[pymethods]
impl TokenByteBuffer {
    #[new]
    #[pyo3(signature = (tokens, max_token_len=None))]
    fn py_new(tokens: Vec<String>, max_token_len: Option<usize>) -> PyResult<Self> {
        let mut instance = Self {
            core: Arc::new(RwLock::new(TokenBufferCore::new())),
        };
        instance._pack_tokens(tokens, max_token_len)?;
        Ok(instance)
    }

    fn _pack_tokens(&mut self, tokens: Vec<String>, max_token_len: Option<usize>) -> PyResult<()> {
        let max_len = max_token_len.unwrap_or(1024);
        let mut core = self.core.write().unwrap();
        core.pack_tokens(&tokens, max_len)
            .map_err(PyValueError::new_err)
    }

    /// Get the total size of the buffer in bytes
    fn buffer_size(&self) -> usize {
        let core = self.core.read().unwrap();
        core.buffer.len()
    }

    /// Get the number of tokens stored
    fn token_count(&self) -> usize {
        let core = self.core.read().unwrap();
        core.token_count
    }

    /// Get raw buffer as bytes (zero-copy)
    fn get_buffer<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let core = self.core.read().unwrap();
        PyBytes::new(py, &core.buffer)
    }

    /// Get lengths array as numpy array
    fn get_lengths(&self, py: Python) -> PyResult<PyObject> {
        let core = self.core.read().unwrap();
        let array = numpy::PyArray1::from_slice(py, &core.lengths);
        Ok(array.into_pyobject(py)?.into())
    }

    /// Get offsets array as numpy array  
    fn get_offsets(&self, py: Python) -> PyResult<PyObject> {
        let core = self.core.read().unwrap();
        let array = numpy::PyArray1::from_slice(py, &core.offsets);
        Ok(array.into_pyobject(py)?.into())
    }

    /// Extract a specific token by index (for debugging)
    fn get_token(&self, index: usize) -> PyResult<String> {
        let core = self.core.read().unwrap();

        if let Some(token_bytes) = core.get_token_bytes(index) {
            String::from_utf8(token_bytes.to_vec())
                .map_err(|_| PyValueError::new_err("Invalid UTF-8 in token"))
        } else {
            Err(PyValueError::new_err("Token index out of bounds"))
        }
    }

    /// Memory usage statistics
    fn memory_stats(&self) -> PyResult<HashMap<String, usize>> {
        let core = self.core.read().unwrap();
        let mut stats = HashMap::new();
        stats.insert("buffer_bytes".to_string(), core.buffer.len());
        stats.insert("lengths_bytes".to_string(), core.lengths.len() * 2);
        stats.insert("offsets_bytes".to_string(), core.offsets.len() * 4);
        stats.insert(
            "total_bytes".to_string(),
            core.buffer.len() + (core.lengths.len() * 2) + (core.offsets.len() * 4),
        );
        stats.insert("token_count".to_string(), core.token_count);
        Ok(stats)
    }
}
