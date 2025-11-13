use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use lazy_static::lazy_static;
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;

// 编译核心的 GPT-2 BPE 正则表达式
// 这与您 Python 代码中的 re.compile(...) 相同
lazy_static! {
    static ref CORE_BPE_RE: Regex =
        Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap();
}

/// 一个 BPE (Byte Pair Encoding) Tokenizer.
///
/// 这个类在 Rust 中实现，并通过 PyO3 暴露给 Python。
/// 它旨在与 Python Tokenizer 类保持 API 兼容。
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    /// 词汇表: token ID (usize) -> token (bytes)
    vocab: HashMap<usize, Vec<u8>>,

    /// 字典 (反向词汇表): token (bytes) -> token ID (usize)
    dictionary: HashMap<Vec<u8>, usize>,

    /// BPE 合并规则
    merges: Vec<(Vec<u8>, Vec<u8>)>,

    /// 特殊 token (字符串形式)，按长度降序排列
    special_tokens: Vec<String>,

    /// 用于分割特殊 token 的正则表达式
    special_tokens_re: Option<Regex>,
}

/// 内部辅助函数,用于执行一次 BPE 合并
fn merge_pair(pair: &(Vec<u8>, Vec<u8>), seq: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut new_seq = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i < seq.len() - 1 && &seq[i] == &pair.0 && &seq[i + 1] == &pair.1 {
            // 找到匹配，合并
            let mut merged = pair.0.clone();
            merged.extend_from_slice(&pair.1);
            new_seq.push(merged);
            i += 2;
        } else {
            // 未找到，原样 push
            new_seq.push(seq[i].clone());
            i += 1;
        }
    }
    new_seq
}

#[pymethods]
impl PyTokenizer {
    /// Python 的 __init__ 方法。
    /// 从已加载的 vocab 和 merges 数据构造 Tokenizer。
    #[new]
    #[pyo3(signature = (vocab, merges, special_tokens = None))]
    fn new(
        vocab: HashMap<String, usize>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut vocab_int_bytes: HashMap<usize, Vec<u8>> = HashMap::new();
        let mut dictionary_bytes_int: HashMap<Vec<u8>, usize> = HashMap::new();

        // 您的 Python __init__ 假设 vocab 是 dict[int, bytes]，
        // 但 from_files 提供了 dict[str, int]。
        // 这个 Rust 版本修复了这种不一致，并正确处理来自 JSON 的 dict[str, int]。
        for (token_str, &id) in &vocab {
            let token_bytes = token_str.as_bytes().to_vec();
            vocab_int_bytes.insert(id, token_bytes.clone());
            dictionary_bytes_int.insert(token_bytes, id);
        }

        let mut n: usize = vocab_int_bytes.len();
        let mut sorted_special_tokens: Vec<String> = Vec::new();

        if let Some(mut tokens) = special_tokens {
            // 按长度降序排序
            tokens.sort_by_key(|s| std::cmp::Reverse(s.len()));
            sorted_special_tokens = tokens;

            // 将特殊 token 添加到词汇表（如果它们不存在）
            for token_str in &sorted_special_tokens {
                let token_bytes = token_str.as_bytes().to_vec();
                if !dictionary_bytes_int.contains_key(&token_bytes) {
                    dictionary_bytes_int.insert(token_bytes.clone(), n);
                    vocab_int_bytes.insert(n, token_bytes);
                    n += 1;
                }
            }
        }

        // 构建用于分割特殊 token 的正则表达式
        let special_tokens_re = if sorted_special_tokens.is_empty() {
            None
        } else {
            let pattern = format!(
                "({})",
                sorted_special_tokens
                    .iter()
                    .map(|s| regex::escape(s))
                    .collect::<Vec<_>>()
                    .join("|")
            );
            Some(Regex::new(&pattern).map_err(|e| PyValueError::new_err(e.to_string()))?)
        };

        Ok(PyTokenizer {
            vocab: vocab_int_bytes,
            dictionary: dictionary_bytes_int,
            merges,
            special_tokens: sorted_special_tokens,
            special_tokens_re,
        })
    }

    /// 从文件构造 Tokenizer (Python: from_files)
    #[classmethod]
    #[pyo3(signature = (vocab_path, merges_path, special_tokens = None))]
    fn from_files(
        _cls: &PyType,
        vocab_path: String,
        merges_path: String,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // 1. 加载 vocab.json (假设为 dict[str, int] 格式, e.g., GPT-2 encoder.json)
        let vocab_file = File::open(&vocab_path).map_err(|e| {
            PyFileNotFoundError::new_err(format!("Vocab file not found at {}: {}", vocab_path, e))
        })?;
        let reader = BufReader::new(vocab_file);
        let vocab: HashMap<String, usize> = serde_json::from_reader(reader)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse vocab JSON: {}", e)))?;

        // 2. 加载 merges.txt
        let merges_file = File::open(&merges_path).map_err(|e| {
            PyFileNotFoundError::new_err(format!("Merges file not found at {}: {}", merges_path, e))
        })?;
        let reader = BufReader::new(merges_file);

        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for line in reader.lines() {
            let line =
                line.map_err(|e| PyValueError::new_err(format!("Failed to read merges: {}", e)))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Python: tuple(line.strip().encode("utf-8").split(b" "))
            // Rust 中等价的操作：
            let line_bytes = line.as_bytes();
            if let Some(space_idx) = line_bytes.iter().position(|&b| b == b' ') {
                let part1 = line_bytes[..space_idx].to_vec();
                let part2 = line_bytes[space_idx + 1..].to_vec();
                merges.push((part1, part2));
            }
        }

        // 3. 调用 __new__ (Rust中的 new) 来构造实例
        Self::new(vocab, merges, special_tokens)
    }

    /// 将给定文本编码为 token ID 列表
    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        let mut token_seq: Vec<usize> = Vec::new();

        // 1. 按特殊 token 分割
        let parts: Vec<&str> = if let Some(re) = &self.special_tokens_re {
            // 're.split' 的行为与 Python re.split(pattern, text) 类似
            // 它会保留被捕获的分隔符 (即特殊 token)
            let mut last = 0;
            let mut result = Vec::new();
            for mat in re.find_iter(text) {
                if mat.start() > last {
                    result.push(&text[last..mat.start()]);
                }
                result.push(mat.as_str());
                last = mat.end();
            }
            if last < text.len() {
                result.push(&text[last..]);
            }
            result
        } else {
            vec![text]
        };

        for part in parts {
            // 2. 检查 part 是否是特殊 token
            let part_bytes = part.as_bytes();
            if let Some(&id) = self.dictionary.get(part_bytes) {
                // 是特殊 token，直接添加 ID
                token_seq.push(id);
            } else {
                // 3. 不是特殊 token，运行 BPE 核心算法
                for mat in CORE_BPE_RE.find_iter(part) {
                    let token_bytes = mat.as_str().as_bytes();

                    // Python: [bytes([b]) for b in token.encode("utf-8")]
                    let mut encoded_token: Vec<Vec<u8>> =
                        token_bytes.iter().map(|&b| vec![b]).collect();

                    // Python: for pair in self.merges...
                    for pair in &self.merges {
                        encoded_token = merge_pair(pair, &encoded_token);
                    }

                    // Python: [self.dictionary[b] for b in encoded_token]
                    for b in encoded_token {
                        if let Some(&id) = self.dictionary.get(&b) {
                            token_seq.push(id);
                        } else {
                            // 理论上，如果词汇表和合并是正确的，不应该发生
                            return Err(PyValueError::new_err(format!(
                                "Token not in dictionary: {:?}",
                                b
                            )));
                        }
                    }
                }
            }
        }

        Ok(token_seq)
    }

    /// (Python: encode_iterable)
    /// 将一个字符串迭代器编码为一个扁平的 ID 列表。
    /// 注意：Python 版本返回一个生成器。
    /// 这个版本为了简单起见，接收一个迭代器并返回一个扁平的 Vec<usize>。
    fn encode_iterable(&self, py: Python, iterable: &PyAny) -> PyResult<Vec<usize>> {
        let mut flat_ids: Vec<usize> = Vec::new();

        for item in iterable.iter()? {
            let text = item?.extract::<&str>()?;
            let ids = self.encode(text)?; // 调用我们自己的 encode 方法
            flat_ids.extend(ids);
        }

        Ok(flat_ids)
    }

    /// 将 token ID 列表解码回字符串
    fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        let mut byte_seq: Vec<u8> = Vec::new();
        for id in ids {
            if let Some(bytes) = self.vocab.get(&id) {
                byte_seq.extend_from_slice(bytes);
            } else {
                return Err(PyValueError::new_err(format!("Invalid token ID: {}", id)));
            }
        }

        // Python: .decode("utf-8", errors="replace")
        let decoded_string = String::from_utf8_lossy(&byte_seq).to_string();
        Ok(decoded_string)
    }
}

/// Python 模块定义
#[pymodule]
fn bpe_tokenizer_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
