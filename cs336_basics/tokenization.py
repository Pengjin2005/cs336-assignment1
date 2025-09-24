import regex as re
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random


# Tool Functions

def merge(
    indices: list[int], pair: tuple[int, int], new_index: int
) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


# Base Classes

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""

    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for (
            pair,
            new_index,
        ) in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


# API functions


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> BPETokenizerParams:
    """
    训练一个字节级的 BPE 分词器。

    Args:
        input_path: 训练数据文本文件的路径。
        vocab_size: 最终词汇表的目标大小。
        special_tokens: 需要添加到词汇表中的特殊标记列表。

    Returns:
        一个元组，包含：
        - vocab: 词汇表，一个从 token ID (int) 到 token 内容 (bytes) 的映射。
        - merges: 合并规则列表，按生成顺序列出。
    """

    pass


# Example usage

if __name__ == "__main__":
    string = "the quick brown fox jumps over the lazy dog."  # @inspect string
    params = train_bpe(string, num_merges=3)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox jumps over the lazy dog."  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    print(f"Original string: {string}")
    print(f"Encoded indices: {indices}")
    print(f"Reconstructed string: {reconstructed_string}")
