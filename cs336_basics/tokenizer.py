import json
from collections.abc import Iterable
from typing import Self

import regex as re


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from given vocabulary and merges.
        """
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            N = len(self.vocab)
            for token in special_tokens:
                if token.encode("utf-8") not in self.vocab.values():
                    self.vocab[N] = token.encode("utf-8")
                    N += 1
        self.dictionary = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> Self:
        """
        Construct a tokenizer from vocabulary and merges files.
        """
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            merges = [tuple(line.strip().encode("utf-8").split(b" ")) for line in f if line.strip()]
        return cls(vocab, merges, special_tokens)

    def _merge_pair(self, pair: tuple[bytes, bytes], seq: list[bytes]) -> list[bytes]:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(pair[0] + pair[1])
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def encode(self, text: str) -> list[int]:
        """
        Encode a given text into a list of token IDs.
        """
        re_pattern = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        token_seq = []

        if self.special_tokens:
            parts = re.split(f"({'|'.join(map(re.escape, self.special_tokens))})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # token_seq.append(part.encode("utf-8"))
                encoded_token = part.encode("utf-8")
                token_seq.append(self.dictionary[encoded_token])
            else:
                for token in re_pattern.findall(part):
                    encoded_token = [bytes([b]) for b in token.encode("utf-8")]
                    for pair in self.merges:
                        encoded_token = self._merge_pair(pair, encoded_token)
                    token_seq.extend(encoded_token)

        return token_seq

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encode an iterable of texts into a list of lists of token IDs.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        byte_seq = b"".join(self.vocab[id] for id in ids)
        return byte_seq.decode("utf-8", errors="replace")
