"""
Byte Pair Encoding (BPE) Tokenizer Training Implementation
"""

import multiprocessing
import os
from collections import defaultdict
from typing import BinaryIO

import regex as re
from tqdm import tqdm

re_pattern = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(args):
    filepath, start, end, special_tokens = args
    token_freq = defaultdict(int)

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)
    chunk_data = chunk_data.decode("utf-8", errors="ignore").replace("\r\n", "\n").encode("utf-8")
    for part in re.split(f"({'|'.join(map(re.escape, special_tokens))})", chunk_data.decode("utf-8")):
        if part in special_tokens:
            continue
        else:
            for token in re_pattern.findall(part):
                token_freq[token.encode("utf-8")] += 1
    return token_freq


def count_pairs(split: dict[bytes, list[bytes]], token_freq: dict[bytes, int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count frequency of adjacent byte pairs in the token splits.
    """
    pair_freq = defaultdict(int)

    for token, freq in token_freq.items():
        bytes_seq = split[token]
        if len(bytes_seq) < 2:
            continue
        for i in range(len(bytes_seq) - 1):
            pair = (bytes_seq[i], bytes_seq[i + 1])
            pair_freq[pair] += freq

    return pair_freq


def merge_pair_optimized(
    split: dict[bytes, list[bytes]],
    pair: tuple[bytes, bytes],
    token_freq: dict[bytes, int],
    pair_freq: dict[tuple[bytes, bytes], int],
) -> tuple[dict[bytes, list[bytes]], dict[tuple[bytes, bytes], int]]:
    """
    Merge a pair and incrementally update pair frequencies.
    Returns the new split and updated pair_freq.
    Only updates frequencies for words that contain the merged pair.
    """
    new_split = split.copy()

    for word, byte_seq in split.items():
        # Quick check if this word contains the pair to merge
        has_pair = False
        i = 0
        while i < len(byte_seq) - 1:
            if byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
                has_pair = True
                break
            i += 1

        if not has_pair:
            continue

        freq = token_freq[word]

        # Remove old pair counts for this word
        i = 0
        while i < len(byte_seq) - 1:
            old_pair = (byte_seq[i], byte_seq[i + 1])
            pair_freq[old_pair] -= freq
            if pair_freq[old_pair] <= 0:
                del pair_freq[old_pair]
            i += 1

        # Merge the pair in this word
        n_split = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
                n_split.append(pair[0] + pair[1])
                i += 2
            else:
                n_split.append(byte_seq[i])
                i += 1

        new_split[word] = n_split

        # Add new pair counts for this word
        i = 0
        while i < len(n_split) - 1:
            new_pair = (n_split[i], n_split[i + 1])
            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
            i += 1

    return new_split, pair_freq


def merge_pair(split: dict[bytes, list[bytes]], pair: tuple[bytes, bytes]) -> dict[bytes, list[bytes]]:
    new_split = {}
    for word in split.keys():
        byte_seq = split[word]
        n_split = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
                n_split.append(pair[0] + pair[1])
                i += 2
            else:
                n_split.append(byte_seq[i])
                i += 1
        new_split[word] = n_split

    return new_split


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    A Byte-Pair Encoding Tokenizer training function.
    """
    # Vocab Initialization
    print("Initializing vocabulary...")
    vocab = {i: bytes([i]) for i in range(256)}  # Adding byte values
    merges = []
    next_idx = 256

    special_tokens_bytes = []
    for token in special_tokens:
        vocab[next_idx] = token.encode("utf-8")  # Adding special tokens
        special_tokens_bytes.append(token.encode("utf-8"))
        next_idx += 1
    print(f"Added {len(special_tokens)} special tokens.")
    # Pre-tokenization with Parallel Processing
    print("Loading and pre-tokenizing data in parallel...")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, os.cpu_count(), b"<|endoftext|>")

    tasks = [(input_path, boundaries[i], boundaries[i + 1], special_tokens) for i in range(len(boundaries) - 1)]

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(pre_tokenize_chunk, tasks)

    token_freq = defaultdict(int)
    for _token_freq in results:
        for token, freq in _token_freq.items():
            token_freq[token] += freq

    print(f"Pre-tokenization complete. {len(token_freq)} unique tokens found.")

    # Byte Pair Merging
    print("Starting BPE merging...")

    split = {}
    for word in token_freq.keys():
        if word not in special_tokens_bytes:
            byte_seq = [bytes([b]) for b in word]
            split[word] = byte_seq

    # Initial pair frequency count
    pair_freq = count_pairs(split, token_freq)

    # split = {token: bytes([token.encode("utf-8")]) for token in token_freq.keys()}
    for _ in tqdm(range(vocab_size - len(vocab)), desc="Merging pairs"):
        # while len(vocab) < vocab_size:
        if not pair_freq:
            break
        best_pair = max(pair_freq.items(), key=lambda pair: (pair[1], pair[0]))[0]
        merges.append(best_pair)

        # Use optimized merge with incremental updates
        split, pair_freq = merge_pair_optimized(split, best_pair, token_freq, pair_freq)

        new_token = best_pair[0] + best_pair[1]
        if new_token not in vocab.values():
            vocab[next_idx] = new_token
            next_idx += 1
    print("BPE merging complete.")

    return vocab, merges
