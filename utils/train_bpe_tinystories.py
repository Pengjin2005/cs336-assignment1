import json

from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # Save the vocab and merges to files
    print("Saving BPE vocab and merges...")
    json.dump(
        {index: vocab_item.decode("utf-8") for index, vocab_item in vocab.items()},
        open("tinystories_bpe_vocab.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
    with open("tinystories_bpe_merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('utf-8')} {merge[1].decode('utf-8')}\n")
