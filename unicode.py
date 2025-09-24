import regex as re

PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
response = re.finditer(PAT, "some text that i'll pre-tokenize")
for item in response:
    print(item.group(0).strip())