# An Log for implementing cs336-assignment1

> JinPeng pengjin@smail.nju.edu.cn

[TOC]

## Byte-Pair Encoding Tokenizer

> 简单来说，Byte-Pair Encoding (BPE) 是一种将文本切分成更小单元（称为“token”）的算法，它的特点是能够智能地在“单词级别”和“字符级别”之间找到一个完美的平衡点。 这种切分方式被称为“子词（subword）”分词。BPE 最初是一种数据压缩算法，后来被引入自然语言处理（NLP）领域，并成为当今几乎所有大型语言模型（如 GPT 系列）的基础组件之一。想象一下，如果让你用一本字典来表示所有的英文单词：如果字典里只有单词：遇到一个新词（比如 "unfriend"），字典里没有，你就不认识它了。这就是“词汇表外（Out-of-Vocabulary, OOV）”问题。如果字典里只有26个字母：你可以表示任何单词，但一个简单的词 "apple" 就要表示成 ['a', 'p', 'p', 'l', 'e']，信息密度太低，模型处理起来效率很差。BPE 的思想就是：我们不预设字典里应该有什么，而是让算法从语料库中自动学习最高效的“词根”、“前缀”、“后缀”等子词单元。它的工作方式是：不断地找出文本中最高频出现的相邻“字节对”（或字符对），然后将它们合并成一个新的、更大的单元，并加入到词汇表中。

### Process required to construct a BPE tokenizer

- Volcabulary initialization
- Pre-tokenization
- Compute BPE merges
- Special tokens

### 