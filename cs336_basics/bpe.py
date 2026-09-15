import regex
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> Counter[str]:
    """按 GPT-2 风格切分文本，并统计每种预分词的出现次数。"""
    counts = Counter()

    for match in regex.finditer(PAT, text):
        token = match.group()
        counts[token] += 1

    return counts


def token_to_bytes(token):
    """将预分词表示为由单字节 token 组成的序列。"""
    token_bytes = token.encode("utf-8")
    bytes_list = []
    for b in token_bytes:
        byte_token = bytes([b])
        bytes_list.append(byte_token)
    return bytes_list


def count_pairs(token_counts, token_sequences):
    """按预分词频率统计相邻字节 token 对的出现次数。"""
    pair_counts = Counter()
    for token, frequency in token_counts.items():
        tokens = token_sequences[token]
        token_zip = zip(tokens, tokens[1:])
        for pair in token_zip:
            pair_counts[pair] += frequency
    return pair_counts


def merge_pair(tokens, best_pair):
    """从左到右合并指定 token 对的所有不重叠出现位置。"""
    i = 0
    new_token = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
            new_token.append(tokens[i]+tokens[i+1])
            i += 2
        else:
            new_token.append(tokens[i])
            i += 1
    return new_token


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # 先移除特殊 token，避免它们参与预分词和合并。
    parts = [text]

    for special_token in special_tokens:
        new_parts = []

        for part in parts:
            new_parts.extend(part.split(special_token))

        parts = new_parts

    token_counts = Counter()

    for part in parts:
        token_counts += pretokenize(part)

    token_sequences = {}

    for token in token_counts:
        token_sequences[token] = token_to_bytes(token)

    merges = []
    pair_to_tokens = defaultdict(set)

    # 记录每个 token 对出现在哪些预分词中，便于增量更新。
    for token, tokens in token_sequences.items():
        for pair in zip(tokens, tokens[1:]):
            pair_to_tokens[pair].add(token)

    num_merges = vocab_size - 256 - len(special_tokens)
    pair_counts = count_pairs(token_counts, token_sequences)

    # 反复选择最高频 token 对；频率相同时按 tuple 顺序打破平局。
    for _ in range(num_merges):
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair)
        )

        merges.append(best_pair)

        affected_tokens = list(pair_to_tokens[best_pair])

        for token in affected_tokens:
            tokens = token_sequences[token]
            frequency = token_counts[token]

            # 移除当前预分词对旧 token 对计数的贡献。
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= frequency
                pair_to_tokens[pair].discard(token)

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if not pair_to_tokens[pair]:
                    del pair_to_tokens[pair]

            # 合并选中的 token 对，并加入新的 token 对计数。
            new_tokens = merge_pair(tokens, best_pair)
            token_sequences[token] = new_tokens

            for pair in zip(new_tokens, new_tokens[1:]):
                pair_counts[pair] += frequency
                pair_to_tokens[pair].add(token)

    # 词表最开始的 256 项对应原始字节 token。
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    idx = 256
    for pair in merges:
        vocab[idx] = pair[0] + pair[1]
        idx += 1

    for special_token in special_tokens:
        vocab[idx] = special_token.encode("utf-8")
        idx += 1

    return vocab, merges
