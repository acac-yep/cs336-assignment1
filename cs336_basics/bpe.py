import regex
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> Counter[str]:
    """Split text into GPT-2-style pre-token strings and count them."""
    counts = Counter()

    for match in regex.finditer(PAT, text):
        token = match.group()
        counts[token] += 1

    return counts


def token_to_bytes(token):
    """Represent a pre-token as a sequence of one-byte tokens."""
    token_bytes = token.encode("utf-8")
    bytes_list = []
    for b in token_bytes:
        byte_token = bytes([b])
        bytes_list.append(byte_token)
    return bytes_list


def count_pairs(token_counts, token_sequences):
    """Count adjacent byte-token pairs, weighted by token frequency."""
    pair_counts = Counter()
    for token, frequency in token_counts.items():
        tokens = token_sequences[token]
        token_zip = zip(tokens, tokens[1:])
        for pair in token_zip:
            pair_counts[pair] += frequency
    return pair_counts


def merge_pair(tokens, best_pair):
    """Merge every non-overlapping occurrence of one pair from left to right."""
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

    # Remove special tokens before pre-tokenization so they are never merged.
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

    # Track which pre-tokens contain each pair for efficient updates.
    for token, tokens in token_sequences.items():
        for pair in zip(tokens, tokens[1:]):
            pair_to_tokens[pair].add(token)

    num_merges = vocab_size - 256 - len(special_tokens)
    pair_counts = count_pairs(token_counts, token_sequences)

    # Repeatedly choose the most frequent pair; tuple order breaks ties.
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

            # Remove this token's old pair contributions.
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= frequency
                pair_to_tokens[pair].discard(token)

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                if not pair_to_tokens[pair]:
                    del pair_to_tokens[pair]

            # Merge the selected pair and add the new pair contributions.
            new_tokens = merge_pair(tokens, best_pair)
            token_sequences[token] = new_tokens

            for pair in zip(new_tokens, new_tokens[1:]):
                pair_counts[pair] += frequency
                pair_to_tokens[pair].add(token)

    # The first 256 vocabulary entries are the raw byte tokens.
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
