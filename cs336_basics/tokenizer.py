import heapq
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

from .bpe import PAT


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return GPT-2's printable-character representation of all bytes."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra_value = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_value)
            extra_value += 1
    return {chr(unicode_value): byte_value for byte_value, unicode_value in zip(byte_values, unicode_values)}


class Tokenizer:
    """GPT-2-style byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self._merge_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self._special_token_ids: dict[str, int] = {}
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            token_id = self._token_to_id.get(token_bytes)
            if token_id is None:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = token_bytes
                self._token_to_id[token_bytes] = token_id
            self._special_token_ids[special_token] = token_id

        self._special_pattern = None
        if self.special_tokens:
            alternatives = "|".join(
                regex.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_pattern = regex.compile(alternatives)

        self._byte_token_ids = {}
        for byte_value in range(256):
            byte_token = bytes([byte_value])
            if byte_token in self._token_to_id:
                self._byte_token_ids[byte_value] = self._token_to_id[byte_token]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a tokenizer from GPT-2-style vocabulary and merge files."""
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        byte_decoder = _gpt2_byte_decoder()
        vocab = {
            int(token_id): bytes(byte_decoder[symbol] for symbol in token)
            for token, token_id in serialized_vocab.items()
        }

        merges = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip("\n").split(" ")
                if len(parts) != 2:
                    continue
                merges.append(
                    (
                        bytes(byte_decoder[symbol] for symbol in parts[0]),
                        bytes(byte_decoder[symbol] for symbol in parts[1]),
                    )
                )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_piece(self, piece: str) -> Iterator[int]:
        symbols = [bytes([byte_value]) for byte_value in piece.encode("utf-8")]
        if not symbols:
            return

        previous = [-1] + list(range(len(symbols) - 1))
        following = list(range(1, len(symbols))) + [-1]
        heap: list[tuple[int, int]] = []

        for position in range(len(symbols) - 1):
            rank = self._merge_ranks.get((symbols[position], symbols[position + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, position))

        while heap:
            rank, position = heapq.heappop(heap)
            right = following[position]
            if right == -1:
                continue
            if self._merge_ranks.get((symbols[position], symbols[right])) != rank:
                continue

            left = previous[position]
            right_after = following[right]
            symbols[position] += symbols[right]
            following[position] = right_after
            if right_after != -1:
                previous[right_after] = position
            previous[right] = -1
            following[right] = -1

            if left != -1:
                left_rank = self._merge_ranks.get((symbols[left], symbols[position]))
                if left_rank is not None:
                    heapq.heappush(heap, (left_rank, left))
            if right_after != -1:
                right_rank = self._merge_ranks.get((symbols[position], symbols[right_after]))
                if right_rank is not None:
                    heapq.heappush(heap, (right_rank, position))

        position = 0
        while position != -1:
            token_id = self._token_to_id.get(symbols[position])
            if token_id is None:
                raise KeyError(f"BPE token is missing from vocabulary: {symbols[position]!r}")
            yield token_id
            position = following[position]

    def _encode_non_special(self, text: str) -> Iterator[int]:
        for match in regex.finditer(PAT, text):
            yield from self._encode_piece(match.group())

    def _encode_text(self, text: str) -> Iterator[int]:
        if self._special_pattern is None:
            yield from self._encode_non_special(text)
            return

        start = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > start:
                yield from self._encode_non_special(text[start : match.start()])
            yield self._special_token_ids[match.group()]
            start = match.end()
        if start < len(text):
            yield from self._encode_non_special(text[start:])

    def encode(self, text: str) -> list[int]:
        """Encode an entire string into token IDs."""
        return list(self._encode_text(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode strings one at a time without retaining the whole input."""
        for text in iterable:
            yield from self._encode_text(text)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs, replacing invalid UTF-8 byte sequences."""
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
