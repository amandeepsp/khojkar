from typing import Callable

import tiktoken

from chunking.commons import Chunker


class RecursiveChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = ["\n\n", "\n", ".", "?", "!", " ", ""],
        tokenizer: Callable[[str], list[int]] = tiktoken.get_encoding(
            "cl100k_base"
        ).encode,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.tokenizer = tokenizer
        if not chunk_overlap < chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        return [
            separator + split if i > 0 else split
            for i, split in enumerate(text.split(separator))
        ]

    def _chunk(self, text: str, separators: list[str]) -> list[str]:
        """Produces all chunks smaller than chunk_size"""
        if len(self.tokenizer(text)) <= self.chunk_size:
            return [text]

        if not separators:
            return [text]

        splits = self._split_with_separator(text, separators[0])
        chunks = []
        for split in splits:
            if not split:
                continue

            chunks.extend(self._chunk(split, separators[1:]))

        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        merged_chunks = []
        current_chunk = [chunks[0]]
        current_chunk_size = len(self.tokenizer(chunks[0]))

        for i in range(1, len(chunks)):
            if current_chunk_size + len(self.tokenizer(chunks[i])) > self.chunk_size:
                full_chunk = "".join(current_chunk)
                if full_chunk:
                    merged_chunks.append(full_chunk)

                overlap_text = (
                    full_chunk[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                )

                current_chunk = [overlap_text, chunks[i]]
                current_chunk_size = len(self.tokenizer(overlap_text)) + len(
                    self.tokenizer(chunks[i])
                )
            else:
                current_chunk.append(chunks[i])
                current_chunk_size += len(self.tokenizer(chunks[i]))

        if current_chunk:
            merged_chunks.append("".join(current_chunk))

        return merged_chunks

    def chunk(self, text: str) -> list[str]:
        chunks = self._chunk(text, self.separators)
        return self._merge_chunks(chunks)
