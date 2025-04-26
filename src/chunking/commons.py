from typing import Protocol


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...
