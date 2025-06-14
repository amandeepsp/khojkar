from typing import Protocol


class Memory(Protocol):
    async def add(self, text: str, metadata: dict): ...

    async def query(self, query: str): ...

    async def clear(self): ...
