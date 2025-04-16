from typing import Protocol


class Memory(Protocol):
    async def add(self, memory: str): ...

    async def query(self, query: str): ...

    async def clear(self): ...
