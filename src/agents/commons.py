from typing import Protocol


class Researcher(Protocol):
    model: str

    async def research(self, topic: str) -> str: ...
