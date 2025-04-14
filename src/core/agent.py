from typing import Optional, Protocol, runtime_checkable

from core.tool import ToolRegistry


@runtime_checkable
class Agent(Protocol):
    """
    Protocol defining the interface for an agent.
    Any object that satisfies this protocol can be used as an agent.
    """

    name: str
    description: str
    model: str
    tool_registry: ToolRegistry = ToolRegistry()
    children: list["Agent"] = []
    parent: Optional["Agent"] = None

    async def run(self, **kwargs): ...

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
        }
