from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from core.tool import ToolRegistry

T = TypeVar("T")  # Define a type variable for the return type


class Agent(Generic[T], ABC):
    def __init__(
        self,
        name: str,
        model: str,
        tool_registry: ToolRegistry = ToolRegistry(),
        next_agents: list["Agent[Any]"] = [],
    ) -> None:
        self.name = name
        self.model = model
        self.tool_registry = tool_registry
        self.next_agents = next_agents

    @abstractmethod
    async def run(self) -> T:
        pass
