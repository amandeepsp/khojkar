from typing import Any, Optional, Protocol, Type, runtime_checkable

from pydantic import BaseModel

from core.tool import Toolbox


@runtime_checkable
class Agent(Protocol):
    """
    Protocol defining the interface for an agent.
    Any object that satisfies this protocol can be used as an agent.
    """

    name: str
    description: str
    model: str
    tool_registry: Toolbox = Toolbox()
    children: list["Agent"] = []
    parent: Optional["Agent"] = None
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None

    async def run(self, **kwargs): ...

    def to_json(self) -> dict:
        json_dict: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }

        if self.input_schema:
            json_dict["input_schema"] = self.input_schema.model_json_schema()

        if self.output_schema:
            json_dict["output_schema"] = self.output_schema.model_json_schema()

        return json_dict
