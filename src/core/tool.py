import inspect
import logging
from typing import (
    Any,
    Callable,
    Protocol,
    Tuple,
    Unpack,
    get_type_hints,
    runtime_checkable,
)

import docstring_parser
import jsonref
from pydantic import ConfigDict, Field, create_model

logger = logging.getLogger(__name__)


@runtime_checkable
class Tool(Protocol):
    name: str
    func: Callable[..., Any]
    schema: dict
    max_result_length: int | None = None
    description: str | None = None

    def formatted_signature(self) -> str: ...

    async def __call__(self, **kwargs: Any) -> Any: ...


class FunctionTool:
    """
    A tool that is a **pure-ish** function.
    Same input can return different outputs, but still should not have side effects.
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        max_result_length: int | None = None,
        description: str | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(func):
            raise ValueError(
                f"Tool {name} must use async functions. Please use asynchronous callables only."
            )
        self.name = name
        self.func = func
        self.max_result_length = max_result_length
        self.description = description
        self.schema = self._generate_schema()

    def _remove_additional_properties(self, schema: dict | list) -> None:
        if isinstance(schema, dict):
            schema.pop("additionalProperties", None)
            for value in schema.values():
                self._remove_additional_properties(value)
        elif isinstance(schema, list):
            for item in schema:
                self._remove_additional_properties(item)

    def _generate_schema(self) -> dict:
        """
        Converts a Python function into a JSON-serializable dictionary
        that describes the function's signature, including its name,
        description, and parameters.

        Args:
            func: The function to be converted.

        Returns:
            A dictionary representing the function's signature in JSON format.
        """

        doc = inspect.getdoc(self.func)
        parsed_docstring = docstring_parser.parse(doc or "")
        docstring_params = parsed_docstring.params
        docstring_description = parsed_docstring.short_description
        type_hints = get_type_hints(self.func)

        try:
            signature = inspect.signature(self.func)
        except ValueError as e:
            raise ValueError(
                f"Failed to get signature for function {self.func.__name__}: {str(e)}"
            )

        parameters = {}
        for param in signature.parameters.values():
            annotation = type_hints.get(param.name, param.annotation)

            parameter_description = ""
            try:
                parameter_description = next(
                    p.description for p in docstring_params if p.arg_name == param.name
                )
            except StopIteration:
                logger.warning(
                    f"Missing docstring description for parameter {param.name} in tool {self.name}"
                )

            parameters[param.name] = (
                annotation,
                Field(description=parameter_description),
            )

        params_model = create_model(
            f"{self.name}_args", **parameters, __config__=ConfigDict(extra="forbid")
        )
        params_schema = params_model.model_json_schema()
        self._remove_additional_properties(params_schema)
        description = self.description or docstring_description or ""

        params_schema = jsonref.replace_refs(params_schema)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": params_schema,
            },
        }

    def formatted_signature(self):
        params = self.schema["function"]["parameters"]["properties"]
        params_str = ", ".join(
            f"{k}: {v.get('description') or v.get('type', 'unknown')}"
            for k, v in params.items()
        )
        return f"{self.name}(" + params_str + ")"

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class Toolbox:
    def __init__(self, *args: Unpack[Tuple[Tool, ...]]) -> None:
        self.tools: dict[str, Tool] = {}
        for tool in args:
            self.register(tool)

    @staticmethod
    def from_tools(*args: Unpack[Tuple[Tool, ...]]) -> "Toolbox":
        return Toolbox(*args)

    @staticmethod
    def from_tool_registries(
        *args: Unpack[Tuple["Toolbox", ...]],
    ) -> "Toolbox":
        return Toolbox(*[tool for registry in args for tool in registry.tools.values()])

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def tool_schemas(self) -> list[dict]:
        return [tool.schema for tool in self.tools.values()]

    def _tool_descriptions(self) -> str:
        return "\n".join(
            f"- `{tool.formatted_signature()}`" for tool in self.tools.values()
        )

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def __getitem__(self, key: str) -> Tool:
        return self.tools[key]

    def with_tools(self, *tools: Tool) -> "Toolbox":
        return Toolbox(*list(self.tools.values()) + list(tools))
