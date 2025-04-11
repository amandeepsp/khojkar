import inspect
import logging
from typing import Any, Callable

import docstring_parser

logger = logging.getLogger(__name__)


class Tool:
    def __init__(
        self, name: str, func: Callable, max_result_length: int | None = None
    ) -> None:
        if not inspect.iscoroutinefunction(func):
            raise ValueError(
                f"Tool {name} must use async functions. Please use asynchronous callables only."
            )
        self.name = name
        self.func = func
        self._schema = self._generate_schema()
        self.max_result_length = max_result_length

    def _generate_schema(self) -> dict:
        """
        Converts a Python function into a JSON-serializable dictionary
        that describes the function's signature, including its name,
        description, and parameters.

        Taken from https://github.com/openai/swarm/

        Args:
            func: The function to be converted.

        Returns:
            A dictionary representing the function's signature in JSON format.
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        doc = inspect.getdoc(self.func)
        parsed_docstring = docstring_parser.parse(doc or "")

        try:
            signature = inspect.signature(self.func)
        except ValueError as e:
            raise ValueError(
                f"Failed to get signature for function {self.func.__name__}: {str(e)}"
            )

        parameters_descriptions = parsed_docstring.params
        parameters = {}
        for param in signature.parameters.values():
            try:
                param_type = type_map.get(param.annotation)
            except KeyError as e:
                raise KeyError(
                    f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
                )

            parameter_description = next(
                p.description
                for p in parameters_descriptions
                if p.arg_name == param.name
            )
            parameters[param.name] = {
                "type": param_type,
                "description": parameter_description,
            }

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        description = parsed_docstring.short_description

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description or "",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }

    @property
    def schema(self) -> dict:
        return self._schema

    def formatted_signature(self):
        params = self.schema["function"]["parameters"]["properties"]
        params_str = ", ".join(
            f"{k}: {v['description'] or v['type']}" for k, v in params.items()
        )
        return f"{self.name}(" + params_str + ")"

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {}

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
