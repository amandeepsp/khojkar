import json
from typing import Any, Callable
import inspect

import litellm

import utils

class Tool:
    def __init__(self, name: str, func: Callable) -> None:
        if not inspect.iscoroutinefunction(func):
            raise ValueError(f"Tool {name} must use async functions. Please use asynchronous callables only.")
        self.name = name
        self.func = func
        self.schema = self._generate_schema()

    def _generate_schema(self) -> dict:
        sig = inspect.signature(self.func)
        if sig.return_annotation is inspect.Signature.empty:
            raise ValueError(f"Tool {self.func.__name__} has no return annotation")
        
        doc = inspect.getdoc(self.func)
        if doc is None:
            raise ValueError(f"Tool {self.func.__name__} has no docstring")
        
        param_descriptions = {
            line.split(":")[0].strip(): line.split(":", 1)[1].strip()
            for line in doc.splitlines()
            if ":" in line
        }

        return {
            "params": {
                name: {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "str",
                    "description": param_descriptions.get(name, "")
                }
                for name, param in sig.parameters.items()
            },
        }
    

    def formatted_signature(self):
        return f"{self.name}(" + ", ".join(
            f"{k}: {v['description'] or v['type']}" for k, v in self.schema["params"].items()
        ) + ")"
    

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def _tool_descriptions(self) -> str:
        return "\n".join(
        f"- `{tool.formatted_signature()}`"
        for tool in self.tools.values()
    )

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def __getitem__(self, key: str) -> Tool:
        return self.tools[key]


class ToolAgent:
    def __init__(self, model: str, tools: ToolRegistry, max_iterations: int = 3) -> None:
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_response_model = {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "tool_args": {"type": "object"},
            },
        }


    def _generate_prompt(self, query: str) -> str:
        return f"""
        You are a helpful assistant. You can use the following tools to answer the user's question. Follow the response format strictly.
        {self.tools._tool_descriptions()}

        Query: {query}

        Response Format:
        {{
            "tool_name": "tool_name",
            "tool_args": {{"arg_name": "arg_value"}}
        }}
        """
        
    def find_tool(self, query: str) -> tuple[Tool, dict]:
        prompt = self._generate_prompt(query)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format=self.tool_response_model,
        )

        tool_response = utils.extract_json_block(response.choices[0].message.content)
        tool_response = json.loads(tool_response)

        # Parse according to the tool response model
        tool = self.tools.get(tool_response["tool_name"])
        if tool is None:
            raise ValueError(f"Tool {tool_response.tool_name} not found")
        return tool, tool_response["tool_args"]
        
        
        