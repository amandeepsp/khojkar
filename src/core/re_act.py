import asyncio
import json
import logging
from typing import Any, Optional

import litellm
from pydantic import BaseModel

from core.agent import Agent
from core.tool import ToolRegistry
from memory.context import InContextMemory

logger = logging.getLogger(__name__)


class ReActAgent:
    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        prompt: str,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        default_temperature: float = 0.3,
        max_concurrent_tool_calls: int = 3,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt = prompt
        self.messages = InContextMemory(system_prompt=prompt, max_tokens=1_000_000_000)
        self.current_step = 0
        self.default_temperature = default_temperature
        self.max_concurrent_tool_calls = max_concurrent_tool_calls
        # For compatibility with Agent protocol
        self.children: list[Agent] = []
        self.parent: Optional[Agent] = None

    def _safe_json_serialize(self, obj):
        if isinstance(obj, str):
            return obj

        if isinstance(obj, BaseModel):
            return obj.model_dump_json()

        try:
            return json.dumps(obj)
        except Exception as e:
            logger.warning(f"Failed to serialize object: {obj}, error: {e}")
            return str(obj)

    def _throttle_tool_calls(self, tool_calls):
        semaphore = asyncio.Semaphore(self.max_concurrent_tool_calls)

        async def semaphore_wrapper(tool_call):
            async with semaphore:
                return await self._call_tool(tool_call)

        return [semaphore_wrapper(tool_call) for tool_call in tool_calls]

    async def _call_tool(self, tool_call):
        assert tool_call.function.name is not None
        tool = self.tool_registry.get(tool_call.function.name)
        tool_args = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id

        logger.info(f"Using tool: {tool_call.function.name}, args: {tool_args}")
        tool_call_result = None
        try:
            tool_call_result = await tool(**tool_args)
            tool_call_result = self._safe_json_serialize(tool_call_result)
        except Exception as e:
            logger.warning(
                f"Tool call {tool_call.function.name} failed, args: {tool_args}, error: {e}"
            )
            tool_call_result = f"Tool call {tool_call.function.name} failed, please try some other tool"

        if not tool_call_result:
            logger.warning(
                f"Tool call {tool_call.function.name} failed, args: {tool_args}"
            )
            tool_call_result = f"Tool call {tool_call.function.name} failed, please try some other tool"

        if tool.max_result_length and len(tool_call_result) > tool.max_result_length:
            tool_call_result = (
                tool_call_result[: tool.max_result_length]
                + "\n\n[Content truncated due to length...]"
            )

        self.messages.add({
            "role": "tool",
            "content": tool_call_result,
            "tool_call_id": tool_call_id,
        })

    async def run(self, **kwargs) -> Any:
        self.messages.clear()

        for _ in range(self.max_steps):
            self.current_step += 1
            logger.info(f"Running ReACT agent with {len(self.messages)} messages")

            response = litellm.completion(
                model=self.model,
                messages=self.messages.get(),
                tools=self.tool_registry.tool_schemas(),
                tool_choice="auto",
                temperature=self.default_temperature,
            )

            logger.info(f"Thinking...\n\n{response.choices[0].message.content}")  # type: ignore

            self.messages.add(response.choices[0].message)  # type: ignore

            tool_calls = response.choices[0].message.tool_calls  # type: ignore

            if not tool_calls:
                logger.info("No further actions needed, finalizing results")
                return response.choices[0].message  # type: ignore

            await asyncio.gather(*self._throttle_tool_calls(tool_calls))

        # If we reach max_steps without the LLM deciding to stop, return the last response
        logger.info("Reached maximum number of steps, returning last response")
        return response.choices[0].message  # type: ignore
