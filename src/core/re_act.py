import asyncio
import json
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

import llm
import utils
from core.agent import Agent
from core.tool import ToolRegistry
from memory.context import MessagesMemory

logger = logging.getLogger(__name__)

console = Console()


class AgentLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if self.extra is None or self.extra["agent_name"] is None:
            return msg, kwargs
        return f"[agent: {self.extra['agent_name']}] {msg}", kwargs


class ReActAgent(Agent):
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
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt = prompt
        self.messages = MessagesMemory(system_prompt=prompt, max_tokens=1_000_000_000)
        self.current_step = 0
        self.default_temperature = default_temperature
        self.max_concurrent_tool_calls = max_concurrent_tool_calls
        self.output_schema = output_schema
        self.input_schema = input_schema

    def _safe_json_serialize(self, obj: Any) -> str:
        """Tool calls can return non-serializable objects, so we need to serialize them to a string"""
        if isinstance(obj, BaseModel):
            return obj.model_dump_json()
        if isinstance(obj, (dict, list)):
            return json.dumps(obj)
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
            logger.info(f"Tool call {tool_call.function.name} succeeded.")
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
            logger.info("Tool call result truncated due to length.")

        self.messages.add({
            "role": "tool",
            "content": tool_call_result,
            "tool_call_id": tool_call_id,
        })

    async def _prompt_model_to_format_response(self) -> Any:
        assert self.output_schema is not None
        self.messages.add({
            "role": "user",
            "content": "Please format the response as JSON according to the following JSON Schema: \n"
            + json.dumps(self.output_schema.model_json_schema()),
        })

        response = await llm.acompletion(
            model=self.model,
            messages=self.messages.get_all(),
            tool_choice="auto",
            temperature=self.default_temperature,
            response_format=self.output_schema.model_json_schema(),
        )

        sanitized_response = utils.remove_thinking_output(
            response.choices[0].message.content  # type: ignore
        )

        return self.output_schema.model_validate_json(
            utils.extract_lang_block(
                sanitized_response,  # type: ignore
                language="json",
            )
        )

    async def run(self, **kwargs) -> Any:
        self.messages.clear()

        if kwargs:
            self.messages.add({
                "role": "user",
                "content": self._safe_json_serialize(kwargs),
            })

        for _ in range(self.max_steps):
            self.current_step += 1
            logger.info(f"Running ReACT agent with {len(self.messages)} messages")

            response = await llm.acompletion(
                model=self.model,
                messages=self.messages.get_all(),
                tools=self.tool_registry.tool_schemas(),
                tool_choice="auto",
                temperature=self.default_temperature,
            )

            if response.choices[0].message.content:  # type: ignore
                console.print()
                console.print("Thinking...")
                console.print(Markdown(response.choices[0].message.content))  # type: ignore
                console.print()

            sanitized_response = utils.remove_thinking_output(
                response.choices[0].message.content  # type: ignore
            )

            self.messages.add({
                "role": "assistant",
                "content": sanitized_response,
                "tool_calls": response.choices[0].message.tool_calls,  # type: ignore
            })

            tool_calls = response.choices[0].message.tool_calls  # type: ignore
            finish_reason = response.choices[0].finish_reason  # type: ignore

            if not tool_calls:
                if finish_reason != "stop":
                    logger.error(f"Agent failed to stop, stop reason - {finish_reason}")
                    raise StopIteration(
                        f"Agent failed to stop, stop reason - {finish_reason}"
                    )

                logger.info("No further actions needed, finalizing results")

                if self.output_schema:
                    return await self._prompt_model_to_format_response()

                return response.choices[0].message  # type: ignore

            logger.info(f"Processing {len(tool_calls)} tool call(s)")

            await asyncio.gather(*self._throttle_tool_calls(tool_calls))

        # If we reach max_steps without the LLM deciding to stop, return the last response
        logger.error("Reached maximum number of steps, still tool calls remaining")
        raise StopIteration("Reached maximum number of steps")
