import json
import logging
from typing import override

import litellm

from core.agent import Agent
from core.tool import ToolRegistry

logger = logging.getLogger(__name__)


class ReActAgent(Agent[litellm.Message]):
    def __init__(
        self,
        model: str,
        prompt: str,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        super().__init__(model, tool_registry, next_agents=[])
        self.max_steps = max_steps
        self.prompt = prompt

    @override
    async def run(self) -> litellm.Message:
        messages = [{"role": "user", "content": self.prompt}]

        for _ in range(self.max_steps):
            logger.info(f"Running ReACT agent with {len(messages)} messages")

            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.tool_registry.tool_schemas(),
                tool_choice="auto",
                temperature=0.0,
            )

            messages.append(response.choices[0].message)  # type: ignore

            tool_calls = response.choices[0].message.tool_calls  # type: ignore

            if not tool_calls:
                logger.info("No tool calls found, exiting")
                return response.choices[0].message  # type: ignore

            for tool_call in tool_calls:
                assert tool_call.function.name is not None
                tool = self.tool_registry.get(tool_call.function.name)
                tool_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

                logger.info(
                    f"Tool call: {tool_call.function.name} with args: {tool_args}"
                )
                tool_call_result = await tool(**tool_args)
                if tool_call_result:
                    # Truncate to max_result_length characters with a note if truncated
                    if (
                        tool.max_result_length is not None
                        and len(tool_call_result) > tool.max_result_length
                    ):
                        tool_call_result = (
                            tool_call_result[: tool.max_result_length]
                            + "\n\n[Content truncated due to length...]"
                        )
                    logger.info(
                        f"Tool call {tool_call.function.name} succeeded, args: {tool_args}, result: {tool_call_result}"
                    )
                    messages.append({
                        "role": "tool",
                        "content": tool_call_result,
                        "tool_call_id": tool_call_id,
                    })
                else:
                    logger.warning(
                        f"Tool call {tool_call.function.name} failed, args: {tool_args}, result: {tool_call_result}"
                    )
                    messages.append({
                        "role": "tool",
                        "content": f"Tool call {tool_call.function.name} failed, please try some other tool",
                        "tool_call_id": tool_call_id,
                    })

        # If we reach max_steps without the LLM deciding to stop, return the last response
        logger.info("Reached maximum number of steps, returning last response")
        return response.choices[0].message  # type: ignore
