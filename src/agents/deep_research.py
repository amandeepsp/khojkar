import logging
from typing import override

import litellm

from core.agent import Agent
from core.re_act import ReActAgent
from core.tool import ToolRegistry

logger = logging.getLogger(__name__)


class DeepResearchAgent(Agent[litellm.Message]):
    def __init__(
        self,
        name: str,
        model: str,
        prompt: str,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        super().__init__(name, model, tool_registry, next_agents=[])
        # Create a ReActAgent instance as a component
        self.react_agent = ReActAgent(
            name=f"{name}_react",
            model=model,
            prompt=prompt,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )
        self.max_steps = max_steps

    @override
    async def run(self) -> litellm.Message:
        last_message = await self.react_agent.run()

        if (
            last_message.content is None
            or self.react_agent.current_step > self.max_steps
        ):
            # If the last message content is None or the current step is greater than the max steps, raise an error
            raise ValueError("Last message content is None")

        # Add a new message to the ReActAgent's message context
        self.react_agent.messages.add({
            "role": "user",
            "content": "Please create the report, only return the report content, nothing else",
        })

        logger.info("Generating final report")

        confirm_report = litellm.completion(
            model=self.model,
            messages=self.react_agent.messages.get(),
            tools=self.tool_registry.tool_schemas(),
            tool_choice="auto",
            temperature=0.0,
        )

        return confirm_report.choices[0].message  # type: ignore
