import logging

import litellm

import llm
from core.re_act import ReActAgent
from core.tool import ToolRegistry

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    def __init__(
        self,
        name: str,
        model: str,
        prompt: str,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        self.name = name
        self.description = (
            "A research agent that uses the web to find information about a topic"
        )
        self.model = model
        self.tool_registry = tool_registry
        self.children = []
        self.parent = None

        self._delegate_agent = ReActAgent(
            name=f"{name}_react",
            description=self.description,
            model=model,
            prompt=prompt,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )
        self.max_steps = max_steps

    async def run(self) -> litellm.Message:
        last_message = await self._delegate_agent.run()

        if (
            last_message.content is None
            or self._delegate_agent.current_step > self.max_steps
        ):
            # If the last message content is None or the current step is greater than the max steps, raise an error
            raise ValueError("Last message content is None")

        # Add a new message to the ReActAgent's message context
        self._delegate_agent.messages.add({
            "role": "user",
            "content": "Please create the report, only return the report content, nothing else",
        })

        logger.info("Generating final report")

        confirm_report = await llm.acompletion(
            model=self._delegate_agent.model,
            messages=self._delegate_agent.messages.get_all(),
            tools=self._delegate_agent.tool_registry.tool_schemas(),
            tool_choice="auto",
            temperature=0.0,
        )

        return confirm_report.choices[0].message  # type: ignore
