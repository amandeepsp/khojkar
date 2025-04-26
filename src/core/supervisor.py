import json
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from core.agent import Agent
from core.re_act import ReActAgent
from core.tool import FunctionTool, Toolbox

logger = logging.getLogger(__name__)


class SupervisorAgent(Agent):
    """
    A generic supervisor agent that manages and coordinates other agents.

    Inspired by LangGraph's supervisor implementation, this agent can:
    1. Route tasks between specialized agents based on their capabilities and required inputs.
    2. Maintain shared state across agent executions (implicitly via context passed).
    3. Make decisions about workflow progression.
    4. Process and integrate results from different agents.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        system_prompt: str,
        tool_registry: Toolbox = Toolbox(),
        children: list[Agent] = [],
        max_steps: int = 10,
    ):
        self.children = children
        system_prompt += self._agent_schemas()  # Add agent details to system prompt
        self._delegate: ReActAgent = ReActAgent(
            name=f"{name}_supervisor",
            description=description,
            model=model,
            prompt=system_prompt,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )

        # TODO: Create an agent factory so that we can have multiple agents running concurrently
        #       Currently, you can handoff again and again to the same agent
        self.agent_registry = {agent.name: agent for agent in children}

        # Handoff tool - updated description and signature expectation
        handoff_tool = FunctionTool(
            name="handoff_to_agent",
            func=self._route_to_agent,  # Maps to the updated routing method
            description=(
                "Delegate the task to a specific agent. "
                "Use this tool to pass control to another agent for a specific step. "
                "You MUST provide the `agent_name` of the target agent. "
                "If the target agent requires specific inputs (check its `input_schema` in the AVAILABLE AGENTS list), "
                "provide them as a JSON dictionary in the `agent_input` argument."
            ),
            # The underlying ReActAgent will expect parameters named 'agent_name' and 'agent_input'
        )

        if "handoff_to_agent" not in tool_registry.tools:
            tool_registry.register(handoff_tool)

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def description(self) -> str:
        return self._delegate.description

    @property
    def model(self) -> str:
        return self._delegate.model

    @property
    def input_schema(self) -> Optional[Type[BaseModel]]:
        return self._delegate.input_schema

    @property
    def output_schema(self) -> Optional[Type[BaseModel]]:
        return self._delegate.output_schema

    def _agent_schemas(self) -> str:
        return f"""
        -------
        AVAILABLE AGENTS:
        {json.dumps([agent.to_json() for agent in self.children])}
        Use the `handoff_to_agent` tool to delegate tasks to these agents.
        Make sure to provide the required inputs as specified by the agent's `input_schema`.
        -------
        """

    async def _route_to_agent(self, agent_name: str, agent_input: dict = {}):
        """
        Route the task to a specific agent, validating input if schema is defined.

        Args:
            agent_name: The name of the agent to route the task to.
            agent_input: A json containing the input parameters for the target agent.
        """

        if agent_name not in self.agent_registry:
            raise ValueError(f"Agent {agent_name} not found in registry")

        logger.info(f"Handing off to agent '{agent_name}' with input: {agent_input}")
        agent = self.agent_registry[agent_name]
        agent_input = self._validate_input_schema(agent, agent_input)
        agent_result = await agent.run(**agent_input)
        logger.info(f"Agent {agent_name} returned: {agent_result}")
        return agent_result

    def _validate_input_schema(self, agent: Agent, agent_input: dict) -> dict:
        if agent.input_schema:
            try:
                validated_input = agent.input_schema.model_validate(agent_input)
                logger.debug(f"Validated input for {agent.name}: {validated_input}")
                return validated_input.model_dump()
            except ValidationError as e:
                raise ValueError(
                    f"Input validation failed for agent '{agent.name}'. Details: {e}. Please provide the correct input according to the schema: {agent.input_schema.model_json_schema()}"
                )
        return agent_input

    async def run(self, **kwargs) -> Any:
        # The supervisor's own run method delegates to the internal ReAct agent
        # The ReAct agent will use the system prompt (with agent schemas) and tools (including handoff_to_agent)
        return await self._delegate.run(**kwargs)
