import json
import logging
from typing import Any

from langfuse.decorators import observe

from core.agent import Agent
from core.re_act import ReActAgent
from core.tool import FunctionTool, ToolRegistry

logger = logging.getLogger(__name__)


class SupervisorAgent(Agent):
    """
    A generic supervisor agent that manages and coordinates other agents.

    Inspired by LangGraph's supervisor implementation, this agent can:
    1. Route tasks between specialized agents
    2. Maintain shared state across agent executions
    3. Make decisions about workflow progression
    4. Process and integrate results from different agents
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        system_prompt: str,
        tool_registry: ToolRegistry = ToolRegistry(),
        children: list[Agent] = [],
        max_steps: int = 10,
    ):
        self.children = children
        system_prompt += self._agent_schemas()
        self._delegate: ReActAgent = ReActAgent(
            name=f"{name}_supervisor",
            description=description,
            model=model,
            prompt=system_prompt,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )

        # Create a registry of available agents by name
        self.agent_registry = {agent.name: agent for agent in children}

        # Handoff tool
        handoff_tool = FunctionTool(
            name="handoff_to_agent",
            func=self._route_to_agent,
            description="Route the task to a specific agent or finish the workflow.",
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

    def _agent_schemas(self) -> str:
        return f"""
        -------
        AVAILABLE AGENTS:
        {json.dumps([agent.to_json() for agent in self.children])}
        -------
        """

    async def _route_to_agent(self, agent_name: str, **kwargs):
        """
        Route the task to a specific agent or finish the workflow.
        Args:
            agent_name: The name of the agent to route the task to.
        """

        if agent_name not in self.agent_registry:
            raise ValueError(f"Agent {agent_name} not found in registry")

        logger.info(f"Handing off to agent {agent_name}")
        agent = self.agent_registry[agent_name]

        actual_kwargs = kwargs["kwargs"]
        if not actual_kwargs:
            return await agent.run()
        return await agent.run(extra_prompt=json.dumps(actual_kwargs))

    @observe(name="supervisor.run")
    async def run(self, **kwargs) -> Any:
        return await self._delegate.run(**kwargs)
