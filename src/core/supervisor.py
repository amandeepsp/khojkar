import logging
from typing import Any

from core.agent import Agent
from core.re_act import ReActAgent
from core.tool import FunctionTool, ToolRegistry

logger = logging.getLogger(__name__)


class SupervisorAgent:
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
        self.name = name
        self.model = model
        self.children = children
        self.parent = None
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self._internal_agent: Agent = ReActAgent(
            name=f"{name}_supervisor",
            description=description,
            model=model,
            prompt=system_prompt,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )

        # Create a registry of available agents by name
        self.agent_registry = {agent.name: agent for agent in children}
        self._append_agent_schemas()

        # Handoff tool
        handoff_tool = FunctionTool(
            name="handoff_to_agent",
            func=self._route_to_agent,
            description="Route the task to a specific agent or finish the workflow.",
        )

        if "handoff_to_agent" not in tool_registry.tools:
            tool_registry.register(handoff_tool)

    def _append_agent_schemas(self):
        # TODO: Add in JSON format for sub_agents
        self.system_prompt += f"""
        <sub_agents>
        {
            "\n".join([
                f"Agent: {agent.name}\nDescription: {agent.description}"
                for agent in self.children
            ])
        }
        </sub_agents>
        """

    async def _route_to_agent(self, agent_name: str):
        """
        Route the task to a specific agent or finish the workflow.
        Args:
            agent_name: The name of the agent to route the task to.
        """

        if agent_name not in self.agent_registry:
            raise ValueError(f"Agent {agent_name} not found in registry")

        logger.info(f"Handing off to agent {agent_name}")
        agent = self.agent_registry[agent_name]
        return await agent.run()

    async def run(self, **kwargs) -> Any:
        return await self._internal_agent.run(**kwargs)
