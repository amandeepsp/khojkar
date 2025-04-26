import logging

from agents.commons import Researcher
from agents.deep_research.models import (
    Plan,
    PlannerInput,
    ReflectionInput,
    Report,
    RetrievalInput,
    SynthesisInput,
)
from agents.deep_research.prompts import (
    PLANNER_PROMPT,
    REFLECTOR_PROMPT,
    RETREIVER_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIS_PROMPT,
)
from agents.deep_research.tools import (
    add_note_tool,
    add_vector_store_tool,
    get_all_notes_tool,
    query_vector_store_tool,
    search_scrape_tools,
)
from core.re_act import ReActAgent
from core.supervisor import SupervisorAgent
from core.tool import Toolbox

logger = logging.getLogger(__name__)


class MultiAgentResearcher(Researcher):
    def __init__(self, model: str):
        self.model = model

    async def research(self, topic: str) -> str:
        planner_agent = ReActAgent(
            name="planner",
            description="Agent that uses search tools to understand the topic and identify key subtopics for research.",
            model=self.model,
            tool_registry=search_scrape_tools,
            prompt=PLANNER_PROMPT,
            input_schema=PlannerInput,
            output_schema=Plan,
            max_steps=10,
        )

        retriever_agent = ReActAgent(
            name="retriever",
            description="Agent that generates search queries for a subtopic, retrieves information, and processes the findings.",
            model=self.model,
            tool_registry=search_scrape_tools.with_tools(
                add_vector_store_tool, add_note_tool
            ),
            prompt=RETREIVER_PROMPT,
            input_schema=RetrievalInput,
            max_steps=50,
        )

        reflection_agent = ReActAgent(
            name="reflection",
            description="Agent that reflects on the information gathered across all subtopics, identifies gaps, and suggests refinements.",
            model=self.model,
            tool_registry=Toolbox(
                get_all_notes_tool, add_note_tool, query_vector_store_tool
            ),
            prompt=REFLECTOR_PROMPT,
            input_schema=ReflectionInput,
            max_steps=50,
        )

        synthesis_agent = ReActAgent(
            name="synthesis",
            description="Agent that synthesizes the gathered information into a final markdown report.",
            model=self.model,
            tool_registry=Toolbox(get_all_notes_tool, query_vector_store_tool),
            prompt=SYNTHESIS_PROMPT.format(original_topic=topic, num_words=2000),
            input_schema=SynthesisInput,
            output_schema=Report,
            max_steps=30,
        )

        supervisor_agent = SupervisorAgent(
            name="storm",
            description="Supervisor agent for the research workflow",
            model=self.model,
            children=[
                planner_agent,
                retriever_agent,
                reflection_agent,
                synthesis_agent,
            ],
            system_prompt=SUPERVISOR_PROMPT.format(topic=topic),
            max_steps=50,
        )

        _agent_result = await supervisor_agent.run()
        return _agent_result.content
