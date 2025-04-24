import logging

import diskcache
import requests

from agents.commons import Researcher
from agents.multi_agent.models import (
    Plan,
    PlannerInput,
    ReflectionInput,
    RetrievalInput,
    Synthesis,
    SynthesisInput,
)
from agents.multi_agent.prompts import (
    PLANNER_PROMPT,
    REFLECTOR_PROMPT,
    RETREIVER_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIS_PROMPT,
)
from core.cached_tool import CachedTool
from core.re_act import ReActAgent
from core.supervisor import SupervisorAgent
from core.tool import FunctionTool, ToolRegistry
from memory.scratchpad import Scratchpad
from memory.vector_store import VectorStoreMemory
from scraping.universal_scraper import UniversalScraper
from search.arxiv import ArxivSearchEngine
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine, ProgrammableSearchNoResults

logger = logging.getLogger(__name__)


class MultiAgentResearcher(Researcher):
    def __init__(self, model: str):
        self.model = model

    async def research(self, topic: str) -> str:
        tool_cache = diskcache.Cache(".cache")
        vector_store = VectorStoreMemory(
            db_path=".memory/research", collection_name="research"
        )
        try:
            google_search = GoogleProgrammableSearchEngine(num_results=10)
            google_scraping_search = GoogleProgrammableScrapingSearchEngine(
                num_results=10, slow_mo=100
            )

            search = FallbackSearchEngine(
                primary_engine=google_search,
                fallback_engine=google_scraping_search,
                error_conditions=[requests.HTTPError, ProgrammableSearchNoResults],
            )

            arxiv_search = ArxivSearchEngine(num_results=10)

            scraper = UniversalScraper(memory=vector_store)

            scratchpad = Scratchpad()

            add_vector_store_tool = FunctionTool(
                name="add_to_memory",
                func=vector_store.add,
                description="Use this tool to add a document to the memory for semantic retrieval.",
            )
            query_vector_store_tool = FunctionTool(
                name="query_memory",
                func=vector_store.query,
                description="Use this tool to query the memory for relevant documents given a query_text",
            )

            get_all_research_tool = FunctionTool(
                name="get_all_research",
                func=vector_store.get_all_documents,
                description="Use this tool to get all research snippets from the memory",
            )

            add_todo_tool = FunctionTool(
                name="add_todo",
                func=scratchpad.add_todos,
                description="Use this tool to add a list of todos to the scratchpad",
            )

            mark_todo_as_done_tool = FunctionTool(
                name="mark_todo_as_done",
                func=scratchpad.mark_todos_as_done,
                description="Use this tool to mark a list of todos as done in the scratchpad",
            )

            google_search_tool = CachedTool(
                FunctionTool(
                    name="google_search",
                    func=search.search_and_stitch,
                ),
                cache=tool_cache,
            )

            google_search_tool = CachedTool(
                FunctionTool(
                    name="google_search",
                    func=search.search_and_stitch,
                    description="Use this tool to search the web for general information. Useful for getting a broad overview of a topic.",
                ),
                cache=tool_cache,
            )

            arxiv_search_tool = CachedTool(
                FunctionTool(
                    name="arxiv_search",
                    func=arxiv_search.search_and_stitch,
                    description="Use this tool to search Arxiv for academic papers, research papers, and other scholarly articles. Useful for more technical and academic topics.",
                ),
                cache=tool_cache,
            )

            web_scrape_tool = FunctionTool(
                name="scrape_url",
                func=scraper.scrape_url,
                description="Use this tool to scrape a specific URL for information. Useful for getting detailed information from a specific website or PDF.",
            )

            add_note_tool = FunctionTool(
                name="add_note",
                func=scratchpad.add_note,
                description="Use this tool to add a note to the scratchpad, prefer to add in a markdown format",
            )

            get_notes_tool = FunctionTool(
                name="get_notes",
                func=scratchpad.get_notes,
                description="Use this tool to get the notes from the scratchpad",
            )

            tool_registry = ToolRegistry(
                google_search_tool,
                arxiv_search_tool,
                web_scrape_tool,
            )

            planner_agent = ReActAgent(
                name="planner",
                description="Agent that uses search tools to understand the topic and identify key subtopics for research.",
                model=self.model,
                tool_registry=tool_registry,
                prompt=PLANNER_PROMPT,
                input_schema=PlannerInput,
                output_schema=Plan,
                max_steps=10,
            )

            retriever_agent = ReActAgent(
                name="retriever",
                description="Agent that generates search queries for a subtopic, retrieves information, and processes the findings.",
                model=self.model,
                tool_registry=tool_registry.with_tools(
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
                tool_registry=tool_registry.with_tools(
                    query_vector_store_tool, get_notes_tool
                ),
                prompt=REFLECTOR_PROMPT,
                input_schema=ReflectionInput,
                max_steps=50,
            )

            synthesis_agent = ReActAgent(
                name="synthesis",
                description="Agent that synthesizes the gathered information into a final markdown report.",
                model=self.model,
                tool_registry=ToolRegistry(
                    query_vector_store_tool, get_all_research_tool, get_notes_tool
                ),
                prompt=SYNTHESIS_PROMPT.format(original_topic=topic),
                input_schema=SynthesisInput,
                output_schema=Synthesis,
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
                tool_registry=ToolRegistry(
                    add_todo_tool,
                    mark_todo_as_done_tool,
                ),
                system_prompt=SUPERVISOR_PROMPT.format(topic=topic),
                max_steps=50,
            )

            _agent_result = await supervisor_agent.run()
            return _agent_result.content
        finally:
            tool_cache.close()
            vector_store.clear()
