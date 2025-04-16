import requests

from agents.commons import Researcher
from agents.multi_agent.models import (
    Questions,
    Reflection,
    Retrievals,
    Subtopics,
)
from agents.multi_agent.prompts import (
    PLANNER_PROMPT,
    QUESTION_GENERATOR_PROMPT,
    REFLECTOR_PROMPT,
    RETREIVER_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIS_PROMPT,
)
from core.re_act import ReActAgent
from core.supervisor import SupervisorAgent
from core.tool import FunctionTool, ToolRegistry
from scraping.universal_scraper import UniversalScraper
from search.arxiv import ArxivSearchEngine
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine


class MultiAgentResearcher(Researcher):
    def __init__(self, model: str):
        self.model = model

    async def research(self, topic: str) -> str:
        tool_registry = ToolRegistry()

        google_search = GoogleProgrammableSearchEngine(num_results=10)
        google_scraping_search = GoogleProgrammableScrapingSearchEngine(
            num_results=10, slow_mo=100
        )

        search = FallbackSearchEngine(
            primary_engine=google_search,
            fallback_engine=google_scraping_search,
            error_conditions=[requests.HTTPError],
        )

        arxiv_search = ArxivSearchEngine(num_results=10)

        scraper = UniversalScraper()

        google_search_tool = FunctionTool(
            name="google_search",
            func=search.search_and_stitch,
        )

        google_search_tool = FunctionTool(
            name="google_search",
            func=search.search_and_stitch,
            description="Use this tool to search the web for general information. Useful for getting a broad overview of a topic.",
        )

        arxiv_search_tool = FunctionTool(
            name="arxiv_search",
            func=arxiv_search.search_and_stitch,
            description="Use this tool to search Arxiv for academic papers, research papers, and other scholarly articles. Useful for more technical and academic topics.",
        )

        web_scrape_tool = FunctionTool(
            name="scrape_url",
            func=scraper.scrape_url,
            description="Use this tool to scrape a specific URL for information. Useful for getting detailed information from a specific website or PDF.",
        )

        tool_registry = ToolRegistry()
        tool_registry.register(google_search_tool)
        tool_registry.register(arxiv_search_tool)
        tool_registry.register(web_scrape_tool)

        planner_agent = ReActAgent(
            name="planner",
            description="Agent that plans the research project, breaking it into subtopics.",
            model=self.model,
            tool_registry=tool_registry,
            prompt=PLANNER_PROMPT,
            output_format=Subtopics,
            max_steps=30,
        )

        question_generator_agent = ReActAgent(
            name="question_generator",
            description="Agent that generates questions for the research project.",
            model=self.model,
            tool_registry=tool_registry,
            prompt=QUESTION_GENERATOR_PROMPT,
            output_format=Questions,
            max_steps=30,
        )

        retriever_agent = ReActAgent(
            name="retriever",
            description="Agent that retrieves information from the web to answer questions.",
            model=self.model,
            tool_registry=tool_registry,
            prompt=RETREIVER_PROMPT,
            output_format=Retrievals,
            max_steps=30,
        )

        reflection_agent = ReActAgent(
            name="reflection",
            description="Agent that reflects on the information from the web to answer questions.",
            model=self.model,
            tool_registry=tool_registry,
            prompt=REFLECTOR_PROMPT,
            output_format=Reflection,
            max_steps=30,
        )

        synthesis_agent = ReActAgent(
            name="synthesis",
            description="Agent that synthesizes the information from the web to answer questions.",
            model=self.model,
            tool_registry=tool_registry,
            prompt=SYNTHESIS_PROMPT.format(original_topic=topic),
            max_steps=30,
        )

        supervisor_agent = SupervisorAgent(
            name="storm",
            description="Supervisor agent for the STORM research workflow",
            model=self.model,
            children=[
                planner_agent,
                question_generator_agent,
                retriever_agent,
                reflection_agent,
                synthesis_agent,
            ],
            system_prompt=SUPERVISOR_PROMPT.format(topic=topic),
            max_steps=50,
        )

        response = await supervisor_agent.run()

        return response.content
