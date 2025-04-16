import logging
from datetime import datetime, timezone

import litellm
import requests
from diskcache import Cache

import llm
from agents.commons import Researcher
from agents.naive.prompt import deep_research_prompt
from core.cached_tool import CachedTool
from core.re_act import ReActAgent
from core.tool import FunctionTool, ToolRegistry
from scraping.universal_scraper import UniversalScraper
from search.arxiv import ArxivSearchEngine
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine

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


class SingleAgentResearcher(Researcher):
    def __init__(self, model: str):
        self.model = model

    async def research(self, topic: str) -> str:
        tool_registry = ToolRegistry()

        tool_cache = Cache(".cache/tool_cache")

        google_search = GoogleProgrammableSearchEngine(num_results=10)
        google_scraping_search = GoogleProgrammableScrapingSearchEngine(
            num_results=10, headless=True, slow_mo=100
        )

        search = FallbackSearchEngine(
            primary_engine=google_search,
            fallback_engine=google_scraping_search,
            error_conditions=[requests.HTTPError],
        )

        arxiv_search = ArxivSearchEngine(num_results=10)

        scraper = UniversalScraper()

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

        web_scrape_tool = CachedTool(
            FunctionTool(
                name="scrape_url",
                func=scraper.scrape_url,
                description="Use this tool to scrape a specific URL for information. Useful for getting detailed information from a specific website or PDF.",
            ),
            cache=tool_cache,
        )

        tool_registry.register(google_search_tool)
        tool_registry.register(web_scrape_tool)
        tool_registry.register(arxiv_search_tool)

        tool_descriptions = tool_registry._tool_descriptions()
        current_date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

        prompt = deep_research_prompt.format(
            question=topic,
            report_format="apa",
            current_date=current_date_str,
            tool_descriptions=tool_descriptions,
        )

        agent = DeepResearchAgent(
            name="research_agent",
            model=self.model,
            prompt=prompt,
            tool_registry=tool_registry,
            max_steps=30,
        )
        result = await agent.run()

        if result.content is None:
            raise ValueError("No content found in the result")

        return result.content
