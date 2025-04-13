#!/usr/bin/env python
import asyncio
import logging
import sys
from datetime import datetime, timezone

import click
import requests
from diskcache import Cache

import utils
from agents.deep_research import DeepResearchAgent
from core.cached_tool import CachedTool
from core.tool import FunctionTool, Tool, ToolRegistry
from prompts import deep_research_prompt
from scraping.trafilatura_scaper import TrafilaturaScraper
from search.arxiv import ArxivSearchEngine
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Caching Setup ---
CACHE_DIR = ".cache/tool_cache"
tool_cache = Cache(CACHE_DIR)


@click.group()
def cli():
    """Khojkar - Conduct deep research on a topic using LLMs"""
    pass


@cli.command()
@click.option("--topic", "-t", required=True, help="The topic to research")
@click.option(
    "--model",
    "-m",
    default="gemini/gemini-2.0-flash",
    help="The LLM model to use (default: gemini/gemini-2.0-flash)",
)
@click.option("--output", "-o", required=True, help="Output file path")
@click.option(
    "--max-steps",
    "-s",
    default=10,
    help="The maximum number of steps to take (default: 10)",
)
def research(topic: str, model: str, output: str, max_steps: int):
    """Research a topic and generate a markdown report"""
    logger.info(f"CLI invoked for research on topic: '{topic}' using model: {model}")

    click.echo(f"Starting research for topic: {topic}")
    click.echo(f"Using model: {model}")

    try:
        # Run the async research function
        report_path = asyncio.run(execute_research(topic, model, output, max_steps))

        click.echo(f"Research complete. Report saved to: {report_path}")

    except Exception as e:
        logger.error(
            f"Research failed with an unhandled exception in orchestrator: {e}"
        )
        click.echo(f"An error occurred during the research: {e}", err=True)
        raise e


async def execute_research(
    topic: str, model: str, output_path: str, max_steps: int
) -> str:
    """Execute the research using a ReACT agent"""
    tool_registry = ToolRegistry()

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

    scraper = TrafilaturaScraper()

    # 1. Initialize original Tool instances
    google_search_tool: Tool = CachedTool(
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
            description="Use this tool to scrape a specific URL for information. Useful for getting detailed information from a specific website.",
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
        model=model,
        prompt=prompt,
        tool_registry=tool_registry,
        max_steps=max_steps,
    )
    result = await agent.run()

    if result.content is None:
        raise ValueError(
            "No content found in the result, try increasing the number of steps or using a different model"
        )

    markdown_report = utils.extract_lang_block(result.content, "markdown")

    with open(output_path, "w") as f:
        f.write(markdown_report)

    return output_path


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error at top level: {e}")
        sys.exit(1)
