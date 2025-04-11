#!/usr/bin/env python
import asyncio
import logging
import sys
from typing import Optional

import click
import requests

import utils
from core.re_act import ReActAgent
from core.tool import Tool, ToolRegistry
from logger import configure_global_logging
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine
from search.scrape import Scraper

logger = logging.getLogger(__name__)
configure_global_logging()


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
@click.option(
    "--max-steps",
    "-s",
    default=10,
    help="The maximum number of steps to take (default: 10)",
)
@click.option(
    "--output", "-o", default=None, help="Output file path (default: {topic}.md)"
)
def research(topic: str, model: str, output: Optional[str], max_steps: int):
    """Research a topic and generate a markdown report"""
    logger.info(f"CLI invoked for research on topic: '{topic}' using model: {model}")
    click.echo(f"Starting research for topic: {topic}")
    click.echo(f"Using model: {model}")

    try:
        # Run the async research function
        report_path = asyncio.run(execute_research(topic, model, output, max_steps))

        if report_path:
            click.echo(f"Research complete. Report saved to: {report_path}")
        else:
            click.echo("Research process did not complete successfully or was aborted.")

    except Exception as e:
        logger.error(
            f"Research failed with an unhandled exception in orchestrator: {e}"
        )
        click.echo(f"An error occurred during the research: {e}", err=True)
        raise e


async def execute_research(
    topic: str, model: str, output: Optional[str], max_steps: int
) -> str:
    """Execute the research using a ReACT agent"""
    # Create empty tool registry (to be populated later)
    tool_registry = ToolRegistry()

    google_search = GoogleProgrammableSearchEngine(num_results=10)
    google_scraping_search = GoogleProgrammableScrapingSearchEngine(num_results=10)

    search = FallbackSearchEngine(
        primary_engine=google_search,
        fallback_engine=google_scraping_search,
        error_conditions=[requests.HTTPError],
    )

    scraper = Scraper()

    google_search_tool = Tool(
        name="google_search",
        func=search.search_and_stitch,
    )

    web_scrape_tool = Tool(
        name="scrape_url",
        func=scraper.scrape_url,
    )

    tool_registry = ToolRegistry()
    tool_registry.register(google_search_tool)
    tool_registry.register(web_scrape_tool)

    # Create comprehensive research prompt
    prompt = f"""Research '{topic}' thoroughly using the available tools.

IMPORTANT: You DO NOT have any information about this topic yet. You MUST use the provided tools to gather information BEFORE you can create a report. You CANNOT rely on internal knowledge to generate your report.

REQUIRED WORKFLOW - YOU MUST FOLLOW THIS PROCESS:
1. Start by using google_search to learn about the topic and discover 3-5 key subtopics
2. For EACH subtopic, use google_search again to gather more specific information
3. Use scrape_url on at least 3 different high-quality sources to get detailed content about the topic
4. Only after you have collected sufficient information using tools, create your report

Step 1: EXPLORE BREADTH
- Use google_search with general queries about {topic}
- Identify 3-5 major subtopics or perspectives based on search results
- Document what you've learned and what questions remain

Step 2: EXPLORE DEPTH
- For each subtopic:
  * Formulate specific search queries
  * Use google_search to find detailed information
  * Use scrape_url on at least 1-2 authoritative sources per subtopic
  * Document key findings for each subtopic

Step 3: ANALYZE & SYNTHESIZE
- Create a comprehensive markdown report with the following structure:
  1. Introduction (explain the topic and why it matters)
  2. Subtopic 1 (with evidence from your research)
  3. Subtopic 2 (with evidence from your research)
  4. Subtopic 3 (with evidence from your research)
  5. [Additional subtopics as needed]
  6. Comparison/Analysis (how the subtopics relate to each other)
  7. Conclusion (key takeaways)
  8. Sources (list of sources used)

Your report should be approximately 1,000-2,000 words and include direct evidence from your research. Focus on factual information and different perspectives on the topic.

IMPORTANT: When you've completed your research and are ready to provide the final report, DO NOT stop at saying "I will now create the report" or "I have enough information." IMMEDIATELY provide the complete report as your final response.

REMINDER: YOU MUST USE BOTH google_search AND scrape_url TOOLS. A high-quality report requires detailed information from actual web pages, not just search results.
"""

    agent = ReActAgent(
        model=model, prompt=prompt, tool_registry=tool_registry, max_steps=max_steps
    )
    result = await agent.run()
    assert result.content is not None
    markdown_report = utils.extract_lang_block(result.content, "markdown")

    output_path = output or f"{topic.replace(' ', '_')}.md"

    with open(output_path, "w") as f:
        f.write(markdown_report)

    return output_path


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error at top level: {e}")
        sys.exit(1)
