"""CLI command for deep research on a topic."""

import asyncio
import logging

import click

import utils
from agents.deep_research.multi_agent_research import MultiAgentResearcher
from agents.deep_research.single_agent_research import SingleAgentResearcher

logger = logging.getLogger(__name__)


@click.command()
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
@click.option("--multi-agent", "-a", is_flag=True, help="Use multi-agent research")
def research(topic: str, model: str, output: str, max_steps: int, multi_agent: bool):
    """Research a topic and generate a markdown report"""
    logger.info(f"CLI invoked for research on topic: '{topic}' using model: {model}")

    click.echo(f"Starting research for topic: {topic}")
    click.echo(f"Using model: {model}")

    researcher = SingleAgentResearcher(model=model)
    if multi_agent:
        researcher = MultiAgentResearcher(model=model)

    try:
        # Run the async research function
        report_content = asyncio.run(researcher.research(topic))
        if report_content is None:
            raise ValueError("No report content returned from the research")

        # Extract the markdown report from the report content
        markdown_report = utils.extract_lang_block(report_content, "markdown")
        if markdown_report is None:
            raise ValueError("No markdown report returned from the research")

        with open(output, "w") as f:
            f.write(markdown_report)

        click.echo(f"Research complete. Report saved to: {output}")

    except Exception as e:
        logger.error(
            f"Research failed with an unhandled exception in orchestrator: {e}"
        )
        click.echo(f"An error occurred during the research: {e}", err=True)
        raise e
