#!/usr/bin/env python
import asyncio
import logging
import sys

import click
from rich.logging import RichHandler

import utils
from agents.multi_agent.agents import MultiAgentResearcher
from agents.naive.deep_research import SingleAgentResearcher

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            level=logging.INFO,
            omit_repeated_times=False,
        )
    ],
)


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


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error at top level: {e}")
        sys.exit(1)
