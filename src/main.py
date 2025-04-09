#!/usr/bin/env python
import asyncio
import click
import logging
from typing import Optional

from agents.orchestrator import ResearchOrchestrator

from logger import configure_global_logging

logger = logging.getLogger(__name__)

configure_global_logging()


@click.group()
def cli():
    """Khojkar - Conduct deep research on a topic using LLMs"""
    pass


@cli.command()
@click.option(
    "--topic", "-t", 
    required=True, 
    help="The topic to research"
)
@click.option(
    "--model", "-m", 
    default="gemini/gemini-2.0-flash",
    help="The LLM model to use (default: gemini/gemini-2.0-flash)"
)
@click.option(
    "--queries", "-q", 
    default=5, 
    type=int, 
    help="Number of search queries to generate (default: 5)"
)
@click.option(
    "--output", "-o", 
    default=None, 
    help="Output file path (default: {topic}.md)"
)
def research(topic: str, model: str, queries: int, output: Optional[str]):
    """Research a topic and generate a markdown report"""
    logger.info(f"CLI invoked for research on topic: '{topic}' using model: {model}")
    click.echo(f"Starting research for topic: {topic}")
    click.echo(f"Using model: {model}")
    
    # Initialize the orchestrator
    orchestrator = ResearchOrchestrator(
        topic=topic,
        model=model,
        num_queries=queries,
        output_path=output
    )
    
    # Run the research process via the orchestrator
    try:
        # Use asyncio.run() to execute the async method
        report_path = asyncio.run(orchestrator.execute_research())
        
        if report_path:
             click.echo(f"Research complete. Report saved to: {report_path}")
        else:
             click.echo("Research process did not complete successfully or was aborted.")
             
    except Exception as e:
        # The orchestrator logs the exception details, here we just inform the user via CLI
        logger.error(f"Research failed with an unhandled exception in orchestrator: {e}") 
        click.echo(f"An error occurred during the research: {e}", err=True)
        # Optionally re-raise or exit with an error code
        # raise # Or import sys; sys.exit(1)


if __name__ == "__main__":
    # The try-except block here can catch exceptions from cli() setup itself,
    # or exceptions deliberately re-raised from the research command.
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error at top level: {e}")
        # Optionally provide a user-friendly message here as well
        # click.echo(f"A critical error occurred: {e}", err=True)
        # Consider exiting with a non-zero status code
        # import sys
        # sys.exit(1) 
