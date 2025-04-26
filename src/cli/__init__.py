"""CLI command group for the deep-researcher tool."""

import logging
import sys

import click
from rich.logging import RichHandler

from cli.codebase import analyze_codebase
from cli.research import research

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
    """Deep Researcher - Conduct deep research and codebase analysis using LLMs"""
    pass


cli.add_command(research, name="research")
cli.add_command(analyze_codebase, name="analyze-codebase")


def run_cli():
    """Run the CLI application."""
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error at top level: {e}")
        sys.exit(1)
