#!/usr/bin/env python
import asyncio
import click
import logging
from typing import Optional
from pathlib import Path

from agents.planner import QueryPlanner, ScrapePlanner
from agents.reporter import Reporter
from search.google import GoogleProgrammableSearchEngine
from search.scrape import Scraper

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
    logger.info(f"Starting research on topic: '{topic}' using model: {model}")
    click.echo(f"Researching topic: {topic}")
    click.echo(f"Using model: {model}")
    
    # Generate search queries
    logger.info("Starting query generation phase")
    click.echo("Generating search queries...")
    query_planner = QueryPlanner(model=model, topic=topic, number_of_queries=queries)
    search_queries = query_planner.generate_queries()
    
    # Search for each query
    logger.info(f"Starting search phase with {len(search_queries)} queries")
    click.echo("Searching the web...")
    search_engine = GoogleProgrammableSearchEngine()
    
    async def run_searches():
        logger.info(f"Executing {len(search_queries)} parallel search tasks")
        tasks = [search_engine.search_and_stitch_context(query) for query in search_queries]
        return await asyncio.gather(*tasks)
    
    search_results = asyncio.run(run_searches())
    mapped_search_results = {query: result for query, result in zip(search_queries, search_results)}
    logger.info(f"Completed search phase, received results for {len(mapped_search_results)} queries")
    
    # Analyze and select the best sources
    logger.info("Starting source analysis phase")
    click.echo("Analyzing search results...")
    scrape_planner = ScrapePlanner(model=model, topic=topic, search_results=mapped_search_results)
    selected_sources = scrape_planner.analyze_and_select_sources()
    
    # Scrape the selected sources
    logger.info(f"Starting scraping phase for {len(selected_sources)} sources")
    click.echo("Scraping selected sources...")
    scraper = Scraper(concurrency=5)
    urls = [entry["url"] for entry in selected_sources]
    
    async def run_scraping():
        logger.info(f"Executing scraping for {len(urls)} URLs")
        return await scraper.scrape_urls(urls)
    
    scrape_results = asyncio.run(run_scraping())
    
    # Count non-empty results
    non_empty_results = sum(1 for content in scrape_results if content)
    logger.info(f"Completed scraping phase, {non_empty_results}/{len(scrape_results)} sources yielded content")
    
    # Generate the report
    logger.info("Starting report generation phase")
    click.echo("Generating report...")
    reporter = Reporter(model=model, topic=topic, scraped_content=scrape_results)
    report = reporter.generate_report()
    
    # Save the report
    if output is None:
        output = f"{topic.replace(' ', '_')}.md"
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path.absolute()}")
    click.echo(f"Report saved to {output_path}")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.exception(f"Unhandled error during research: {e}")
        raise
