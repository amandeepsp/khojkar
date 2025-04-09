import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .planner import QueryPlanner, ScrapePlanner
from .reporter import Reporter
from search.google import GoogleProgrammableSearchEngine
from search.scrape import Scraper

logger = logging.getLogger(__name__)

class ResearchOrchestrator:
    """Orchestrates the entire research process."""

    def __init__(
        self, 
        topic: str, 
        model: str, 
        num_queries: int, 
        output_path: Optional[str] = None,
        scraper_concurrency: int = 5
    ):
        self.topic = topic
        self.model = model
        self.num_queries = num_queries
        self.output_path = output_path
        self.scraper_concurrency = scraper_concurrency

        # Initialize necessary components
        self.query_planner = QueryPlanner(model=self.model, topic=self.topic, number_of_queries=self.num_queries)
        self.search_engine = GoogleProgrammableSearchEngine()
        self.scrape_planner = ScrapePlanner(model=self.model, topic=self.topic)
        self.scraper = Scraper(concurrency=self.scraper_concurrency)
        self.reporter = Reporter(model=self.model, topic=self.topic)
        
        logger.info(f"ResearchOrchestrator initialized for topic: '{self.topic}' with model: {self.model}")

    async def _generate_queries(self) -> List[str]:
        logger.info("Starting query generation phase")
        search_queries = self.query_planner.generate_queries()
        logger.info(f"Generated {len(search_queries)} queries.")
        return search_queries

    async def _perform_searches(self, search_queries: List[str]) -> Dict[str, str]:
        logger.info(f"Starting search phase with {len(search_queries)} queries")
        
        logger.info(f"Executing {len(search_queries)} parallel search tasks")
        tasks = [self.search_engine.search_and_stitch_context(query) for query in search_queries]
        search_results_list = await asyncio.gather(*tasks)
        
        mapped_search_results = {query: result for query, result in zip(search_queries, search_results_list)}
        logger.info(f"Completed search phase, received results for {len(mapped_search_results)} queries")
        return mapped_search_results

    async def _select_sources(self, mapped_search_results: Dict[str, str]) -> List[Dict[str, Any]]:
        logger.info("Starting source analysis phase")
        selected_sources = self.scrape_planner.analyze_and_select_sources(mapped_search_results)
        logger.info(f"Selected {len(selected_sources)} sources for scraping.")
        return selected_sources

    async def _scrape_sources(self, selected_sources: List[Dict[str, Any]]) -> List[str]:
        logger.info(f"Starting scraping phase for {len(selected_sources)} sources")
        urls = [entry["url"] for entry in selected_sources]
        
        logger.info(f"Executing scraping for {len(urls)} URLs")
        scrape_results = await self.scraper.scrape_urls(urls)
        
        non_empty_results = sum(1 for content in scrape_results if content)
        logger.info(f"Completed scraping phase, {non_empty_results}/{len(scrape_results)} sources yielded content")
        return scrape_results

    async def _generate_report(self, scrape_results: List[str]) -> str:
        logger.info("Starting report generation phase")
        report = self.reporter.generate_report(scrape_results)
        logger.info("Report generation complete.")
        return report
        
    def _save_report(self, report: str) -> Path:
        """Saves the report and returns the path object."""
        if self.output_path is None:
            output_filename = f"{self.topic.replace(' ', '_')}.md"
        else:
            output_filename = self.output_path
            
        output_path_obj = Path(output_filename)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, "w") as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path_obj.absolute()}")
        return output_path_obj


    async def execute_research(self) -> Optional[Path]:
        """Executes the full research workflow asynchronously and returns the report path."""
        try:
            logger.info(f"Executing research for topic: '{self.topic}'")
            
            search_queries = await self._generate_queries()
            if not search_queries:
                logger.warning("No search queries were generated. Aborting research.")
                return None

            search_results = await self._perform_searches(search_queries)
            if not search_results:
                logger.warning("No search results were obtained. Aborting research.")
                return None
                
            selected_sources = await self._select_sources(search_results)
            if not selected_sources:
                logger.warning("No sources were selected for scraping. Aborting research.")
                return None

            scraped_content = await self._scrape_sources(selected_sources)
            if not any(scraped_content): # Check if any content was actually scraped
                 logger.warning("Scraping yielded no content. Aborting report generation.")
                 return None

            report = await self._generate_report(scraped_content)
            report_path = self._save_report(report)
            
            logger.info(f"Research completed successfully for topic: '{self.topic}'")
            return report_path

        except Exception as e:
            logger.exception(f"An error occurred during the research process for topic '{self.topic}': {e}")
            raise 