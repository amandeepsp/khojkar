import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from src.memory.memory import Memory  # Import Memory protocol

logger = logging.getLogger(__name__)


class ScrapeResult(BaseModel):
    """Scrape result"""

    url: str = Field(description="The URL of the scraped website")
    title: str = Field(description="The title of the scraped website")
    content: str = Field(description="The extracted content of the scraped website")
    author: str | None = Field(description="The author of the scraped website")
    published_date: str | None = Field(
        description="The published date of the scraped website"
    )
    website_name: str | None = Field(description="The name of the website")


class BaseScraper(ABC):
    """Base class for scrapers"""

    def __init__(self, memory: Optional[Memory] = None):
        """Initializes the scraper with an optional memory object.

        Args:
            memory: An optional object conforming to the Memory protocol.
        """
        self.memory = memory

    @abstractmethod
    async def _scrape_url(self, url: str) -> ScrapeResult:
        """Scrape the website and return the content"""
        pass

    async def scrape_url(self, url: str) -> str:
        """Scrape the website, optionally add to memory, and return the content

        Uses the memory object provided during initialization, if any.

        Args:
            url: The URL to scrape

        Returns:
            The content of the website and metadata as a json string
        """
        logger.info(f"Scraping URL: {url}")
        try:
            scrape_result = await self._scrape_url(url)
            if self.memory is not None:
                logger.info(f"Adding scraped content from {url} to memory.")
                metadata_raw = scrape_result.model_dump(exclude={"content"})
                # Filter out keys with None values
                metadata_filtered = {
                    k: v for k, v in metadata_raw.items() if v is not None
                }
                await self.memory.add(
                    text=scrape_result.content, metadata=metadata_filtered
                )
            return scrape_result.model_dump_json()
        except Exception as e:
            logger.error(f"Error scraping URL: {url}, error: {e}")
            return ""
