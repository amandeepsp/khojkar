import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

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

    @abstractmethod
    async def _scrape_url(self, url: str) -> ScrapeResult:
        """Scrape the website and return the content"""
        pass

    async def scrape_url(self, url: str) -> str:
        """Scrape the website and return the content

        Args:
            url: The URL to scrape

        Returns:
            The content of the website and metadata as a json string
        """
        logger.info(f"Scraping URL: {url}")
        try:
            scrape_result = await self._scrape_url(url)
            return scrape_result.model_dump_json()
        except Exception as e:
            logger.error(f"Error scraping URL: {url}, error: {e}")
            return ""
