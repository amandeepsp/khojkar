import logging
from typing import Optional

import requests

from memory.memory import Memory

from .commons import BaseScraper, ScrapeResult
from .pdf_scraper import PdfScraper
from .trafilatura_scaper import TrafilaturaScraper

logger = logging.getLogger(__name__)


class UniversalScraper(BaseScraper):
    """Scraper that delegates to specific scrapers based on content type."""

    def __init__(self, memory: Optional[Memory] = None):
        self.html_scraper = TrafilaturaScraper()
        self.pdf_scraper = PdfScraper()
        super().__init__(memory)
        logger.info("Initialized UniversalScraper")

    async def _scrape_url(self, url: str) -> ScrapeResult:
        """Determine content type and delegate to the appropriate scraper."""
        logger.info(f"Determining content type for URL: {url}")
        try:
            # Use a HEAD request to check Content-Type without downloading the full content
            response = requests.head(url, allow_redirects=True, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            logger.debug(f"Content-Type for {url}: {content_type}")

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                logger.info(
                    f"Detected PDF content for {url}. Delegating to PdfScraper."
                )
                return await self.pdf_scraper._scrape_url(url)
            else:
                logger.info(
                    f"Detected non-PDF content ({content_type}) for {url}. Delegating to TrafilaturaScraper."
                )
                return await self.html_scraper._scrape_url(url)

        except requests.exceptions.RequestException as e:
            # If HEAD request fails, log error and potentially try default scraper or re-raise
            logger.warning(
                f"HEAD request failed for {url}: {e}. Falling back to default (HTML) scraper."
            )
            # Optionally, you could raise the error or implement more sophisticated fallback
            # For now, we'll try the HTML scraper as a default fallback.
            try:
                return await self.html_scraper._scrape_url(url)
            except Exception as fallback_e:
                logger.error(
                    f"Fallback HTML scraping also failed for {url}: {fallback_e}"
                )
                raise fallback_e  # Re-raise the exception from the fallback attempt

        except Exception as e:
            logger.error(f"Error determining scraper or during scraping for {url}: {e}")
            raise  # Re-raise other exceptions
