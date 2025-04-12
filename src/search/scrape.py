import asyncio
import logging
from io import BytesIO

import requests
from pypdf import PdfReader
from trafilatura import extract

logger = logging.getLogger(__name__)


class Scraper:
    """Scrape the website and return the content"""

    def __init__(self, concurrency: int = 4):
        self.concurrency = concurrency
        logger.info(f"Initialized Scraper with concurrency={concurrency}")

    async def scrape_urls(self, urls: list[str]) -> list[str]:
        """Scrape the websites and return the content.

        Args:
            urls: The URLs to scrape

        Returns:
            The content of the websites as a list of strings
        """
        logger.info(f"Scraping {len(urls)} URLs with concurrency {self.concurrency}")
        semaphore = asyncio.Semaphore(self.concurrency)

        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url)

        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        logger.info(f"Completed scraping {len(urls)} URLs")

        # Count non-empty results
        non_empty = sum(1 for content in results if content)
        logger.info(f"Successfully scraped {non_empty}/{len(urls)} URLs")

        return results

    @staticmethod
    async def scrape_url(url: str, skip_errors: bool = True) -> str:
        """Scrape the website and return the content.
        Uses trafilatura to parse the html and pypdf to parse the pdf.

        Args:
            url: The URL to scrape
            skip_errors: If True, return empty string on error instead of raising

        Returns:
            The content of the website as a string

        """
        logger.info(f"Scraping URL: {url}")
        # Browser-like headers to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            error_msg = f"Error scraping URL: {url}, error: {e}"
            logger.error(error_msg)
            if skip_errors:
                return ""
            raise ValueError(error_msg)

        content_type = response.headers.get("Content-Type", "")

        if "text/html" in content_type:
            # Use more aggressive options to extract content even when main content isn't easily detected
            content = extract(response.text)

            if content:
                logger.info(
                    f"Successfully extracted HTML content from {url}, size: {len(content)} chars"
                )
            else:
                logger.warning(f"Extracted empty content from HTML at {url}")
            return content or ""

        if "application/pdf" in content_type:
            try:
                pdf_reader = PdfReader(BytesIO(response.content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                logger.info(
                    f"Successfully extracted PDF content from {url}, size: {len(text)} chars"
                )
                return text
            except Exception as e:
                error_msg = f"Error extracting PDF content from {url}: {e}"
                logger.error(error_msg)
                if skip_errors:
                    return ""
                raise ValueError(error_msg)

        error_msg = f"Unsupported content type: {content_type} for URL: {url}"
        logger.error(error_msg)
        if skip_errors:
            return ""
        raise ValueError(error_msg)
