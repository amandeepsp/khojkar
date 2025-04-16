import io
import logging

import requests
from pypdf import PdfReader

from .commons import BaseScraper, ScrapeResult

logger = logging.getLogger(__name__)


class PdfScraper(BaseScraper):
    """Scraper for PDF files"""

    async def _scrape_url(self, url: str) -> ScrapeResult:
        """Download and extract text from a PDF URL"""
        logger.info(f"Attempting to scrape PDF from URL: {url}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Check if the content type is PDF
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                # Log a warning if the content type is not PDF, but still try parsing
                logger.warning(
                    f"URL {url} did not return PDF content-type, got {content_type}. Attempting parse anyway."
                )
                # Or raise an error if we want to be strict:
                # raise ValueError(f"Expected PDF content-type, but got {content_type}")

            pdf_content = io.BytesIO(response.content)
            reader = PdfReader(pdf_content)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() or ""

            # Extract metadata if available (often limited in PDFs)
            metadata = reader.metadata
            title = (
                metadata.get("/Title", url.split("/")[-1])
                if metadata
                else url.split("/")[-1]
            )
            author = metadata.get("/Author") if metadata else None
            # Published date and website name are typically not standard PDF metadata
            published_date = None
            website_name = url

            if not text_content:
                logger.warning(f"Could not extract text from PDF: {url}")

            return ScrapeResult(
                url=url,
                title=str(title),
                content=text_content,
                author=str(author) or None,
                published_date=published_date,
                website_name=website_name,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error downloading PDF from {url}: {e}")
            raise  # Re-raise the exception to be caught by the base class
        except Exception as e:
            logger.error(f"Error parsing PDF from {url}: {e}")
            raise  # Re-raise the exception
