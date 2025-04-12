from trafilatura import extract, extract_metadata, fetch_url

from scraping.commons import BaseScraper, ScrapeResult


class TrafilaturaScraper(BaseScraper):
    """Scraper that uses trafilatura to scrape the website"""

    async def _scrape_url(self, url: str) -> ScrapeResult:
        response = fetch_url(url)
        if not response:
            raise ValueError(f"Failed to fetch URL: {url}")

        content = extract(response)

        if not content:
            raise ValueError(f"No content found for URL: {url}")

        metadata = extract_metadata(response)
        return ScrapeResult(
            url=url,
            title=metadata.title or "Could not find title",
            content=content,
            author=metadata.author,
            published_date=metadata.date,
            website_name=metadata.sitename,
        )
