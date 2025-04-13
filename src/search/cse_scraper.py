"""Google Custom Search Engine scraper implementation."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import override

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from search.commons import SearchEngine
from search.models import SearchResult, SearchResults

logger = logging.getLogger(__name__)

# Standard user agent to use
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"

# Standard viewport size
VIEWPORT = {"width": 1920, "height": 1080}


class GoogleProgrammableScrapingSearchEngine(SearchEngine):
    """Google Programmable Search Engine scraper implementation.

    Uses Playwright to scrape Google Programmable Search Engine results directly from the web interface.
    Designed to avoid rate limits and API quotas.
    """

    def __init__(
        self,
        num_results: int = 10,
        link_site: str | None = None,
        headless: bool = True,
        slow_mo: int = 0,
    ) -> None:
        """Initialize the Google Programmable Search Engine Scraper.

        Args:
            num_results: Number of results to return
            link_site: Optional site to filter results by
            headless: Whether to run in headless mode (True by default)
            slow_mo: Slows down Playwright operations by the specified amount of milliseconds (0 by default)
        """
        self.search_engine_id = os.environ["SEARCH_ENGINE_ID"]
        self.cse_url = "https://cse.google.com/cse"
        self.num_results = num_results
        self.link_site = link_site
        self.headless = headless
        self.slow_mo = slow_mo
        logger.info(
            f"Initialized GoogleProgrammableScrapingSearchEngine with num_results={num_results}, headless={headless}, slow_mo={slow_mo}"
        )
        if link_site:
            logger.info(f"Filtering search results to site: {link_site}")

    def _get_timestamp_filename(self, base_name: str) -> str:
        """Generate a filename with timestamp for screenshots.

        Args:
            base_name: Base name for the screenshot

        Returns:
            Filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f".playwright/{base_name}_{timestamp}.png"

    async def _setup_playwright_context(
        self, playwright: Playwright
    ) -> tuple[Browser, BrowserContext]:
        """Set up and configure the Playwright browser and context.

        Returns:
            Tuple of (browser, context)
        """
        # Configure browser with basic anti-bot detection
        browser = await playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                f"--user-agent={USER_AGENT}",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        # Create context with fixed settings
        context = await browser.new_context(
            viewport=VIEWPORT,  # type: ignore
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
        )

        return browser, context

    @asynccontextmanager
    async def _browser_context_manager(self, playwright: Playwright):
        """Context manager for browser and context setup/teardown."""
        browser, context = await self._setup_playwright_context(playwright)
        try:
            yield context
        finally:
            await self._cleanup_playwright(context, browser)

    async def _setup_page(self, context: BrowserContext) -> Page:
        """Create and configure a new page with basic anti-bot detection measures.

        Returns:
            Configured Playwright Page
        """
        page = await context.new_page()

        # Basic webdriver detection evasion
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
        """)

        return page

    @asynccontextmanager
    async def _page_manager(self, context: BrowserContext):
        """Context manager for page setup."""
        page = await self._setup_page(context)
        try:
            yield page
        finally:
            await page.close()

    async def _navigate_to_results(self, page: Page, search_url: str):
        """Navigate directly to search results."""
        await page.goto(search_url)
        # Wait for document load state
        await page.wait_for_load_state("domcontentloaded")

    async def _extract_search_results(self, page: Page) -> list[SearchResult]:
        """Extract search results from the page.

        Returns:
            List of SearchResult objects
        """
        # Wait for the main results container to be visible
        results_container_selector = ".gsc-results"
        try:
            await page.wait_for_selector(
                results_container_selector,
                state="visible",
                timeout=15 * 1000,
            )
            logger.debug("Results container is visible.")
        except Exception as e:
            logger.error(
                f"Timeout or error waiting for results container '{results_container_selector}': {e}"
            )
            # Capture a screenshot for debugging with timestamp
            await page.screenshot(
                path=self._get_timestamp_filename("debug_container_timeout")
            )
            return []  # Return empty list if container doesn't appear

        result_selector = ".gsc-webResult.gsc-result"
        result_elements = await page.query_selector_all(result_selector)

        if not result_elements:
            logger.warning(
                f"Results container found, but no results matching '{result_selector}' found within it."
            )
            await page.screenshot(path=self._get_timestamp_filename("debug_no_results"))
            return []

        scraped_results = []
        seen_urls = set()

        # Limit to requested number of results
        result_elements = result_elements[: self.num_results]

        for element in result_elements:
            is_visible = await element.is_visible()
            if not is_visible:
                logger.warning("Skipping hidden result element.")
                continue

            title_element = await element.query_selector(".gs-title")
            title = await title_element.text_content() if title_element else "No title"
            title = title.strip() if title else "No title"

            url_element = await element.query_selector(".gs-title a")
            url = await url_element.get_attribute("href") if url_element else None
            url = url.strip() if url else "No URL"

            desc_element = await element.query_selector(".gs-snippet")
            description = (
                await desc_element.text_content() if desc_element else "No description"
            )
            description = description.strip() if description else "No description"

            if url and url != "No URL" and url not in seen_urls:
                seen_urls.add(url)
                scraped_results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        description=description,
                    )
                )

        return scraped_results

    async def _cleanup_playwright(self, context: BrowserContext, browser: Browser):
        """Close Playwright context and browser."""
        await context.close()
        await browser.close()

    async def _run_cse_search(
        self, playwright: Playwright, search_url: str
    ) -> list[SearchResult]:
        """Run the CSE search using Playwright and return search results.

        Args:
            playwright: Playwright instance
            search_url: Full search URL including query

        Returns:
            List of SearchResult objects
        """
        try:
            async with self._browser_context_manager(playwright) as context:
                async with self._page_manager(context) as page:
                    logger.info("Navigating to CSE URL")

                    # Navigate to search results
                    await self._navigate_to_results(page, search_url)

                    # Extract and return results
                    return await self._extract_search_results(page)
        except Exception as e:
            logger.error(f"Error scraping CSE results: {str(e)}")
            return []

    @override
    async def _search(self, query: str) -> SearchResults:
        """Search using Google Programmable Search Engine scraping and return relevant links.

        Args:
            query: The search query

        Returns:
            A SearchResults object containing search results
        """
        logger.info(f"Scraping Programmable Search Engine for query: {query}")

        # Add site restriction if specified
        if self.link_site:
            query = f"site:{self.link_site} {query}"

        full_cse_url = f"{self.cse_url}?cx={self.search_engine_id}&q={query}"

        async with async_playwright() as playwright:
            results = await self._run_cse_search(playwright, full_cse_url)

        logger.info(f"Scraped {len(results)} results from CSE for query: {query}")
        return SearchResults(results=results)
