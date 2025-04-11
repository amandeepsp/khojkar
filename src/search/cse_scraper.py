"""Google Custom Search Engine scraper implementation."""

import asyncio
import logging
import os
import random
from contextlib import asynccontextmanager
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

# Common user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
]

# Common screen resolutions
VIEWPORT_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 2560, "height": 1440},
    {"width": 1280, "height": 720},
]

# Common locales and timezones
LOCALES = ["en-US", "en-GB", "en-CA"]
TIMEZONES = ["America/New_York", "Europe/London", "Asia/Tokyo"]


class GoogleProgrammableScrapingSearchEngine(SearchEngine):
    """Google Programmable Search Engine scraper implementation.

    Uses Playwright to scrape Google Programmable Search Engine results directly from the web interface.
    Designed to avoid rate limits and API quotas.
    """

    def __init__(self, num_results: int = 10, link_site: str | None = None) -> None:
        """Initialize the Google Programmable Search Engine Scraper.

        Args:
            num_results: Number of results to return
            link_site: Optional site to filter results by
        """
        self.search_engine_id = os.environ["SEARCH_ENGINE_ID"]
        self.cse_url = "https://cse.google.com/cse"
        self.num_results = num_results
        self.link_site = link_site

        logger.info(
            f"Initialized GoogleProgrammableScrapingSearchEngine with num_results={num_results}"
        )
        if link_site:
            logger.info(f"Filtering search results to site: {link_site}")

    async def _random_sleep(self, min_seconds=1.0, max_seconds=4.0):
        """Sleep for a random amount of time to simulate human behavior."""
        await asyncio.sleep(random.uniform(min_seconds, max_seconds))

    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the predefined list."""
        return random.choice(USER_AGENTS)

    def _get_random_viewport(self) -> dict:
        """Get a random viewport size from the predefined list."""
        return random.choice(VIEWPORT_SIZES)

    def _get_random_locale(self) -> str:
        """Get a random locale from the predefined list."""
        return random.choice(LOCALES)

    def _get_random_timezone(self) -> str:
        """Get a random timezone from the predefined list."""
        return random.choice(TIMEZONES)

    async def _setup_playwright_context(
        self, playwright: Playwright
    ) -> tuple[Browser, BrowserContext]:
        """Set up and configure the Playwright browser and context.

        Returns:
            Tuple of (browser, context)
        """
        # Select random user agent and viewport
        user_agent = self._get_random_user_agent()
        viewport = self._get_random_viewport()

        # Configure browser to appear more human-like
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                f"--user-agent={user_agent}",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # Create context with specific viewport and additional settings
        context = await browser.new_context(
            viewport=viewport,  # type: ignore
            user_agent=user_agent,
            locale=self._get_random_locale(),
            timezone_id=self._get_random_timezone(),
            has_touch=random.choice([True, False]),
            is_mobile=random.choice([
                True,
                False,
                False,
                False,
            ]),  # Less likely to be mobile
        )

        # Add random cookies
        await context.add_cookies([
            {
                "name": "session_pref",
                "value": f"lang=en-{random.choice(['US', 'GB', 'CA'])}",
                "url": "https://cse.google.com",
            }
        ])

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
        """Create and configure a new page with anti-bot detection measures.

        Returns:
            Configured Playwright Page
        """
        page = await context.new_page()

        # Modify JS properties to avoid detection
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
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
        """Navigate to search results with human-like behavior."""
        # Initial navigation to Google
        await page.goto("https://www.google.com")
        await self._random_sleep(2.0, 5.0)

        # Navigate to the search URL
        await page.goto(search_url)

    async def _simulate_human_behavior(self, page: Page):
        """Perform random actions to simulate human-like browsing behavior."""
        # Random scroll before results load
        await page.mouse.move(random.randint(100, 700), random.randint(100, 300))

        # Random scrolling to simulate reading
        for _ in range(random.randint(1, 3)):
            await page.mouse.wheel(0, random.randint(100, 300))
            await self._random_sleep(0.5, 2.0)

        # More random scrolling after results appear
        for _ in range(random.randint(2, 4)):
            await page.mouse.wheel(0, random.randint(200, 600))
            await self._random_sleep(1.0, 3.0)

        # Final random scrolling
        if random.random() < 0.7:  # 70% chance
            for _ in range(random.randint(1, 3)):
                await page.mouse.wheel(0, random.randint(-200, 500))
                await self._random_sleep(0.3, 1.5)

    async def _extract_search_results(self, page: Page) -> list[SearchResult]:
        """Extract search results from the page.

        Returns:
            List of SearchResult objects
        """
        # Wait for search results with jitter in timeout
        await page.wait_for_selector(
            ".gsc-webResult.gsc-result", timeout=random.randint(8000, 15000)
        )

        # Extract search results
        result_elements = await page.query_selector_all(".gsc-webResult.gsc-result")

        scraped_results = []
        seen_urls = set()

        # Limit to requested number of results
        result_elements = result_elements[: self.num_results]

        for i, element in enumerate(result_elements):
            # Add jitter between processing elements
            if i > 0 and random.random() < 0.3:  # 30% chance
                await self._random_sleep(0.2, 1.0)

            # Hover over element to simulate user interest
            element_box = await element.bounding_box()
            if element_box:
                await page.mouse.move(
                    element_box["x"]
                    + random.randint(10, int(element_box["width"] - 10)),
                    element_box["y"]
                    + random.randint(10, int(element_box["height"] - 10)),
                )

            # Extract title
            title_element = await element.query_selector(".gs-title")
            title = await title_element.text_content() if title_element else "No title"
            title = title.strip() if title else "No title"

            # Extract URL
            url_element = await element.query_selector(".gs-title a")
            url = await url_element.get_attribute("href") if url_element else None
            url = url.strip() if url else "No URL"

            # Extract description
            desc_element = await element.query_selector(".gs-snippet")
            description = (
                await desc_element.text_content() if desc_element else "No description"
            )
            description = description.strip() if description else "No description"
            if url and url not in seen_urls:
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
                    logger.info("Navigating to CSE URL with randomized browser profile")

                    # Navigate to search results
                    await self._navigate_to_results(page, search_url)

                    # Simulate human behavior
                    await self._simulate_human_behavior(page)

                    # Extract and return results
                    return await self._extract_search_results(page)
        except Exception as e:
            logger.error(f"Error scraping CSE results: {str(e)}")
            return []

    @override
    async def search(self, query: str) -> SearchResults:
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
