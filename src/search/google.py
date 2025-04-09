"""Google Programmable Search Engine implementation."""
import os
import logging
import requests

from search.models import SearchResult, SearchResults
from search.commons import _BaseSearchEngine

logger = logging.getLogger(__name__)

class GoogleProgrammableSearchEngine(_BaseSearchEngine):
    """Google Programmable Search Engine implementation.
    
    Uses Google's Custom Search API to perform searches.
    """

    def __init__(self, num_results: int = 10, link_site: str | None = None) -> None:
        """Initialize the Google Programmable Search Engine.
        
        Args:
            num_results: Number of results to return (max 10)
            link_site: Optional site to filter results by

        """
        self.search_engine_id = os.environ["SEARCH_ENGINE_ID"]
        self.search_engine_api_key = os.environ["SEARCH_ENGINE_API_KEY"]
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
        self.num_results = min(num_results, 10) # Limit to 10 results per query
        self.link_site = link_site
        logger.info(f"Initialized GoogleProgrammableSearchEngine with num_results={num_results}")
        if link_site:
            logger.info(f"Filtering search results to site: {link_site}")

    async def search(self, query: str) -> SearchResults:
        """Search using Google Programmable Search API and return relevant links.
        
        Args:
            query: The search query
            
        Returns:
            A SearchResults object containing search results
            
        """
        logger.info(f"Searching for query: {query}")
        params: dict[str, str | int] = {
            "q": query, 
            "cx": self.search_engine_id, 
            "key": self.search_engine_api_key,
            "num": self.num_results  
        }
        
        if self.link_site:
            params["linkSite"] = self.link_site
        
        response = requests.get(self.base_url, params=params)
        response_json = response.json()
        
        results: list[SearchResult] = []

        if "items" not in response_json:
            error_msg = f"No items found in response for query '{query}', response_code={response.status_code}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        items = response_json["items"]
        logger.debug(f"Received {len(items)} raw search results")

        seen_urls = set()
        for item in items:
            url = item["link"]
            if url in seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue
            
            seen_urls.add(url)
            result = SearchResult(
                url=url,
                title=item["title"],
                description=item["snippet"],
            )
            results.append(result)

        logger.info(f"Returning {len(results)} unique search results for query: {query}")
        return SearchResults(results=results)