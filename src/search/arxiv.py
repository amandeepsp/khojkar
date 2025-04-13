import logging
from typing import override

import arxiv

from search.commons import SearchEngine
from search.models import SearchResult, SearchResults

logger = logging.getLogger(__name__)


class ArxivSearchEngine(SearchEngine):
    def __init__(self, num_results: int = 10):
        self.client = arxiv.Client(page_size=num_results, delay_seconds=5)
        self.num_results = num_results

    @override
    async def _search(self, query: str) -> SearchResults:
        search_query = arxiv.Search(
            query=query,
            max_results=self.num_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = self.client.results(search_query)

        search_results = []
        for result in results:
            if result.pdf_url is None:
                logger.warning(f"No PDF URL found for result: {result.title}")
                continue

            search_results.append(
                SearchResult(
                    title=result.title,
                    url=result.pdf_url,
                    description=result.summary,
                )
            )

        return SearchResults(results=search_results)
