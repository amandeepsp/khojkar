from typing import override
import arxiv

from search.commons import SearchEngine
from search.models import SearchResult, SearchResults


class ArxivSearchEngine(SearchEngine):
    def __init__(self, num_results: int = 10):
        self.client = arxiv.Client()
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
            search_results.append(
                SearchResult(
                    title=result.title,
                    url=f"https://arxiv.org/html/{result.entry_id}",
                    description=result.summary,
                )
            )

        return SearchResults(results=search_results)
