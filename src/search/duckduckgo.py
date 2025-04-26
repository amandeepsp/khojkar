from duckduckgo_search import DDGS

from search.commons import SearchEngine
from search.models import SearchResult, SearchResults


class DuckDuckGoSearchEngine(SearchEngine):
    def __init__(self) -> None:
        self.search_engine = DDGS()

    async def _search(self, query: str) -> SearchResults:
        results = []
        with self.search_engine as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append(
                    SearchResult(title=r["title"], url=r["href"], description=r["body"])
                )
        return SearchResults(results=results)
