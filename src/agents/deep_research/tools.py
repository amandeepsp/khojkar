import diskcache
import requests

from agents.deep_research.models import Notes
from core.cached_tool import CachedTool
from core.tool import FunctionTool, Toolbox
from memory.vector_store import VectorStoreMemory
from scraping.universal_scraper import UniversalScraper
from search.arxiv import ArxivSearchEngine
from search.cse_scraper import GoogleProgrammableScrapingSearchEngine
from search.duckduckgo import DuckDuckGoSearchEngine
from search.fallback import FallbackSearchEngine
from search.google import GoogleProgrammableSearchEngine, ProgrammableSearchNoResults

tool_cache = diskcache.Cache(".cache")
vector_store = VectorStoreMemory(db_path=".memory/research", collection_name="research")

google_search = GoogleProgrammableSearchEngine(num_results=10)
google_scraping_search = GoogleProgrammableScrapingSearchEngine(
    num_results=10, slow_mo=100
)
duckduckgo_search = DuckDuckGoSearchEngine()

search = FallbackSearchEngine(
    primary_engine=google_search,
    fallback_engine=duckduckgo_search,
    error_conditions=[requests.HTTPError, ProgrammableSearchNoResults],
)

arxiv_search = ArxivSearchEngine(num_results=10)
scraper = UniversalScraper(memory=vector_store)


# Search and scraping tools
google_search_tool = CachedTool(
    FunctionTool(
        name="google_search",
        func=search.search_and_stitch,
        description="Search the web for general information. Useful for getting a broad overview of a topic.",
    ),
    cache=tool_cache,
)

arxiv_search_tool = CachedTool(
    FunctionTool(
        name="arxiv_search",
        func=arxiv_search.search_and_stitch,
        description="Search Arxiv for academic papers, research papers, and other scholarly articles. Useful for more technical and academic topics.",
    ),
    cache=tool_cache,
)

duckduckgo_search_tool = CachedTool(
    FunctionTool(
        name="duckduckgo_search",
        func=duckduckgo_search.search_and_stitch,
        description="Search DuckDuckGo for general information. Useful for getting a broad overview of a topic.",
    ),
    cache=tool_cache,
)

web_scrape_tool = FunctionTool(
    name="scrape_url",
    func=scraper.scrape_url,
    description="Scrape a specific URL for information. Useful for getting detailed information from a specific website or PDF.",
)

search_scrape_tools = Toolbox(
    google_search_tool,
    arxiv_search_tool,
    web_scrape_tool,
    duckduckgo_search_tool,
)

# Vector store tools
add_vector_store_tool = FunctionTool(
    name="add_to_memory",
    func=vector_store.add,
    description="Add a document to the memory for semantic retrieval",
)

query_vector_store_tool = FunctionTool(
    name="query_memory",
    func=vector_store.query,
    description="Query the memory for relevant documents given a query_text",
)

vector_store_tools = Toolbox(add_vector_store_tool, query_vector_store_tool)

notes = Notes()

add_note_tool = FunctionTool(
    name="add_note",
    func=notes.add,
    description="Add a structured note with title, content, and citations",
)

get_all_notes_tool = FunctionTool(
    name="get_all_notes",
    func=notes.get_all,
    description="Get all notes",
)

note_tools = Toolbox(add_note_tool, get_all_notes_tool)
