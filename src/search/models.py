"""Models for search functionality."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A model representing a single search result."""

    url: str = Field(description="URL of the search result")
    title: str = Field(description="Title of the search result")
    description: str = Field(description="Description of the search result")


class SearchResults(BaseModel):
    """A model representing a collection of search results."""

    results: list[SearchResult] = Field(description="List of search results")


class SearchQueries(BaseModel):
    """A model representing a collection of search queries."""

    queries: list[str] = Field(description="List of queries to search for")
