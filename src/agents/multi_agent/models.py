from typing import Literal, Optional

from pydantic import BaseModel


class Subtopics(BaseModel):
    subtopics: list[str]


class Questions(BaseModel):
    subtopic: str
    questions: list[str]


class Citation(BaseModel):
    title: str
    author: Optional[str] = None
    published_date: Optional[str] = None
    website: Optional[str] = None
    url: str


class Retrieval(BaseModel):
    summary: str
    full_text: str
    citations: list[Citation]


class Retrievals(BaseModel):
    question: str
    retrievals: list[Retrieval]


class Comparison(BaseModel):
    question: str
    comparison_summary: str
    agreement: bool
    technical_detail: Literal["high", "medium", "low"]
    bias: Literal["high", "medium", "low", "unknown"]
    recency: Literal["high", "medium", "low", "unknown"]
    credibility: Literal["high", "medium", "low", "unknown"]


class Reflection(BaseModel):
    question: str
    gaps: list[str]
    improvements: list[str]
