from typing import Optional

from pydantic import BaseModel, Field


class Subtopics(BaseModel):
    subtopics: list[str] = Field(description="List of subtopics to research")


class Questions(BaseModel):
    subtopic: str = Field(description="Subtopic to research")
    questions: list[str] = Field(
        description="List of questions to generated from the subtopic"
    )


class Citation(BaseModel):
    title: str = Field(description="Title of the citation")
    author: Optional[str] = Field(description="Author of the citation")
    published_date: Optional[str] = Field(description="Published date of the citation")
    website: Optional[str] = Field(description="Website of the citation")
    url: str = Field(description="URL of the citation")


class Retrieval(BaseModel):
    summary: str = Field(description="Summary of the retrieval")
    citations: list[Citation] = Field(
        description="List of citations from the retrieval"
    )
    memory_id: Optional[str] = Field(description="ID of the memory")


class Retrievals(BaseModel):
    question: str = Field(description="Question to research")
    retrievals: list[Retrieval] = Field(
        description="List of retrievals from the question"
    )


class Reflection(BaseModel):
    question: str = Field(description="Question to research")
    gaps: list[str] = Field(description="List of gaps in the retrieval")
    improvements: list[str] = Field(description="List of improvements to the retrieval")
