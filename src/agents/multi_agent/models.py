from typing import Optional

from pydantic import BaseModel, Field


class PlannerInput(BaseModel):
    topic: str = Field(description="Topic to research")


class Subtopic(BaseModel):
    title: str = Field(description="Title of the subtopic")
    description: str = Field(description="Description of the subtopic")


class Plan(BaseModel):
    subtopics: list[Subtopic] = Field(description="List of subtopics to research")


class RetrievalInput(BaseModel):
    subtopic: Subtopic = Field(description="Subtopic to research")


class Citation(BaseModel):
    subtopic: str = Field(description="Subtopic of the citation")
    title: str = Field(description="Title of the citation")
    url: str = Field(description="URL of the citation")
    author: Optional[str] = Field(description="Author of the citation, None if unknown")
    published_date: Optional[str] = Field(
        description="Published date of the citation, None if unknown"
    )
    website: Optional[str] = Field(
        description="Website of the citation, None if unknown"
    )


class ReflectionInput(BaseModel):
    subtopics: list[Subtopic] = Field(description="List of subtopics to reflect on")


class Reflection(BaseModel):
    subtopic: Subtopic = Field(description="Subtopic to reflect on")
    gaps: Optional[list[str]] = Field(
        description="List of gaps in the retrieval, None if no gaps"
    )
    contradictions: Optional[list[str]] = Field(
        description="List of contradictions in the retrieval, None if no contradictions"
    )
    follow_ups: Optional[list[str]] = Field(
        description="List of follow-ups to the reflection, None if no follow-ups"
    )


class Reflections(BaseModel):
    reflections: list[Reflection] = Field(
        description="List of reflections to incorporate into the synthesis"
    )


class SynthesisInput(BaseModel):
    subtopics: list[Subtopic] = Field(description="List of subtopics to synthesize")


class Synthesis(BaseModel):
    report: str = Field(description="Report of the synthesis")
