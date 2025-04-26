import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    title: str = Field(description="Title of the reflection")
    content: str = Field(description="Content of the reflection")
    type: Literal["insight", "gap", "contradiction", "follow_up"] = Field(
        description="Type of the reflection, can be insight, gap, contradiction, or follow_up"
    )


class Reflections(BaseModel):
    reflections: list[Reflection] = Field(
        description="List of reflections to incorporate into the synthesis"
    )


class SynthesisInput(BaseModel):
    subtopics: list[Subtopic] = Field(description="List of subtopics to synthesize")
    reflections: list[Reflection] = Field(
        description="List of reflections to incorporate into the synthesis"
    )


class Section(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section")


class Report(BaseModel):
    report: str = Field(description="Report text, in markdown format")


class Note(BaseModel):
    title: str = Field(description="Title of the note")
    content: str = Field(description="Content of the note")
    citation: Citation = Field(
        description="Citation to include in the note, optional, if no citation, omit the field",
    )


class Notes(BaseModel):
    notes: list[Note] = Field(
        default_factory=list, description="List of notes to include in the report"
    )

    async def add(self, note: Note) -> None:
        """Add a note to the list

        Args:
            note (Note): Note to add
        """
        validated_note = Note.model_validate(note)
        self.notes.append(validated_note)

    async def get_all(self) -> str:
        """Get all notes"""
        markdown = "## Notes\n\n"
        for note in self.notes:
            markdown += f"### {note.title}\n\n{note.content}\n\n"
            if note.citation:
                markdown += (
                    f"**Citation:** {note.citation.model_dump_json(indent=2)}\n\n"
                )
            markdown += "\n\n---\n\n"
        return markdown
