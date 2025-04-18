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


class Retrievals(BaseModel):
    subtopic: Subtopic = Field(description="Subtopic to research")
    insights: list[str] = Field(
        description="List of insights from the subtopic in bullet points"
    )
    citations: list[str] = Field(
        description="List of citations from the subtopic in APA format like; Doe, J. (2021). Title of the citation. Journal Name, 1(1), 1-10."
    )


class ReflectionInput(BaseModel):
    retrievals: list[Retrievals] = Field(description="List of retrievals to reflect on")


class Reflection(BaseModel):
    gaps: list[str] = Field(description="List of gaps in the retrieval")
    improvements: list[str] = Field(description="List of improvements to the retrieval")


class SynthesisInput(BaseModel):
    reflections: list[Reflection] = Field(
        description="List of reflections to incorporate into the synthesis"
    )


class Synthesis(BaseModel):
    report: str = Field(description="Report of the synthesis")
