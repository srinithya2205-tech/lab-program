from typing import List

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_LENGTH = 500


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH, description="A question about the portfolio.")

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class AskResponse(BaseModel):
    answer: str
    matched: bool = Field(description="False when the portfolio has nothing relevant to the question.")
    topics: List[str]
    sources: List[str] = Field(description="Portfolio sections the answer was built from.")
    suggestions: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    portfolio_loaded: bool
