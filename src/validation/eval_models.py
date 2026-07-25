from __future__ import annotations

from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic_evals import Case, Dataset


class TextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[list[str], Field(min_length=1)]


class NumberOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float


class TableOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table_path: str
    ordered: bool


class ExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str


ExpectedItem = TextOutput | NumberOutput | TableOutput | ExplanationOutput
ExpectedOutput = Annotated[list[ExpectedItem], Field(min_length=1)]
BenchmarkOutput = str | ExpectedOutput


class TaskCategory(str, Enum):
    DATA_DISCOVERY = "data discovery"
    DATA_TRANSFORMATIONS = "data transformations"
    SEMANTIC_INTEGRATION = "semantic integration"
    DOMAIN_KNOWLEDGE = "domain knowledge"


class CaseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: UUID
    query_code: str
    category: TaskCategory


BenchmarkCase = Case[str, BenchmarkOutput, CaseMetadata]
BenchmarkDataset = Dataset[str, BenchmarkOutput, CaseMetadata]
