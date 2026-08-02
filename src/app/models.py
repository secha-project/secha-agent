from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import ModelMessage


class TextDisplayPart(BaseModel):
    kind: Literal["text"] = "text"
    content: str


class ToolDisplayPart(BaseModel):
    kind: Literal["tool"] = "tool"
    tool_id: str
    title: str
    args: str
    output: str | None = None
    state: Literal["running", "complete", "error"] = "running"


class ArtifactDisplayPart(BaseModel):
    kind: Literal["artifact"] = "artifact"
    artifact_id: str


DisplayPart = Annotated[
    TextDisplayPart | ToolDisplayPart | ArtifactDisplayPart,
    Field(discriminator="kind"),
]


class UserDisplayMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantDisplayMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    parts: list[DisplayPart] = Field(default_factory=list)


DisplayMessage = Annotated[
    UserDisplayMessage | AssistantDisplayMessage,
    Field(discriminator="role"),
]

DISPLAY_MESSAGES_ADAPTER = TypeAdapter(list[DisplayMessage])


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class User(BaseModel):
    id: str
    username: str
    created_at: str


class Conversation(BaseModel):
    id: str
    user_id: str
    title: str
    display_messages: list[DisplayMessage] = Field(default_factory=list)
    model_messages: list[ModelMessage] = Field(default_factory=list)
    created_at: str
    updated_at: str
