from typing import Literal, Optional
from pydantic import BaseModel


class DeltaModel(BaseModel):
    content: str
    type: Literal["thinking", "text"] = "thinking"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.type
        }


class ChoicesModel(BaseModel):
    index: int
    delta: DeltaModel

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "delta": self.delta.to_dict()
        }


class LLMResponseModel(BaseModel):
    choices: ChoicesModel
    model: str
    chunk_token_usage: int
    created: int
    message_id: int
    parent_id: int
    time_elapsed: int

    def to_dict(self) -> dict:
        choices = self.choices.to_dict()
        return {
            "choices": choices,
            "model": self.model,
            "chunk_token_usage": self.chunk_token_usage,
            "created": self.created,
            "message_id": self.message_id,
            "parent_id": self.parent_id,
            "time_elapsed": self.time_elapsed
        }
