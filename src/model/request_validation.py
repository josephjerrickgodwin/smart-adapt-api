from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class ModelNames(str, Enum):
    """Enum for supported model names."""
    LLAMA_3_2 = "meta-llama/Llama-3.2-1B-instruct"


class DatasetInsertRequest(BaseModel):
    """Request model for dataset insertion."""
    dataset: List[int]


class LoRAHyperparametersRequest(BaseModel):
    """Request model for calculating LoRA hyperparameters."""
    model_size: int
    dataset_size: int


class MessageModel(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class InferenceRequest(BaseModel):
    """Request model for model inference."""
    user_id: str
    history: List[MessageModel]


class FineTuningRequest(BaseModel):
    """Request model for fine-tuning process."""
    dataset: List[int]
    num_epochs: Optional[int] = Field(default=10, ge=1, le=50)
    learning_rate: Optional[float] = Field(default=2e-4, ge=1e-5, le=1e-3)
