from pydantic import BaseModel, Field
from typing import List, Optional
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


class InferenceRequest(BaseModel):
    """Request model for model inference."""
    query: str = Field(..., description="Input query for model")
    max_new_tokens: Optional[int] = Field(default=512, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    use_lora: Optional[bool] = Field(default=False, ge=True)


class FineTuningRequest(BaseModel):
    """Request model for fine-tuning process."""
    dataset: List[int]
    model_name: ModelNames = ModelNames.LLAMA_3_2
    num_epochs: Optional[int] = Field(default=10, ge=1, le=50)
    learning_rate: Optional[float] = Field(default=2e-4, ge=1e-5, le=1e-3)
