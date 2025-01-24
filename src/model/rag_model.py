from pydantic import BaseModel
from typing import Optional


class RAGSearchModel(BaseModel):
    query: str
    top_k: Optional[int] = None
