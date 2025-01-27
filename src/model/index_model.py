from typing import List

from pydantic import BaseModel


class IndexModel(BaseModel):
    email: str
    data: List[int]
