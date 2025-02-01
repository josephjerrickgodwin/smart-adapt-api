from typing import List

from pydantic import BaseModel


class IndexModel(BaseModel):
    user_id: str
    data: List[int]
