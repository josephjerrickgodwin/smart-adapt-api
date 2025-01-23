from typing import List

from pydantic import BaseModel


class IndexModel(BaseModel):
    data: List[str]
