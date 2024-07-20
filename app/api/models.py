from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid

class PostPatchRecordSchema(BaseModel):
    brain_id: int
    section_id: int
    x: int
    y: int
    store_path: str


class LabelSchema(BaseModel):
    region_id: int
    area: float
    polygon: List[List[List[int]]] | List[List[List[List[int]]]]

class PostPatchRecordsSchema(BaseModel):
    patches: Dict[str,PostPatchRecordSchema]