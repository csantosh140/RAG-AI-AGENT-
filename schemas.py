from pydantic import BaseModel
from typing import List, Optional


class ChatMessageCreate(BaseModel):
    message: str

class DocumentResponse(BaseModel):
    filename: str
    total_chunks: int
    doc_id: str

    class Config:
        from_attributes = True
