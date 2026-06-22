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

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
