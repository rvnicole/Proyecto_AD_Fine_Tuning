from pydantic import BaseModel

class Prompt(BaseModel):
    message: str
    text_history: str