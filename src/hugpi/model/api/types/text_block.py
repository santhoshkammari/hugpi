__all__ = ["TextBlock"]

from typing_extensions import Literal

from pydantic import BaseModel


class TextBlock(BaseModel):
    text: str

    type: Literal["text"]