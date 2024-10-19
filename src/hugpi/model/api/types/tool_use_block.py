__all__ = ["ToolUseBlock"]

from typing_extensions import Literal

from pydantic import BaseModel


class ToolUseBlock(BaseModel):
    id: str

    input: object

    name: str

    type: Literal["tool_use","tool_result"]