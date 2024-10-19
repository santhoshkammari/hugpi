__all__ = ["Usage"]

from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int
    """The number of input tokens which were used."""

    output_tokens: int
    """The number of output tokens which were used."""