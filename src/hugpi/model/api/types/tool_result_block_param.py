# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable
from typing_extensions import Literal, Required, TypeAlias, TypedDict

from .text_block_param import TextBlockParam

__all__ = ["ToolResultBlockParam", "Content"]

Content: TypeAlias = Union[TextBlockParam]


class ToolResultBlockParam(TypedDict, total=False):
    tool_use_id: Required[str]

    type: Required[Literal["tool_result"]]

    content: Union[str, Iterable[Content]]

    is_error: bool
