import json
from typing import Callable

from typing_extensions import Iterable

from transformers.utils import get_json_schema

from .types.tool_param import ToolParam


class ToolPrepare:
    @staticmethod
    def _prepare_tool_prompt(tools: Iterable[ToolParam]) -> str:
        tools_instructions = ""
        tools_list = []
        for tool in tools:
            tools_list.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
            tools_instructions += (
                f"{tool['name']}: Call this tool to interact with the {tool['name']} API. "
                f"What is the {tool['name']} API useful for? {tool['description']}. "
                f"Parameters: {tool['parameters']}\n"
                f"Required parameters: {tool['parameters'].get('required', [])}\n\n"
            )

        TOOL_EXAMPLE = """You will receive a JSON string containing a list of callable tools. Please parse
    this JSON string and return a JSON object containing the tool name and tool parameters. Here
    is an example of the tool list:

    {"tools": [{"name": "plus_one", "description": "Add one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}},{"name": "minus_one", "description": "Minus one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}}]}

    Based on this tool list, generate a JSON object to call a tool. For example, if you need to add one to number 77, return:

    {"tool": "plus_one", "parameters": {"number": "77"}}

    Please note that the above is just an example and does not mean that the plus_one and minus_one tools are currently available."""

        RETURN_FORMAT = '{"tool": "tool name", "parameters": {"parameter name": "parameter value"}}'

        INSTRUCTION = f"""
    {TOOL_EXAMPLE}
    Answer the following questions as best you can. You have access to the following APIs:
    {tools_instructions}
    Use the following format:
    '''tool_json
    {RETURN_FORMAT}
    '''
    Please choose the appropriate tool according to the user's question. If you don't need to call it,
    please reply directly to the user's question. When the user communicates with you in a
    language other than English, you need to communicate with the user in the same language.
    When you have enough information from the tool results, respond directly to the user with a text
    message without having to call the tool again.
    """
        return INSTRUCTION

    @staticmethod
    def _default_tool_schema_func(tools):
        available_tools = ""
        for tool in tools:
            available_tools += "\n"
            if isinstance(tool, dict):
                available_tools += json.dumps({"type": "function", "function": tool})
            elif callable(tool):
                available_tools += json.dumps(get_json_schema(tool))
        return available_tools

    @staticmethod
    def _transformers_prepare_tool_prompt(tools: Iterable[ToolParam],
                                          tool_prompt: str | None = None,
                                          tool_schema_func: Callable | None = None,
                                          ) -> str:
        tool_schema_func = ToolPrepare._default_tool_schema_func if tool_schema_func is None else tool_schema_func
        tools = tool_schema_func(tools)

        if tool_prompt is None:
            tool_prompt = """You also have ability of function calling. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. 
    Here are the available tools: <tools> {tools}
    For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {"name": <function-name>, "arguments": <args-dict>}
    </tool_call>
    example:
    <tool_call>{"name": "addition", "arguments":{"x": 1, "y":2}</tool_call>
    <tool_call>{"name": "sum_of_ages", "arguments":{"p1_age":10,"p2_age":20}</tool_call>
            """
        return tool_prompt.replace("{tools}",tools)
