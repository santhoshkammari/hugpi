import uuid
import json
import re
from typing import Union, Dict, List, Generator, Iterable

from langchain_core.utils import print_text

from .._tool_calling import ToolPrepare
from ..types._message import Message
from ..types._model_types import MODELS_TYPE
from ..types.tool_param import ToolParam
from ..types.usage import Usage
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level='DEBUG')



class Messages:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = ""
        self.tools = None



    def create(
            self,
            *,
            max_tokens: int | None = None,
            model: Union[MODELS_TYPE, None] = None,
            prompt: str | None = None,
            messages: List[Dict[str, Union[str, int, float]]] | None = None,
            tools: Iterable[ToolParam] | None = None,
            stream: bool = False,
            conversation: bool = False,
            **kwargs
    ) -> Message | Generator:
        if tools:
            self.tools = tools
            self.tool_prompt = ToolPrepare._transformers_prepare_tool_prompt(tools)
        if kwargs.get("debug",False):
            logger.debug(f"System Prompt: {self.system_prompt}")
            logger.debug(f"User Prompt: {prompt or messages}")
            logger.debug(f"Tools: {self.tools}")
            logger.debug(f"Max Tokens: {max_tokens}")
            logger.debug(f"Model: {model}")
            logger.debug(f"Stream: {stream}")
            logger.debug(f"Conversation: {conversation}")
            logger.debug(f"kwargs: {kwargs}")

        if stream:
            return self.stream(model=model, messages=prompt or messages, conversation=conversation,
                               max_tokens=max_tokens, **kwargs)
        else:
            return self.invoke(model=model, messages=messages or prompt, conversation=conversation,
                               max_tokens=max_tokens, **kwargs)

    def _get_sys_and_user_prompt(self, messages: Union[List[Dict], str]):
        if isinstance(messages, str):
            return self.system_prompt+self.tool_prompt if self.tools else "", messages
        sp, up = self.system_prompt+self.tool_prompt if self.tools else "", ""
        for m in messages:
            if m['role'] == "system":
                sp += m['content']
            elif m['role'] == "user":
                up = m['content']
        return sp, up

    def _parse_tool_use(self, content: str) -> Dict[str, Union[str, Dict]]:
        pattern = r"<tool_call>(.*)</tool_call>"
        match = re.findall(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match[0].replace("\n", ""))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in tool_call: {match}")
                return {}
        return {}

    def _execute_tool(self, tool_call: Dict[str, Union[str, Dict]]) -> str:
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        result = None
        for tool in self.tools:
            if callable(tool) and tool.__name__ == tool_name:
                try:
                    result = tool(**arguments)
                except Exception as e:
                    result = f"Error {e} for {tool_name} with args: {arguments}"
        return result

    def invoke(self, messages: Union[List, str], model: str = None,
               max_tokens: int | None = None,
               conversation: Union[bool, None] = None,
               **kwargs):
        response: str = ""
        for value in self.stream(model=model,messages=messages,
                             conversation=conversation,max_tokens=max_tokens,
                             **kwargs):
            response += value.content[0]["text"]

        if kwargs.get("debug", False):
            logger.debug('########################')
            logger.debug("### Invoke Response ###")
            logger.debug(response)
            logger.debug('########################')

        tool_call = self._parse_tool_use(response)

        if kwargs.get("debug", False):
            logger.debug("### tool_call Parsed ###")
            logger.debug(json.dumps(tool_call, indent=4))
            logger.debug('################')

        usage = Usage(input_tokens=10, output_tokens=10)

        if tool_call:
            results = self._execute_tool(tool_call)
            if results and tool_call.get("name") and tool_call.get("arguments"):
                return  Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": f"tool_{str(uuid.uuid4)}",
                            "content":results
                        }
                    ],
                    type='message',
                    model=model if model else 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                    conversation=conversation,
                    usage = usage,
                    **kwargs
                )
            else:
                return Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=[
                        {
                            "type": "tool_use",
                            "id": f"tool_{str(uuid.uuid4)}",
                            "name":tool_call.get("name",""),
                            "input":tool_call.get("arguments",{})
                        }
                    ],
                    type='message',
                    model=model if model else 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                    conversation=conversation,
                    usage=usage,
                    **kwargs
                )
        else:
            return  Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=[
                    {
                        "type": "text",
                        "text": response
                    }
                ],
                type='message',
                model=model if model else 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                conversation=conversation,
                usage=usage,

                **kwargs
            )

    def stream(
            self,
            messages: Union[List, str],
            model: str = None,
            max_tokens: int | None = None,
            conversation: Union[bool, None] = None,
            **kwargs):
        user_prompt = self._update_dependencies(model_name=model,
                                                messages=messages,
                                                conversation=conversation)
        res = self.llm.chat(user_prompt, stream=True, **kwargs)
        for token in self._stream_with_chat(res, max_tokens):
            if token == '<MAX_TOKEN_REACHED>':
                break
            usage = Usage(input_tokens=10, output_tokens=10)
            yield Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=[{"type": "text", "text": token}],
                type='message',
                model=model if model else 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                conversation=conversation,
                usage=usage,
                **kwargs
            )



    def _update_dependencies(self, model_name, messages, conversation):
        conversation = False if conversation is None else conversation
        curr_sys_prompt, user_prompt = self._get_sys_and_user_prompt(messages)

        if not conversation:
            if curr_sys_prompt and (curr_sys_prompt != self.system_prompt):
                self.system_prompt = curr_sys_prompt + self.tool_prompt if self.tools else ""
            self.llm.new_conversation(modelIndex=self.llm.MODELS.index(model_name) if model_name else 0,
                                      system_prompt=self.system_prompt,
                                      switch_to=True)
        return user_prompt

    def _stream_with_chat(self, res, max_tokens):
        token_count = 0
        for x in res:
            if x and isinstance(x, dict):
                res = x.get('token', "")
                if max_tokens and token_count >= max_tokens:
                    yield "<MAX_TOKEN_REACHED>"
                token_count += 1
                yield res
            else:
                yield "<MAX_TOKEN_REACHED>"