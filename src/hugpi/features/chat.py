import json
from typing import List, Iterator, Dict, Any, Optional, Union, Callable
from pydantic import Field, BaseModel
from phi.model.base import Model
from phi.model.message import Message
from phi.model.response import ModelResponse
from phi.tools.function import FunctionCall
from phi.utils.timer import Timer
from phi.utils.log import logger
from src.hugpi.model.api._client import HUGPIClient
from src.hugpi.model.api.types.tool_param import ToolParam
from src.hugpi.model.api.types._model_types import MODELS_TYPE

class Metrics:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.time_to_first_token = None
        self.response_timer = Timer()

    def log(self):
        logger.debug("**************** METRICS START ****************")
        if self.time_to_first_token is not None:
            logger.debug(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        logger.debug(f"* Time to generate response:   {self.response_timer.elapsed:.4f}s")
        logger.debug(f"* Tokens per second:           {self.output_tokens / self.response_timer.elapsed:.4f} tokens/s")
        logger.debug(f"* Input tokens:                {self.input_tokens}")
        logger.debug(f"* Output tokens:               {self.output_tokens}")
        logger.debug(f"* Total tokens:                {self.total_tokens}")
        logger.debug("**************** METRICS END ******************")


class HugpiChat(Model):
    client: HUGPIClient = Field(default_factory=HUGPIClient)
    conversation: bool = False
    tools: List[ToolParam] = []
    debug: bool = False

    def __init__(
            self,
            model: MODELS_TYPE = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
            name: str = "HugpiChat",
            provider: str = "HUGPI",
            conversation: bool = False,
            api_key: str = 'backupsanthosh1@gmail.com_SK99@pass',
            cookie_dir_path: str = "./cookies/",
            save_cookies: bool = True,
            **kwargs
    ):
        super().__init__(model=model, name=name, provider=provider, **kwargs)
        self.conversation = conversation
        _hf_email, _hf_password = api_key.split("@gmail.com_")
        self.client = HUGPIClient(
            model=model,
            api_key=api_key,
            cookie_dir_path=cookie_dir_path,
            save_cookies=save_cookies
        )

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        kwargs = {"conversation": self.conversation,"debug":self.debug}
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _create_assistant_message(self, response: Any, metrics: Metrics) -> Message:
        if response.content[0]["type"]=='text':
            content = response.content[0]["text"]
        elif response.content[0]["type"]=='tool_result':
            content = response.content[0]["content"]
        elif response.content[0]["type"]=="tool_use":
            content = json.dumps(response,indent=2)
        assistant_message = Message(role="assistant", content=content)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            assistant_message.tool_calls = response.tool_calls
        self._update_usage_metrics(assistant_message, metrics)
        return assistant_message

    def _update_usage_metrics(self, assistant_message: Message, metrics: Metrics):
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        if metrics.time_to_first_token is not None:
            assistant_message.metrics["time_to_first_token"] = metrics.time_to_first_token
            self.metrics.setdefault("time_to_first_token", []).append(metrics.time_to_first_token)
        assistant_message.metrics["input_tokens"] = metrics.input_tokens
        assistant_message.metrics["output_tokens"] = metrics.output_tokens
        assistant_message.metrics["total_tokens"] = metrics.total_tokens
        self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + metrics.input_tokens
        self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + metrics.output_tokens
        self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens

    def _handle_tool_calls(self, assistant_message: Message, messages: List[Message], model_response: ModelResponse) -> \
    Optional[ModelResponse]:
        if assistant_message.tool_calls and self.run_tools:
            model_response.content = assistant_message.get_content_string() + "\n\n"
            function_calls_to_run = self._get_function_calls_to_run(assistant_message, messages)
            function_call_results: List[Message] = []

            if self.show_tool_calls:
                model_response.content += "Running:\n"
                for f in function_calls_to_run:
                    model_response.content += f" - {f.get_call_str()}\n"
                model_response.content += "\n"

            for _ in self.run_function_calls(function_calls=function_calls_to_run,
                                             function_call_results=function_call_results):
                pass

            self._format_function_call_results(function_call_results, messages)
            return model_response
        return None

    def invoke(self, messages: List[Message]) -> Any:
        hugpi_messages = self._format_messages(messages)
        response = self.client.messages.create(
            messages=hugpi_messages,
            **self.request_kwargs
        )
        return response

    async def ainvoke(self, messages: List[Message]) -> Any:
        return self.invoke(messages)  # HUGPIClient doesn't have native async support

    def invoke_stream(self, messages: List[Message]) -> Iterator[Any]:
        hugpi_messages = self._format_messages(messages)
        return self.client.messages.create(
            messages=hugpi_messages,
            stream=True,
            **self.request_kwargs
        )

    async def ainvoke_stream(self, messages: List[Message]) -> Any:
        yield await self.ainvoke(messages)

    def response(self, messages: List[Message]) -> ModelResponse:
        logger.debug("---------- HugpiChat Response Start ----------")
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()

        metrics.response_timer.start()
        response = self.invoke(messages)
        metrics.response_timer.stop()

        assistant_message = self._create_assistant_message(response, metrics)
        messages.append(assistant_message)

        assistant_message.log()
        metrics.log()

        if self._handle_tool_calls(assistant_message, messages, model_response):
            response_after_tool_calls = self.response(messages)
            if response_after_tool_calls.content:
                model_response.content = (model_response.content or "") + response_after_tool_calls.content
            return model_response

        model_response.content = assistant_message.get_content_string()

        logger.debug("---------- HugpiChat Response End ----------")
        return model_response

    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        return self.response(messages)  # HUGPIClient doesn't have native async support

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        logger.debug("---------- HugpiChat Response Stream Start ----------")
        self._log_messages(messages)
        metrics = Metrics()

        metrics.response_timer.start()
        for response in self.invoke_stream(messages):
            metrics.output_tokens += 1
            if metrics.output_tokens == 1:
                metrics.time_to_first_token = metrics.response_timer.elapsed

            yield ModelResponse(content=response.content[0]["text"])

            if hasattr(response, 'tool_calls') and response.tool_calls:
                assistant_message = self._create_assistant_message(response, metrics)
                messages.append(assistant_message)
                yield from self._handle_stream_tool_calls(assistant_message, messages)
                yield from self.response_stream(messages)

        metrics.response_timer.stop()
        logger.debug("---------- HugpiChat Response Stream End ----------")

    async def aresponse_stream(self, messages: List[Message]) -> Any:
        async for response in self.ainvoke_stream(messages):
            yield ModelResponse(content=response.content[0]["text"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.id,
            "provider": self.provider,
            "conversation": self.conversation,
        }

    def add_tool(self, tool: Union[ToolParam, Dict[str, Any]]) -> None:
        if isinstance(tool, dict):
            self.tools.append(ToolParam(**tool))
        else:
            self.tools.append(tool)

    def get_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        if not self.tools:
            return None
        return [tool.dict() for tool in self.tools]

    def clear(self) -> None:
        self.metrics = {}
        self.function_call_stack = None

    def get_system_message_for_model(self) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.instructions

    def _get_function_calls_to_run(self, assistant_message: Message, messages: List[Message]) -> List[FunctionCall]:
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_call = self._get_function_call_for_tool_call(tool_call)
                if function_call:
                    function_calls_to_run.append(function_call)
                else:
                    messages.append(Message(role="system", content="Could not find function to call."))
        return function_calls_to_run

    def _get_function_call_for_tool_call(self, tool_call: Dict[str, Any]) -> Optional[FunctionCall]:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})

        for tool in self.tools:
            if tool.name == tool_name:
                return FunctionCall(name=tool_name, arguments=tool_args)
        return None

    def _format_function_call_results(self, function_call_results: List[Message], messages: List[Message]) -> None:
        messages.extend(function_call_results)

    def _handle_stream_tool_calls(self, assistant_message: Message, messages: List[Message]) -> Iterator[ModelResponse]:
        if assistant_message.tool_calls and self.run_tools:
            yield ModelResponse(content="\n\n")
            function_calls_to_run = self._get_function_calls_to_run(assistant_message, messages)
            function_call_results: List[Message] = []

            if self.show_tool_calls:
                yield ModelResponse(content="Running:\n")
                for f in function_calls_to_run:
                    yield ModelResponse(content=f" - {f.get_call_str()}\n")
                yield ModelResponse(content="\n")

            for intermediate_model_response in self.run_function_calls(
                    function_calls=function_calls_to_run, function_call_results=function_call_results
            ):
                yield intermediate_model_response

            self._format_function_call_results(function_call_results, messages)

    def run_function_calls(self, function_calls: List[FunctionCall], function_call_results: List[Message]) -> Iterator[
        ModelResponse]:
        for function_call in function_calls:
            tool = next((t for t in self.tools if t.name == function_call.name), None)
            if tool and callable(tool.function):
                try:
                    result = tool.function(**function_call.arguments)
                    function_call_results.append(Message(role="function", content=str(result), name=function_call.name))
                    yield ModelResponse(content=f"Function {function_call.name} executed successfully.")
                except Exception as e:
                    error_message = f"Error executing function {function_call.name}: {str(e)}"
                    function_call_results.append(
                        Message(role="function", content=error_message, name=function_call.name))
                    yield ModelResponse(content=error_message)
            else:
                error_message = f"Function {function_call.name} not found or not callable."
                function_call_results.append(Message(role="function", content=error_message, name=function_call.name))
                yield ModelResponse(content=error_message)