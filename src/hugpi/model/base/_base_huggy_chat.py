from typing import List, Dict

from ._base_llm import BaseHuggyLLM
from ..utils.const import AVAILABLE_MODEL_LIST

from typing import Literal, Union

_AVAILABLE_MODELS = Literal[ 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'CohereForAI/c4ai-command-r-plus-08-2024',
    'Qwen/Qwen2.5-72B-Instruct',
    'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'NousResearch/Hermes-3-Llama-3.1-8B',
    'mistralai/Mistral-Nemo-Instruct-2407',
    'microsoft/Phi-3.5-mini-instruct']


class HuggyLLM():
    def __init__(
        self,
        model_name:Union[_AVAILABLE_MODELS, None] = None,
        hf_email='backupsanthosh1@gmail.com',
        hf_password='SK99@pass',
        cookie_dir_path="./cookies/",
        save_cookies=True,
        system_prompt:str = "",
        _llm: BaseHuggyLLM|None = None
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = _llm if _llm is not None else BaseHuggyLLM(hf_email=hf_email,
                                                              hf_password=hf_password,
                                                              cookie_dir_path=cookie_dir_path,
                                                              save_cookies=save_cookies,
                                                              system_prompt=system_prompt,
                                                              default_llm=AVAILABLE_MODEL_LIST.index(model_name))
    @property
    def models(self):
        return self.llm.MODELS

    def _get_sys_and_user_prompt(self,messages: List[Dict]|str):
        if isinstance(messages, str):
            return "",messages
        sp ,up = "",""
        for m in messages:
            if m['role'] == "system":
                sp = m['content']
            elif m['role'] == "user":
                up =m['content']
        return sp,up

    def invoke(self,messages: List|str, model_name:_AVAILABLE_MODELS=None,
               conversation: Union[bool, None] = None,
               **kwargs):
        user_prompt = self._update_dependencies(model_name=model_name,
                                                messages=messages,
                                                conversation=conversation)
        res = self.llm.chat(user_prompt,**kwargs)
        res.wait_until_done()
        return res.text

    def stream(
            self,
            messages: List|str,
            model_name:_AVAILABLE_MODELS=None,
            conversation:Union[bool,None] = None,
            **kwargs):
        """new chat """

        user_prompt = self._update_dependencies(model_name=model_name,
                                  messages=messages,
                                  conversation=conversation)
        res = self.llm.chat(user_prompt,stream = True,**kwargs)
        for x in res:
            if x and isinstance(x,dict):
                yield x.get('token',"")

    def pstream(self,messages:List[Dict]|str,model_name:_AVAILABLE_MODELS=None,
                conversation = None,
                **kwargs):
        for _ in self.stream(messages,model_name=model_name,
                             conversation=conversation,
                             **kwargs):
            print(_,end = "",flush=True)

    def _update_dependencies(self,model_name,messages,conversation):
        # Setting Defaults
        conversation = False if conversation is None else conversation

        curr_sys_prompt, user_prompt = self._get_sys_and_user_prompt(messages)

        if not conversation: #Default conversation is False
            if curr_sys_prompt and (curr_sys_prompt!=self.system_prompt):
                 self.system_prompt = curr_sys_prompt
            self.llm.new_conversation(modelIndex=self.llm.MODELS.index(model_name) if model_name else 0,
                                      system_prompt=self.system_prompt,
                                      switch_to=True)
        return user_prompt