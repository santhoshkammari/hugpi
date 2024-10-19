from typing import List


from ._base_login import HFCredentialManager
from ..utils.const import AVAILABLE_MODEL_LIST
from ._hf_chatbot_custom import HFChatBot

class BaseHuggyLLM(HFCredentialManager, HFChatBot):
    MODELS: List[str] = AVAILABLE_MODEL_LIST

    def __new__(cls, hf_email='backupsanthosh1@gmail.com', hf_password='SK99@pass', cookie_dir_path="./cookies/", save_cookies=True,
                system_prompt:str = "",default_llm:int = 3):
        instance = super().__new__(cls)
        instance.__init__(hf_email, hf_password, cookie_dir_path, save_cookies)
        instance.model = cls.MODELS[default_llm]
        return HFChatBot(default_llm=default_llm,system_prompt=system_prompt,cookies=instance.cookies.get_dict())
