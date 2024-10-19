from . import resources
from ._models import HUGPiLLM

class HUGPIClient:
    def __init__(
            self,
            api_key:str = 'backupsanthosh1@gmail.com_SK99@pass',
            cookie_dir_path: str = "./cookies/",
            save_cookies: bool = True
    ):
        _hf_email,_hf_password = api_key.split("@gmail.com_")
        self.llm = HUGPiLLM(
            hf_email=_hf_email+'@gmail.com',
            hf_password=_hf_password,
            cookie_dir_path=cookie_dir_path,
            save_cookies=save_cookies
        )
        self.messages = resources.Messages(self.llm)

