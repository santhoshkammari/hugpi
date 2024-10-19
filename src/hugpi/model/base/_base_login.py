import os
from hugchat.login import Login

class HFCredentialManager:
    def __init__(self,hf_email=None,hf_password=None,cookie_dir_path = "./cookies/",save_cookies = True):
        self.hf_email = hf_email
        self.hf_password = hf_password
        self.cooke_dir_path = cookie_dir_path
        self.save_cookies = save_cookies
        self.cookies = self._get_cookies()

    def _get_cookies(self):
        sign = Login(self.hf_email, self.hf_password)
        cookies = sign.login(cookie_dir_path=self.cooke_dir_path, save_cookies=self.save_cookies)
        return cookies


