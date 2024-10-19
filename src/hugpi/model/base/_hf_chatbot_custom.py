import typing

from hugchat.hugchat import ChatBot
from hugchat.types.message import Conversation
import requests
import json

import logging
import typing
import traceback
from hugchat import exceptions

class HFChatBot(ChatBot):
    def _stream_query(
            self,
            text: str,
            web_search: bool = False,
            is_retry: bool = False,
            retry_count: int = 5,
            conversation: Conversation = None,
            message_id: str = None,
    ) -> typing.Generator[dict, None, None]:
        if conversation is None:
            conversation = self.current_conversation

        if retry_count <= 0:
            raise Exception(
                "the parameter retry_count must be greater than 0.")
        if len(conversation.history) == 0:
            raise Exception(
                "conversation history is empty, but we need the root message id of this conversation to continue.")

        if not message_id:
            # get last message id
            message_id = conversation.history[-1].id

        logging.debug(f'message_id: {message_id}')

        req_json = {
            "id": message_id,
            "inputs": text,
            "is_continue": False,
            "is_retry": is_retry,
            "web_search": web_search,
            "tools": []
        }
        headers = {
            'authority': 'huggingface.co',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,en-GB;q=0.6',
            'origin': 'https://huggingface.co',
            'sec-ch-ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        }
        final_answer = {}

        break_flag = False

        while retry_count > 0:
            resp = self.session.post(
                self.hf_base_url + f"/chat/conversation/{conversation}",
                files={"data": (None, json.dumps(req_json))},
                stream=True,
                headers=headers,
                cookies=self.session.cookies.get_dict(),
            )
            resp.encoding = 'utf-8'

            if resp.status_code != 200:

                retry_count -= 1
                if retry_count <= 0:
                    raise exceptions.ChatError(
                        f"Failed to chat. ({resp.status_code})")

            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    res = line
                    obj = json.loads(res)
                    if obj.__contains__("type"):
                        _type = obj["type"]

                        if _type == "finalAnswer":
                            final_answer = obj
                            break_flag = True
                            break
                    else:
                        logging.error(f"No `type` found in response: {obj}")
                    yield obj
            except requests.exceptions.ChunkedEncodingError:
                pass
            except BaseException as e:
                print(e)
                pass
                # traceback.print_exc()
                # if "Model is overloaded" in str(e):
                #     raise exceptions.ModelOverloadedError(
                #         "Model is overloaded, please try again later or switch to another model."
                #     )
                logging.debug(resp.headers)
                # if "Conversation not found" in str(res):
                #     raise exceptions.InvalidConversationIDError("Conversation id invalid")
                # raise exceptions.ChatError(f"Failed to parse response: {res}")
            if break_flag:
                break

        # update the history of current conversation
        self.get_conversation_info(conversation)
        yield final_answer



