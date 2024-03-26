from .base_model_inference import *
import requests
import sys, os
from PIL import Image


class GPT4Inference(BaseModelInference):
    def __init__(self, model_name, local_save_path=""):
        super().__init__(model_name, local_save_path)

    def load_model(self, **kwargs):
        self.api_key = kwargs["api_key"]
        self.header = self._make_headers()

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        payload = self._make_payload()
        self.response = self._request_gpt_api(payload)

    def _request_gpt_api(self, payload):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.header,
            json=payload,
        )
        return response.json()

    def extract_answers(self):
        if "error" in self.response:
            raise Exception("error on gpt4 api" + str(self.response))
        return self.response["choices"][0]["message"]["content"]

    def _make_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _make_payload(self):
        return {
            "model": "gpt-4-vision-preview",
            "messages": [
                self._make_system_prompt(),
                self._make_user_prompt(),
                {"role": "assistant", "content": ["In the video,"]},
            ],
            "max_tokens": self.max_tokens,
        }

    def _make_system_prompt(self):
        return {"role": "system", "content": [self.system_prompt]}

    def _make_user_prompt(self):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": self.user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.base64_image}",
                        "detail": "high",
                    },
                },
            ],
        }

    def _extract_arguments(self, **kwargs):
        self.system_prompt = kwargs["system_prompt"]
        self.user_prompt = kwargs["user_prompt"]
        self.base64_image = kwargs["base64_img"]
        self.max_tokens = kwargs.get("max_tokens", 500)
