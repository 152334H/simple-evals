from typing import Any
from random import choice

from ..types import MessageList, SamplerBase
from openai import OpenAI, BadRequestError

SYSTEM_PROMPT = (
    "You are a helpful assistant."
    # + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class VLLMCompletionSampler(SamplerBase):
    def __init__(
        self,
        model: str,
        ports: list[int],
        *,
        system_message: str | None = SYSTEM_PROMPT,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.message_start = [] if system_message is None else [self._pack_message("system", system_message)]
        self.clients = [OpenAI(api_key="EMPTY", base_url=f"http://localhost:{p}/v1") for p in ports]
        self.kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
        self.image_format = "url" # IDK

    def f(self, msg_ls: MessageList):
        c: OpenAI = choice(self.clients) # randomly select one port
        return c.chat.completions.create(messages=self.message_start + msg_ls, **self.kwargs)

    def _handle_image(self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768):
        raise NotImplementedError

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        try:
            chat_response = self.f(message_list)
        except BadRequestError as e:
            print(e)
            return ''
        return chat_response.choices[0].message.content
