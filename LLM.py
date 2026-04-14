import os
from typing import Any, Union, Optional
import openai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, temperature: float = 0.1, model: str = "openai/gpt-4o-mini"):
        self.temperature = temperature
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в .env")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
        )

    def chat(
        self,
        messages: list,
        response_format: Optional[Union[dict, BaseModel, type[BaseModel]]] = None
    ) -> Any:
        extra_params = {}

        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                extra_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }
            else:
                extra_params["response_format"] = response_format

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **extra_params
        )

        message = response.choices[0].message

        if extra_params.get("response_format", {}).get("type") == "json_schema":
            try:
                if message.content:
                    return openai.types.chat.chat_completion_message.ChatCompletionMessage.model_validate({
                        "content": message.content,
                        "role": "assistant"
                    }).content 
                return {}
            except Exception:
                return message.content

        return message.content