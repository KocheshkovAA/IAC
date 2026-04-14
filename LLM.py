import requests
import os
import uuid
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
        self.access_token = None
        self.token_expires_at = None
        self.auth_key = os.getenv("GIGACHAT_AUTH_KEY")
        self.base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat-Pro")

    def _get_access_token(self) -> str:
        if (self.access_token and self.token_expires_at and 
            datetime.now() < (self.token_expires_at - timedelta(minutes=2))):
            return self.access_token

        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.auth_key}"
        }
        data = {"scope": "GIGACHAT_API_PERS"}

        resp = requests.post(url, headers=headers, data=data, timeout=15, verify=True)
        resp.raise_for_status()
        token_data = resp.json()

        self.access_token = token_data["access_token"]
        expires_at = token_data.get("expires_at")

        if expires_at:
            self.token_expires_at = datetime.fromtimestamp(expires_at / 1000.0)
        else:
            self.token_expires_at = datetime.now() + timedelta(minutes=28)

        return self.access_token

    def chat(self, messages: list, response_format: dict = None):
        token = self._get_access_token()

        client = openai.OpenAI(
            api_key=token,
            base_url=self.base_url,
            timeout=90.0
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=response_format
        )
        return response.choices[0].message.content
