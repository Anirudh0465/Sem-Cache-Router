from __future__ import annotations

import time

import httpx

from app.config import get_settings
from app.models import ChatCompletionRequest, Message, ProviderResponse, Usage
from app.providers.base import LLMProvider, ProviderError


class OpenAIClient(LLMProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.http_client = http_client
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.pricing = {
            "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
            "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000}
        }

    async def complete(self, request: ChatCompletionRequest) -> ProviderResponse:
        start_time = time.monotonic()

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = request.model_dump(exclude_unset=True)

        try:
            response = await self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            raise ProviderError(
                f"OpenAI HTTP error: {e.response.text}",
                provider=self.name,
                status_code=status_code,
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"OpenAI Network error: {str(e)}",
                provider=self.name,
                status_code=None,
            ) from e

        latency_ms = (time.monotonic() - start_time) * 1000.0

        choice = data["choices"][0]
        usage = data["usage"]

        provider_usage = Usage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"]
        )

        cost = self.price_of(provider_usage, model=request.model)

        return ProviderResponse(
            provider=self.name,
            model=request.model,
            message=Message(role=choice["message"]["role"], content=choice["message"]["content"]),
            usage=provider_usage,
            cost_usd=cost,
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms
        )

    def estimate_tokens(self, messages: list[Message]) -> int:
        characters = sum(len(message.content) for message in messages)
        return max(1, characters // 4)

    def price_of(self, usage: Usage, model: str = "gpt-4o-mini") -> float:
        rates = self.pricing.get(model, self.pricing["gpt-4o-mini"])
        return (usage.prompt_tokens * rates["input"]) + (usage.completion_tokens * rates["output"])
