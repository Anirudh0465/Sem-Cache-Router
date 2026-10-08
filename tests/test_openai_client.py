
import httpx
import pytest

from app.models import Usage
from app.providers.openai_client import OpenAIClient


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    from app.config import get_settings
    get_settings.cache_clear()
    return get_settings()

@pytest.mark.asyncio
async def test_openai_client_price_of(mock_settings):
    client = OpenAIClient(httpx.AsyncClient())
    usage = Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
    cost = client.price_of(usage, "gpt-4o-mini")
    assert cost == pytest.approx(0.00075)

@pytest.mark.asyncio
async def test_openai_client_complete(mock_settings):
    client = OpenAIClient(httpx.AsyncClient())
    assert client.name == "openai"
