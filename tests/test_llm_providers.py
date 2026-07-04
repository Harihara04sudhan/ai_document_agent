"""Tests for OpenAI and Ollama provider clients (all HTTP mocked)."""

import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClientFactory, OllamaClient, OpenAIClient


def _response(payload):
    resp = Mock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestOpenAIClient:
    def test_requires_api_key(self):
        import pytest
        with pytest.raises(ValueError):
            OpenAIClient(api_key="")

    @patch("utils.llm_client.requests.post")
    def test_generate_response(self, mock_post):
        mock_post.return_value = _response(
            {"choices": [{"message": {"content": "hello!"}}]}
        )
        client = OpenAIClient(api_key="test-key")
        out = client.generate_response("hi", system_message="be brief")

        assert out == "hello!"
        url = mock_post.call_args[0][0]
        body = mock_post.call_args[1]["json"]
        assert url.endswith("/chat/completions")
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}

    @patch("utils.llm_client.requests.post")
    def test_generate_embeddings(self, mock_post):
        mock_post.return_value = _response({"data": [{"embedding": [0.1, 0.2]}]})
        client = OpenAIClient(api_key="test-key")
        assert client.generate_embeddings("text") == [0.1, 0.2]

    @patch("utils.llm_client.requests.post")
    def test_custom_base_url(self, mock_post):
        mock_post.return_value = _response(
            {"choices": [{"message": {"content": "ok"}}]}
        )
        client = OpenAIClient(api_key="k", base_url="https://my-proxy.example/v1/")
        client.generate_response("hi")
        assert mock_post.call_args[0][0] == "https://my-proxy.example/v1/chat/completions"


class TestOllamaClient:
    @patch("utils.llm_client.requests.post")
    def test_generate_response(self, mock_post):
        mock_post.return_value = _response({"message": {"content": "local hello"}})
        client = OllamaClient()
        out = client.generate_response("hi")

        assert out == "local hello"
        url = mock_post.call_args[0][0]
        body = mock_post.call_args[1]["json"]
        assert url == "http://localhost:11434/api/chat"
        assert body["stream"] is False

    @patch("utils.llm_client.requests.post")
    def test_generate_embeddings(self, mock_post):
        mock_post.return_value = _response({"embedding": [1.0, 2.0, 3.0]})
        client = OllamaClient()
        assert client.generate_embeddings("text") == [1.0, 2.0, 3.0]

    @patch("utils.llm_client.requests.post")
    def test_embedding_failure_returns_zero_vector(self, mock_post):
        mock_post.side_effect = ConnectionError("ollama down")
        client = OllamaClient()
        out = client.generate_embeddings("text")
        assert len(out) == 768
        assert set(out) == {0.0}


class TestFactory:
    def test_unknown_provider_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMClientFactory.create_client("nonexistent")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "k"})
    def test_creates_openai(self):
        client = LLMClientFactory.create_client("openai")
        assert isinstance(client, OpenAIClient)

    def test_creates_ollama_without_keys(self):
        client = LLMClientFactory.create_client("ollama")
        assert isinstance(client, OllamaClient)
