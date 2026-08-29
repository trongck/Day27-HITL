"""Unit tests for chat_tracer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from chat_tracer import (
    clear_trace_logs,
    load_trace_logs,
    save_trace_entry,
    chat_with_retention_ai,
)


def test_save_and_load_trace_logs(tmp_path: Path):
    test_file = tmp_path / "test_trace.json"
    entry = {
        "trace_id": "test-uuid-1",
        "timestamp": "2026-08-29T10:00:00Z",
        "model": "gpt-4o-mini",
        "user_query": "Khách hàng ECOMM_10001 có rủi ro gì?",
        "response": "Khách hàng có nguy cơ churn thấp.",
        "latency_ms": 320.5,
    }
    saved = save_trace_entry(entry, filepath=test_file)
    assert saved["trace_id"] == "test-uuid-1"

    logs = load_trace_logs(filepath=test_file)
    assert len(logs) == 1
    assert logs[0]["trace_id"] == "test-uuid-1"

    clear_trace_logs(filepath=test_file)
    assert load_trace_logs(filepath=test_file) == []


def test_chat_without_api_key():
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        with pytest.raises(ValueError, match="Chưa cấu hình OpenAI API Key"):
            chat_with_retention_ai([{"role": "user", "content": "Hello"}], api_key="")


@patch("chat_tracer.OpenAI")
def test_chat_mock_success(mock_openai, tmp_path: Path):
    test_file = tmp_path / "test_trace_mock.json"
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Xin chào! Tôi có thể giúp gì cho bạn?"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 70
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    resp, trace = chat_with_retention_ai(
        [{"role": "user", "content": "Xin chào"}],
        api_key="mock-key-123",
        model="gpt-4o-mini",
        trace_filepath=test_file,
    )
    assert resp == "Xin chào! Tôi có thể giúp gì cho bạn?"
    assert trace["total_tokens"] == 70
    assert trace["model"] == "gpt-4o-mini"
    assert test_file.exists()
