"""Unit tests for the GPT validator module."""

from unittest.mock import MagicMock, patch
import pytest
from validator import validate_scraped_data_with_gpt


def test_validator_missing_key():
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        with pytest.raises(ValueError, match="Chưa cấu hình OpenAI API Key"):
            validate_scraped_data_with_gpt(api_key="")


@patch("validator.OpenAI")
def test_validator_mock_success(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "### Kết luận: THÀNH CÔNG\nDữ liệu 200 bản ghi chuẩn xác."
    mock_response.choices = [mock_choice]
    mock_response.usage.total_tokens = 450
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    result = validate_scraped_data_with_gpt(api_key="test-key-123", model="gpt-4o-mini")
    assert result["success"] is True
    assert result["total_records"] == 200
    assert result["model"] == "gpt-4o-mini"
    assert "THÀNH CÔNG" in result["analysis"]
    assert result["tokens_used"] == 450
