"""Module xử lý hội thoại trực tiếp với AI (OpenAI GPT-4o-mini) và ghi trace log an toàn."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any
import uuid

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


TRACE_LOG_FILE = Path(__file__).parent / "chat_trace_logs.json"
_TRACE_LOCK = Lock()


def load_trace_logs(filepath: str | os.PathLike[str] = TRACE_LOG_FILE) -> list[dict[str, Any]]:
    """Đọc toàn bộ lịch sử trace logs."""
    path = Path(filepath)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return data


def save_trace_entry(
    entry: dict[str, Any],
    filepath: str | os.PathLike[str] = TRACE_LOG_FILE,
) -> dict[str, Any]:
    """Append trace log an toàn bằng lock và atomic replace."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        logs = load_trace_logs(path)
        logs.append(entry)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(logs, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    return entry


def clear_trace_logs(filepath: str | os.PathLike[str] = TRACE_LOG_FILE) -> None:
    """Xóa lịch sử trace logs."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOCK:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump([], handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


def chat_with_retention_ai(
    messages: list[dict[str, str]],
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    context_metadata: dict[str, Any] | None = None,
    trace_filepath: str | os.PathLike[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    if OpenAI is None:
        raise ImportError("Chưa cài đặt thư viện openai. Hãy chạy: pip install openai")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key or key.strip() in ("", "your_openai_api_key_here"):
        raise ValueError("Chưa cấu hình OpenAI API Key! Vui lòng nhập API Key để chat với AI.")

    system_prompt = (
        "Bạn là Trợ lý AI Cấp cao chuyên gia về Giữ chân Khách hàng (Customer Retention Specialist) "
        "và Hệ thống tương tác Người - Máy (Human-in-the-Loop - HITL) cho sàn Thương mại điện tử.\n\n"
        "BẠN PHẢI TUÂN THỦ NGHIÊM NGẶT QUY TRÌNH RA QUYẾT ĐỊNH 3 TẦNG HITL SAU ĐÂY KHI TƯƠNG TÁC:\n\n"
        "1. 🟢 TẦNG 1 - TỰ ĐỘNG THỰC THI (Confidence >= 85% & Low-risk):\n"
        "   - Khi câu hỏi/thông tin khách hàng rõ ràng, rủi ro thấp (ví dụ: gửi email chăm sóc, mã giảm giá nhỏ).\n"
        "   - Gắn thẻ: `[🟢 HITL: AUTO-EXECUTE - Confidence >= 85%]` và trình bày kế hoạch thực thi tự động.\n\n"
        "2. 🟡 TẦNG 2 - DỪNG CHỜ CON NGƯỜI DUYỆT (70% <= Confidence < 85% HOẶC Hành động tài chính / Tăng tín dụng):\n"
        "   - Khi đề xuất tăng hạn mức tín dụng (Hard Policy) hoặc gửi ưu đãi giá trị cao với confidence 70-85%.\n"
        "   - Gắn thẻ: `[🟡 HITL: CHỜ CON NGƯỜI PHÊ DUYỆT]`.\n"
        "   - BẮT BUỘC CHỦ ĐỘNG HỎI NGƯỜI DÙNG: Yêu cầu Reviewer chọn **Approve (Duyệt)**, **Reject (Từ chối)**, hoặc **Edit (Chỉnh sửa số tiền/nội dung)**.\n\n"
        "3. 🔴 TẦNG 3 - CHỦ ĐỘNG HỎI LẠI / LÀM RÕ (Confidence < 70% hoặc Thông tin mơ hồ / thiếu dữ liệu):\n"
        "   - Khi người dùng hỏi chung chung, thiếu thông tin cụ thể (chưa rõ lịch sử khiếu nại, ngành hàng, ngân sách, số ngày ngưng mua hàng).\n"
        "   - Gắn thẻ: `[🔴 HITL: YÊU CẦU LÀM RÕ / HỎI LẠI - Confidence < 70%]`.\n"
        "   - BẮT BUỘC ĐẶT 2-3 CÂU HỎI LÀM RÕ CỤ THỂ để người dùng bổ sung thông tin trước khi AI có thể đưa ra đề xuất giữ chân an toàn.\n\n"
        "Hãy luôn trả lời mạch lạc, chuyên nghiệp bằng tiếng Việt, định dạng Markdown đẹp mắt và thể hiện rõ tinh thần Human-in-the-Loop."
    )

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    client = OpenAI(api_key=key.strip())
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.4,
    )
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    assistant_content = response.choices[0].message.content or "Không nhận được phản hồi."
    usage = response.usage

    last_user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_query = msg.get("content", "")
            break

    trace_entry = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "user_query": last_user_query,
        "system_prompt": system_prompt,
        "response": assistant_content,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "latency_ms": latency_ms,
        "context_metadata": context_metadata or {},
    }

    save_trace_entry(trace_entry, filepath=trace_filepath or TRACE_LOG_FILE)
    return assistant_content, trace_entry
