"""Module kiểm tra và thẩm định chất lượng dữ liệu cào bằng OpenAI GPT-4o-mini."""

from __future__ import annotations

import json
import os
import time
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from data_loader import DATA_FILE, load_cached_customers


def validate_scraped_data_with_gpt(
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    sample_size: int = 5,
) -> dict[str, Any]:
    """Sử dụng model gpt-4o-mini để thẩm định chất lượng và tính toàn vẹn của dataset."""
    if OpenAI is None:
        raise ImportError("Chưa cài đặt thư viện openai. Hãy chạy: pip install openai")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key or key.strip() in ("", "your_openai_api_key_here"):
        raise ValueError(
            "Chưa cấu hình OpenAI API Key! Vui lòng điền OPENAI_API_KEY vào file .env hoặc nhập trực tiếp trên giao diện."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {DATA_FILE}")

    customers = load_cached_customers()
    total_count = len(customers)
    
    # Tính toán các chỉ số thống kê
    categories = {}
    high_risk = 0
    med_risk = 0
    low_risk = 0
    has_null_or_invalid = 0

    required_fields = ["customer_id", "toi", "churn_probability", "category", "action_payload"]

    for c in customers:
        cat = c.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
        
        prob = c.get("churn_probability", 0.0)
        if prob >= 0.75:
            high_risk += 1
        elif prob >= 0.40:
            med_risk += 1
        else:
            low_risk += 1

        for rf in required_fields:
            if rf not in c or c[rf] is None:
                has_null_or_invalid += 1
                break

    samples = customers[:sample_size]

    prompt = f"""
Bạn là chuyên gia Thẩm định Dữ liệu AI (AI Data Quality Auditor) & Chuyên gia Phân tích Churn E-Commerce.
Hãy kiểm tra và đánh giá xem tập dữ liệu cào từ sàn Thương mại điện tử trên Hugging Face có thành công, toàn vẹn và phù hợp cho bài lab Customer Retention Human-in-the-Loop (HITL) hay không.

### THÔNG TIN TỔNG QUAN TẬP DỮ LIỆU ĐÃ CÀO:
- Tổng số bản ghi đã cào: {total_count} bản ghi (Mục tiêu: 200 bản ghi)
- Số trường bị thiếu / không hợp lệ: {has_null_or_invalid}
- Phân bổ ngành hàng: {json.dumps(categories, ensure_ascii=False)}
- Phân bổ mức độ rủi ro Churn:
  + Rủi ro cao (Churn >= 75% -> Kích hoạt tăng hạn mức tín dụng & Dừng duyệt HITL): {high_risk} khách hàng
  + Rủi ro vừa (40% <= Churn < 75% -> Dừng duyệt gửi Email): {med_risk} khách hàng
  + Rủi ro thấp (Churn < 40% -> Tự động thực thi): {low_risk} khách hàng

### MẪU DỮ LIỆU ĐẠI DIỆN ({sample_size} bản ghi đầu tiên):
{json.dumps(samples, ensure_ascii=False, indent=2)}

### YÊU CẦU ĐÁNH GIÁ (Trả lời bằng tiếng Việt, định dạng Markdown rõ ràng):
1. **Kết luận trạng thái**: [THÀNH CÔNG / CẦN CẢI THIỆN / THẤT BẠI] kèm 1-2 câu tóm tắt tổng quan.
2. **Đánh giá độ toàn vẹn & Phân bổ dữ liệu**: Đánh giá tính đầy đủ của các trường, sự cân đối giữa các phân khúc rủi ro và các ngành hàng.
3. **Độ tương thích với bài lab HITL**: Phân tích xem dữ liệu này có đáp ứng hoàn hảo 4 kịch bản kiểm thử (Auto-execute, Low-risk Review, High-risk Hard Policy Interrupt, Edit Payload) hay không.
4. **Nhận xét chuyên môn & Đề xuất**: Đưa ra 2-3 gợi ý tối ưu (nếu có).
"""

    client = OpenAI(api_key=key.strip())
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia thẩm định dữ liệu và kiến trúc sư hệ thống AI."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    
    duration = time.time() - start_time
    analysis_text = response.choices[0].message.content or "Không nhận được phản hồi từ mô hình."
    usage = response.usage

    return {
        "success": True,
        "total_records": total_count,
        "high_risk": high_risk,
        "med_risk": med_risk,
        "low_risk": low_risk,
        "categories": categories,
        "invalid_count": has_null_or_invalid,
        "analysis": analysis_text,
        "model": model,
        "duration_seconds": round(duration, 2),
        "tokens_used": usage.total_tokens if usage else None,
    }
