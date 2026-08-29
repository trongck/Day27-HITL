# Customer Retention Human-in-the-Loop

Lab chính triển khai workflow LangGraph an toàn cho hai action:

- `increase_credit_limit`: hard policy, luôn dừng chờ human review.
- `send_email`: chỉ auto-execute khi confidence `>= 0.85`; mọi trường hợp khác fail-closed sang review.

## Chạy ứng dụng

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Mỗi workflow dùng một `thread_id` riêng. `MemorySaver` giữ state khi graph bị interrupt và UI resume bằng cùng `thread_id` qua `update_state(...)` rồi `invoke(None, config=...)`.

> `MemorySaver` chỉ lưu trong RAM của process. Restart Streamlit sẽ mất checkpoint. Production nên dùng SQLite/PostgreSQL checkpointer; audit nên chuyển sang append-only database.

## Tự kiểm tra

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q --cov=graph --cov=models --cov-report=term-missing
```

Kiểm tra thủ công bốn kịch bản:

1. Customer churn `0.25` → `send_email`, confidence `0.92` → tự thực thi.
2. Customer churn `0.55` → `send_email`, confidence `0.82` → dừng review → Approve → thực thi.
3. Customer churn `0.82` → `increase_credit_limit`, confidence `0.96` → vẫn dừng review → Reject → không thực thi.
4. Kịch bản 3 → Edit `50,000,000` thành `20,000,000` → chỉ payload mới được thực thi và ghi audit.

Khi pending, mở `LangGraph state snapshot` để xác nhận đủ `customer_id`, `proposed_action`, `confidence_score`, `reasoning`, `human_decision`; action high-risk chưa chạy. Sau quyết định, `audit_log.json` phải được append entry gồm `timestamp`, `agent_id`, `action`, `confidence`, `reviewer_id`, `decision`.

## Lỗi thường gặp

- Mất state sau interrupt: bảo đảm compile với `MemorySaver()` và cùng `thread_id` khi `get_state`/resume.
- High-risk chạy trước review: phải dùng `interrupt_before=["execute_high_risk_action"]`.
- Confidence override hard rule: kiểm tra `increase_credit_limit` trước threshold.
- Button không resume: gọi `update_state(config, ...)`, sau đó `invoke(None, config=config)`.
- Audit bị ghi đè/hỏng: luôn đọc lịch sử, append entry và atomic replace; không thay bằng một object đơn lẻ.
"# Day27-HITL" 
