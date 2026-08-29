# Lab 27 — Customer Retention Human-in-the-Loop

Hệ thống mô phỏng một AI agent đánh giá Total Operating Income (TOI) và xác suất churn của khách hàng, đề xuất hành động giữ chân, tự động thực thi tác vụ an toàn và bắt buộc con người phê duyệt tác vụ rủi ro cao.

## Thành phần chính

- `GraphState`: giữ `customer_id`, `proposed_action`, `confidence_score`, `reasoning` và `human_decision` xuyên suốt workflow.
- `AuditEntry`: lưu đầy đủ timestamp, agent, action, confidence, reviewer và quyết định.
- `evaluate_customer`: đánh giá churn, dùng TOI để hiệu chỉnh confidence trước routing.
- `route_action`: áp dụng hard policy trước confidence threshold.
- `execute_low_risk_action`: mô phỏng gửi email an toàn.
- `execute_high_risk_action`: chỉ thực thi sau Approve/Edit hợp lệ; Reject sẽ hủy.
- `MemorySaver` và `interrupt_before`: giữ state và dừng graph trước high-risk action.
- Streamlit: hiển thị Action Card, Approve, Reject, Edit và lịch sử audit.

## Cài đặt

Yêu cầu Python 3.11 trở lên. Từ thư mục repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu muốn chạy test và coverage:

```powershell
python -m pip install -r requirements-dev.txt
```

Tạo file `.env` từ `.env.example` nếu sử dụng GPT Validator hoặc Live Chat. Workflow HITL chính dùng mock deterministic nên không cần API key.

```powershell
Copy-Item .env.example .env
```

Không commit `.env` hoặc credential thật lên GitHub.

## Chạy LangGraph workflow

Có thể chạy workflow trực tiếp, không cần Streamlit:

```powershell
@'
from graph import start_customer_workflow

result = start_customer_workflow(
    customer_id="CUST001",
    toi="low",
    churn_probability=0.25,
)
print(result)
'@ | python -
```

Với churn `0.25`, TOI `low`, agent đề xuất `send_email`, confidence `0.90` và tự thực thi. Với high-risk hoặc confidence thấp, kết quả chứa `is_paused=True` và `next=["execute_high_risk_action"]`.

Mỗi workflow dùng một `thread_id` riêng. `MemorySaver` giữ snapshot khi graph bị interrupt; resume phải dùng đúng `thread_id` qua `update_state(...)` rồi `invoke(None, config=...)`.

> `MemorySaver` chỉ lưu trong RAM của process. Restart ứng dụng sẽ mất pending checkpoint. Production nên dùng SQLite/PostgreSQL checkpointer và append-only database cho audit.

## Chạy Streamlit UI

```powershell
streamlit run app.py
```

Quy trình sử dụng:

1. Chọn một khách hàng trong dataset hoặc nhập `customer_id`, TOI và churn probability.
2. Nhấn **Run agent**.
3. Nếu action an toàn và confidence đủ cao, hệ thống tự thực thi.
4. Nếu graph dừng review, nhập `Reviewer ID` và chọn:
   - **Approve**: chấp nhận action/payload gốc và resume graph.
   - **Reject**: nhập lý do, hủy action và resume graph theo nhánh abort.
   - **Edit**: sửa số tiền hoặc email template rồi thực thi payload mới.
5. Mở tab **Audit log** để kiểm tra quyết định đã được lưu.

Compiled graph được giữ trong `st.session_state` để `MemorySaver` và pending state không bị tạo lại trong mỗi Streamlit rerun.

## Confidence threshold và hard policy

Threshold auto-execute là:

```python
AUTO_EXECUTE_THRESHOLD = 0.85
```

Routing áp dụng theo thứ tự:

1. `increase_credit_limit` luôn route tới `execute_high_risk_action`, bất kể confidence, kể cả `0.99`.
2. Chỉ `send_email` có confidence `>= 0.85` được route tới `execute_low_risk_action`.
3. Confidence `< 0.85` hoặc action không xác định đều fail-closed sang human review.

TOI hiệu chỉnh confidence trước routing: `low=-0.02`, `medium=0.00`, `high=+0.02`. Hard policy luôn có quyền override confidence sau hiệu chỉnh.

## Audit log

Audit được lưu tại:

```text
audit_log.json
```

File khởi tạo là một JSON array rỗng `[]`. Sau mỗi quyết định Approve, Reject hoặc Edit, hệ thống append một entry gồm tối thiểu:

```json
{
  "timestamp": "2026-08-29T09:00:00Z",
  "agent_id": "customer-retention-agent",
  "action": "increase_credit_limit",
  "confidence": 0.98,
  "reviewer_id": "operator_01",
  "decision": "approve"
}
```

Audit sử dụng lock và atomic replace để không ghi đè lịch sử. Nếu ghi audit thất bại, high-risk action bị chặn.

## Tự kiểm tra kết quả

Chạy toàn bộ test:

```powershell
python -m pytest -q --cov=graph --cov=models --cov-report=term-missing
```

Kiểm tra thủ công bốn kịch bản:

1. Churn `0.25`, TOI `low` → `send_email`, confidence `0.90` → auto-execute.
2. Churn `0.55`, TOI `medium` → `send_email`, confidence `0.82` → dừng review → Approve → execute.
3. Churn `0.82`, TOI `high` → `increase_credit_limit`, confidence `0.98` → vẫn dừng review → Reject → abort.
4. Kịch bản 3 → Edit `50,000,000` thành `20,000,000` → chỉ payload mới được execute và ghi audit.

Khi pending, mở `LangGraph state snapshot` và xác nhận:

- Có `customer_id`, `proposed_action`, `confidence_score`, `reasoning`, `human_decision`.
- `next` là `execute_high_risk_action`.
- `executed_payload` vẫn là `null`.
- Sau quyết định, `audit_log.json` có entry mới và lịch sử cũ không bị overwrite.

## Lỗi thường gặp

### Mất state sau interrupt

Kiểm tra graph dùng `MemorySaver()`, truyền `checkpointer=memory` và tất cả lệnh `invoke`, `get_state`, `update_state` dùng cùng `thread_id`.

### High-risk action chạy trước review

Phải compile bằng:

```python
interrupt_before=["execute_high_risk_action"]
```

### Hard rule bị confidence override

Kiểm tra `increase_credit_limit` trước confidence threshold. Không auto-execute action này dù confidence là `0.99`.

### Streamlit bấm nút nhưng graph không tiếp tục

Sau khi cập nhật human decision phải gọi cả hai bước:

```python
graph.update_state(config, state_update)
graph.invoke(None, config=config)
```

### Không lấy được pending state

Dùng `graph.get_state(config)` với đúng `thread_id` đã dùng khi invoke lần đầu.

### Audit log bị ghi đè hoặc hỏng

Đọc danh sách cũ, append `AuditEntry`, rồi ghi atomic toàn bộ danh sách. Không ghi một object đơn lẻ đè lên file.

### Lỗi `No module named datasets`

Cài lại đầy đủ dependency bằng `python -m pip install -r requirements.txt`. Thư viện `datasets` đã được khai báo trực tiếp trong requirements.

## Reflection Questions

### Câu 1 — Dùng `interrupt_before` hay `interrupt_after` để human rewrite email?

Dùng `interrupt_after` trên node generate email. Khi interrupt xảy ra, email đã được tạo và lưu vào state nhưng routing node chưa chạy. Human có thể đọc, sửa nội dung bằng `update_state`, sau đó resume để routing sử dụng bản đã chỉnh sửa. Nếu dùng `interrupt_before` node generate thì chưa có email để review; nếu đặt `interrupt_before` routing node cũng có thể đạt mục đích nhưng không diễn đạt trực tiếp yêu cầu “dừng ngay sau khi generate”.

### Câu 2 — Giảm alert fatigue khi có 500 email confidence 0.82

- Gom các email tương đồng thành batch và hỗ trợ bulk approve.
- Xếp hàng ưu tiên theo churn, TOI, giá trị khách hàng và mức độ bất thường.
- Dùng template đã được phê duyệt; chỉ review phần biến đổi hoặc diff.
- Sau khi có đủ dữ liệu đúng, áp dụng sampling review cho nhóm low-risk thay vì review 100%.
- Theo dõi false-escalation rate và điều chỉnh threshold dựa trên calibrated confidence, không hạ threshold tùy tiện.
- Hard policy như `increase_credit_limit` vẫn phải review 100%, không được đưa vào bulk auto-execute.

### Câu 3 — Vì sao self-reported confidence nguy hiểm và cách calibrate?

Confidence do LLM tự báo không phải xác suất thống kê đã được hiệu chỉnh. Model có thể rất tự tin khi dữ liệu thu nhập sai, cũ hoặc thiếu; vì vậy confidence cao không chứng minh recommendation đúng và không được phép bypass policy.

Trước routing cần đánh giá trên validation set có nhãn, đo Brier score, Expected Calibration Error và reliability curve. Sau đó có thể dùng Platt scaling, isotonic regression hoặc temperature scaling để chuyển raw score thành calibrated probability. Confidence cuối cùng cũng nên kết hợp chất lượng, nguồn gốc và độ mới của dữ liệu. Dù calibrated score cao, hard policy cho action tài chính vẫn giữ nguyên.

## Cấu trúc dữ liệu

Dataset duy nhất được lưu tại:

```text
data/ecommerce_customers_200.json
```

Không lưu thêm bản sao ở thư mục gốc repository.
