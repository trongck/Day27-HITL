"""Streamlit approval console for the customer-retention HITL workflow."""

from __future__ import annotations

import uuid
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from graph import (
    MAX_CREDIT_LIMIT_INCREASE,
    build_hitl_graph,
    get_current_workflow_state,
    start_customer_workflow,
    submit_human_decision,
)
from models import HumanDecision, load_audit_logs


st.set_page_config(page_title="Customer Retention HITL", page_icon="🛡️", layout="wide")
st.title("Customer Retention — Human-in-the-Loop")
st.caption("LangGraph checkpoint, hard-policy routing, human approval and append-only audit")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "hitl_graph" not in st.session_state:
    st.session_state.hitl_graph = build_hitl_graph()


def current_workflow() -> dict | None:
    thread_id = st.session_state.thread_id
    return (
        get_current_workflow_state(thread_id, graph=st.session_state.hitl_graph)
        if thread_id
        else None
    )


def apply_decision(decision: HumanDecision) -> None:
    try:
        submit_human_decision(
            st.session_state.thread_id,
            decision,
            graph=st.session_state.hitl_graph,
        )
    except Exception as exc:
        st.error(f"Không thể xử lý quyết định: {exc}")
    else:
        st.rerun()


import os
from chat_tracer import chat_with_retention_ai, clear_trace_logs, load_trace_logs
from data_loader import load_cached_customers
from validator import validate_scraped_data_with_gpt

workflow_tab, audit_tab, dataset_tab, validator_tab, chat_tab, help_tab = st.tabs([
    "Workflow", 
    "Audit log", 
    "📦 200 E-Commerce Data", 
    "🤖 AI Validator (GPT-4o-mini)",
    "💬 Live Chat & Trace Log",
    "Tự kiểm tra"
])

with workflow_tab:
    workflow = current_workflow()
    values = workflow["values"] if workflow else None

    # Thanh test nhanh 4 kịch bản chuẩn của bài Lab HITL
    st.markdown("#### ⚡ Thử nghiệm nhanh 4 kịch bản chuẩn của Lab HITL:")
    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    
    if sc_col1.button("🟢 1. Low Churn (Auto-run)", use_container_width=True, help="Churn=0.25, TOI=low -> send_email (Confidence=90% >= 85%) -> Tự động thực thi"):
        thread_id = f"customer-CUST_AUTO-{uuid.uuid4().hex}"
        start_customer_workflow(customer_id="CUST_LOW_01", toi="low", churn_probability=0.25, thread_id=thread_id, graph=st.session_state.hitl_graph)
        st.session_state.thread_id = thread_id
        st.rerun()

    if sc_col2.button("🟡 2. Med Churn (Dừng duyệt Email)", use_container_width=True, help="Churn=0.55 -> send_email (Confidence=82% < 85%) -> DỪNG CHỜ DUYỆT"):
        thread_id = f"customer-CUST_REVIEW-{uuid.uuid4().hex}"
        start_customer_workflow(customer_id="CUST_MED_02", toi="medium", churn_probability=0.55, thread_id=thread_id, graph=st.session_state.hitl_graph)
        st.session_state.thread_id = thread_id
        st.rerun()

    if sc_col3.button("🚨 3. High Churn (Dừng Tín dụng)", use_container_width=True, help="Churn=0.82 -> increase_credit_limit -> HARD POLICY: LUÔN DỪNG CHỜ DUYỆT"):
        thread_id = f"customer-CUST_HIGHRISK-{uuid.uuid4().hex}"
        start_customer_workflow(customer_id="CUST_HIGH_03", toi="high", churn_probability=0.82, thread_id=thread_id, graph=st.session_state.hitl_graph)
        st.session_state.thread_id = thread_id
        st.rerun()

    if sc_col4.button("✏️ 4. Chỉnh sửa số tiền (Edit)", use_container_width=True, help="Tạo ca Churn cao 0.85 để bạn vào tab Edit chỉnh sửa số tiền từ 50tr thành 20tr"):
        thread_id = f"customer-CUST_EDIT-{uuid.uuid4().hex}"
        start_customer_workflow(customer_id="CUST_EDIT_04", toi="high", churn_probability=0.85, thread_id=thread_id, graph=st.session_state.hitl_graph)
        st.session_state.thread_id = thread_id
        st.rerun()

    st.divider()

    if values is None:
        st.subheader("1. Customer input")
        input_mode = st.radio(
            "Chọn phương thức nạp dữ liệu:",
            ["📦 Chọn từ 200 khách hàng E-Commerce (Hugging Face)", "✍️ Nhập thủ công"],
            horizontal=True,
        )

        if input_mode == "📦 Chọn từ 200 khách hàng E-Commerce (Hugging Face)":
            customers = load_cached_customers()
            
            filter_risk = st.selectbox(
                "Lọc theo mức độ rủi ro Churn:",
                [
                    "Tất cả (200 khách hàng)",
                    "🚨 Rủi ro cao (Churn ≥ 75% -> Dừng duyệt tăng tín dụng BNPL)",
                    "⚠️ Rủi ro vừa (40% ≤ Churn < 75% -> Dừng duyệt gửi Email)",
                    "✅ Rủi ro thấp (Churn < 40% -> Tự động thực thi gửi Email)",
                ],
            )
            
            if "Rủi ro cao" in filter_risk:
                filtered_customers = [c for c in customers if c["churn_probability"] >= 0.75]
            elif "Rủi ro vừa" in filter_risk:
                filtered_customers = [c for c in customers if 0.40 <= c["churn_probability"] < 0.75]
            elif "Rủi ro thấp" in filter_risk:
                filtered_customers = [c for c in customers if c["churn_probability"] < 0.40]
            else:
                filtered_customers = customers

            customer_options = {
                f"{c['customer_id']} | {c['category']} | Churn: {c['churn_probability']:.0%} | TOI: {c['toi'].upper()} | Khiếu nại: {'Có' if c['complain'] else 'Không'}": c
                for c in filtered_customers
            }
            
            selected_label = st.selectbox("Chọn khách hàng từ danh sách:", list(customer_options.keys()))
            selected_cust = customer_options[selected_label]

            # Hiển thị thông tin tóm tắt của khách hàng
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mã khách hàng", selected_cust["customer_id"])
            c2.metric("Ngành hàng", selected_cust["category"])
            c3.metric("Xác suất Churn", f"{selected_cust['churn_probability']:.0%}")
            c4.metric("Thời gian gắn bó", f"{selected_cust['tenure_months']} tháng")

            if st.button("🚀 Chạy Agent với khách hàng này", type="primary", use_container_width=True):
                thread_id = f"customer-{selected_cust['customer_id'].strip()}-{uuid.uuid4().hex}"
                try:
                    start_customer_workflow(
                        customer_id=selected_cust["customer_id"],
                        toi=selected_cust["toi"],
                        churn_probability=selected_cust["churn_probability"],
                        action_payload=selected_cust.get("action_payload"),
                        thread_id=thread_id,
                        graph=st.session_state.hitl_graph,
                    )
                except Exception as exc:
                    st.error(f"Không thể khởi chạy workflow: {exc}")
                else:
                    st.session_state.thread_id = thread_id
                    st.rerun()

        else:
            with st.form("customer_input"):
                customer_id = st.text_input("Customer ID", value="CUST001", max_chars=100)
                toi = st.selectbox("TOI", ["high", "medium", "low"])
                churn_probability = st.slider(
                    "Churn probability", min_value=0.0, max_value=1.0, value=0.25, step=0.01
                )
                submitted = st.form_submit_button("Run agent", type="primary")
            if submitted:
                thread_id = f"customer-{customer_id.strip()}-{uuid.uuid4().hex}"
                try:
                    start_customer_workflow(
                        customer_id=customer_id,
                        toi=toi,
                        churn_probability=churn_probability,
                        thread_id=thread_id,
                        graph=st.session_state.hitl_graph,
                    )
                except Exception as exc:
                    st.error(f"Không thể khởi chạy workflow: {exc}")
                else:
                    st.session_state.thread_id = thread_id
                    st.rerun()
    else:
        st.subheader("2. Agent reasoning")
        col1, col2, col3 = st.columns(3)
        col1.metric("Customer ID", values["customer_id"])
        col2.metric("Proposed action", values["proposed_action"])
        col3.metric("Confidence score", f"{values['confidence_score']:.0%}")
        st.info(values["reasoning"])
        st.json({"action_payload": values.get("action_payload", {})})

        if workflow["is_paused"]:
            st.warning("Pending human review — high-risk action has not run.")
            reviewer_id = st.text_input("Reviewer ID (required)", max_chars=100)
            approve_tab, reject_tab, edit_tab = st.tabs(["Approve", "Reject", "Edit"])

            with approve_tab:
                approve_note = st.text_input("Approval note", key="approve_note", max_chars=1000)
                if st.button("Approve", type="primary", use_container_width=True):
                    try:
                        decision = HumanDecision(
                            action="approve", reviewer_id=reviewer_id, reason=approve_note or None
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        apply_decision(decision)

            with reject_tab:
                reject_reason = st.text_area("Rejection reason (required)", max_chars=1000)
                if st.button("Reject", use_container_width=True):
                    try:
                        decision = HumanDecision(
                            action="reject", reviewer_id=reviewer_id, reason=reject_reason
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        apply_decision(decision)

            with edit_tab:
                old_payload = values.get("action_payload", {})
                st.write("Giá trị cũ:")
                st.json(old_payload)
                if values["proposed_action"] == "increase_credit_limit":
                    edited_amount = st.number_input(
                        "New credit-limit amount (VND)",
                        min_value=1,
                        max_value=MAX_CREDIT_LIMIT_INCREASE,
                        value=int(old_payload.get("amount", 50_000_000)),
                        step=1_000_000,
                    )
                    edited_payload = {"amount": edited_amount}
                else:
                    edited_template = st.text_area(
                        "New email template",
                        value=str(old_payload.get("template", "retention_offer")),
                        max_chars=10_000,
                    )
                    edited_payload = {"template": edited_template}
                edit_reason = st.text_input("Edit reason", max_chars=1000)
                st.write("Giá trị mới:")
                st.json(edited_payload)
                if st.button("Edit and execute", type="primary", use_container_width=True):
                    try:
                        decision = HumanDecision(
                            action="edit",
                            reviewer_id=reviewer_id,
                            reason=edit_reason or None,
                            edited_payload=edited_payload,
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        apply_decision(decision)
        else:
            status = values.get("execution_status", "unknown")
            human_dec = values.get("human_decision")
            if status == "completed":
                if human_dec:
                    st.success(f"✅ {values.get('execution_message', 'Completed')} (Đã thực thi sau khi Human phê duyệt: {human_dec.get('action')})")
                else:
                    st.info(f"🟢 {values.get('execution_message', 'Completed')} (Tự động thực thi vì `send_email` có Confidence {values.get('confidence_score', 0):.0%} ≥ 85%)")
            elif status == "aborted":
                st.error(f"🛑 {values.get('execution_message', 'Rejected')} (Đã hủy bỏ hành động do Human Reviewer từ chối)")
            else:
                st.warning(values.get("execution_message", status))
            st.write("Executed payload:")
            st.json(values.get("executed_payload"))

        with st.expander("LangGraph state snapshot"):
            st.json(values)
        st.code(st.session_state.thread_id, language="text")
        if st.button("Start new workflow"):
            st.session_state.thread_id = None
            st.rerun()

with audit_tab:
    st.subheader("Human decision audit history")
    try:
        logs = load_audit_logs()
    except Exception as exc:
        st.error(f"Audit log không đọc được: {exc}")
    else:
        if logs:
            required_columns = [
                "timestamp",
                "agent_id",
                "action",
                "confidence",
                "reviewer_id",
                "decision",
            ]
            st.dataframe(
                [{key: item.get(key) for key in required_columns} for item in logs],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Full audit payload"):
                st.json(logs)
with dataset_tab:
    st.subheader("📦 200 Khách hàng E-Commerce (Hugging Face Hub)")
    st.caption("Dữ liệu được trích xuất và chuẩn hóa từ tập dữ liệu e-commerce churn thực tế.")
    try:
        all_data = load_cached_customers()
        import pandas as pd
        df_display = pd.DataFrame(all_data)
        
        # Thống kê nhanh
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng số khách hàng", len(df_display))
        m2.metric("Số khách hàng High-Risk", len(df_display[df_display["churn_probability"] >= 0.75]))
        m3.metric("Số khách hàng Med-Risk", len(df_display[(df_display["churn_probability"] >= 0.40) & (df_display["churn_probability"] < 0.75)]))
        m4.metric("Số khách hàng Low-Risk", len(df_display[df_display["churn_probability"] < 0.40]))

        st.dataframe(
            df_display[[
                "customer_id", "category", "tenure_months", "satisfaction_score",
                "complain", "day_since_last_order", "cashback_amount",
                "toi", "churn_probability"
            ]],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error(f"Không thể tải dữ liệu: {exc}")

with validator_tab:
    st.subheader("🤖 Thẩm định chất lượng dữ liệu cào bằng OpenAI GPT-4o-mini")
    st.caption("Sử dụng mô hình GPT-4o-mini để tự động kiểm tra tính toàn vẹn, độ phân bổ rủi ro và tính tương thích của 200 bản ghi với bài lab HITL (API Key được tự động nạp từ file `.env`).")

    if st.button("🔍 Tiến hành thẩm định dữ liệu với GPT-4o-mini", type="primary", use_container_width=True):
        api_key_to_use = os.getenv("OPENAI_API_KEY")
        if not api_key_to_use or api_key_to_use in ("", "your_openai_api_key_here"):
            st.error("⚠️ Chưa tìm thấy `OPENAI_API_KEY`! Vui lòng điền API Key vào file `.env` của dự án.")
        else:
            with st.spinner("🤖 Đang kết nối tới OpenAI API (gpt-4o-mini) và phân tích 200 bản ghi dữ liệu..."):
                try:
                    result = validate_scraped_data_with_gpt(
                        api_key=api_key_to_use,
                        model="gpt-4o-mini",
                        sample_size=5,
                    )
                except Exception as exc:
                    st.error(f"❌ Quá trình thẩm định gặp lỗi: {exc}")
                else:
                    st.success("✅ Đã hoàn thành thẩm định dữ liệu thành công!")
                    
                    # Hiển thị metrics phản hồi
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Tổng bản ghi", f"{result['total_records']} / 200")
                    m_col2.metric("Số bản ghi lỗi", result["invalid_count"])
                    m_col3.metric("Thời gian phản hồi", f"{result['duration_seconds']}s")
                    m_col4.metric("Tokens sử dụng", result.get("tokens_used", "N/A"))

                    # Kết quả phân tích chi tiết
                    st.markdown("### 📋 Báo cáo đánh giá từ GPT-4o-mini:")
                    st.markdown(result["analysis"])

with chat_tab:
    st.subheader("💬 Chat trực tiếp với AI Retention & Theo dõi Trace Log")
    st.caption("Trò chuyện trực tiếp với AI chuyên gia Giữ chân khách hàng và kiểm tra toàn bộ dữ liệu Trace Logs (Request, Response, Tokens, Latency).")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI Giữ chân khách hàng (Customer Retention AI). Bạn có thể hỏi tôi về phân tích dữ liệu 200 khách hàng E-commerce, cơ chế duyệt Human-in-the-Loop hoặc gợi ý chính sách ưu đãi."}
        ]

    chat_col, trace_col = st.columns([3, 2])

    with chat_col:
        st.markdown("#### 🗨️ Khung trò chuyện")
        
        # Gợi ý câu hỏi nhanh
        st.write("💡 *Thử nghiệm 3 tầng phản hồi HITL của Bot:*")
        q_cols = st.columns(3)
        if q_cols[0].button("🔴 1. Thiếu data (Bot sẽ HỎI LẠI)", use_container_width=True, help="Câu hỏi mơ hồ -> Bot gắn thẻ Tầng 3 và đặt 2-3 câu hỏi làm rõ"):
            st.session_state.chat_messages.append({"role": "user", "content": "Có một khách hàng đang muốn rời bỏ sàn, tôi có nên cấp ưu đãi cho họ không?"})
            st.rerun()
        if q_cols[1].button("🟡 2. Tín dụng (Bot DỪNG HỎI DUYỆT)", use_container_width=True, help="Đề xuất tăng tín dụng -> Bot gắn thẻ Tầng 2 và yêu cầu Reviewer Approve/Reject/Edit"):
            st.session_state.chat_messages.append({"role": "user", "content": "Khách hàng VIP ECOMM_10008 có nguy cơ rời sàn cao. Hãy đề xuất tăng hạn mức tín dụng 50,000,000 VND cho họ."})
            st.rerun()
        if q_cols[2].button("🟢 3. Low-Risk (Bot AUTO-EXECUTE)", use_container_width=True, help="Khách hàng an toàn -> Bot gắn thẻ Tầng 1 và xác nhận tự động thực thi"):
            st.session_state.chat_messages.append({"role": "user", "content": "Khách hàng ECOMM_10001 có mức độ gắn bó cao, không có khiếu nại. Hãy gửi email chăm sóc định kỳ."})
            st.rerun()

        # Hiển thị lịch sử chat
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Input gửi tin nhắn
        user_input = st.chat_input("Nhập câu hỏi cho AI về khách hàng hoặc luồng HITL...")
        
        # Xử lý khi có tin nhắn mới hoặc trigger từ gợi ý
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.rerun()

        if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
            api_key_chat = os.getenv("OPENAI_API_KEY")
            if not api_key_chat or api_key_chat == "your_openai_api_key_here":
                st.error("⚠️ Chưa tìm thấy `OPENAI_API_KEY`! Vui lòng cấu hình API Key trong file `.env` của dự án.")
            else:
                with st.spinner("🤖 AI đang suy nghĩ và tạo câu trả lời..."):
                    try:
                        reply, trace = chat_with_retention_ai(
                            messages=st.session_state.chat_messages,
                            api_key=api_key_chat,
                            model="gpt-4o-mini",
                        )
                    except Exception as exc:
                        st.error(f"Lỗi gọi AI: {exc}")
                    else:
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                        st.rerun()

        if st.button("🧹 Xóa đoạn chat", use_container_width=True):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Đoạn chat đã được làm mới. Tôi sẵn sàng hỗ trợ bạn!"}
            ]
            st.rerun()

    with trace_col:
        st.markdown("#### 🔬 Bảng kiểm tra Trace Logs")
        trace_logs = load_trace_logs()
        
        if trace_logs:
            total_traces = len(trace_logs)
            avg_latency = round(sum(t.get("latency_ms", 0) for t in trace_logs) / total_traces, 1)
            total_tokens = sum(t.get("total_tokens", 0) or 0 for t in trace_logs)

            t1, t2, t3 = st.columns(3)
            t1.metric("Tổng lượt trace", total_traces)
            t2.metric("Độ trễ TB", f"{avg_latency} ms")
            t3.metric("Tổng Tokens", total_tokens)

            # Danh sách Trace logs gần nhất
            st.write("**Danh sách Trace gần nhất:**")
            trace_table_data = [
                {
                    "Thời gian": t.get("timestamp", "")[:19].replace("T", " "),
                    "Model": t.get("model", ""),
                    "Câu hỏi": (t.get("user_query", "")[:35] + "...") if len(t.get("user_query", "")) > 35 else t.get("user_query", ""),
                    "Độ trễ (ms)": t.get("latency_ms", 0),
                    "Tokens": t.get("total_tokens", "N/A"),
                }
                for t in reversed(trace_logs[-15:])
            ]
            st.dataframe(trace_table_data, use_container_width=True, hide_index=True)

            with st.expander("🔍 Xem chi tiết Payload Trace Log mới nhất"):
                st.json(trace_logs[-1])

            if st.button("🗑️ Xóa toàn bộ Trace Logs", use_container_width=True):
                clear_trace_logs()
                st.rerun()
        else:
            st.info("Chưa có trace log nào. Hãy gửi tin nhắn ở khung chat bên trái để tạo trace đầu tiên!")

with help_tab:
    st.subheader("Kiểm tra nhanh")
    st.markdown(
        """
1. `increase_credit_limit` luôn dừng review, kể cả confidence rất cao.
2. `send_email` với confidence từ `0.85` tự chạy; thấp hơn phải dừng review.
3. Khi pending, snapshot còn customer/action/confidence/reasoning và `next` là high-risk node.
4. Approve/Edit thực thi một lần; Reject hủy; mỗi quyết định thêm một audit entry.
5. Chạy tự động: `python -m pytest -q --cov=graph --cov=models --cov-report=term-missing`.

`MemorySaver` chỉ giữ checkpoint trong RAM. Khởi động lại Streamlit sẽ mất pending state;
production nên dùng SQLite/PostgreSQL checkpointer và append-only database cho audit.
"""
    )
