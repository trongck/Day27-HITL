import uuid

import pytest

from graph import (
    get_current_workflow_state,
    start_customer_workflow,
    start_hitl_workflow,
    workflow_config,
)


def unique_thread(label):
    return f"test-{label}-{uuid.uuid4().hex}"


def test_high_risk_interrupt_preserves_state_before_execution():
    thread_id = unique_thread("hard-policy")
    snapshot = start_customer_workflow(
        customer_id="CUST002",
        toi="medium",
        churn_probability=0.82,
        thread_id=thread_id,
    )
    values = snapshot["values"]

    assert snapshot["next"] == ["execute_high_risk_action"]
    assert snapshot["is_paused"] is True
    assert values["execution_status"] == "evaluated"
    assert values["executed_payload"] is None
    assert values["customer_id"] == "CUST002"
    assert values["proposed_action"] == "increase_credit_limit"
    assert values["confidence_score"] == 0.96
    assert values["reasoning"]
    assert values["human_decision"] is None

    restored = get_current_workflow_state(thread_id)
    for field in ("customer_id", "proposed_action", "confidence_score", "reasoning"):
        assert restored["values"][field] == values[field]


def test_low_confidence_email_interrupts_but_high_confidence_auto_executes():
    pending = start_customer_workflow(
        customer_id="CUST003",
        toi="high",
        churn_probability=0.55,
        thread_id=unique_thread("low-confidence"),
    )
    assert pending["values"]["proposed_action"] == "send_email"
    assert pending["values"]["confidence_score"] == 0.82
    assert pending["is_paused"] is True

    automatic = start_customer_workflow(
        customer_id="CUST001",
        toi="high",
        churn_probability=0.25,
        thread_id=unique_thread("auto"),
    )
    assert automatic["next"] == []
    assert automatic["values"]["execution_status"] == "completed"
    assert automatic["values"]["executed_payload"] == {"template": "customer_care"}


def test_thread_ids_do_not_share_customer_state():
    first = start_customer_workflow(
        customer_id="ONE", toi="high", churn_probability=0.25, thread_id=unique_thread("one")
    )
    second = start_customer_workflow(
        customer_id="TWO", toi="medium", churn_probability=0.82, thread_id=unique_thread("two")
    )
    assert first["values"]["customer_id"] == "ONE"
    assert second["values"]["customer_id"] == "TWO"


def test_thread_guards_and_legacy_start_adapter():
    with pytest.raises(ValueError, match="thread_id"):
        workflow_config(" ")
    assert get_current_workflow_state(unique_thread("missing")) is None

    thread_id = unique_thread("legacy")
    values = start_hitl_workflow(thread_id, "HIGH", "0.25")
    assert values["customer_id"] == thread_id
    assert values["execution_status"] == "completed"
    with pytest.raises(ValueError, match="already exists"):
        start_customer_workflow(
            customer_id="duplicate", toi="high", churn_probability=0.25, thread_id=thread_id
        )
