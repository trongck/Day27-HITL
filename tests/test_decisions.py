import uuid

import pytest

import graph
from graph import execute_high_risk_action, start_customer_workflow, submit_human_decision
from models import HumanDecision


@pytest.fixture
def captured_audit(monkeypatch):
    entries = []
    monkeypatch.setattr(graph, "save_audit_entry", lambda entry: entries.append(entry) or entry)
    return entries


def start_pending(churn=0.82):
    thread_id = f"decision-{uuid.uuid4().hex}"
    start_customer_workflow(
        customer_id="CUST-REVIEW",
        toi="medium",
        churn_probability=churn,
        thread_id=thread_id,
    )
    return thread_id


def test_approve_resumes_and_executes_once(captured_audit):
    thread_id = start_pending()
    result = submit_human_decision(
        thread_id, HumanDecision(action="approve", reviewer_id="reviewer-1")
    )
    assert result["execution_status"] == "completed"
    assert result["executed_payload"] == {"amount": 50_000_000}
    assert captured_audit[0].decision == "approve"
    with pytest.raises(ValueError, match="already processed"):
        submit_human_decision(
            thread_id, HumanDecision(action="approve", reviewer_id="reviewer-1")
        )
    assert len(captured_audit) == 1


def test_reject_resumes_and_aborts(captured_audit):
    result = submit_human_decision(
        start_pending(),
        HumanDecision(action="reject", reviewer_id="reviewer-2", reason="Insufficient evidence"),
    )
    assert result["execution_status"] == "aborted"
    assert result["executed_payload"] is None
    assert captured_audit[0].decision == "reject"


def test_edit_executes_only_valid_new_payload_and_audits_it(captured_audit):
    result = submit_human_decision(
        start_pending(),
        HumanDecision(
            action="edit",
            reviewer_id="reviewer-3",
            edited_payload={"amount": 20_000_000},
            reason="Reduce exposure",
        ),
    )
    assert result["execution_status"] == "completed"
    assert result["action_payload"] == {"amount": 20_000_000}
    assert result["executed_payload"] == {"amount": 20_000_000}
    assert captured_audit[0].action_payload == {"amount": 20_000_000}
    assert captured_audit[0].decision == "edit"


def test_invalid_edit_and_missing_decision_never_execute(captured_audit):
    state = {
        "customer_id": "C1",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.99,
        "reasoning": "test",
        "human_decision": None,
        "action_payload": {"amount": 50_000_000},
    }
    assert execute_high_risk_action(state)["execution_status"] == "blocked"
    state["human_decision"] = HumanDecision(
        action="edit", reviewer_id="reviewer", edited_payload={"amount": -1}
    ).model_dump(mode="json")
    assert execute_high_risk_action(state)["execution_status"] == "blocked"
    assert len(captured_audit) == 1
    assert captured_audit[0].decision == "edit"


def test_audit_failure_blocks_execution(monkeypatch):
    monkeypatch.setattr(graph, "save_audit_entry", lambda entry: (_ for _ in ()).throw(OSError("disk full")))
    state = {
        "customer_id": "C1",
        "proposed_action": "increase_credit_limit",
        "confidence_score": 0.99,
        "reasoning": "test",
        "human_decision": HumanDecision(
            action="approve", reviewer_id="reviewer"
        ).model_dump(mode="json"),
        "action_payload": {"amount": 10_000_000},
    }
    result = execute_high_risk_action(state)
    assert result["execution_status"] == "blocked"
    assert result["executed_payload"] is None
    assert "Audit write failed" in result["execution_message"]


def test_submission_for_unknown_thread_is_rejected():
    with pytest.raises(ValueError, match="workflow not found"):
        submit_human_decision(
            f"missing-{uuid.uuid4().hex}",
            HumanDecision(action="approve", reviewer_id="reviewer"),
        )
