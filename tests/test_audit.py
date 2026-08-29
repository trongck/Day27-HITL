from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from models import (
    AuditEntry,
    HumanDecision,
    clear_audit_logs,
    get_thread_audit_logs,
    load_audit_logs,
    save_audit_entry,
)


def entry(decision, reviewer):
    return AuditEntry(
        timestamp=datetime.now(timezone.utc),
        agent_id="agent-1",
        action="increase_credit_limit",
        confidence=0.99,
        reviewer_id=reviewer,
        decision=decision,
        action_payload={"amount": 20_000_000},
    )


def test_audit_appends_without_overwriting_and_has_required_fields(tmp_path):
    path = tmp_path / "audit.json"
    save_audit_entry(entry("approve", "r1"), path)
    save_audit_entry(entry("reject", "r2"), path)
    save_audit_entry(entry("edit", "r3"), path)

    logs = load_audit_logs(path)
    assert [item["decision"] for item in logs] == ["approve", "reject", "edit"]
    required = {"timestamp", "agent_id", "action", "confidence", "reviewer_id", "decision"}
    assert all(required <= item.keys() for item in logs)
    assert json.loads(path.read_text(encoding="utf-8")) == logs


def test_audit_schema_rejects_missing_fields_and_naive_timestamp():
    with pytest.raises(ValidationError):
        AuditEntry(timestamp=datetime.now(), agent_id="agent")
    with pytest.raises(ValidationError, match="timezone"):
        entry_values = entry("approve", "r1").model_dump()
        entry_values["timestamp"] = datetime.now()
        AuditEntry(**entry_values)


def test_malformed_audit_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        save_audit_entry(entry("approve", "r1"), path)
    assert path.read_text(encoding="utf-8") == "not-json"


def test_human_decision_validation():
    with pytest.raises(ValidationError, match="reviewer_id"):
        HumanDecision(action="approve")
    with pytest.raises(ValidationError, match="rejection reason"):
        HumanDecision(action="reject", reviewer_id="r1")
    with pytest.raises(ValidationError, match="edited_payload"):
        HumanDecision(action="edit", reviewer_id="r1")


def test_legacy_decision_alias_and_audit_helpers(tmp_path):
    decision = HumanDecision(
        action="edit_and_approve",
        reviewer="legacy-reviewer",
        feedback="updated",
        edited_payload={"amount": 1},
    )
    assert decision.action == "edit"
    assert decision.reviewer_id == "legacy-reviewer"
    assert decision.reason == "updated"

    path = tmp_path / "audit.json"
    audit = entry("approve", "r1")
    audit.thread_id = "thread-1"
    save_audit_entry(audit, path)
    save_audit_entry(audit, path)
    assert len(get_thread_audit_logs("thread-1", path)) == 1
    clear_audit_logs(path)
    assert load_audit_logs(path) == []


def test_audit_requires_json_array(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_audit_logs(path)
