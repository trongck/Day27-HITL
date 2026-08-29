"""Validated state, human-decision and append-only audit models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, NotRequired, TypedDict
import json
import os
import tempfile
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ActionName = Literal["send_email", "increase_credit_limit"]
DecisionName = Literal["approve", "reject", "edit"]


class GraphState(TypedDict):
    """LangGraph state. The first five fields are the acceptance-contract fields."""

    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: dict[str, Any] | None

    thread_id: NotRequired[str]
    toi: NotRequired[str]
    churn_probability: NotRequired[float]
    action_payload: NotRequired[dict[str, Any]]
    reviewer_id: NotRequired[str]
    execution_status: NotRequired[str]
    execution_message: NotRequired[str]
    executed_payload: NotRequired[dict[str, Any] | None]
    agent_id: NotRequired[str]


class CustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_id: str = Field(min_length=1, max_length=100)
    toi: Literal["low", "medium", "high"]
    churn_probability: float = Field(ge=0.0, le=1.0)
    action_payload: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    proposed_action: str = Field(min_length=1, max_length=100)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=2000)


class HumanDecision(BaseModel):
    """Validated decision supplied by Streamlit or another trusted reviewer UI."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["approve", "reject", "edit", "edit_and_approve"]
    reviewer_id: str | None = Field(default=None, max_length=100)
    reviewer: str | None = Field(default=None, max_length=100)  # legacy API alias
    reason: str | None = Field(default=None, max_length=1000)
    feedback: str | None = Field(default=None, max_length=1000)  # legacy API alias
    edited_payload: dict[str, Any] | None = None
    edited_content: str | None = Field(default=None, max_length=10_000)  # legacy API

    @model_validator(mode="after")
    def validate_decision(self) -> "HumanDecision":
        reviewer_id = self.reviewer_id or self.reviewer
        if not reviewer_id:
            raise ValueError("reviewer_id is required")
        object.__setattr__(self, "reviewer_id", reviewer_id)

        if self.action == "edit_and_approve":
            object.__setattr__(self, "action", "edit")
        reason = self.reason or self.feedback
        object.__setattr__(self, "reason", reason)

        if self.action == "reject" and not reason:
            raise ValueError("a rejection reason is required")
        if self.action == "edit" and self.edited_payload is None:
            raise ValueError("edited_payload is required for edit")
        return self


class AuditEntry(BaseModel):
    """One human decision. The six required rubric fields have no defaults."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    timestamp: datetime
    agent_id: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
    reviewer_id: str = Field(min_length=1, max_length=100)
    decision: DecisionName
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str | None = Field(default=None, max_length=100)
    thread_id: str | None = Field(default=None, max_length=200)
    action_payload: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = Field(default=None, max_length=1000)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


AUDIT_LOG_FILE = Path(__file__).with_name("audit_log.json")
_AUDIT_LOCK = Lock()


def load_audit_logs(filepath: str | os.PathLike[str] = AUDIT_LOG_FILE) -> list[dict[str, Any]]:
    """Read the complete audit history; malformed data is surfaced, never discarded."""

    path = Path(filepath)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("audit log must contain a JSON array")
    return data


def save_audit_entry(
    entry: AuditEntry,
    filepath: str | os.PathLike[str] = AUDIT_LOG_FILE,
) -> AuditEntry:
    """Append using a lock and atomic replace to avoid truncation/corruption."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
        logs = load_audit_logs(path)
        if any(item.get("id") == entry.id for item in logs):
            return entry
        logs.append(entry.to_dict())
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


def get_thread_audit_logs(
    thread_id: str,
    filepath: str | os.PathLike[str] = AUDIT_LOG_FILE,
) -> list[dict[str, Any]]:
    return [item for item in load_audit_logs(filepath) if item.get("thread_id") == thread_id]


def clear_audit_logs(filepath: str | os.PathLike[str] = AUDIT_LOG_FILE) -> None:
    """Legacy maintenance API. The workflow itself never clears audit history."""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
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


def new_audit_entry(**values: Any) -> AuditEntry:
    return AuditEntry(timestamp=datetime.now(timezone.utc), **values)
