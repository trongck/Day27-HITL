import pytest
from pydantic import ValidationError

from graph import _validate_payload, evaluate_customer, execute_low_risk_action, route_action
from models import AgentEvaluation, CustomerInput, GraphState


@pytest.mark.parametrize(
    ("action", "confidence", "expected"),
    [
        ("send_email", 0.91, "execute_low_risk_action"),
        ("send_email", 0.85, "execute_low_risk_action"),
        ("send_email", 0.8499, "execute_high_risk_action"),
        ("increase_credit_limit", 0.20, "execute_high_risk_action"),
        ("increase_credit_limit", 0.99, "execute_high_risk_action"),
        ("unknown_action", 0.99, "execute_high_risk_action"),
    ],
)
def test_route_action_is_fail_closed(action, confidence, expected):
    state = {"proposed_action": action, "confidence_score": confidence}
    assert route_action(state) == expected


def test_agent_reasoning_is_deterministic_and_complete():
    state: GraphState = {
        "customer_id": "CUST001",
        "toi": "high",
        "churn_probability": 0.25,
        "action_payload": {},
        "proposed_action": "pending",
        "confidence_score": 0.0,
        "reasoning": "pending",
        "human_decision": None,
    }
    first = evaluate_customer(state)
    second = evaluate_customer(state)
    assert first == second
    assert first["proposed_action"] == "send_email"
    assert first["confidence_score"] == 0.92
    assert "TOI=high" in first["reasoning"]
    assert "churn_probability=0.25" in first["reasoning"]


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (AgentEvaluation, {"proposed_action": "send_email", "confidence_score": 1.01, "reasoning": "x"}),
        (AgentEvaluation, {"proposed_action": "send_email", "confidence_score": -0.01, "reasoning": "x"}),
        (CustomerInput, {"customer_id": "C1", "toi": "high", "churn_probability": 1.01}),
    ],
)
def test_confidence_and_probability_validation(model, values):
    with pytest.raises(ValidationError):
        model(**values)


def test_low_risk_node_has_its_own_safety_guard():
    blocked = execute_low_risk_action(
        {"proposed_action": "increase_credit_limit", "confidence_score": 0.99}
    )
    assert blocked["execution_status"] == "blocked"
    assert blocked["executed_payload"] is None


def test_payload_validation_covers_each_action_and_fail_closed_default():
    assert _validate_payload("send_email", {"template": "  offer  "}) == {"template": "offer"}
    with pytest.raises(ValueError, match="email template"):
        _validate_payload("send_email", {"template": ""})
    with pytest.raises(ValueError, match="numeric"):
        _validate_payload("increase_credit_limit", {"amount": True})
    with pytest.raises(ValueError, match="unknown"):
        _validate_payload("other", {})
