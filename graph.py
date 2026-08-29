"""Customer-retention HITL workflow with fail-closed routing."""

from __future__ import annotations

from threading import Lock
from typing import Any
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

from models import (
    AgentEvaluation,
    CustomerInput,
    GraphState,
    HumanDecision,
    new_audit_entry,
    save_audit_entry,
)


AUTO_EXECUTE_THRESHOLD = 0.85
AGENT_ID = "customer-retention-agent"
MAX_CREDIT_LIMIT_INCREASE = 500_000_000


def evaluate_customer(state: GraphState) -> dict[str, Any]:
    """Return a deterministic, validated recommendation for lab/demo repeatability."""

    customer = CustomerInput(
        customer_id=state["customer_id"],
        toi=state.get("toi", "medium"),
        churn_probability=state.get("churn_probability", 0.0),
        action_payload=state.get("action_payload", {}),
    )

    if customer.churn_probability >= 0.75:
        action = "increase_credit_limit"
        confidence = 0.96
        default_payload = {"amount": 50_000_000}
        explanation = "high churn probability requires a stronger retention incentive"
    elif customer.churn_probability >= 0.40:
        action = "send_email"
        confidence = 0.82
        default_payload = {"template": "retention_offer"}
        explanation = "moderate churn signal makes outreach useful but needs review"
    else:
        action = "send_email"
        confidence = 0.92
        default_payload = {"template": "customer_care"}
        explanation = "low churn signal and non-financial outreach are low risk"

    # TOI is an independent business signal, not merely text echoed in the
    # explanation. It calibrates the recommendation confidence before routing.
    toi_adjustment = {"low": -0.02, "medium": 0.0, "high": 0.02}[customer.toi]
    confidence = round(max(0.0, min(1.0, confidence + toi_adjustment)), 2)
    adjustment_text = f"TOI={customer.toi} adjusts confidence by {toi_adjustment:+.2f}"

    evaluation = AgentEvaluation(
        proposed_action=action,
        confidence_score=confidence,
        reasoning=(
            f"Customer {customer.customer_id} has TOI={customer.toi} and "
            f"churn_probability={customer.churn_probability:.2f}; {explanation}; "
            f"{adjustment_text}."
        ),
    )
    payload = customer.action_payload or default_payload
    return {
        **evaluation.model_dump(),
        "action_payload": payload,
        "agent_id": AGENT_ID,
        "execution_status": "evaluated",
        "execution_message": "Agent evaluation completed.",
    }


def route_action(state: GraphState) -> str:
    """Hard policy first, confidence second, and unknown actions fail closed."""

    action = state.get("proposed_action", "")
    confidence = float(state.get("confidence_score", 0.0))

    if action == "increase_credit_limit":
        return "execute_high_risk_action"
    if action == "send_email" and confidence >= AUTO_EXECUTE_THRESHOLD:
        return "execute_low_risk_action"
    return "execute_high_risk_action"


def execute_low_risk_action(state: GraphState) -> dict[str, Any]:
    """Execute only the narrowly allowed low-risk action."""

    if not (
        state.get("proposed_action") == "send_email"
        and AUTO_EXECUTE_THRESHOLD <= float(state.get("confidence_score", 0.0)) <= 1.0
    ):
        return {
            "execution_status": "blocked",
            "execution_message": "Low-risk guard rejected an invalid or unsafe action.",
            "executed_payload": None,
        }
    return {
        "execution_status": "completed",
        "execution_message": "Retention email sent successfully (simulated).",
        "executed_payload": dict(state.get("action_payload", {})),
    }


def _validate_payload(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    if action == "increase_credit_limit":
        amount = payload.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("credit-limit amount must be numeric")
        if amount <= 0 or amount > MAX_CREDIT_LIMIT_INCREASE:
            raise ValueError(
                f"credit-limit amount must be between 0 and {MAX_CREDIT_LIMIT_INCREASE:,} VND"
            )
        return {"amount": amount}
    if action == "send_email":
        template = payload.get("template")
        if not isinstance(template, str) or not template.strip() or len(template) > 10_000:
            raise ValueError("email template must be a non-empty string up to 10,000 characters")
        return {"template": template.strip()}
    raise ValueError("unknown actions cannot be executed")


def execute_high_risk_action(state: GraphState) -> dict[str, Any]:
    """Second safety layer: no execution without a valid, audited human decision."""

    raw_decision = state.get("human_decision")
    try:
        decision = HumanDecision.model_validate(raw_decision)
    except Exception:
        return {
            "execution_status": "blocked",
            "execution_message": "A valid human decision is required before execution.",
            "executed_payload": None,
        }

    action = state.get("proposed_action", "")
    payload = dict(state.get("action_payload", {}))
    if decision.action == "edit":
        payload = dict(decision.edited_payload or {})

    validation_error: ValueError | None = None
    if decision.action != "reject":
        try:
            payload = _validate_payload(action, payload)
        except ValueError as exc:
            validation_error = exc

    try:
        save_audit_entry(
            new_audit_entry(
                agent_id=state.get("agent_id", AGENT_ID),
                action=action,
                confidence=float(state.get("confidence_score", 0.0)),
                reviewer_id=decision.reviewer_id,
                decision=decision.action,
                customer_id=state.get("customer_id"),
                thread_id=state.get("thread_id"),
                action_payload=payload,
                reason=decision.reason,
            )
        )
    except Exception as exc:
        return {
            "execution_status": "blocked",
            "execution_message": f"Audit write failed; action was not executed: {exc}",
            "executed_payload": None,
        }

    if validation_error is not None:
        return {
            "execution_status": "blocked",
            "execution_message": str(validation_error),
            "executed_payload": None,
        }
    if decision.action == "reject":
        return {
            "action_payload": payload,
            "execution_status": "aborted",
            "execution_message": f"Action rejected: {decision.reason}",
            "executed_payload": None,
        }
    return {
        "action_payload": payload,
        "execution_status": "completed",
        "execution_message": f"{action} executed after human {decision.action} (simulated).",
        "executed_payload": payload,
    }


def build_hitl_graph():
    builder = StateGraph(GraphState)
    builder.add_node("evaluate_customer", evaluate_customer)
    builder.add_node("execute_low_risk_action", execute_low_risk_action)
    builder.add_node("execute_high_risk_action", execute_high_risk_action)
    builder.add_edge(START, "evaluate_customer")
    builder.add_conditional_edges(
        "evaluate_customer",
        route_action,
        {
            "execute_low_risk_action": "execute_low_risk_action",
            "execute_high_risk_action": "execute_high_risk_action",
        },
    )
    builder.add_edge("execute_low_risk_action", END)
    builder.add_edge("execute_high_risk_action", END)

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["execute_high_risk_action"],
    )
    return graph


hitl_graph = build_hitl_graph()
_SUBMISSION_LOCK = Lock()


def workflow_config(thread_id: str) -> dict[str, dict[str, str]]:
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id is required")
    return {"configurable": {"thread_id": thread_id.strip()}}


def start_customer_workflow(
    *,
    customer_id: str,
    toi: str,
    churn_probability: float,
    action_payload: dict[str, Any] | None = None,
    thread_id: str | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    customer = CustomerInput(
        customer_id=customer_id,
        toi=toi,
        churn_probability=churn_probability,
        action_payload=action_payload or {},
    )
    resolved_thread_id = thread_id or f"customer-{customer.customer_id}-{uuid.uuid4().hex}"
    config = workflow_config(resolved_thread_id)
    workflow_graph = graph or hitl_graph
    existing = workflow_graph.get_state(config)
    if existing and existing.values:
        raise ValueError("thread_id already exists; create a unique workflow thread")

    initial_state: GraphState = {
        "customer_id": customer.customer_id,
        "proposed_action": "pending_evaluation",
        "confidence_score": 0.0,
        "reasoning": "Pending agent evaluation.",
        "human_decision": None,
        "thread_id": resolved_thread_id,
        "toi": customer.toi,
        "churn_probability": customer.churn_probability,
        "action_payload": customer.action_payload,
        "reviewer_id": "",
        "execution_status": "pending",
        "execution_message": "Workflow started.",
        "executed_payload": None,
        "agent_id": AGENT_ID,
    }
    workflow_graph.invoke(initial_state, config=config)
    return get_current_workflow_state(resolved_thread_id, graph=workflow_graph) or {}


def submit_human_decision(
    thread_id: str,
    decision: HumanDecision | dict[str, Any],
    graph: Any | None = None,
) -> dict[str, Any]:
    validated = HumanDecision.model_validate(decision)
    config = workflow_config(thread_id)
    workflow_graph = graph or hitl_graph
    with _SUBMISSION_LOCK:
        snapshot = workflow_graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise ValueError(f"workflow not found for thread_id: {thread_id}")
        if tuple(snapshot.next) != ("execute_high_risk_action",):
            raise ValueError("workflow is not pending human review or was already processed")

        state_update = {
            "human_decision": validated.model_dump(mode="json"),
            "reviewer_id": validated.reviewer_id,
        }
        workflow_graph.update_state(config, state_update)
        workflow_graph.invoke(None, config=config)
        final = workflow_graph.get_state(config)
        return dict(final.values)


def get_current_workflow_state(
    thread_id: str,
    graph: Any | None = None,
) -> dict[str, Any] | None:
    workflow_graph = graph or hitl_graph
    snapshot = workflow_graph.get_state(workflow_config(thread_id))
    if not snapshot or not snapshot.values:
        return None
    return {
        "values": dict(snapshot.values),
        "next": list(snapshot.next),
        "is_paused": bool(snapshot.next),
        "thread_id": thread_id,
    }


# Backward-compatible public names retained for existing callers.
def submit_human_approval(thread_id: str, decision: HumanDecision) -> dict[str, Any]:
    return submit_human_decision(thread_id, decision)


def start_hitl_workflow(
    thread_id: str,
    task_type: str,
    prompt: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Legacy adapter; new code should call ``start_customer_workflow``."""

    del api_key
    normalized_toi = task_type.lower() if task_type.lower() in {"low", "medium", "high"} else "medium"
    try:
        churn_probability = float(prompt)
    except (TypeError, ValueError):
        churn_probability = 0.5
    result = start_customer_workflow(
        customer_id=thread_id,
        toi=normalized_toi,
        churn_probability=churn_probability,
        thread_id=thread_id,
    )
    return result["values"]
