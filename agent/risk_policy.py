"""
Cost-aware risk policy: decides, per event, whether it's worth spending a
warehouse API call (and possibly LLM tokens) to check for drift.

This is the "should not check the warehouse on every event" requirement.
The policy is two-tiered on purpose:

  1. A free, deterministic heuristic scorer (base weight by event type +
     contextual adjustments) runs on *every* event. This costs zero tokens
     and zero API calls.
  2. Only when the heuristic lands in an AMBIGUOUS band does the agent
     spend an LLM call to reason about whether to escalate. Events the
     heuristic is already confident about (clearly low-risk or clearly
     high-risk) never touch the LLM -- there's nothing for a model to add
     over a threshold check, and every avoided call is tokens saved.

Base weights encode a simple idea: events that touch money or physical
fulfillment (payment, shipment) are inherently riskier to get wrong than
events that only set up an expectation (order created) or reduce scope
(cancellation, usually).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .events import Event, EventType
from .system_state import OrderBelief

BASE_RISK: Dict[EventType, float] = {
    EventType.ORDER_CREATED: 0.10,
    EventType.PAYMENT_PROCESSED: 0.70,
    EventType.SHIPMENT_DISPATCHED: 0.75,
    EventType.INVENTORY_ADJUSTED: 0.50,
    EventType.ORDER_CANCELLED: 0.35,
}

SKIP_BELOW = 0.30
QUERY_ABOVE = 0.65  # heuristic is confident enough to skip the LLM entirely


class Decision(str, Enum):
    SKIP = "skip"                 # no query, no LLM -- heuristic confident it's low risk
    LLM_ESCALATE = "llm_escalate"  # ambiguous -- spend an LLM call to decide
    QUERY = "query"                # heuristic confident it's high risk -- queue a warehouse check, no LLM needed


@dataclass
class RiskScore:
    event: Event
    score: float
    reasons: list


def score_event(event: Event, belief: OrderBelief, recent_high_risk_orders: set) -> RiskScore:
    reasons = []
    score = BASE_RISK.get(event.type, 0.4)
    reasons.append(f"base weight for {event.type.value} = {score:.2f}")

    # Contextual escalation: a shipment dispatched *after* the order was
    # cancelled is a strong signal something is wrong -- max it out.
    if event.type == EventType.SHIPMENT_DISPATCHED and belief.cancelled:
        score = 0.95
        reasons.append("shipment dispatched on a cancelled order -> escalated to 0.95")

    # Contextual escalation: a second high-risk event on the same order in
    # a short window compounds risk (e.g. payment processed, then inventory
    # adjusted before shipment) -- more moving parts, more chance of drift.
    if event.order_id in recent_high_risk_orders and score >= SKIP_BELOW:
        score = min(1.0, score + 0.15)
        reasons.append("second risky event on this order recently -> +0.15")

    # Contextual de-escalation: order_created is inherently low risk because
    # there is nothing yet to have drifted -- the warehouse can't disagree
    # with an expectation that was only just set.
    if event.type == EventType.ORDER_CREATED:
        reasons.append("order_created establishes expectation only, nothing to reconcile yet")

    return RiskScore(event=event, score=score, reasons=reasons)


def decide(risk: RiskScore) -> Decision:
    if risk.score < SKIP_BELOW:
        return Decision.SKIP
    if risk.score >= QUERY_ABOVE:
        return Decision.QUERY
    return Decision.LLM_ESCALATE
