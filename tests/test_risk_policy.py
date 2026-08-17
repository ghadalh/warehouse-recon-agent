import time

from agent.events import Event, EventType
from agent.risk_policy import Decision, decide, score_event
from agent.system_state import SystemState


def _belief(state, order_id):
    return state.get(order_id)


def test_order_created_is_low_risk_and_skipped():
    state = SystemState()
    e = Event(1, EventType.ORDER_CREATED, "A", time.time(), {"items": 1, "amount_cents": 100})
    belief = state.apply(e)
    risk = score_event(e, belief, set())
    assert risk.score < 0.30
    assert decide(risk) == Decision.SKIP


def test_payment_processed_is_high_risk_and_queried_without_llm():
    state = SystemState()
    e = Event(1, EventType.PAYMENT_PROCESSED, "A", time.time(), {})
    belief = state.apply(e)
    risk = score_event(e, belief, set())
    assert risk.score >= 0.65
    assert decide(risk) == Decision.QUERY


def test_inventory_adjusted_is_ambiguous_by_default():
    state = SystemState()
    e = Event(1, EventType.INVENTORY_ADJUSTED, "A", time.time(), {"new_items": 5})
    belief = state.apply(e)
    risk = score_event(e, belief, set())
    assert 0.30 <= risk.score < 0.65
    assert decide(risk) == Decision.LLM_ESCALATE


def test_shipment_after_cancellation_is_escalated_to_critical():
    state = SystemState()
    cancel = Event(1, EventType.ORDER_CANCELLED, "A", time.time(), {})
    state.apply(cancel)
    dispatch = Event(2, EventType.SHIPMENT_DISPATCHED, "A", time.time(), {})
    belief = state.apply(dispatch)
    risk = score_event(dispatch, belief, set())
    assert risk.score >= 0.9
    assert decide(risk) == Decision.QUERY


def test_repeat_high_risk_on_same_order_compounds_score():
    state = SystemState()
    e1 = Event(1, EventType.PAYMENT_PROCESSED, "A", time.time(), {})
    belief = state.apply(e1)
    risk1 = score_event(e1, belief, set())

    e2 = Event(2, EventType.SHIPMENT_DISPATCHED, "A", time.time(), {})
    belief2 = state.apply(e2)
    risk2 = score_event(e2, belief2, recent_high_risk_orders={"A"})

    base_only = score_event(e2, belief2, recent_high_risk_orders=set())
    assert risk2.score > base_only.score
