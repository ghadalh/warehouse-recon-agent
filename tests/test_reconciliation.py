from agent.reconciliation import detect
from agent.system_state import OrderBelief


def test_no_warehouse_record_yet_is_not_a_mismatch():
    belief = OrderBelief(order_id="A", items_expected=3)
    mismatches = detect("A", belief, {"found": False})
    assert mismatches == []


def test_matching_record_has_no_mismatch():
    belief = OrderBelief(
        order_id="A", items_expected=3, payment_expected_cents=1000,
        payment_processed=True, shipment_status="dispatched",
    )
    wr = {
        "items_shipped": 3, "payment_settled_cents": 1000,
        "shipment_status": "dispatched",
    }
    assert detect("A", belief, wr) == []


def test_small_item_delta_is_classified_minor():
    belief = OrderBelief(order_id="A", items_expected=10)
    wr = {"items_shipped": 9, "payment_settled_cents": 0, "shipment_status": "not_dispatched"}
    mismatches = detect("A", belief, wr)
    assert len(mismatches) == 1
    assert mismatches[0].severity == "minor"


def test_large_item_delta_is_classified_major():
    belief = OrderBelief(order_id="A", items_expected=10)
    wr = {"items_shipped": 2, "payment_settled_cents": 0, "shipment_status": "not_dispatched"}
    mismatches = detect("A", belief, wr)
    assert len(mismatches) == 1
    assert mismatches[0].severity == "major"


def test_dispatched_belief_vs_not_dispatched_warehouse_is_always_major():
    belief = OrderBelief(order_id="A", shipment_status="dispatched")
    wr = {"items_shipped": 0, "payment_settled_cents": 0, "shipment_status": "not_dispatched"}
    mismatches = detect("A", belief, wr)
    assert any(m.field == "shipment_status" and m.severity == "major" for m in mismatches)


def test_cancelled_order_shipped_anyway_is_flagged():
    belief = OrderBelief(order_id="A", cancelled=True, shipment_status="not_dispatched")
    wr = {"items_shipped": 1, "payment_settled_cents": 0, "shipment_status": "dispatched"}
    mismatches = detect("A", belief, wr)
    assert any("cancelled" in m.description.lower() for m in mismatches)
    assert all(m.severity == "major" for m in mismatches)


def test_small_payment_rounding_is_minor():
    belief = OrderBelief(order_id="A", payment_processed=True, payment_expected_cents=1000)
    wr = {"items_shipped": 0, "payment_settled_cents": 970, "shipment_status": "not_dispatched"}
    mismatches = detect("A", belief, wr)
    assert len(mismatches) == 1
    assert mismatches[0].severity == "minor"
