"""
Detects inconsistencies between the live system's belief and the
warehouse's reported record, classifies severity, and decides + executes a
corrective action.

Two severities, two different cost profiles -- another cost-saving
decision worth defending in the write-up:

  * MINOR drift (e.g. a 1-2 unit shipment shortfall, a few cents of payment
    rounding) is auto-healed deterministically: patch the belief, log a
    correction record, done. No LLM call. Spending tokens to have a model
    "explain" that 1 item out of 40 shipped short should be auto-corrected
    is not worth it.
  * MAJOR drift (missing shipment on a paid order, a payment that settled
    for meaningfully less than expected, a shipment dispatched against a
    cancelled order) gets an LLM call to produce a human-readable
    reconciliation rationale, because a person may need to review it and
    "warehouse said so" is not an adequate audit trail on its own for a
    financially significant mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .system_state import OrderBelief

MINOR_ITEM_DELTA = 2
MINOR_PAYMENT_DELTA_CENTS = 100  # $1.00


@dataclass
class Mismatch:
    order_id: str
    field: str
    system_value: Any
    warehouse_value: Any
    severity: str  # "minor" | "major"
    description: str


@dataclass
class Correction:
    order_id: str
    field: str
    old_value: Any
    new_value: Any
    rationale: str
    reviewed_by_llm: bool


def detect(order_id: str, belief: OrderBelief, warehouse_record: Optional[Dict[str, Any]]) -> List[Mismatch]:
    mismatches: List[Mismatch] = []
    if warehouse_record is None or warehouse_record.get("found") is False:
        # Nothing exported yet for this order -- not a mismatch, just not
        # observable. Common right after order_created.
        return mismatches

    wr = warehouse_record

    if belief.items_expected and wr["items_shipped"] != belief.items_expected:
        delta = abs(wr["items_shipped"] - belief.items_expected)
        severity = "minor" if delta <= MINOR_ITEM_DELTA else "major"
        mismatches.append(
            Mismatch(
                order_id=order_id,
                field="items_shipped",
                system_value=belief.items_expected,
                warehouse_value=wr["items_shipped"],
                severity=severity,
                description=(
                    f"System expects {belief.items_expected} items shipped; "
                    f"warehouse export shows {wr['items_shipped']} (delta={delta})."
                ),
            )
        )

    if belief.payment_processed and wr["payment_settled_cents"] != belief.payment_expected_cents:
        delta = abs(wr["payment_settled_cents"] - belief.payment_expected_cents)
        severity = "minor" if delta <= MINOR_PAYMENT_DELTA_CENTS else "major"
        mismatches.append(
            Mismatch(
                order_id=order_id,
                field="payment_settled_cents",
                system_value=belief.payment_expected_cents,
                warehouse_value=wr["payment_settled_cents"],
                severity=severity,
                description=(
                    f"System expects {belief.payment_expected_cents}c settled; "
                    f"warehouse export shows {wr['payment_settled_cents']}c (delta={delta}c)."
                ),
            )
        )

    if belief.shipment_status == "dispatched" and wr["shipment_status"] not in ("dispatched",):
        mismatches.append(
            Mismatch(
                order_id=order_id,
                field="shipment_status",
                system_value=belief.shipment_status,
                warehouse_value=wr["shipment_status"],
                severity="major",  # a status mismatch is never "minor"
                description=(
                    f"System believes shipment is 'dispatched'; warehouse export shows "
                    f"'{wr['shipment_status']}'."
                ),
            )
        )

    if belief.cancelled and wr["shipment_status"] == "dispatched":
        mismatches.append(
            Mismatch(
                order_id=order_id,
                field="shipment_status",
                system_value="cancelled",
                warehouse_value="dispatched",
                severity="major",
                description="Order was cancelled in the live system but warehouse shipped it anyway.",
            )
        )

    return mismatches


def build_reconciliation_prompt(mismatch: Mismatch) -> str:
    return (
        "You are a reconciliation agent for an order-management system. "
        f"Order {mismatch.order_id}: field '{mismatch.field}' -- system believed "
        f"'{mismatch.system_value}', warehouse export reports '{mismatch.warehouse_value}'. "
        f"Severity classified as {mismatch.severity}. Explain, in two sentences, whether the "
        "warehouse export should be treated as ground truth here, what corrective action to take, "
        "and what would need manual review."
    )
