"""
The live system's *belief* about each order -- i.e. what the operational
system thinks happened, built purely from the events it has seen. This is
deliberately dumb and cheap: no LLM, no I/O, just a fold over the event
stream. It's the thing we reconcile against the warehouse's ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .events import Event, EventType


@dataclass
class OrderBelief:
    order_id: str
    items_expected: int = 0
    payment_expected_cents: int = 0
    payment_processed: bool = False
    shipment_status: str = "not_dispatched"
    cancelled: bool = False
    last_event_seq: int = -1


class SystemState:
    def __init__(self) -> None:
        self.orders: Dict[str, OrderBelief] = {}

    def get(self, order_id: str) -> OrderBelief:
        if order_id not in self.orders:
            self.orders[order_id] = OrderBelief(order_id=order_id)
        return self.orders[order_id]

    def apply(self, event: Event) -> OrderBelief:
        belief = self.get(event.order_id)
        belief.last_event_seq = event.seq

        if event.type == EventType.ORDER_CREATED:
            belief.items_expected = event.payload.get("items", 0)
            belief.payment_expected_cents = event.payload.get("amount_cents", 0)

        elif event.type == EventType.PAYMENT_PROCESSED:
            belief.payment_processed = True

        elif event.type == EventType.SHIPMENT_DISPATCHED:
            belief.shipment_status = "dispatched"

        elif event.type == EventType.INVENTORY_ADJUSTED:
            belief.items_expected = event.payload.get("new_items", belief.items_expected)

        elif event.type == EventType.ORDER_CANCELLED:
            belief.cancelled = True

        return belief
