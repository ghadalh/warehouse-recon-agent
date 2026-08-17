"""
Event models for the operational event stream.

The agent receives these events in real time from the "live system" (an
order-management service, in a real deployment). Each event type carries a
different, hand-picked risk weight in the risk policy -- see risk_policy.py
for why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    ORDER_CREATED = "order_created"
    PAYMENT_PROCESSED = "payment_processed"
    SHIPMENT_DISPATCHED = "shipment_dispatched"
    INVENTORY_ADJUSTED = "inventory_adjusted"  # 4th type, beyond the minimum 3
    ORDER_CANCELLED = "order_cancelled"        # 5th type, used to test batching/no-op paths


@dataclass
class Event:
    seq: int
    type: EventType
    order_id: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Event(#{self.seq} {self.type.value} order={self.order_id} {self.payload})"
