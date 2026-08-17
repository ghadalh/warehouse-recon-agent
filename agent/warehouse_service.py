"""
Mock warehouse service.

Stands in for a real warehouse management system (WMS). It exposes a tiny
HTTP API (via FastAPI) so the agent talks to it exactly the way it would
talk to a real warehouse endpoint -- real HTTP calls, real latency, real
request/response payloads -- which is what lets us *count* warehouse API
calls honestly instead of just incrementing a counter in Python.

The warehouse holds its own, independently-updated view of order state
(inventory counts, shipment records, payment settlement records). Normally
this is exported / synced periodically (hence "periodic warehouse data
exports" in the brief) and is a source of truth for what physically
happened in the warehouse -- which can drift from what the live system
*believes* happened.

For the demo we seed the warehouse with state that matches the live system,
then apply a handful of independent warehouse-side mutations (a partial
shipment, a payment that settled for less than expected) to simulate real
drift. The agent never sees these mutations directly -- it can only learn
about them by querying /warehouse/order/{id} or /warehouse/batch.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class WarehouseRecord(BaseModel):
    order_id: str
    items_shipped: int
    items_expected: int
    payment_settled_cents: int
    payment_expected_cents: int
    shipment_status: str  # "not_dispatched" | "dispatched" | "partial" | "returned"
    last_export_ts: float


class WarehouseService:
    """In-memory warehouse ground truth + call counters."""

    def __init__(self) -> None:
        self._records: Dict[str, WarehouseRecord] = {}
        self.lock = threading.Lock()
        self.call_count = 0          # total HTTP calls served (single + batch)
        self.order_lookups = 0       # total distinct order lookups served (batch counts as N)
        self.call_log: List[Dict[str, Any]] = []

    def seed(self, record: WarehouseRecord) -> None:
        self._records[record.order_id] = record

    def mutate(self, order_id: str, **changes: Any) -> None:
        """Simulate a warehouse-side event the live system doesn't know about yet."""
        rec = self._records[order_id]
        data = rec.model_dump()
        data.update(changes)
        data["last_export_ts"] = time.time()
        self._records[order_id] = WarehouseRecord(**data)

    def _lookup(self, order_id: str) -> Optional[WarehouseRecord]:
        with self.lock:
            self.call_count += 1
            self.order_lookups += 1
            rec = self._records.get(order_id)
            self.call_log.append({"op": "lookup", "order_id": order_id, "found": rec is not None})
            return rec

    def _batch_lookup(self, order_ids: List[str]) -> Dict[str, Optional[WarehouseRecord]]:
        with self.lock:
            self.call_count += 1  # ONE HTTP call regardless of batch size
            self.order_lookups += len(order_ids)
            out = {oid: self._records.get(oid) for oid in order_ids}
            self.call_log.append({"op": "batch_lookup", "order_ids": order_ids})
            return out


def build_app(service: WarehouseService) -> FastAPI:
    app = FastAPI(title="Mock Warehouse Service")

    @app.get("/warehouse/order/{order_id}")
    def get_order(order_id: str):
        rec = service._lookup(order_id)
        return rec.model_dump() if rec else {"order_id": order_id, "found": False}

    @app.post("/warehouse/batch")
    def batch(order_ids: List[str]):
        results = service._batch_lookup(order_ids)
        return {
            oid: (rec.model_dump() if rec else {"order_id": oid, "found": False})
            for oid, rec in results.items()
        }

    @app.get("/warehouse/_stats")
    def stats():
        return {"call_count": service.call_count, "order_lookups": service.order_lookups}

    return app


def run_in_thread(service: WarehouseService, port: int = 8799) -> threading.Thread:
    app = build_app(service)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run():
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # wait until the server is actually accepting connections
    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    return t
