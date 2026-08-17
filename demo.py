#!/usr/bin/env python3
"""
Demo: streams a realistic sequence of operational events through the
reconciliation agent against a mock warehouse service, deliberately seeded
with two real inconsistencies (a shipment marked dispatched in the live
system that the warehouse never actually received, and a shipment that
went out for an order that had just been cancelled), plus one benign
rounding discrepancy that should be auto-healed. Prints a full decision
trace and a final cost report.

Run:  python3 demo.py
Options:
  --llm anthropic   use a real Claude API call instead of the deterministic
                     stand-in (requires `pip install anthropic` and
                     ANTHROPIC_API_KEY in the environment)
  --json out.json   also write the full cost report as JSON
"""
from __future__ import annotations

import argparse
import sys
import time

from agent.agent_core import ReconciliationAgent
from agent.cost_tracker import CostTracker
from agent.events import Event, EventType
from agent.llm_client import build_llm_client
from agent.system_state import SystemState
from agent.warehouse_service import WarehouseRecord, WarehouseService, run_in_thread

WAREHOUSE_PORT = 8799
WAREHOUSE_URL = f"http://127.0.0.1:{WAREHOUSE_PORT}"


def seed_warehouse(service: WarehouseService) -> None:
    now = time.time()
    # ORD-1: live system will believe items=3, paid, and (incorrectly)
    # dispatched. Warehouse ground truth: still NOT dispatched -- the
    # physical shipment never actually left the building even though the
    # live system's dispatch event fired. This is the headline
    # inconsistency the demo detects and corrects.
    service.seed(WarehouseRecord(
        order_id="ORD-1", items_shipped=3, items_expected=3,
        payment_settled_cents=4999, payment_expected_cents=4999,
        shipment_status="not_dispatched", last_export_ts=now,
    ))
    # ORD-2: everything matches once inventory is adjusted mid-stream.
    service.seed(WarehouseRecord(
        order_id="ORD-2", items_shipped=2, items_expected=2,
        payment_settled_cents=1200, payment_expected_cents=1200,
        shipment_status="not_dispatched", last_export_ts=now,
    ))
    # ORD-3: shipment matches; payment settles 50c short (card processor
    # rounding) -- a genuinely minor discrepancy that should be auto-healed
    # without spending an LLM call on it.
    service.seed(WarehouseRecord(
        order_id="ORD-3", items_shipped=5, items_expected=5,
        payment_settled_cents=8750, payment_expected_cents=8800,
        shipment_status="dispatched", last_export_ts=now,
    ))
    # ORD-4: gets cancelled in the live system, but a race condition means
    # the warehouse ships it anyway. Second real inconsistency, and an edge
    # case beyond the minimum ask (cancel-after-dispatch race).
    service.seed(WarehouseRecord(
        order_id="ORD-4", items_shipped=2, items_expected=2,
        payment_settled_cents=2500, payment_expected_cents=2500,
        shipment_status="dispatched", last_export_ts=now,
    ))


def build_event_stream() -> list:
    E = EventType
    return [
        Event(1, E.ORDER_CREATED, "ORD-1", time.time(), {"items": 3, "amount_cents": 4999}),
        Event(2, E.ORDER_CREATED, "ORD-2", time.time(), {"items": 1, "amount_cents": 1200}),
        Event(3, E.ORDER_CREATED, "ORD-4", time.time(), {"items": 2, "amount_cents": 2500}),
        Event(4, E.PAYMENT_PROCESSED, "ORD-1", time.time(), {}),
        Event(5, E.ORDER_CREATED, "ORD-3", time.time(), {"items": 5, "amount_cents": 8800}),
        Event(6, E.INVENTORY_ADJUSTED, "ORD-2", time.time(), {"new_items": 2}),
        Event(7, E.PAYMENT_PROCESSED, "ORD-2", time.time(), {}),
        Event(8, E.SHIPMENT_DISPATCHED, "ORD-1", time.time(), {}),
        Event(9, E.SHIPMENT_DISPATCHED, "ORD-3", time.time(), {}),
        Event(10, E.ORDER_CANCELLED, "ORD-4", time.time(), {}),
        Event(11, E.SHIPMENT_DISPATCHED, "ORD-4", time.time(), {}),
        Event(12, E.PAYMENT_PROCESSED, "ORD-3", time.time(), {}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="deterministic", choices=["deterministic", "anthropic"])
    parser.add_argument("--json", default=None, help="write full cost report JSON to this path")
    parser.add_argument("--quiet-server", action="store_true", default=True)
    args = parser.parse_args()

    print("Starting mock warehouse service on http://127.0.0.1:%d ..." % WAREHOUSE_PORT)
    warehouse = WarehouseService()
    seed_warehouse(warehouse)
    run_in_thread(warehouse, port=WAREHOUSE_PORT)
    time.sleep(0.3)

    llm = build_llm_client(args.llm)
    print(f"LLM backend: {llm.__class__.__name__} (model={getattr(llm, 'model_name', '?')})\n")

    state = SystemState()
    cost = CostTracker()
    agent = ReconciliationAgent(
        system_state=state,
        llm_client=llm,
        cost_tracker=cost,
        warehouse_base_url=WAREHOUSE_URL,
    )

    print("Streaming events into the agent")
    print("-" * 72)
    for event in build_event_stream():
        agent.process_event(event)
    agent.finalize()
    print("-" * 72)

    print("\nCORRECTIONS APPLIED")
    print("-" * 72)
    if not agent.corrections:
        print("(none)")
    for c in agent.corrections:
        reviewer = "LLM-reviewed" if c.reviewed_by_llm else "auto-healed"
        print(f"order={c.order_id} field={c.field}: {c.old_value!r} -> {c.new_value!r}  [{reviewer}]")
        print(f"    rationale: {c.rationale}")

    cost.print_report()

    if args.json:
        with open(args.json, "w") as f:
            f.write(cost.to_json())
        print(f"\nFull machine-readable cost report written to {args.json}")

    totals = cost.totals()
    ok = totals["all_cycles_within_token_budget"] and totals["all_cycles_within_query_budget"]
    print(f"\nBUDGET COMPLIANCE (<=5000 tokens/cycle, <=10 warehouse queries/cycle): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
