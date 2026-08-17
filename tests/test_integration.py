"""
End-to-end test: runs the full demo scenario against a real (mock)
warehouse HTTP service and asserts the whole thing stays within the
brief's example budget (<=5000 tokens/cycle, <=10 warehouse queries/cycle)
while still detecting and correcting both injected inconsistencies.
"""
import time

import pytest

import demo
from agent.agent_core import ReconciliationAgent
from agent.cost_tracker import CostTracker
from agent.llm_client import build_llm_client
from agent.system_state import SystemState
from agent.warehouse_service import WarehouseService, run_in_thread

TEST_PORT = 8798
TEST_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture(scope="module")
def running_warehouse():
    service = WarehouseService()
    demo.seed_warehouse(service)
    run_in_thread(service, port=TEST_PORT)
    time.sleep(0.3)
    return service


def _silent(*args, **kwargs):
    pass


def test_full_demo_scenario_stays_within_budget_and_fixes_both_bugs(running_warehouse):
    state = SystemState()
    cost = CostTracker()
    llm = build_llm_client("deterministic")
    agent = ReconciliationAgent(
        system_state=state, llm_client=llm, cost_tracker=cost,
        warehouse_base_url=TEST_URL, narrate=_silent,
    )

    for event in demo.build_event_stream():
        agent.process_event(event)
    agent.finalize()

    totals = cost.totals()

    # Budget compliance -- the brief's example thresholds.
    assert totals["all_cycles_within_token_budget"], cost.to_json()
    assert totals["all_cycles_within_query_budget"], cost.to_json()
    assert totals["llm_total_tokens"] < 5000

    # Not every event should have triggered a warehouse check.
    assert totals["events_skipped_no_check"] > 0
    assert totals["events_that_triggered_a_query"] < totals["events_seen"]

    # Both injected inconsistencies must be found and corrected.
    corrected_fields = {(c.order_id, c.field) for c in agent.corrections}
    assert ("ORD-1", "shipment_status") in corrected_fields  # premature dispatch belief
    assert ("ORD-4", "shipment_status") in corrected_fields  # shipped-after-cancel
    assert ("ORD-3", "payment_settled_cents") in corrected_fields  # minor rounding, auto-healed

    # ORD-1's belief must actually have been patched back to match the
    # warehouse's ground truth, not just logged.
    assert state.get("ORD-1").shipment_status == "not_dispatched"

    # The minor correction must NOT have used an LLM call (cost discipline).
    minor = [c for c in agent.corrections if c.order_id == "ORD-3" and c.field == "payment_settled_cents"]
    assert minor and minor[0].reviewed_by_llm is False

    # The two real inconsistencies must have used an LLM-reviewed rationale.
    major = [c for c in agent.corrections if c.field == "shipment_status"]
    assert all(c.reviewed_by_llm for c in major)


def test_agent_never_calls_warehouse_more_than_the_example_budget_per_cycle(running_warehouse):
    state = SystemState()
    cost = CostTracker()
    llm = build_llm_client("deterministic")
    agent = ReconciliationAgent(
        system_state=state, llm_client=llm, cost_tracker=cost,
        warehouse_base_url=TEST_URL, narrate=_silent,
    )
    for event in demo.build_event_stream():
        agent.process_event(event)
    agent.finalize()

    for cycle in cost.cycles:
        assert cycle.warehouse_order_lookups <= 10
        assert (cycle.llm_input_tokens + cycle.llm_output_tokens) <= 5000
