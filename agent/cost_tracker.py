"""
Records the actual, measured cost of every decision the agent makes, and
enforces the per-cycle budget from the brief:

    "demonstrate that the agent completed a full reconciliation within a
    defined budget (for example, under 5000 tokens or 10 warehouse queries
    per cycle)"

A "cycle" here = one batch-flush-and-reconcile round: the set of events
accumulated since the last flush, the (at most one) warehouse HTTP call
that resolves them, and any reconciliation/LLM work that call triggers.
Defining the cycle boundary at the flush (not at the whole demo run) is
what makes the budget numbers meaningful -- ten cheap events and one flush
is one cycle, not the same "cycle" as fifty events across ten flushes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

TOKEN_BUDGET_PER_CYCLE = 5000
WAREHOUSE_QUERY_BUDGET_PER_CYCLE = 10  # order lookups, not raw HTTP calls


@dataclass
class DecisionRecord:
    seq: int
    order_id: str
    event_type: str
    risk_score: float
    reasons: List[str]
    decision: str
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CycleRecord:
    cycle_id: int
    order_ids_flushed: List[str]
    warehouse_http_calls: int
    warehouse_order_lookups: int
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int
    llm_cost_usd: float
    mismatches_found: int
    corrections_applied: int
    within_token_budget: bool
    within_query_budget: bool


class CostTracker:
    def __init__(self) -> None:
        self.decisions: List[DecisionRecord] = []
        self.cycles: List[CycleRecord] = []

    def record_decision(self, rec: DecisionRecord) -> None:
        self.decisions.append(rec)

    def record_cycle(
        self,
        cycle_id: int,
        order_ids_flushed: List[str],
        warehouse_http_calls: int,
        warehouse_order_lookups: int,
        llm_calls: int,
        llm_input_tokens: int,
        llm_output_tokens: int,
        llm_cost_usd: float,
        mismatches_found: int,
        corrections_applied: int,
        escalation_input_tokens: int = 0,
        escalation_output_tokens: int = 0,
        escalation_llm_calls: int = 0,
        escalation_cost_usd: float = 0.0,
    ) -> CycleRecord:
        # A cycle's token budget must include BOTH the risk-escalation LLM
        # calls that queued events into this cycle AND the reconciliation
        # LLM calls spent resolving it -- the caller passes the escalation
        # totals it accumulated since the last flush (by sequence, not by
        # order_id, since the same order can appear in more than one
        # cycle's flush list and order_id matching would double count).
        llm_input_tokens += escalation_input_tokens
        llm_output_tokens += escalation_output_tokens
        llm_calls += escalation_llm_calls
        llm_cost_usd += escalation_cost_usd
        total_tokens = llm_input_tokens + llm_output_tokens
        rec = CycleRecord(
            cycle_id=cycle_id,
            order_ids_flushed=order_ids_flushed,
            warehouse_http_calls=warehouse_http_calls,
            warehouse_order_lookups=warehouse_order_lookups,
            llm_calls=llm_calls,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            llm_cost_usd=llm_cost_usd,
            mismatches_found=mismatches_found,
            corrections_applied=corrections_applied,
            within_token_budget=total_tokens <= TOKEN_BUDGET_PER_CYCLE,
            within_query_budget=warehouse_order_lookups <= WAREHOUSE_QUERY_BUDGET_PER_CYCLE,
        )
        self.cycles.append(rec)
        return rec

    # -- reporting -----------------------------------------------------

    def totals(self) -> Dict[str, Any]:
        # LLM totals are summed from cycles, which is the single source of
        # truth for "every token spent" -- each cycle already folds in both
        # the risk-escalation calls that fed it and its own reconciliation
        # calls (see record_cycle), attributed by sequence rather than by
        # order_id so nothing is double-counted or dropped.
        total_llm_calls = sum(c.llm_calls for c in self.cycles)
        total_input = sum(c.llm_input_tokens for c in self.cycles)
        total_output = sum(c.llm_output_tokens for c in self.cycles)
        total_cost = sum(c.llm_cost_usd for c in self.cycles)
        total_events = len(self.decisions)
        free_skipped = sum(1 for d in self.decisions if d.decision == "skip")
        llm_skipped = sum(1 for d in self.decisions if d.decision == "llm_escalate_skip")
        skipped = free_skipped + llm_skipped
        queried = sum(1 for d in self.decisions if d.decision in ("query", "llm_escalate_query"))
        total_http_calls = sum(c.warehouse_http_calls for c in self.cycles)
        total_lookups = sum(c.warehouse_order_lookups for c in self.cycles)
        return {
            "events_seen": total_events,
            "events_skipped_no_check": skipped,
            "events_skipped_free_heuristic": free_skipped,
            "events_skipped_after_llm_reasoning": llm_skipped,
            "events_that_triggered_a_query": queried,
            "skip_rate_pct": round(100 * skipped / total_events, 1) if total_events else 0.0,
            "llm_calls": total_llm_calls,
            "llm_input_tokens": total_input,
            "llm_output_tokens": total_output,
            "llm_total_tokens": total_input + total_output,
            "llm_cost_usd": round(total_cost, 6),
            "warehouse_http_calls": total_http_calls,
            "warehouse_order_lookups": total_lookups,
            "cycles": len(self.cycles),
            "all_cycles_within_token_budget": all(c.within_token_budget for c in self.cycles),
            "all_cycles_within_query_budget": all(c.within_query_budget for c in self.cycles),
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "decisions": [asdict(d) for d in self.decisions],
                "cycles": [asdict(c) for c in self.cycles],
                "totals": self.totals(),
            },
            indent=2,
        )

    def print_report(self) -> None:
        t = self.totals()
        print("\n" + "=" * 72)
        print("COST REPORT")
        print("=" * 72)
        print(f"Events seen:                  {t['events_seen']}")
        print(f"  -> skipped, free heuristic (0 tokens, 0 calls): {t['events_skipped_free_heuristic']}")
        print(f"  -> skipped, after LLM reasoning ($ tokens spent to avoid a call): "
              f"{t['events_skipped_after_llm_reasoning']}")
        print(f"  -> total skipped:                               {t['events_skipped_no_check']} "
              f"({t['skip_rate_pct']}%)")
        print(f"  -> triggered a warehouse query:                 {t['events_that_triggered_a_query']}")
        print(f"LLM calls made:                {t['llm_calls']}")
        print(f"LLM tokens (in/out/total):     {t['llm_input_tokens']} / {t['llm_output_tokens']} / "
              f"{t['llm_total_tokens']}")
        print(f"LLM cost (USD, illustrative):  ${t['llm_cost_usd']}")
        print(f"Warehouse HTTP calls:          {t['warehouse_http_calls']}")
        print(f"Warehouse order lookups:       {t['warehouse_order_lookups']}")
        print(f"Reconciliation cycles:         {t['cycles']}")
        print(f"All cycles within {TOKEN_BUDGET_PER_CYCLE}-token budget:   {t['all_cycles_within_token_budget']}")
        print(f"All cycles within {WAREHOUSE_QUERY_BUDGET_PER_CYCLE}-query budget:      {t['all_cycles_within_query_budget']}")
        print("-" * 72)
        for c in self.cycles:
            print(
                f"cycle {c.cycle_id}: orders={c.order_ids_flushed} "
                f"http_calls={c.warehouse_http_calls} lookups={c.warehouse_order_lookups} "
                f"tokens={c.llm_input_tokens + c.llm_output_tokens} "
                f"mismatches={c.mismatches_found} corrections={c.corrections_applied} "
                f"budget_ok={c.within_token_budget and c.within_query_budget}"
            )
        print("=" * 72)
