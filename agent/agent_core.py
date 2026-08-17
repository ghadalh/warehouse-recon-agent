"""
The agent itself: consumes events one at a time, applies the risk policy,
batches warehouse checks, runs reconciliation when a batch flush reveals a
mismatch, and records every decision's cost.

This is deliberately NOT "check warehouse -> reconcile -> repeat" on a
fixed schedule. Control flow is decision-driven:

    event -> risk score -> {skip | escalate to LLM | queue for query}
    queue -> flush trigger -> ONE batched warehouse call for N orders
    flush result -> per order: detect mismatch -> {auto-heal | LLM-explained correction}

Batching policy: the queue flushes when it reaches BATCH_SIZE queued
orders, or when MAX_WAIT_EVENTS events have been processed since the last
flush (whichever comes first), or immediately if an event scores as
critical (>=0.9) since letting a near-certain problem sit in a queue
defeats the point of catching it. This bounds both worst-case staleness
and worst-case batch size, and turns "N risky events" into a small number
of warehouse HTTP calls instead of N of them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import requests

from .cost_tracker import CostTracker, DecisionRecord
from .events import Event
from .llm_client import LLMClient
from .reconciliation import Correction, build_reconciliation_prompt, detect
from .risk_policy import Decision, decide, score_event
from .system_state import SystemState

BATCH_SIZE = 3
MAX_WAIT_EVENTS = 5
CRITICAL_FLUSH_THRESHOLD = 0.9
RECENT_WINDOW = 4  # events


class ReconciliationAgent:
    def __init__(
        self,
        system_state: SystemState,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        warehouse_base_url: str,
        narrate=print,
    ) -> None:
        self.state = system_state
        self.llm = llm_client
        self.cost = cost_tracker
        self.warehouse_url = warehouse_base_url.rstrip("/")
        self.narrate = narrate

        self._pending: List[str] = []          # order_ids queued for a warehouse check
        self._events_since_flush = 0
        self._recent_high_risk: Set[str] = set()
        self._recent_high_risk_order: List[str] = []  # to expire from the set
        self._cycle_id = 0
        self.corrections: List[Correction] = []

        # Accumulate risk-escalation LLM spend since the last flush, so it
        # can be folded into the cycle it belongs to (see _flush). Reset on
        # every flush.
        self._esc_calls_since_flush = 0
        self._esc_input_since_flush = 0
        self._esc_output_since_flush = 0
        self._esc_cost_since_flush = 0.0

    # -- main entry point -------------------------------------------------

    def process_event(self, event: Event) -> None:
        belief = self.state.apply(event)
        risk = score_event(event, belief, self._recent_high_risk)
        decision = decide(risk)

        record = DecisionRecord(
            seq=event.seq,
            order_id=event.order_id,
            event_type=event.type.value,
            risk_score=round(risk.score, 3),
            reasons=risk.reasons,
            decision=decision.value,
        )

        if decision == Decision.SKIP:
            self.narrate(
                f"[event {event.seq:>2}] {event.type.value:<20} order={event.order_id}  "
                f"risk={risk.score:.2f}  -> SKIP (below {0.30} threshold, no warehouse call, $0)"
            )

        elif decision == Decision.LLM_ESCALATE:
            prompt = self._risk_escalation_prompt(event, belief, risk.reasons)
            result = self.llm.complete(prompt, purpose="risk_escalation")
            record.llm_calls.append(
                {
                    "purpose": "risk_escalation",
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                    "model": result.model,
                }
            )
            self._esc_calls_since_flush += 1
            self._esc_input_since_flush += result.input_tokens
            self._esc_output_since_flush += result.output_tokens
            self._esc_cost_since_flush += result.cost_usd
            llm_says_skip = "Recommendation: SKIP" in result.text
            if llm_says_skip:
                record.decision = "llm_escalate_skip"
                self.narrate(
                    f"[event {event.seq:>2}] {event.type.value:<20} order={event.order_id}  "
                    f"risk={risk.score:.2f}  -> AMBIGUOUS, asked LLM "
                    f"({result.input_tokens}+{result.output_tokens} tok) -> LLM says SKIP, no warehouse call"
                )
            else:
                record.decision = "llm_escalate_query"
                self.narrate(
                    f"[event {event.seq:>2}] {event.type.value:<20} order={event.order_id}  "
                    f"risk={risk.score:.2f}  -> AMBIGUOUS, asked LLM "
                    f"({result.input_tokens}+{result.output_tokens} tok) -> LLM says QUERY, queued"
                )
                self._enqueue(event.order_id)

        else:  # Decision.QUERY
            self.narrate(
                f"[event {event.seq:>2}] {event.type.value:<20} order={event.order_id}  "
                f"risk={risk.score:.2f}  -> HIGH RISK, queued for warehouse check (no LLM needed)"
            )
            self._enqueue(event.order_id)

        self._track_recent_high_risk(event.order_id, risk.score)
        self.cost.record_decision(record)

        self._events_since_flush += 1
        if risk.score >= CRITICAL_FLUSH_THRESHOLD:
            self.narrate(f"           risk >= {CRITICAL_FLUSH_THRESHOLD} -> flushing immediately")
            self._flush()
        elif len(self._pending) >= BATCH_SIZE:
            self.narrate(f"           batch size reached ({BATCH_SIZE}) -> flushing")
            self._flush()
        elif self._events_since_flush >= MAX_WAIT_EVENTS and self._pending:
            self.narrate(f"           max wait reached ({MAX_WAIT_EVENTS} events) -> flushing")
            self._flush()

    def finalize(self) -> None:
        """Flush any remaining queued orders at the end of the stream."""
        if self._pending:
            self._flush()

    # -- internals ----------------------------------------------------

    def _enqueue(self, order_id: str) -> None:
        if order_id not in self._pending:
            self._pending.append(order_id)

    def _track_recent_high_risk(self, order_id: str, score: float) -> None:
        if score >= 0.5:
            self._recent_high_risk.add(order_id)
            self._recent_high_risk_order.append(order_id)
            if len(self._recent_high_risk_order) > RECENT_WINDOW:
                expired = self._recent_high_risk_order.pop(0)
                # only drop from the set if it's not still present later in the window
                if expired not in self._recent_high_risk_order:
                    self._recent_high_risk.discard(expired)

    def _flush(self) -> None:
        order_ids = list(self._pending)
        self._pending.clear()
        self._events_since_flush = 0
        self._cycle_id += 1

        resp = requests.post(f"{self.warehouse_url}/warehouse/batch", json=order_ids, timeout=5)
        resp.raise_for_status()
        records = resp.json()

        cycle_input_tokens = 0
        cycle_output_tokens = 0
        cycle_cost = 0.0
        cycle_llm_calls = 0
        mismatches_found = 0
        corrections_applied = 0

        for order_id in order_ids:
            belief = self.state.get(order_id)
            wr = records.get(order_id)
            mismatches = detect(order_id, belief, wr)
            if not mismatches:
                self.narrate(f"           reconcile order={order_id}: no mismatch, in sync")
                continue

            for m in mismatches:
                mismatches_found += 1
                if m.severity == "minor":
                    correction = self._auto_heal(m)
                    self.narrate(
                        f"           reconcile order={order_id}: MINOR mismatch on '{m.field}' "
                        f"({m.system_value} -> {m.warehouse_value}) -> auto-healed, no LLM"
                    )
                else:
                    prompt = build_reconciliation_prompt(m)
                    result = self.llm.complete(prompt, purpose="reconciliation_plan")
                    cycle_input_tokens += result.input_tokens
                    cycle_output_tokens += result.output_tokens
                    cycle_cost += result.cost_usd
                    cycle_llm_calls += 1
                    correction = self._apply_correction(m, reviewed_by_llm=True, rationale=result.text)
                    self.narrate(
                        f"           reconcile order={order_id}: MAJOR mismatch on '{m.field}' "
                        f"({m.system_value} -> {m.warehouse_value}) -> LLM reconciliation plan "
                        f"({result.input_tokens}+{result.output_tokens} tok), correction applied"
                    )
                self.corrections.append(correction)
                corrections_applied += 1

        self.cost.record_cycle(
            cycle_id=self._cycle_id,
            order_ids_flushed=order_ids,
            warehouse_http_calls=1,
            warehouse_order_lookups=len(order_ids),
            llm_calls=cycle_llm_calls,
            llm_input_tokens=cycle_input_tokens,
            llm_output_tokens=cycle_output_tokens,
            llm_cost_usd=cycle_cost,
            mismatches_found=mismatches_found,
            corrections_applied=corrections_applied,
            escalation_input_tokens=self._esc_input_since_flush,
            escalation_output_tokens=self._esc_output_since_flush,
            escalation_llm_calls=self._esc_calls_since_flush,
            escalation_cost_usd=self._esc_cost_since_flush,
        )

        # reset the since-last-flush escalation accumulators -- their spend
        # has now been folded into this cycle.
        self._esc_calls_since_flush = 0
        self._esc_input_since_flush = 0
        self._esc_output_since_flush = 0
        self._esc_cost_since_flush = 0.0

    def _auto_heal(self, m) -> Correction:
        self._patch_belief(m)
        return Correction(
            order_id=m.order_id,
            field=m.field,
            old_value=m.system_value,
            new_value=m.warehouse_value,
            rationale=f"Minor drift ({m.description}) auto-healed deterministically; below manual-review threshold.",
            reviewed_by_llm=False,
        )

    def _apply_correction(self, m, reviewed_by_llm: bool, rationale: str) -> Correction:
        self._patch_belief(m)
        return Correction(
            order_id=m.order_id,
            field=m.field,
            old_value=m.system_value,
            new_value=m.warehouse_value,
            rationale=rationale,
            reviewed_by_llm=reviewed_by_llm,
        )

    def _patch_belief(self, m) -> None:
        belief = self.state.get(m.order_id)
        if m.field == "items_shipped":
            belief.items_expected = m.warehouse_value
        elif m.field == "payment_settled_cents":
            belief.payment_expected_cents = m.warehouse_value
        elif m.field == "shipment_status":
            belief.shipment_status = m.warehouse_value

    @staticmethod
    def _risk_escalation_prompt(event: Event, belief, reasons: List[str]) -> str:
        return (
            "You are a cost-aware reconciliation agent deciding whether an operational "
            f"event warrants a warehouse API call. Event: {event.type.value} for order "
            f"{event.order_id}. Current belief: items_expected={belief.items_expected}, "
            f"payment_processed={belief.payment_processed}, shipment_status={belief.shipment_status}, "
            f"cancelled={belief.cancelled}. Heuristic reasons: {'; '.join(reasons)}. "
            "Decide QUERY or SKIP and justify briefly, weighing the cost of one warehouse call "
            "against the risk of missing a real inconsistency."
        )
