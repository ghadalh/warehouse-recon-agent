# Warehouse Reconciliation Agent

An agent that watches a stream of operational events (orders, payments,
shipments...) and reconciles the live system's belief about what happened
against a warehouse's independently-recorded ground truth — without
checking the warehouse on every event, and while measuring the exact token
and API cost of every decision it makes.

Built for the LEC AI build assessment. This README explains how to run it,
how it's built, the cost-tradeoff decisions behind the design, what the
demo actually proves, and what's unfinished.

## TL;DR

```bash
pip install -r requirements.txt
python3 demo.py
```

That starts a mock warehouse HTTP service on `localhost:8799`, streams 12
events across 5 event types through the agent, and prints a full decision
trace followed by a cost report. It finds and fixes two real, deliberately
injected inconsistencies, auto-heals one benign one, and finishes the
whole run in **973 tokens across 4 LLM calls and 4 warehouse HTTP calls
(6 order lookups)** — well inside the brief's example budget of 5,000
tokens / 10 warehouse queries *per cycle*. A saved copy of an actual run is
in [`assets/demo_run_output.txt`](assets/demo_run_output.txt) and
[`assets/cost_report.json`](assets/cost_report.json).

Run the tests with:

```bash
python3 -m pytest -q
```

19 tests: risk-policy unit tests, reconciliation-detection unit tests,
cost-tracker budget-enforcement unit tests, and two end-to-end integration
tests that run the full demo scenario and assert budget compliance plus
correct detection/correction of both injected bugs.

## The scenario

Four orders move through the live system. Two of them have a real bug
baked into the warehouse's ground truth on purpose, so the demo has
something genuine to catch:

- **ORD-1**: created, paid, then the live system fires a
  `shipment_dispatched` event — but the warehouse's own export shows the
  shipment never actually left the building. The operational system's
  belief is *wrong*. This is the headline inconsistency.
- **ORD-2**: created, its item count gets revised mid-stream
  (`inventory_adjusted`), then paid. Warehouse matches throughout — a
  clean "nothing wrong here" case, deliberately included so the demo
  doesn't only show positives.
- **ORD-3**: created, shipped, matches on items — but settles for 50c
  less than expected (card-processor rounding). A genuinely minor
  discrepancy.
- **ORD-4**: created, then **cancelled** — but a race condition means the
  warehouse ships it anyway. Second real inconsistency, and an edge case
  beyond the minimum ask (cancel-after-dispatch race).

## Architecture

```
event stream ──▶ risk_policy.score_event()  (free, deterministic, every event)
                        │
        ┌───────────────┼────────────────────┐
        ▼                ▼                    ▼
      SKIP          LLM_ESCALATE            QUERY
   (no cost)     (ambiguous — spend an   (heuristic already
                  LLM call to decide)     confident — no LLM
                        │                  needed, just queue)
                        ▼
              QUERY or SKIP (LLM's call)
                        │
                        ▼
              enqueue order_id ──▶ batch flush trigger:
                                     batch_size reached, OR
                                     max-wait events elapsed, OR
                                     risk >= critical (flush now)
                                        │
                                        ▼
                          ONE warehouse HTTP call for N orders
                                        │
                                        ▼
                       reconciliation.detect() per order
                          │                        │
                        no mismatch           mismatch found
                                              │            │
                                            minor         major
                                        (auto-heal,   (LLM builds a
                                         no LLM)       rationale, then
                                                        belief is patched)
```

Everything is in `agent/`:

| File | Responsibility |
|---|---|
| `events.py` | Event types and the `Event` dataclass |
| `system_state.py` | The live system's belief — a pure fold over events, no I/O |
| `warehouse_service.py` | Mock warehouse: a real FastAPI HTTP service (not just a Python object) with a `/warehouse/batch` endpoint, so "API calls" are literal HTTP requests we can count |
| `risk_policy.py` | The free heuristic risk scorer + the skip/escalate/query decision |
| `llm_client.py` | Pluggable LLM interface with honest token accounting (see below) |
| `cost_tracker.py` | Records every decision's cost, rolls it up per reconciliation cycle, enforces the budget |
| `reconciliation.py` | Detects mismatches, classifies severity, builds the reconciliation prompt |
| `agent_core.py` | Ties it together: batching, flush triggers, applying corrections |

`demo.py` seeds the warehouse and system, builds the 12-event stream, runs
it through the agent, and prints the trace + cost report.

## The cost-aware decisions (and why)

This is the part of the brief that matters most, so here's the reasoning
spelled out rather than left implicit in the code:

**1. A free heuristic runs on every event; the LLM only runs on the
genuinely ambiguous ones.** `risk_policy.py` assigns a base risk weight per
event type (payment/shipment events are inherently riskier to get wrong
than "order created," which has nothing yet to have drifted on) plus a
couple of contextual adjustments (a shipment dispatched on a cancelled
order is escalated to critical; repeated high-risk activity on the same
order compounds). Events the heuristic is already confident about — clearly
low risk (`order_created`) or clearly high risk (`payment_processed`,
`shipment_dispatched` in the normal case) — **never touch the LLM**. There
is nothing for a model to add over a threshold check on those, and every
avoided call is tokens not spent. Only the middle band (`0.30 ≤ score <
0.65`) spends an LLM call to reason about it. In the demo run, that's 2 of
12 events.

**2. The LLM's answer actually changes control flow, both ways.** The
`order_cancelled` event for ORD-4 lands in the ambiguous band (base weight
0.35) — but at the moment it fires, nothing has shipped yet, so there's
nothing for the warehouse to have diverged on. The LLM call reasons about
exactly that and recommends **SKIP**, and the agent honors it — no
warehouse call spent on confirming something that's definitionally not yet
inconsistent. (When the *next* event, a dispatch on that now-cancelled
order, arrives a moment later, the heuristic alone — no LLM needed — scores
it as critical and flushes immediately.) This is the answer to "could you
have detected the same inconsistency more cheaply": for the cancellation
event alone, yes — a warehouse call there would have been wasted, and the
agent knows that without spending one.

**3. Queries are batched, not fired one per event.** `agent_core.py` queues
order IDs and flushes them as a single `/warehouse/batch` HTTP call, either
when the batch reaches `BATCH_SIZE=3`, when `MAX_WAIT_EVENTS=5` events have
passed since the last flush, or immediately if an event scores as critical
(≥0.9) — a near-certain problem shouldn't sit in a queue. In the demo, 7
events triggered a query but only **4 HTTP calls** were made to resolve
them (6 total order lookups), because the queue collapses back-to-back
risky events on different orders into one round trip.

**4. Minor drift is auto-healed without an LLM call; major drift gets one.**
`reconciliation.py` classifies severity by the size of the discrepancy (≤2
items, ≤$1 payment delta = minor; a shipment-status contradiction is always
major, since "the system thinks it shipped and the warehouse says it
didn't" is never a rounding error). Minor mismatches are patched
deterministically and logged — spending tokens to have a model "explain"
that 1 unit out of 40 shipped short should just be corrected isn't worth
it. Major mismatches get an LLM call to produce a reviewable rationale,
because a financially/operationally significant correction shouldn't have
"warehouse said so" as its only audit trail.

**5. A "reconciliation cycle" is defined at the flush boundary, not the
whole run**, specifically so the budget check means something: the tokens
attributed to cycle N are the risk-escalation calls that fed events into
that cycle *plus* the reconciliation calls spent resolving it, attributed
by event sequence (not by order ID, since the same order can legitimately
appear across more than one cycle and ID-based attribution would double
count). See `cost_tracker.record_cycle` for the accounting; there's a unit
test (`test_escalation_tokens_are_folded_into_cycle_totals`) that pins this
down because I actually got it wrong on the first pass — the reconciliation
LLM calls (made inside `_flush`) weren't being rolled into `totals()` at
all, which silently under-reported real spend by about half. Worth
flagging here because "measure the actual cost of every decision" only
means something if the measurement itself is correct — I'd rather show the
bug and the fix than pretend the first version was right.

## Honest token accounting without a live API key

The brief wants real token counts and real dollar costs. Requiring a paid
API key just to *run* the submission felt like the wrong tradeoff — it also
means the demo's numbers aren't reproducible run-to-run without paying for
them. So `llm_client.py` ships a `DeterministicLLM` by default:

- It builds the **exact same prompts** a real call would use (see
  `agent_core._risk_escalation_prompt` and
  `reconciliation.build_reconciliation_prompt`).
- It counts them with **tiktoken's real `cl100k_base` tokenizer** — the
  same tokenizer real API-side accounting is based on.
- The *completion* is template-based rather than sampled, but its tokens
  are counted for real too. Nothing in the cost report is fabricated —
  only the reasoning text is canned instead of generated.
- Pricing constants (`PRICE_PER_M_INPUT` / `PRICE_PER_M_OUTPUT`) are
  clearly labeled as illustrative, not a claim about actual current rates.

Swapping in a real model is a one-line change: `python3 demo.py --llm
anthropic` with `pip install anthropic` and `ANTHROPIC_API_KEY` set uses
`AnthropicLLM`, which calls the real Claude Messages API and reports real
`usage.input_tokens` / `usage.output_tokens` from the response. Every
downstream piece — the cost tracker, the budget check, the report — is
completely unchanged, because that's the point of the `LLMClient`
abstraction: the agent's decision logic doesn't know or care which backend
answered.

One more honesty note: `tiktoken`'s `cl100k_base` encoding lazily
downloads its vocab file on first use — completely normal, that's how it
ships on a fresh `pip install` on any real machine with internet access.
The sandbox this was *developed* in blocks that one specific download
domain, so `llm_client.py` catches that and falls back to a regex-based
approximate counter, printing a clear warning when it does. On your
machine this fallback almost certainly won't trigger — you'll get the real
tokenizer. I'm calling this out explicitly rather than leaving a mystery
warning in the output.

## What the demo actually proves

Run `python3 demo.py` and you'll see, in order:

1. 4 `order_created` events skipped for free (risk 0.10, no warehouse call,
   no tokens).
2. A `payment_processed` event scored high-risk and queued directly — no
   LLM needed.
3. An `inventory_adjusted` event landing in the ambiguous band, an LLM call
   deciding to query, and that query getting batched with the next event.
4. A `shipment_dispatched` event on ORD-1 pushing risk to 0.90 (base risk
   plus a "second risky event on this order recently" compounding
   adjustment) and triggering an **immediate** flush rather than waiting
   for the batch to fill.
5. That flush's warehouse response revealing the real inconsistency:
   system believes ORD-1 is dispatched, warehouse says it never left the
   building. An LLM call produces a reconciliation rationale, and the
   agent **patches the system's belief back to the warehouse's ground
   truth** (verified by an integration test, not just printed).
6. A cancellation event where the LLM reasons its way to *not* querying,
   saving a warehouse call.
7. The very next event — a dispatch on that now-cancelled order — scoring
   as critical from the heuristic alone and immediately triggering the
   second real inconsistency: warehouse shows it shipped anyway.
8. A final minor payment-rounding mismatch on ORD-3, auto-healed with zero
   LLM calls.
9. The full cost report: **12 events, 5 skipped (4 free + 1 after LLM
   reasoning), 7 queried, 4 LLM calls, 973 total tokens, ~$0.0023
   illustrative cost, 4 warehouse HTTP calls (6 order lookups) across 4
   reconciliation cycles — every single cycle inside the 5,000-token /
   10-query example budget**, most of them by a wide margin.

## What I'd do next with more time

- **Learn the risk weights instead of hand-setting them.** Right now
  `BASE_RISK` is a small hardcoded table I picked based on "which fields
  are financially/fulfillment-critical." A real version would fit these
  from historical data — which event types actually preceded a warehouse
  mismatch — and update them online as the false-positive/false-negative
  rate on queries becomes observable. That's a genuinely different (and
  better) system; I didn't want to fake a "learned" policy that's secretly
  the same three if-statements.
- **Idempotency and replay.** The agent currently assumes events arrive
  once, in order, and processes them synchronously. A production version
  needs to handle redelivery (event IDs + a seen-set), out-of-order
  arrival, and crash recovery mid-cycle (what happens if the process dies
  after a warehouse call but before applying the correction?).
- **Concurrency.** Events are processed one at a time on a single thread.
  Real order volume would need the risk scoring and queuing to be
  thread-safe / async, with the warehouse batching logic protected against
  races between two events for the same order landing in different
  batches.
- **A real warehouse export model.** The mock currently exposes a
  synchronous "give me current state" endpoint. Real warehouse integrations
  are usually genuinely periodic (a CSV/SFTP drop every N minutes), which
  changes the cost story: you might not be able to force a check even when
  risk is critical, only mark an order "pending verification" until the
  next natural export lands. Worth modeling that latency explicitly rather
  than assuming synchronous query access.
- **Structured LLM output instead of text parsing.** The agent currently
  looks for the literal substring `"Recommendation: SKIP"` in the LLM's
  response to decide control flow. That's fine for a deterministic
  template but fragile against a real model's phrasing variance — a real
  version should use tool calling / structured output (e.g. a forced
  `{"action": "skip"|"query", "reason": str}` schema) instead of string
  matching.
- **More severity nuance in reconciliation.** Right now severity is a
  single delta threshold per field. A real version would want per-order
  context (a $1 delta on a $5 order is not the same signal as a $1 delta on
  a $5,000 order) and probably a third "needs human review, don't
  auto-resolve either way" tier for cases like a shipment status of
  `"returned"` that no simple patch should silently paper over.
- **Persisting the decision/cost log.** `CostTracker` is in-memory for the
  demo. A real deployment would stream `DecisionRecord`/`CycleRecord` to
  somewhere queryable (even just an append-only JSONL file) so the cost
  report can be computed over days of production traffic, not one run.

## Repo layout

```
agent/
  events.py             event types
  system_state.py        live system's belief (pure event fold)
  warehouse_service.py   mock warehouse, real FastAPI HTTP service
  risk_policy.py         free heuristic risk scoring + skip/escalate/query decision
  llm_client.py          pluggable LLM client, honest token accounting
  reconciliation.py      mismatch detection, severity, correction
  cost_tracker.py        per-decision and per-cycle cost accounting + budget checks
  agent_core.py          the agent: batching, flush triggers, orchestration
demo.py                  scenario runner (the thing to actually run)
tests/                    19 unit + integration tests
assets/
  demo_run_output.txt     a real captured run of `python3 demo.py`
  cost_report.json        the machine-readable cost report from that run
requirements.txt
pytest.ini
```
