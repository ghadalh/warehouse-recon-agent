from agent.cost_tracker import CostTracker, DecisionRecord, TOKEN_BUDGET_PER_CYCLE, WAREHOUSE_QUERY_BUDGET_PER_CYCLE


def test_cycle_within_budget_flagged_correctly():
    tracker = CostTracker()
    cycle = tracker.record_cycle(
        cycle_id=1, order_ids_flushed=["A", "B"], warehouse_http_calls=1,
        warehouse_order_lookups=2, llm_calls=1, llm_input_tokens=100,
        llm_output_tokens=50, llm_cost_usd=0.001, mismatches_found=1,
        corrections_applied=1,
    )
    assert cycle.within_token_budget is True
    assert cycle.within_query_budget is True


def test_cycle_over_token_budget_is_flagged():
    tracker = CostTracker()
    cycle = tracker.record_cycle(
        cycle_id=1, order_ids_flushed=["A"], warehouse_http_calls=1,
        warehouse_order_lookups=1, llm_calls=1,
        llm_input_tokens=TOKEN_BUDGET_PER_CYCLE, llm_output_tokens=1,
        llm_cost_usd=1.0, mismatches_found=0, corrections_applied=0,
    )
    assert cycle.within_token_budget is False


def test_cycle_over_query_budget_is_flagged():
    tracker = CostTracker()
    cycle = tracker.record_cycle(
        cycle_id=1, order_ids_flushed=[f"O{i}" for i in range(WAREHOUSE_QUERY_BUDGET_PER_CYCLE + 1)],
        warehouse_http_calls=1, warehouse_order_lookups=WAREHOUSE_QUERY_BUDGET_PER_CYCLE + 1,
        llm_calls=0, llm_input_tokens=0, llm_output_tokens=0, llm_cost_usd=0.0,
        mismatches_found=0, corrections_applied=0,
    )
    assert cycle.within_query_budget is False


def test_escalation_tokens_are_folded_into_cycle_totals():
    tracker = CostTracker()
    cycle = tracker.record_cycle(
        cycle_id=1, order_ids_flushed=["A"], warehouse_http_calls=1,
        warehouse_order_lookups=1, llm_calls=1, llm_input_tokens=100,
        llm_output_tokens=50, llm_cost_usd=0.001, mismatches_found=1,
        corrections_applied=1, escalation_input_tokens=40,
        escalation_output_tokens=20, escalation_llm_calls=1,
        escalation_cost_usd=0.0005,
    )
    assert cycle.llm_input_tokens == 140
    assert cycle.llm_output_tokens == 70
    assert cycle.llm_calls == 2
    assert abs(cycle.llm_cost_usd - 0.0015) < 1e-9


def test_totals_sum_across_cycles_not_decisions():
    tracker = CostTracker()
    tracker.record_decision(DecisionRecord(
        seq=1, order_id="A", event_type="payment_processed", risk_score=0.7,
        reasons=[], decision="query",
    ))
    tracker.record_cycle(
        cycle_id=1, order_ids_flushed=["A"], warehouse_http_calls=1,
        warehouse_order_lookups=1, llm_calls=2, llm_input_tokens=200,
        llm_output_tokens=100, llm_cost_usd=0.002, mismatches_found=1,
        corrections_applied=1,
    )
    totals = tracker.totals()
    assert totals["llm_calls"] == 2
    assert totals["llm_total_tokens"] == 300
    assert totals["warehouse_order_lookups"] == 1
