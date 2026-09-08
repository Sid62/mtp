"""Regression tests for Phase 3: Replan Trigger Logic and Baseline Isolation."""

import pytest
from src.env.daca_env import DACAEnv
from src.coordination.replan_trigger import PlanState, should_replan, update_plan_state


def test_static_baseline_ignores_cqi_drift():
    env = DACAEnv("logistics", {}, seed=0)
    plan_state = PlanState()
    subtasks = env.subtask_list
    fleet = env.fleet

    assignments = {s.subtask_id: [fleet.agents[0].agent_id] for s in subtasks}
    update_plan_state(plan_state, subtasks, fleet, [], assignments, mode=0, sys_cqi=1.0, current_step=0)

    # Simulate severe CQI drop from 1.0 to 0.2
    # Dynamic architecture would trigger cqi_changed_significantly
    # Static baseline (is_static_baseline=True) MUST NOT trigger
    for step in range(1, 10):
        replan, reason = should_replan(
            plan_state, subtasks, fleet, [],
            mode=0, sys_cqi=0.2, current_step=step,
            is_static_baseline=True
        )
        assert replan is False
        assert reason == ""


def test_static_baseline_ignores_packet_loss_threshold():
    env = DACAEnv("logistics", {}, seed=0)
    plan_state = PlanState()
    subtasks = env.subtask_list
    fleet = env.fleet

    assignments = {s.subtask_id: [fleet.agents[0].agent_id] for s in subtasks}
    update_plan_state(plan_state, subtasks, fleet, [], assignments, mode=0, packet_loss=0.0, current_step=0)

    # Packet loss spikes from 0.0 to 0.8
    replan, reason = should_replan(
        plan_state, subtasks, fleet, [],
        mode=0, packet_loss=0.8, current_step=1,
        is_static_baseline=True
    )
    assert replan is False
    assert reason == ""


def test_no_replan_when_all_tasks_completed():
    env = DACAEnv("logistics", {}, seed=0)
    plan_state = PlanState()
    subtasks = env.subtask_list
    fleet = env.fleet

    assignments = {s.subtask_id: [fleet.agents[0].agent_id] for s in subtasks}
    # Mark all subtasks complete
    for st in subtasks:
        env.mark_subtask_complete(st.subtask_id)

    update_plan_state(plan_state, subtasks, fleet, [], assignments, mode=0, current_step=0)

    # Next step: should NOT replan because mission is already complete
    replan, reason = should_replan(
        plan_state, subtasks, fleet, [],
        mode=0, current_step=1,
        is_static_baseline=True
    )
    assert replan is False
    assert reason == ""
