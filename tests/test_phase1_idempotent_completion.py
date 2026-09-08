"""Regression tests for Phase 1: Idempotent Task Completion."""

import pytest
from src.env.daca_env import DACAEnv
from src.coordination.replan_trigger import PlanState, should_replan, update_plan_state
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def test_complete_same_task_twice():
    env = DACAEnv("logistics", {}, seed=0)
    st = env.subtask_list[0]
    sid = st.subtask_id

    # First completion
    first_res = env.mark_subtask_complete(sid)
    assert first_res is True
    assert st.completed is True
    assert env.state.completed_subtasks == [sid]

    # Second completion
    second_res = env.mark_subtask_complete(sid)
    assert second_res is False
    assert env.state.completed_subtasks == [sid]
    assert len(env.state.completed_subtasks) == 1


def test_complete_same_task_n_times():
    env = DACAEnv("logistics", {}, seed=0)
    st = env.subtask_list[0]
    sid = st.subtask_id

    assert env.mark_subtask_complete(sid) is True
    for _ in range(10):
        assert env.mark_subtask_complete(sid) is False

    assert env.state.completed_subtasks == [sid]
    assert len(env.state.completed_subtasks) == 1


def test_completed_task_never_transitions_back():
    env = DACAEnv("logistics", {}, seed=0)
    st = env.subtask_list[0]
    sid = st.subtask_id

    env.mark_subtask_complete(sid)
    assert st.completed is True
    # Invariant: subtask remains completed across env advances
    env.advance()
    assert st.completed is True
    assert sid in env.state.completed_subtasks


def test_completed_subtasks_has_no_duplicates():
    env = DACAEnv("logistics", {}, seed=0)
    for st in env.subtask_list:
        env.mark_subtask_complete(st.subtask_id)
        env.mark_subtask_complete(st.subtask_id)

    assert len(env.state.completed_subtasks) == len(set(env.state.completed_subtasks))
    assert len(env.state.completed_subtasks) == len(env.subtask_list)


def test_completed_task_does_not_trigger_repeated_replans():
    env = DACAEnv("logistics", {}, seed=0)
    plan_state = PlanState()
    subtasks = env.subtask_list
    fleet = env.fleet

    # Initialize plan
    assignments = {s.subtask_id: [fleet.agents[0].agent_id] for s in subtasks}
    update_plan_state(plan_state, subtasks, fleet, [], assignments, mode=0, current_step=0)
    assert plan_state.known_completed_ids == set()

    # Step 1: Mark T_0 complete
    env.mark_subtask_complete(subtasks[0].subtask_id)
    replan, reason = should_replan(plan_state, subtasks, fleet, [], mode=0, current_step=1)
    assert replan is True
    assert f"task_completed_needs_reassignment:['{subtasks[0].subtask_id}']" in reason

    # Update plan state after replan
    update_plan_state(plan_state, subtasks, fleet, [], assignments, mode=0, current_step=1)
    assert subtasks[0].subtask_id in plan_state.known_completed_ids

    # Step 2: In the next step, no new task is completed
    replan, reason = should_replan(plan_state, subtasks, fleet, [], mode=0, current_step=2)
    assert replan is False
    assert reason == ""
