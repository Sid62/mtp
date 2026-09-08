"""Regression tests for Phase 2: Task Lifecycle and Active Planning Context."""

import pytest
from src.env.daca_env import DACAEnv
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator


def test_completed_task_excluded_from_active_planning():
    env = DACAEnv("logistics", {}, seed=0)
    subtasks = env.subtask_list
    assert len(subtasks) == 6

    # Mark T_0 as completed
    env.mark_subtask_complete("T_0")
    assert subtasks[0].completed is True

    # Active tasks should be 5
    active_subtasks = [s for s in subtasks if not s.completed]
    assert len(active_subtasks) == 5
    assert "T_0" not in [s.subtask_id for s in active_subtasks]

    # Observation should reflect completed status
    obs = env.get_observation()
    active_obs_subtasks = [s for s in obs["subtasks"] if not s["completed"]]
    assert len(active_obs_subtasks) == 5
    assert all(s["id"] != "T_0" for s in active_obs_subtasks)


def test_completed_task_cannot_receive_new_assignment():
    coordinator = CentralizedHybridCoordinator(cloud_llm=None, device_llms={})
    # Simulate fallback assignments containing both active and completed subtasks
    fallback = {
        "T_0": ["uav_0"],
        "T_1": ["uav_1"],
    }
    # T_0 is completed, only T_1 is in valid_subtask_ids
    valid_sids = {"T_1"}
    exec_assigns = coordinator.extract_executable_assignments(fallback, valid_subtask_ids=valid_sids)

    # T_0 must be excluded
    assert "T_0" not in exec_assigns.values()
    assert "uav_0" not in exec_assigns
    # T_1 must be included
    assert exec_assigns.get("uav_1") == "T_1"


def test_total_task_count_invariant_and_active_count_decrease():
    env = DACAEnv("logistics", {}, seed=0)
    initial_total = len(env.subtask_list)
    assert initial_total == 6

    # Complete tasks one by one
    for idx, st in enumerate(env.subtask_list):
        env.mark_subtask_complete(st.subtask_id)
        # Total tasks invariant
        assert len(env.subtask_list) == initial_total
        # Completed tasks count
        assert len(env.state.completed_subtasks) == idx + 1
        # Active tasks count decreases
        active_remaining = [s for s in env.subtask_list if not s.completed]
        assert len(active_remaining) == initial_total - (idx + 1)
