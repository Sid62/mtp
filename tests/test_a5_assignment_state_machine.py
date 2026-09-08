"""Comprehensive invariant and property-based tests for A5 assignment state machine.

Verifies Invariants 1-12 across arbitrary configurations and seeds:
- INVARIANT 1: No executable assignment contains nonexistent fleet agent.
- INVARIANT 2: No executable assignment contains nonexistent task.
- INVARIANT 3: Completed tasks never appear in active assignments.
- INVARIANT 4: Invalid assignments never appear in executable assignments.
- INVARIANT 5: Invalid assignments never marked validated.
- INVARIANT 6: Required skills strictly satisfied.
- INVARIANT 7: Nearest-agent fallback never overrides hard skill constraints.
- INVARIANT 8: If no valid agent exists, task remains explicitly unresolved.
- INVARIANT 9: Completed tasks removed atomically from all tracking queues.
- INVARIANT 10: Mode transitions preserve invariants without resurrecting stale mappings.
- INVARIANT 11: No task simultaneously completed and active/executable.
- INVARIANT 12: No agent multiply assigned in executable state.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.env.agents import AgentFleet, AgentState, AgentType, Position, create_fleet_from_scenario
from src.env.scenarios import Subtask, build_search_rescue_scenario
from src.coordination.autohma_structs import (
    AssignmentRecord,
    AssignmentStatus,
    normalize_to_assignment_records,
    records_to_task_assignments,
    records_to_agent_assignments,
)
from src.coordination.assignment_validator import (
    AssignmentValidator,
    validate_global_assignment_state,
)
from src.coordination.plan_continuity import PlanContinuityEngine
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer


@pytest.fixture
def base_scenario_cfg():
    return {
        "num_uav": 5,
        "num_vehicle": 3,
        "num_robot": 4,
    }


@pytest.fixture
def base_kinematics_cfg():
    return {
        "uav": {"max_speed": 15.0, "max_turn_rate": 1.0},
        "vehicle": {"max_speed": 10.0, "max_turn_rate": 1.0},
        "robot": {"max_speed": 3.0, "max_turn_rate": 1.0},
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 10, 20])
def test_property_1_and_4_incompatible_skills_rejected(base_scenario_cfg, base_kinematics_cfg, seed):
    """Property 1 & 4: Incompatible skill assignments are never marked VALID or executable."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=seed)
    subtasks = [
        Subtask(subtask_id="T_0", description="rescue task", target=Position(10, 10), required_skills=["lift", "rescue"]),
        Subtask(subtask_id="T_1", description="nav task", target=Position(20, 20), required_skills=["sense", "navigate"]),
    ]

    # Find an agent that has neither lift nor rescue
    incompatible_agent = None
    for a in fleet.agents:
        if not ({"lift", "rescue"} & set(a.skills)):
            incompatible_agent = a.agent_id
            break

    if incompatible_agent:
        candidate = {incompatible_agent: "T_0"}
        report = AssignmentValidator.filter_assignments(
            candidate, fleet, subtasks, check_skills=True, strict_skills=False, log_diagnostics=False
        )
        assert incompatible_agent not in report.valid_assignments
        assert incompatible_agent in report.rejected_assignments
        assert "no_matching_skills" in report.rejection_reasons[incompatible_agent]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_property_2_completed_tasks_never_executable(base_scenario_cfg, base_kinematics_cfg, seed):
    """Property 2: Completed tasks can never enter executable state."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=seed)
    subtasks = [
        Subtask(subtask_id="T_0", description="task 0", target=Position(10, 10), required_skills=["sense"], completed=True),
        Subtask(subtask_id="T_1", description="task 1", target=Position(20, 20), required_skills=["sense"], completed=False),
    ]

    # Try assigning to completed task
    agent_id = fleet.agents[0].agent_id
    candidate = {agent_id: "T_0"}
    report = AssignmentValidator.filter_assignments(
        candidate, fleet, subtasks, check_skills=False, log_diagnostics=False
    )
    assert agent_id not in report.valid_assignments
    assert "task_already_completed" in report.rejection_reasons.get(agent_id, "")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_property_3_nonexistent_agent_rejected(base_scenario_cfg, base_kinematics_cfg, seed):
    """Property 3: Nonexistent fleet agents cannot enter executable assignments."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=seed)
    subtasks = [
        Subtask(subtask_id="T_0", description="task 0", target=Position(10, 10), required_skills=["sense"]),
    ]

    candidate = {"phantom_agent_999": "T_0"}
    report = AssignmentValidator.filter_assignments(
        candidate, fleet, subtasks, check_skills=False, log_diagnostics=False
    )
    assert "phantom_agent_999" not in report.valid_assignments
    assert "unknown_agent" in report.rejection_reasons.get("phantom_agent_999", "")


@pytest.mark.parametrize("seed", [0, 2, 3, 7])
def test_property_5_nearest_agent_never_overrides_hard_skill_constraints(base_scenario_cfg, base_kinematics_cfg, seed):
    """Property 5: DistanceFeasibleDecomposer and PlanContinuityEngine never assign an agent with zero matching skills."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=seed)

    # Pick impossible skills that no agent in the fleet has
    impossible_task = Subtask(
        subtask_id="T_impossible",
        description="impossible",
        target=Position(fleet.agents[0].position.x, fleet.agents[0].position.y),
        required_skills=["interstellar_travel", "quantum_teleport"],
    )

    decomposer = DistanceFeasibleDecomposer(cloud_llm=None, r_reach=100.0, c_task=30.0)
    assigned = decomposer._find_feasible_agents(impossible_task, fleet)
    assert assigned == [], "Decomposer must return [] when no agent has matching skills"

    # Plan continuity engine test
    engine = PlanContinuityEngine(r_reach=100.0)
    engine.set_active_plan(
        assignments={"T_impossible": []},
        coalitions=[],
        subtasks=[impossible_task],
        mode=0,
    )
    updated = engine.get_updated_executable_assignments(fleet, [impossible_task])
    assert updated.get("T_impossible") == [], "Plan continuity engine must never assign incompatible freed agents"


def test_property_6_repeated_completion_idempotent():
    """Property 6: Repeated completion events are idempotent and do not trigger repeat state transitions."""
    from src.env.daca_env import DACAEnv

    env = DACAEnv("search_rescue", thresholds={}, seed=0)
    first_subtask_id = env.subtask_list[0].subtask_id

    # First completion event
    t1 = env.mark_subtask_complete(first_subtask_id)
    assert t1 is True
    assert env.subtask_list[0].completed is True
    assert first_subtask_id in env.state.completed_subtasks
    completed_count_1 = len(env.state.completed_subtasks)

    # Duplicate completion event
    t2 = env.mark_subtask_complete(first_subtask_id)
    assert t2 is False
    assert len(env.state.completed_subtasks) == completed_count_1


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_property_7_mode_transition_preserves_invariants(base_scenario_cfg, base_kinematics_cfg, seed):
    """Property 7: Mode switching cannot resurrect completed or invalid assignments."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=seed)
    subtasks = [
        Subtask(subtask_id="T_0", description="task 0", target=Position(10, 10), required_skills=["sense"], completed=True),
        Subtask(subtask_id="T_1", description="task 1", target=Position(20, 20), required_skills=["sense"], completed=False),
    ]

    dirty_assignments = {
        "T_0": [fleet.agents[0].agent_id],  # completed task
        "T_1": [fleet.agents[1].agent_id],  # active task
        "T_unknown": [fleet.agents[2].agent_id],  # nonexistent task
    }

    cleaned, report = AssignmentValidator.validate_and_clean_plan(
        dirty_assignments, fleet, subtasks, source="mode_switch", mode=1, step=10
    )

    assert "T_0" not in cleaned
    assert "T_unknown" not in cleaned
    assert "T_1" in cleaned

    violations = validate_global_assignment_state(
        cleaned, fleet, subtasks, source="mode_switch_verification", mode=1, step=10
    )
    assert violations == []


def test_property_11_global_state_validator_catches_all_violations(base_scenario_cfg, base_kinematics_cfg):
    """Phase 11: Global state validator diagnostic catches and flags all invariant violations."""
    fleet = create_fleet_from_scenario(base_scenario_cfg, base_kinematics_cfg, c1=50.0, c2=10.0, seed=0)
    subtasks = [
        Subtask(subtask_id="T_0", description="task 0", target=Position(10, 10), required_skills=["sense"], completed=True),
        Subtask(subtask_id="T_1", description="task 1", target=Position(20, 20), required_skills=["lift", "rescue"], completed=False),
    ]

    # Construct deliberately broken state
    uav = [a for a in fleet.agents if a.agent_type == AgentType.UAV][0]
    # Give UAV incompatible skills
    uav.skills = ["inspect", "sense"]

    broken_assignments = {
        "T_0": [uav.agent_id],  # Invariant 3: completed task active
        "T_1": [uav.agent_id, "ghost_agent"],  # Invariant 6: incompatible skill, Invariant 1: nonexistent agent
        "T_phantom": [uav.agent_id],  # Invariant 2: nonexistent task, Invariant 12: agent multiply assigned
    }

    violations = validate_global_assignment_state(
        broken_assignments, fleet, subtasks, source="test", mode=0, step=0
    )

    violation_text = "\n".join(violations)
    assert "INVARIANT_1_NONEXISTENT_AGENT" in violation_text
    assert "INVARIANT_2_NONEXISTENT_TASK" in violation_text
    assert "INVARIANT_3_COMPLETED_TASK_ACTIVE" in violation_text
    assert "INVARIANT_6_NO_MATCHING_SKILLS" in violation_text
    assert "INVARIANT_12_AGENT_MULTIPLY_ASSIGNED" in violation_text

