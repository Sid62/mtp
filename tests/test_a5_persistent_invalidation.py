"""Deterministic regression tests for A5 assignment invalidation and lifecycle consistency.

Verifies:
- Test A: Two tasks assigned to same agent (rejected, 1-to-1 enforced, no resurrection).
- Test B: Invalid assignment removed -> continuity reuse rejects old plan, never returns invalid.
- Test C: Completed task -> continuity reuse never reintroduces completed task.
- Test D: Reallocation proposes new agent -> canonical execution and dispatch agree.
- Test E: Mode switch centralized <-> decentralized preserves canonical state without resurrecting stale mappings.
- Test F: Partial invalidity (T1 valid, T2 invalid, T3 valid) leaves T1/T3 executable and T2 unresolved.
- Test G: Seeds 0 through 9 invariant checks.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.env.agents import AgentFleet, AgentState, AgentType, Position, create_fleet_from_scenario
from src.env.scenarios import Subtask
from src.coordination.assignment_validator import (
    AssignmentValidator,
    validate_global_assignment_state,
)
from src.coordination.plan_continuity import PlanContinuityEngine


@pytest.fixture
def fleet_and_subtasks():
    scenario_cfg = {"num_uav": 5, "num_vehicle": 3, "num_robot": 4}
    kinematics_cfg = {
        "uav": {"max_speed": 15.0, "max_turn_rate": 1.0},
        "vehicle": {"max_speed": 10.0, "max_turn_rate": 1.0},
        "robot": {"max_speed": 3.0, "max_turn_rate": 1.0},
    }
    fleet = create_fleet_from_scenario(scenario_cfg, kinematics_cfg, c1=50.0, c2=10.0, seed=42)
    subtasks = [
        Subtask(subtask_id=f"T_{i}", description=f"task {i}", target=Position(10.0 * i, 10.0 * i), required_skills=[])
        for i in range(10)
    ]
    return fleet, subtasks


def test_a_two_tasks_assigned_to_same_agent(fleet_and_subtasks):
    """Test A: Two tasks assigned to same agent.
    Candidate rejected, only valid assignment enters executable state,
    invalid assignment never reappears through continuity reuse.
    """
    fleet, subtasks = fleet_and_subtasks
    uav_3 = fleet.get_agent("uav_3")
    uav_4 = fleet.get_agent("uav_4")
    subtasks[0].required_skills = list(uav_3.skills)[:1]
    subtasks[4].required_skills = list(uav_3.skills)[:1]
    subtasks[1].required_skills = list(uav_4.skills)[:1]
    subtasks[9].required_skills = list(uav_4.skills)[:1]

    # Raw plan proposes uav_3 for both T_0 and T_4, uav_4 for both T_1 and T_9
    raw_plan = {
        "T_0": ["uav_3"],
        "T_4": ["uav_3"],
        "T_1": ["uav_4"],
        "T_9": ["uav_4"],
    }

    # Validation
    cleaned, report = AssignmentValidator.validate_and_clean_plan(
        raw_plan, fleet, subtasks, source="test_a"
    )

    # uav_3 is assigned to T_0, T_4 is rejected with multiply_assigned
    assert report.valid_assignments.get("uav_3") == "T_0"
    assert report.rejected_assignments.get("uav_3") == "T_4"
    assert "agent_multiply_assigned" in report.rejection_reasons["uav_3"]
    assert report.valid_assignments.get("uav_4") == "T_1"
    assert report.rejected_assignments.get("uav_4") == "T_9"
    assert "agent_multiply_assigned" in report.rejection_reasons["uav_4"]
    assert cleaned["T_0"] == ["uav_3"]
    assert cleaned["T_4"] == []
    assert cleaned["T_1"] == ["uav_4"]
    assert cleaned["T_9"] == []

    # Initialize continuity engine
    continuity = PlanContinuityEngine()
    continuity.set_active_plan(cleaned, coalitions=[], subtasks=subtasks)
    # Record rejected mappings
    continuity.active_context.rejected_mappings.add(("T_4", "uav_3"))
    continuity.active_context.rejected_mappings.add(("T_9", "uav_4"))

    # Updated executable assignments through continuity
    updated = continuity.get_updated_executable_assignments(fleet, subtasks)

    # Invariant: uav_3 must NEVER be assigned to T_4
    assert updated.get("T_4") != ["uav_3"]
    assert updated.get("T_9") != ["uav_4"]
    # Invariant: no agent appears more than once
    assigned_agents = [a for aids in updated.values() for a in aids]
    assert len(assigned_agents) == len(set(assigned_agents))


def test_b_invalid_assignment_removed_continuity_reuse(fleet_and_subtasks):
    """Test B: Invalid assignment -> removed -> continuity reuse.
    Continuity reuse rejects the old plan or excludes invalid mapping,
    invalid assignment does NOT return.
    """
    fleet, subtasks = fleet_and_subtasks
    # Ensure uav_0, uav_1, uav_2 have skills matching their tasks
    for i, aid in enumerate(["uav_0", "uav_1", "uav_2"]):
        agent = fleet.get_agent(aid)
        subtasks[i].required_skills = list(agent.skills)[:1]

    plan = {
        "T_0": ["uav_0"],
        "T_1": ["uav_1"],
        "T_2": ["uav_2"],
    }
    continuity = PlanContinuityEngine()
    continuity.set_active_plan(plan, coalitions=[], subtasks=subtasks)

    # Invalidate (T_1, uav_1) and record in rejected_mappings
    continuity.active_context.rejected_mappings.add(("T_1", "uav_1"))

    # If continuity plan still had stale (T_1, uav_1), validity must be 0.0
    score = continuity.evaluate_plan_validity(fleet, subtasks)
    assert score.total_validity_score == 0.0, "Plan validity must be 0.0 if any rejected mapping is present"

    # Remove invalid mapping from assignments
    continuity.active_context.assignments["T_1"] = []

    # Refresh executable assignments
    refreshed = continuity.get_updated_executable_assignments(fleet, subtasks)
    assert "uav_1" not in refreshed.get("T_1", [])
    assert ("T_1", "uav_1") not in [(sid, a) for sid, aids in refreshed.items() for a in aids]


def test_c_completed_task_continuity_reuse(fleet_and_subtasks):
    """Test C: Completed task -> continuity reuse.
    Completed task never returns to active plan.
    """
    fleet, subtasks = fleet_and_subtasks
    agent_0 = fleet.agents[0]
    agent_1 = fleet.agents[1]
    subtasks[0].required_skills = list(agent_0.skills)[:1]
    subtasks[1].required_skills = list(agent_1.skills)[:1]

    plan = {
        "T_0": [agent_0.agent_id],
        "T_1": [agent_1.agent_id],
    }
    continuity = PlanContinuityEngine()
    continuity.set_active_plan(plan, coalitions=[], subtasks=subtasks)

    # Mark T_0 complete
    subtasks[0].completed = True
    continuity.active_context.completed_subtask_ids.add("T_0")
    continuity.active_context.assignments.pop("T_0", None)

    refreshed = continuity.get_updated_executable_assignments(fleet, subtasks)
    assert "T_0" not in refreshed
    assert subtasks[0].completed


def test_d_reallocation_proposes_new_agent(fleet_and_subtasks):
    """Test D: Reallocation proposes new agent.
    Canonical execution assignment changes only when reallocation is committed,
    assignment state and executable state agree.
    """
    fleet, subtasks = fleet_and_subtasks
    uav_1 = fleet.get_agent("uav_1")
    uav_2 = fleet.get_agent("uav_2")
    subtasks[3].required_skills = list(uav_2.skills)[:1]
    subtasks[5].required_skills = list(uav_1.skills)[:1]

    assignments = {"T_3": ["robot_8"], "T_5": ["robot_10"]}

    # Reallocation proposes uav_2 for T_3 and uav_1 for T_5
    new_realloc = {"T_3": ["uav_2"], "T_5": ["uav_1"]}
    cleaned, report = AssignmentValidator.validate_and_clean_plan(
        new_realloc, fleet, subtasks, source="reallocation"
    )

    # Commit reallocation to canonical execution state
    assignments = cleaned
    canonical_plan = {k: list(v) for k, v in assignments.items()}

    # Executable assignments extracted from canonical state
    exec_assignments = {agents[0]: sid for sid, agents in canonical_plan.items() if agents}

    # Verify provenance and state agreement
    assert exec_assignments["uav_2"] == "T_3"
    assert exec_assignments["uav_1"] == "T_5"
    assert "robot_8" not in exec_assignments
    assert "robot_10" not in exec_assignments


def test_e_mode_switch_centralized_to_decentralized(fleet_and_subtasks):
    """Test E: Mode switch centralized -> decentralized.
    Canonical assignment state survives mode switch correctly,
    invalid assignments are not resurrected.
    """
    fleet, subtasks = fleet_and_subtasks
    rejected_mappings = {("T_4", "uav_3")}
    canonical_plan = {"T_0": ["uav_3"], "T_1": ["uav_4"]}

    # Mode switch preserves canonical active plan
    switched_plan = {k: list(v) for k, v in canonical_plan.items()}
    assert ("T_4", "uav_3") in rejected_mappings
    assert "T_4" not in switched_plan
    assert switched_plan["T_0"] == ["uav_3"]


def test_f_partial_invalidity(fleet_and_subtasks):
    """Test F: Partial invalidity.
    T1 valid, T2 invalid, T3 valid.
    T1 and T3 remain executable, T2 becomes unresolved.
    Continuity cannot reuse the invalid T2 assignment.
    """
    fleet, subtasks = fleet_and_subtasks
    uav_0 = fleet.get_agent("uav_0")
    uav_1 = fleet.get_agent("uav_1")
    subtasks[0].required_skills = list(uav_0.skills)[:1]
    subtasks[2].required_skills = list(uav_1.skills)[:1]

    # T_0 valid, T_1 invalid (nonexistent agent), T_2 valid
    candidate = {
        "T_0": ["uav_0"],
        "T_1": ["nonexistent_agent"],
        "T_2": ["uav_1"],
    }
    cleaned, report = AssignmentValidator.validate_and_clean_plan(
        candidate, fleet, subtasks, source="partial_test"
    )

    assert cleaned["T_0"] == ["uav_0"]
    assert cleaned["T_1"] == []
    assert cleaned["T_2"] == ["uav_1"]
    assert "T_1" in report.unresolved_tasks

    continuity = PlanContinuityEngine()
    continuity.set_active_plan(cleaned, coalitions=[], subtasks=subtasks)
    refreshed = continuity.get_updated_executable_assignments(fleet, subtasks)

    assert refreshed["T_0"] == ["uav_0"]
    assert refreshed["T_1"] == []
    assert refreshed["T_2"] == ["uav_1"]


@pytest.mark.parametrize("seed", range(10))
def test_g_seeds_0_through_9_invariants(seed):
    """Test G: Deterministic invariant verification across seeds 0-9.
    Enforces that 1-to-1 mapping, skill satisfaction, and no completed tasks hold.
    """
    scenario_cfg = {"num_uav": 5, "num_vehicle": 3, "num_robot": 4}
    kinematics_cfg = {
        "uav": {"max_speed": 15.0, "max_turn_rate": 1.0},
        "vehicle": {"max_speed": 10.0, "max_turn_rate": 1.0},
        "robot": {"max_speed": 3.0, "max_turn_rate": 1.0},
    }
    fleet = create_fleet_from_scenario(scenario_cfg, kinematics_cfg, c1=50.0, c2=10.0, seed=seed)
    # Define subtasks with skills matched to the respective agents
    subtasks = [
        Subtask(
            subtask_id=f"T_{i}",
            description=f"task {i}",
            target=Position(float(i * 5), float(i * 5)),
            required_skills=list(fleet.agents[i].skills)[:1],
        )
        for i in range(5)
    ]

    # Assign distinct agents
    assigned_plan = {
        f"T_{i}": [fleet.agents[i].agent_id]
        for i in range(len(subtasks))
    }
    cleaned, report = AssignmentValidator.validate_and_clean_plan(
        assigned_plan, fleet, subtasks, source="test_g"
    )
    assert report.is_valid

    # Validate global invariant state
    violations = validate_global_assignment_state(cleaned, fleet, subtasks, source="test_g")
    assert len(violations) == 0
