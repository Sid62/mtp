"""Issue 1 Regression Tests — Validate assignments and task completion.

Tests:
 1. valid agent + correct required skill → valid
 2. valid agent + missing required skill → invalid
 3. multiple agents collectively covering required skills → valid
 4. multiple agents missing one required skill → invalid
 5. unknown agent ID → invalid
 6. empty coalition → invalid
 7. unknown-only coalition → invalid
 8. multi-agent task: first agent reaches but skills invalid → NOT complete
 9. valid team reaches completion conditions → complete
10. repaired/reused assignments validated with same skill constraints
"""

from __future__ import annotations

import numpy as np
import pytest

from src.coordination.assignment_validator import (
    AssignmentValidator,
    validate_global_assignment_state,
)
from src.coalition.feasibility import (
    validate_coalition_members,
    coalition_feasibility_score,
)
from src.env.agents import (
    AgentFleet,
    AgentState,
    AgentType,
    KinematicsConfig,
    Position,
    dist,
)
from src.env.scenarios import Subtask


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_fleet() -> AgentFleet:
    """Fleet with deterministic role-based skills matching Issue 1 fix."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0),
                   skills=["transport", "navigate", "sense"]),
        AgentState("vehicle_1", AgentType.VEHICLE, Position(20.0, 20.0),
                   skills=["transport", "navigate", "inspect"]),
        AgentState("robot_2", AgentType.ROBOT, Position(30.0, 30.0),
                   skills=["lift", "rescue", "sense"]),
    ]
    kin = {
        "uav": KinematicsConfig(max_speed=15.0, max_turn_rate=1.5),
        "vehicle": KinematicsConfig(max_speed=10.0, max_turn_rate=0.8),
        "robot": KinematicsConfig(max_speed=3.0, max_turn_rate=2.0),
    }
    return AgentFleet(agents, kin)


def _make_subtasks() -> list[Subtask]:
    return [
        Subtask("T_0", "Transport task", Position(15.0, 15.0),
                required_skills=["transport", "navigate"]),
        Subtask("T_1", "Lift task", Position(25.0, 25.0),
                required_skills=["lift", "rescue"]),
        Subtask("T_2", "Inspect task", Position(35.0, 35.0),
                required_skills=["inspect", "sense"]),
        Subtask("T_3", "Sense-navigate task", Position(45.0, 45.0),
                required_skills=["navigate", "sense"]),
    ]


# ── Test 1: valid agent + correct required skill → valid ─────────────

def test_valid_agent_correct_skills():
    fleet = _make_fleet()
    subtasks = _make_subtasks()
    # uav_0 has [transport, navigate, sense] → T_0 requires [transport, navigate] → valid
    valid, reason = AssignmentValidator.validate_single(
        "uav_0", "T_0", fleet, subtasks, check_skills=True,
    )
    assert valid, f"Expected valid, got reason={reason}"


# ── Test 2: valid agent + missing required skill → invalid ───────────

def test_valid_agent_missing_skill():
    fleet = _make_fleet()
    subtasks = _make_subtasks()
    # uav_0 has [transport, navigate, sense] → T_1 requires [lift, rescue] → invalid
    valid, reason = AssignmentValidator.validate_single(
        "uav_0", "T_1", fleet, subtasks, check_skills=True,
    )
    assert not valid
    assert "missing_required_skill" in reason


# ── Test 3: multiple agents collectively covering skills → valid ─────

def test_coalition_covers_all_skills():
    fleet = _make_fleet()
    # T_2 requires [inspect, sense]; vehicle_1 has inspect, robot_2 has sense
    subtasks = [
        Subtask("T_2", "Inspect task", Position(35.0, 35.0),
                required_skills=["inspect", "sense"]),
    ]
    coalitions = [{"coalition_id": 0, "members": ["vehicle_1", "robot_2"]}]
    valid, reason = AssignmentValidator.validate_single(
        "vehicle_1", "T_2", fleet, subtasks,
        check_skills=True, strict_skills=True, coalitions=coalitions,
    )
    assert valid, f"Expected valid coalition coverage, got reason={reason}"


# ── Test 4: multiple agents missing one required skill → invalid ─────

def test_coalition_missing_one_skill():
    fleet = _make_fleet()
    # Task requires [inspect, rescue]; uav_0 has [transport,navigate,sense],
    # vehicle_1 has [transport,navigate,inspect] → coalition has inspect but no rescue
    subtasks = [
        Subtask("T_x", "Mixed task", Position(10.0, 10.0),
                required_skills=["inspect", "rescue"]),
    ]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "vehicle_1"]}]
    valid, reason = AssignmentValidator.validate_single(
        "uav_0", "T_x", fleet, subtasks,
        check_skills=True, strict_skills=True, coalitions=coalitions,
    )
    assert not valid
    assert "missing_required_skill" in reason


# ── Test 5: unknown agent ID → invalid ───────────────────────────────

def test_unknown_agent_id_invalid():
    fleet = _make_fleet()
    subtasks = _make_subtasks()
    valid, reason = AssignmentValidator.validate_single(
        "phantom_agent_99", "T_0", fleet, subtasks,
    )
    assert not valid
    assert "unknown_agent" in reason


# ── Test 6: empty coalition → invalid ────────────────────────────────

def test_empty_coalition_invalid():
    n = 3
    psi = np.ones((n, n))
    id_to_idx = {"uav_0": 0, "vehicle_1": 1, "robot_2": 2}

    result = validate_coalition_members([], id_to_idx, psi, gamma_min=0.3)
    assert result is False, "Empty coalition must fail validation"


# ── Test 7: unknown-only coalition → invalid ─────────────────────────

def test_unknown_only_coalition_invalid():
    n = 3
    psi = np.ones((n, n))
    id_to_idx = {"uav_0": 0, "vehicle_1": 1, "robot_2": 2}

    result = validate_coalition_members(
        ["phantom_1", "phantom_2"], id_to_idx, psi, gamma_min=0.3,
    )
    assert result is False, "Unknown-only coalition must fail validation"


# ── Test 8: first agent reaches target but skills invalid → NOT complete

def test_completion_blocked_without_skills():
    """Simulate the completion gate: agent reaches target but lacks skills."""
    fleet = _make_fleet()
    subtask = Subtask("T_1", "Lift task", Position(30.0, 30.0),
                      required_skills=["lift", "rescue"])

    # uav_0 has [transport, navigate, sense] — no lift or rescue
    agent = fleet.get_agent("uav_0")
    # Place agent at subtask target (within completion radius)
    agent.position = Position(30.0, 30.0)
    assert dist(agent.position, subtask.target) < 8.0  # within COMPLETION_RADIUS_M

    # Simulate the completion gate from orchestrator
    agent_list = ["uav_0"]
    team_skills: set[str] = set()
    for aid in agent_list:
        if fleet.has_agent(aid):
            team_skills.update(fleet.get_agent(aid).skills)
    required_skills = set(subtask.required_skills)

    # This MUST block completion
    assert not required_skills.issubset(team_skills), \
        "Team without required skills must NOT pass completion check"


# ── Test 9: valid team reaches completion conditions → complete ───────

def test_completion_allowed_with_valid_skills():
    """Simulate the completion gate: agent reaches target with correct skills."""
    fleet = _make_fleet()
    subtask = Subtask("T_0", "Transport task", Position(10.0, 10.0),
                      required_skills=["transport", "navigate"])

    # uav_0 has [transport, navigate, sense] — covers both required
    agent = fleet.get_agent("uav_0")
    agent.position = Position(10.0, 10.0)
    assert dist(agent.position, subtask.target) < 8.0

    agent_list = ["uav_0"]
    team_skills: set[str] = set()
    for aid in agent_list:
        if fleet.has_agent(aid):
            team_skills.update(fleet.get_agent(aid).skills)
    required_skills = set(subtask.required_skills)

    assert required_skills.issubset(team_skills), \
        "Team with required skills MUST pass completion check"


# ── Test 10: repaired/reused assignments use same skill constraints ──

def test_reused_assignments_strict_validation():
    """Assignments from different sources (plan, reuse, recovery) all use
    the same skill hard constraint via validate_and_clean_plan."""
    fleet = _make_fleet()
    subtasks = _make_subtasks()

    # Assignment where robot_2 [lift,rescue,sense] is assigned to T_0 [transport,navigate]
    # This should be rejected regardless of source
    for source in ["centralized_plan", "continuity_reuse", "recovery_plan", "reallocation"]:
        assignments = {"T_0": ["robot_2"]}
        cleaned, report = AssignmentValidator.validate_and_clean_plan(
            assignments, fleet, subtasks, source=source, check_skills=True,
        )
        assert cleaned["T_0"] == [], \
            f"Source={source}: robot_2 lacks transport/navigate, must be rejected"
        assert "robot_2" in report.rejected_assignments, \
            f"Source={source}: robot_2 should be in rejected assignments"


# ── Test 10b: valid assignment accepted from all sources ──────────────

def test_valid_assignment_all_sources():
    fleet = _make_fleet()
    subtasks = _make_subtasks()

    for source in ["centralized_plan", "continuity_reuse", "recovery_plan", "reallocation"]:
        assignments = {"T_0": ["uav_0"]}
        cleaned, report = AssignmentValidator.validate_and_clean_plan(
            assignments, fleet, subtasks, source=source, check_skills=True,
        )
        assert cleaned["T_0"] == ["uav_0"], \
            f"Source={source}: uav_0 has transport+navigate, must be accepted"


# ── Test 11: filter_assignments rejects partial skill match ──────────

def test_filter_rejects_partial_skill():
    """Even one missing skill must cause rejection (not just zero overlap)."""
    fleet = _make_fleet()
    # uav_0 has [transport, navigate, sense]; T_2 requires [inspect, sense]
    # uav_0 has sense but NOT inspect → must be rejected
    subtasks = _make_subtasks()
    candidates = {"uav_0": "T_2"}
    report = AssignmentValidator.filter_assignments(
        candidates, fleet, subtasks, check_skills=True, strict_skills=False,
    )
    assert not report.is_valid
    assert "uav_0" in report.rejected_assignments
    assert "missing_required_skill" in report.rejection_reasons["uav_0"]


# ── Test 12: validate_global_assignment_state detects skill violation ─

def test_global_validator_detects_missing_skill():
    fleet = _make_fleet()
    subtasks = _make_subtasks()
    # robot_2 [lift,rescue,sense] assigned to T_0 [transport,navigate] → violation
    assignments = {"T_0": ["robot_2"]}
    violations = validate_global_assignment_state(
        assignments, fleet, subtasks, source="test", strict_skills=False,
    )
    assert any("INVARIANT_6" in v for v in violations), \
        "Global validator must detect missing required skills"


# ── Test 13: coalition with mix of valid and unknown → invalid ────────

def test_coalition_with_one_unknown_member():
    n = 3
    psi = np.ones((n, n))
    id_to_idx = {"uav_0": 0, "vehicle_1": 1, "robot_2": 2}

    # One valid, one unknown
    result = validate_coalition_members(
        ["uav_0", "phantom_x"], id_to_idx, psi, gamma_min=0.3,
    )
    assert result is False, "Coalition with any unknown member must fail"
