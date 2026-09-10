"""Tests for Phase 6 (AssignmentValidator and Failure Guard)."""

from __future__ import annotations

import pytest
import numpy as np
from src.coordination.assignment_validator import AssignmentValidator
from src.env.agents import AgentFleet, AgentState, AgentType, Position, KinematicsConfig
from src.env.scenarios import Subtask


def _create_fleet() -> AgentFleet:
    a0 = AgentState("uav_0", AgentType.UAV, Position(0.0, 0.0), skills=["navigate", "sensor"])
    a1 = AgentState("ugv_0", AgentType.VEHICLE, Position(10.0, 10.0), skills=["transport", "heavy_lift"])
    kin = {
        "uav": KinematicsConfig(max_speed=10.0, max_turn_rate=1.0),
        "vehicle": KinematicsConfig(max_speed=5.0, max_turn_rate=0.5),
    }
    return AgentFleet([a0, a1], kin)


def _create_subtasks() -> list[Subtask]:
    s0 = Subtask("T_0", "Subtask 0", Position(5.0, 5.0), required_skills=["navigate"], priority=1)
    s1 = Subtask("T_1", "Subtask 1", Position(15.0, 15.0), required_skills=["transport"], priority=2)
    s2 = Subtask("T_2", "Subtask 2", Position(20.0, 20.0), required_skills=["underwater"], priority=1)
    return [s0, s1, s2]


def test_validator_accepts_valid_assignments():
    fleet = _create_fleet()
    subtasks = _create_subtasks()
    candidates = {"uav_0": "T_0", "ugv_0": "T_1"}

    report = AssignmentValidator.filter_assignments(
        candidates, fleet, subtasks, check_skills=True
    )
    assert report.is_valid
    assert len(report.valid_assignments) == 2
    assert len(report.rejected_assignments) == 0


def test_validator_rejects_unknown_agent():
    fleet = _create_fleet()
    subtasks = _create_subtasks()
    candidates = {"phantom_agent": "T_0"}

    report = AssignmentValidator.filter_assignments(
        candidates, fleet, subtasks
    )
    assert not report.is_valid
    assert "phantom_agent" in report.rejected_assignments
    assert "unknown_agent" in report.rejection_reasons["phantom_agent"]


def test_validator_rejects_completed_task():
    fleet = _create_fleet()
    subtasks = _create_subtasks()
    subtasks[0].completed = True
    candidates = {"uav_0": "T_0"}

    report = AssignmentValidator.filter_assignments(
        candidates, fleet, subtasks
    )
    assert not report.is_valid
    assert "uav_0" in report.rejected_assignments
    assert "task_already_completed" in report.rejection_reasons["uav_0"]


def test_validator_rejects_duplicate_agent():
    fleet = _create_fleet()
    subtasks = _create_subtasks()
    # Same agent assigned to multiple tasks
    valid1, _ = AssignmentValidator.validate_single("uav_0", "T_0", fleet, subtasks, assigned_agents=set())
    assert valid1
    valid2, reason = AssignmentValidator.validate_single("uav_0", "T_1", fleet, subtasks, assigned_agents={"uav_0"})
    assert not valid2
    assert "agent_multiply_assigned" in reason


def test_validator_rejects_skill_mismatch():
    fleet = _create_fleet()
    subtasks = _create_subtasks()
    # uav_0 only has navigate and sensor, T_2 requires underwater
    candidates = {"uav_0": "T_2"}
    report = AssignmentValidator.filter_assignments(
        candidates, fleet, subtasks, check_skills=True, strict_skills=False
    )
    assert not report.is_valid
    assert "uav_0" in report.rejected_assignments
    assert "missing_required_skill" in report.rejection_reasons["uav_0"]


def test_validator_coalition_skill_coverage():
    fleet = _create_fleet()
    # Task requires both navigate and transport
    s_joint = Subtask("T_joint", "Joint Task", Position(5.0, 5.0), required_skills=["navigate", "transport"], priority=1)
    subtasks = [s_joint]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "ugv_0"]}]

    # Individually uav_0 lacks transport, but coalition covers it
    valid, reason = AssignmentValidator.validate_single(
        "uav_0", "T_joint", fleet, subtasks, check_skills=True, strict_skills=True, coalitions=coalitions
    )
    assert valid
