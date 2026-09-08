"""Regression tests for B1 Baseline Assignment Schema Normalization & Execution Validation.

Validates:
TEST 1: Device object with agent_id/subtask_id parses correctly.
TEST 2: Task-key mapping {"T_6": ["uav_1"]} parses correctly.
TEST 3: The field names agent_id/subtask_id can never become fleet IDs.
TEST 4: Invalid agent_id "agent_id" is rejected.
TEST 5: Unknown agent ID is rejected.
TEST 6: Unknown task ID is rejected.
TEST 7: Completed task cannot be assigned.
TEST 8: The exact DEVICE_RAW structure from the B1 failure log reaches normalized execution correctly.
TEST 9: Valid assignment reaches NMPC without ValueError.
"""

from __future__ import annotations

import pytest
from src.coordination.autohma_structs import (
    ExecutionDirective,
    RESERVED_SCHEMA_KEYS,
    normalize_device_dispatch,
)
from src.coordination.assignment_validator import AssignmentValidator
from src.env.agents import AgentFleet, Position, AgentState, AgentType, KinematicsConfig
from src.env.scenarios import Subtask
from src.control.nmpc import NMPCController
from src.handoff.ca_transfer import CATransferManager


@pytest.fixture
def test_fleet():
    agents = [
        AgentState(agent_id="uav_1", agent_type=AgentType.UAV, position=Position(0, 0), skills=["navigate"]),
        AgentState(agent_id="uav_2", agent_type=AgentType.UAV, position=Position(10, 10), skills=["navigate"]),
        AgentState(agent_id="robot_8", agent_type=AgentType.ROBOT, position=Position(20, 20), skills=["transport"]),
    ]
    kin = {
        "uav": KinematicsConfig(max_speed=10.0, max_turn_rate=1.0),
        "robot": KinematicsConfig(max_speed=5.0, max_turn_rate=0.5),
    }
    return AgentFleet(agents=agents, kinematics=kin)


@pytest.fixture
def test_subtasks():
    return [
        Subtask(subtask_id="T_1", description="Task 1", target=Position(100, 100), required_skills=["navigate"]),
        Subtask(subtask_id="T_6", description="Task 6", target=Position(200, 200), required_skills=["navigate"]),
        Subtask(subtask_id="T_7", description="Task 7", target=Position(300, 300), required_skills=["transport"]),
        Subtask(subtask_id="T_completed", description="Task completed", target=Position(400, 400), required_skills=["navigate"], completed=True),
    ]


# ==============================================================================
# TEST 1: Device object with agent_id/subtask_id parses correctly (Schema B)
# ==============================================================================
def test_schema_b_parsing():
    raw = {
        "agent_id": "uav_1",
        "subtask_id": "T_6",
        "waypoints": [
            {"x": 100, "y": 200},
            {"x": 300, "y": 400},
            {"x": 500, "y": 600},
        ],
    }
    result = normalize_device_dispatch(raw, valid_subtask_ids={"T_6"})
    assert result == {"uav_1": "T_6"}
    assert "agent_id" not in result
    assert "subtask_id" not in result


# ==============================================================================
# TEST 2: Task-key mapping parses correctly (Schema A)
# ==============================================================================
def test_schema_a_parsing():
    raw = {
        "T_6": ["uav_1"],
        "T_7": ["robot_8"],
    }
    result = normalize_device_dispatch(raw, valid_subtask_ids={"T_6", "T_7"})
    assert result == {"uav_1": "T_6", "robot_8": "T_7"}


# ==============================================================================
# TEST 3: The field names agent_id/subtask_id can never become fleet IDs
# ==============================================================================
def test_reserved_schema_field_names_cannot_become_fleet_ids():
    raw = {
        "assignments": {
            "agent_id": "uav_1",
            "subtask_id": "T_6",
            "waypoints": "T_1",
            "dispatched": "T_6",
            "ack": "T_7",
        }
    }
    result = normalize_device_dispatch(raw, valid_subtask_ids={"T_1", "T_6", "T_7"})
    # Only uav_1 is a valid agent ID
    assert result == {"uav_1": "T_6"}
    for reserved in RESERVED_SCHEMA_KEYS:
        assert reserved not in result


# ==============================================================================
# TEST 4: Invalid agent_id "agent_id" is rejected
# ==============================================================================
def test_invalid_agent_id_rejected(test_fleet, test_subtasks):
    valid, reason = AssignmentValidator.validate_single(
        agent_id="agent_id",
        subtask_id="T_6",
        fleet=test_fleet,
        subtasks=test_subtasks,
    )
    assert not valid
    assert "reserved_schema_keyword" in reason


# ==============================================================================
# TEST 5: Unknown agent ID is rejected
# ==============================================================================
def test_unknown_agent_id_rejected(test_fleet, test_subtasks):
    valid, reason = AssignmentValidator.validate_single(
        agent_id="ghost_agent_99",
        subtask_id="T_6",
        fleet=test_fleet,
        subtasks=test_subtasks,
    )
    assert not valid
    assert "unknown_agent" in reason


# ==============================================================================
# TEST 6: Unknown task ID is rejected
# ==============================================================================
def test_unknown_task_id_rejected(test_fleet, test_subtasks):
    valid, reason = AssignmentValidator.validate_single(
        agent_id="uav_1",
        subtask_id="T_nonexistent_99",
        fleet=test_fleet,
        subtasks=test_subtasks,
    )
    assert not valid
    assert "unknown_task" in reason


# ==============================================================================
# TEST 7: Completed task cannot be assigned
# ==============================================================================
def test_completed_task_rejected(test_fleet, test_subtasks):
    valid, reason = AssignmentValidator.validate_single(
        agent_id="uav_1",
        subtask_id="T_completed",
        fleet=test_fleet,
        subtasks=test_subtasks,
    )
    assert not valid
    assert "task_already_completed" in reason


# ==============================================================================
# TEST 8: Exact DEVICE_RAW structure from B1 failure reaches normalized execution
# ==============================================================================
def test_exact_device_raw_conversion(test_fleet, test_subtasks):
    device_raw = {
        "uav": {
            "dispatched": True,
            "assignments": {
                "agent_id": "uav_1",
                "subtask_id": "T_6",
                "waypoints": [
                    {"x": 100, "y": 200},
                    {"x": 300, "y": 400},
                    {"x": 500, "y": 600},
                ],
            },
            "ack": True,
        }
    }
    normalized = normalize_device_dispatch(
        device_raw,
        managed_agent_ids={"uav_1", "uav_2"},
        valid_subtask_ids={"T_1", "T_6", "T_7"},
    )
    assert normalized == {"uav_1": "T_6"}

    # Validate with AssignmentValidator
    report = AssignmentValidator.filter_assignments(
        normalized,
        test_fleet,
        test_subtasks,
        check_skills=True,
    )
    assert report.is_valid
    assert report.valid_assignments == {"uav_1": "T_6"}
    assert "agent_id" not in report.valid_assignments
    assert "subtask_id" not in report.valid_assignments


# ==============================================================================
# TEST 9: Valid assignment reaches NMPC and CA_TRANSFER without ValueError
# ==============================================================================
def test_valid_assignment_reaches_nmpc_without_value_error(test_fleet, test_subtasks):
    valid_assignments = {"uav_1": "T_6"}
    targets = {s.subtask_id: s.target for s in test_subtasks}

    # Verify NMPC direct step
    nmpc = NMPCController()
    nmpc.step(test_fleet, valid_assignments, targets)

    # Verify CATransferManager step
    ca_transfer = CATransferManager()
    ca_transfer.step(test_fleet, mode=0, assignments=valid_assignments, targets=targets)

    # Agent position should have updated toward target
    agent = test_fleet.get_agent("uav_1")
    assert agent.position.x > 0 or agent.position.y > 0
