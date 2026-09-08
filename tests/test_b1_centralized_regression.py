"""Regression tests for B1 Centralized Baseline execution and parsing integrity.

Validates:
1. B1 produces non-zero executable assignments and completes tasks.
2. CloudLLMClient._parse_assignments_response parses diverse real-world LLM schemas
   (lists, direct mappings, inverted mappings, thought blocks, markdown).
3. If planning yields zero executable assignments, the system marks the plan invalid
   and fails loudly rather than silently executing 200 empty simulation steps.
"""

from __future__ import annotations

import pytest
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.exceptions import ExperimentFailed
from src.coordination.replan_trigger import PlanState, update_plan_state, should_replan


def test_b1_parser_handles_real_llm_schemas():
    """Verify that CloudLLMClient robustly parses various LLM response formats."""
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "vehicle_3", "type": "vehicle"},
        {"id": "robot_5", "type": "robot"},
    ]
    subtasks = [
        {"id": "T_0", "required_skills": ["navigate"]},
        {"id": "T_1", "required_skills": ["transport"]},
    ]

    # 1. Dict with list of subtask objects
    res1 = client._parse_assignments_response(
        '{"assignments": [{"subtask_id": "T_0", "agent_id": "uav_0"}, {"subtask_id": "T_1", "agent_id": "vehicle_3"}]}',
        agents=agents,
        subtasks=subtasks,
    )
    assert res1 == {"T_0": ["uav_0"], "T_1": ["vehicle_3"]}

    # 2. Direct subtask to agent list
    res2 = client._parse_assignments_response(
        '{"T_0": ["uav_0"], "T_1": ["vehicle_3"]}',
        agents=agents,
        subtasks=subtasks,
    )
    assert res2 == {"T_0": ["uav_0"], "T_1": ["vehicle_3"]}

    # 3. Direct subtask to string
    res3 = client._parse_assignments_response(
        '{"T_0": "uav_0", "T_1": "vehicle_3"}',
        agents=agents,
        subtasks=subtasks,
    )
    assert res3 == {"T_0": ["uav_0"], "T_1": ["vehicle_3"]}

    # 4. Inverted agent to subtask mapping
    res4 = client._parse_assignments_response(
        '{"assignments": {"uav_0": "T_0", "vehicle_3": "T_1"}}',
        agents=agents,
        subtasks=subtasks,
    )
    assert res4 == {"T_0": ["uav_0"], "T_1": ["vehicle_3"]}

    # 5. Markdown codeblock with reasoning / thought tags (Nemotron / reasoning models)
    res5 = client._parse_assignments_response(
        '<thought>Assigning T_0 to uav_0</thought>\n```json\n{"assignments": {"T_0": ["uav_0"]}}\n```',
        agents=agents,
        subtasks=subtasks,
    )
    assert res5 == {"T_0": ["uav_0"]}

    # 6. Numeric subtask IDs ("0", "1") normalized to "T_0", "T_1"
    res6 = client._parse_assignments_response(
        '{"0": ["uav_0"], "1": ["vehicle_3"]}',
        agents=agents,
        subtasks=subtasks,
    )
    assert res6 == {"T_0": ["uav_0"], "T_1": ["vehicle_3"]}


def test_b1_centralized_produces_executable_assignments():
    """Verify B1 produces executable assignments and completes tasks rather than stalling."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="oscillatory",
        seed=1,
        config=CONFIGS["B1"],
        max_steps=50,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d in orch.device_llms.values():
        d.config["use_mock"] = True

    metrics = orch.run()
    assert metrics.steps > 0
    # At least some tasks must be completed in 50 steps
    assert len(orch.env.state.completed_subtasks) > 0
    assert metrics.success_rate > 0.0


def test_b1_empty_plan_fails_loudly_instead_of_silent_200_steps():
    """Regression test: If centralized planner produces zero executable assignments,
    the plan state is marked invalid and the simulation fails loudly instead of
    silently executing 200 empty steps.
    """
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=0,
        config=CONFIGS["B1"],
        max_steps=10,
    )
    # Simulate an empty decomposition response from the planner
    orch.centralized.plan = lambda env, cqi_matrix=None, device_feedbacks=None: ({}, [], True, False)

    # 1. Verify update_plan_state marks empty plan invalid
    plan_state = PlanState()
    update_plan_state(
        plan_state,
        orch.env.subtask_list,
        orch.env.fleet,
        coalitions=[],
        assignments={},
        mode=0,
        current_step=0,
    )
    assert plan_state.initialized is False
    assert plan_state.has_executable_plan is False
    assert plan_state.executable_assignment_count == 0

    # 2. Verify should_replan does not reuse empty plan
    needs_replan, reason = should_replan(
        plan_state,
        orch.env.subtask_list,
        orch.env.fleet,
        coalitions=[],
        mode=0,
        current_step=1,
    )
    assert needs_replan is True
    assert "initialization" in reason or "empty" in reason

    # 3. Verify orchestrator raises ExperimentFailed instead of running 200 silent empty steps
    with pytest.raises(ExperimentFailed) as excinfo:
        orch.run()
    assert "zero executable assignments" in str(excinfo.value).lower()
