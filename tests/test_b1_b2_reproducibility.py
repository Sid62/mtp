"""Comprehensive reproducibility tests for B1 Centralized and B2 Decentralized baselines.

Covers all 14 reproducibility criteria:
1. reasoning text + valid JSON assignment response
2. fenced JSON assignment response
3. nested JSON
4. malformed JSON
5. invalid task IDs
6. invalid agent IDs
7. empty assignment response
8. Cloud parser failure followed by controlled retry
9. Device assignment propagation
10. canonical task-ID normalization
11. B1 producing executable assignments
12. B2 producing executable assignments
13. success aggregation after a completed task
14. single clean __post_init__ behavior in DeviceLLMClient
"""

from __future__ import annotations

import ast
import json
import tracemalloc
import pytest

from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import AgentFleet, AgentState, AgentType, Position
from src.env.scenarios import Subtask
from src.coordination.autohma_structs import ExecutionDirective
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient
from src.metrics.evaluation import ExperimentMetrics


@pytest.fixture
def agent_roster() -> list[dict]:
    return [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
        {"id": "vehicle_2", "type": "vehicle"},
        {"id": "robot_3", "type": "robot"},
    ]


@pytest.fixture
def subtask_roster() -> list[dict]:
    return [
        {"id": "T_0", "required_skills": ["navigate"]},
        {"id": "T_1", "required_skills": ["transport"]},
        {"id": "T_2", "required_skills": ["inspect"]},
    ]


# 1. Reasoning text + valid JSON assignment response
def test_reasoning_text_plus_valid_json(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = (
        "Here's a thinking process:\n"
        "1. Analyze User Input: We have agents {uav_0, uav_1} and subtasks {T_0, T_1}.\n"
        "2. Form valid assignments.\n"
        "Return JSON:\n"
        '{"assignments": {"T_0": ["uav_0"], "T_1": ["vehicle_2"]}}'
    )
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {"T_0": ["uav_0"], "T_1": ["vehicle_2"]}


# 2. Fenced JSON assignment response
def test_fenced_json_assignment_response(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = (
        "```json\n"
        '{"assignments": {"T_0": ["uav_0"], "T_2": ["robot_3"]}}\n'
        "```"
    )
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {"T_0": ["uav_0"], "T_2": ["robot_3"]}


# 3. Nested JSON
def test_nested_json(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = (
        '{"status": "ok", "data": {"plan": {"assignments": {"T_0": ["uav_0"]}}}}'
    )
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {"T_0": ["uav_0"]}


# 4. Malformed JSON
def test_malformed_json(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = "Here is some invalid text {that is not closed"
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {}


# 5. Invalid task IDs
def test_invalid_task_ids_filtered(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = '{"assignments": {"T_0": ["uav_0"], "T_999": ["uav_1"]}}'
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert "T_999" not in parsed
    assert parsed == {"T_0": ["uav_0"]}


# 6. Invalid agent IDs
def test_invalid_agent_ids_filtered(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = '{"assignments": {"T_0": ["uav_0", "hallucinated_drone_x"]}}'
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {"T_0": ["uav_0"]}


# 7. Empty assignment response
def test_empty_assignment_response(agent_roster, subtask_roster):
    client = CloudLLMClient()
    raw = '{"assignments": {}}'
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {}


# 8. Cloud parser failure followed by controlled retry
def test_cloud_parser_failure_controlled_retry(agent_roster, subtask_roster):
    client = CloudLLMClient()
    client.config["use_mock"] = False
    
    call_count = 0
    def mock_complete(prompt, system="", caller=""):
        nonlocal call_count
        call_count += 1
        if caller == "decompose":
            return "This response is completely malformed and has no JSON."
        elif caller == "decompose_retry":
            return '{"assignments": {"T_0": ["uav_0"]}}'
        return "{}"

    client.complete = mock_complete
    res = client.decompose("instruction", agent_roster, subtask_roster)
    assert call_count == 2
    assert res == {"T_0": ["uav_0"]}


# 9. Device assignment propagation
def test_device_assignment_propagation():
    coord = CentralizedHybridCoordinator(
        cloud_llm=CloudLLMClient(),
        device_llms={"uav": DeviceLLMClient.for_domain("uav", ["uav_0", "uav_1"])},
    )
    # Simulate a directive received from device dispatch
    coord._last_dispatch_directives = {
        "uav": ExecutionDirective(
            domain_id="uav",
            dispatch_result={"dispatched": True, "assignments": {"uav_0": "T_0", "uav_1": "T_1"}},
            coalitions=[],
            agent_assignments={"uav_0": "T_0", "uav_1": "T_1"},
        )
    }
    # Even if global fallback_assignments was empty, device assignments propagate
    executable = coord.extract_executable_assignments(
        fallback_assignments={},
        valid_subtask_ids={"T_0", "T_1"},
    )
    assert executable == {"uav_0": "T_0", "uav_1": "T_1"}


# 10. Canonical task-ID normalization
def test_canonical_task_id_normalization(agent_roster, subtask_roster):
    client = CloudLLMClient()
    # Test numeric ID "0" and prefixed alias "subtask_id_1"
    raw = '{"assignments": {"0": ["uav_0"], "subtask_id_1": ["vehicle_2"]}}'
    parsed = client._parse_assignments_response(raw, agents=agent_roster, subtasks=subtask_roster)
    assert parsed == {"T_0": ["uav_0"], "T_1": ["vehicle_2"]}


# 11. B1 producing executable assignments
def test_b1_producing_executable_assignments():
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=1,
        config=CONFIGS["B1"],
        max_steps=50,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d in orch.device_llms.values():
        d.config["use_mock"] = True

    metrics = orch.run()
    assert metrics.steps > 0
    assert len(orch.env.state.completed_subtasks) > 0
    assert metrics.success_rate > 0.0


# 12. B2 producing executable assignments
def test_b2_producing_executable_assignments():
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=1,
        config=CONFIGS["B2"],
        max_steps=50,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d in orch.device_llms.values():
        d.config["use_mock"] = True

    metrics = orch.run()
    assert metrics.steps > 0
    assert len(orch.env.state.completed_subtasks) > 0
    assert metrics.success_rate > 0.0


# 13. Success aggregation after a completed task
def test_success_aggregation_after_completed_task():
    m = ExperimentMetrics(
        config_name="B1",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=0.833,
        steps=50,
        cloud_tokens=100,
        device_tokens=50,
        total_tokens=150,
        cloud_api_calls=2,
        device_api_calls=3,
        total_api_calls=5,
        device_memory_mb=40.0,
        computation_s=0.5,
    )
    d = m.to_dict()
    assert d["success_rate"] == 83.3
    assert d["config"] == "B1"


# 14. Duplicate __post_init__ behavior in DeviceLLMClient
def test_duplicate_post_init_behavior():
    with open("src/llm/device_llm_client.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    post_inits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeviceLLMClient":
            post_inits = [
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
            ]
    assert len(post_inits) == 1, f"Found {len(post_inits)} __post_init__ methods in DeviceLLMClient; expected exactly 1."

    client = DeviceLLMClient.for_domain("uav", ["uav_0", "uav_1"])
    assert tracemalloc.is_tracing() is True
    assert client.node_state is not None
    assert client.node_state.node_id == "uav"
    assert client.node_state.managed_agent_ids == ["uav_0", "uav_1"]
