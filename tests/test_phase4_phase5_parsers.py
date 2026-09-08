"""Tests for Phase 4 (Structured LLM Parsing) and Phase 5 (Coalition Fallback Behavior)."""

from __future__ import annotations

import pytest
from src.llm.cloud_llm_client import CloudLLMClient, LLMUsage


def test_clean_llm_text():
    client = CloudLLMClient()
    raw = "<think>Let me think about how to allocate UAVs...</think>Here is the answer:\n{\"assignments\": {\"T_0\": [\"uav_0\"]}}"
    cleaned = client._clean_llm_text(raw)
    assert "<think>" not in cleaned
    assert "uav_0" in cleaned


def test_balanced_json_extraction():
    client = CloudLLMClient()
    raw = (
        "Here is the assignment plan:\n"
        "```json\n"
        "{\"assignments\": {\"T_0\": [\"uav_0\"], \"T_1\": [\"ugv_0\"]}}\n"
        "```\n"
        "Hope this helps!"
    )
    parsed = client._parse_relaxed_json(raw)
    assert isinstance(parsed, dict)
    assert "assignments" in parsed
    assert parsed["assignments"]["T_0"] == ["uav_0"]


def test_parse_coalitions_response_success():
    client = CloudLLMClient()
    raw = '{"coalitions": [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}]}'
    agents = [{"id": "uav_0"}, {"id": "uav_1"}, {"id": "ugv_0"}]
    coalitions, fallback_used, stripped = client._parse_coalitions_response(raw, agents)
    assert not fallback_used
    assert len(coalitions) == 1
    assert coalitions[0]["members"] == ["uav_0", "uav_1"]
    assert client.usage.successful_parses >= 1


def test_parse_coalitions_response_failure_b1_b2():
    client = CloudLLMClient()
    client.experiment_architecture = "B1"
    raw = "I am sorry, I cannot form coalitions."
    agents = [{"id": "uav_0"}, {"id": "uav_1"}]
    coalitions, fallback_used, stripped = client._parse_coalitions_response(raw, agents)
    # For B1/B2, it must NOT silently fall back to singletons!
    assert fallback_used
    assert coalitions == []
    assert client.usage.parse_failures >= 1


def test_parse_coalitions_response_fallback_a5():
    client = CloudLLMClient()
    client.experiment_architecture = "A5"
    raw = "I am sorry, I cannot form coalitions."
    agents = [{"id": "uav_0"}, {"id": "uav_1"}]
    coalitions, fallback_used, stripped = client._parse_coalitions_response(raw, agents)
    # For A5, singleton fallback is preserved
    assert fallback_used
    assert len(coalitions) == 2
    assert coalitions[0]["members"] == ["uav_0"]
    assert coalitions[1]["members"] == ["uav_1"]


def test_usage_properties():
    usage = LLMUsage()
    usage.cloud_api_calls = 5
    usage.retried_calls = 2
    usage.parse_failures = 1
    usage.successful_parses = 4
    assert usage.cloud_calls == 5
    assert usage.cloud_retries == 2
    assert usage.parse_failures == 1
    assert usage.successful_parses == 4

    usage.reset()
    assert usage.cloud_calls == 0
    assert usage.cloud_retries == 0
    assert usage.parse_failures == 0
    assert usage.successful_parses == 0
