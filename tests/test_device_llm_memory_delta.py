"""Unit tests for peak per-call Device-LLM RSS-delta memory metric definition and isolation."""

import pytest
from unittest.mock import MagicMock, patch
from src.llm.device_llm_client import DeviceLLMClient, DeviceLLMUsage, aggregate_device_usage
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector
from src.coordination.orchestrator import DACAOrchestrator, CONFIGS


def test_device_llm_usage_per_call_rss_delta():
    """Verify that DeviceLLMUsage records peak per-call RSS delta."""
    usage = DeviceLLMUsage()
    assert usage.memory_mb == 0.0
    assert len(usage.rss_delta_mb_samples) == 0

    # Simulate three Device-LLM calls with varying RSS deltas
    deltas = [0.15, 0.42, 0.28]
    for d in deltas:
        usage.rss_delta_mb_samples.append(d)
        usage.memory_mb = max(usage.memory_mb, d)

    assert usage.memory_mb == pytest.approx(0.42)
    assert usage.rss_delta_mb_samples == [0.15, 0.42, 0.28]


def test_aggregate_device_usage_multi_domain_peak():
    """Verify aggregation across multi-domain device LLMs (uav, vehicle, robot)."""
    uav_client = MagicMock()
    uav_client.usage = DeviceLLMUsage()
    uav_client.usage.rss_delta_mb_samples = [0.10, 0.35, 0.20]
    uav_client.usage.memory_mb = 0.35

    vehicle_client = MagicMock()
    vehicle_client.usage = DeviceLLMUsage()
    vehicle_client.usage.rss_delta_mb_samples = [0.05, 0.58, 0.12]
    vehicle_client.usage.memory_mb = 0.58

    robot_client = MagicMock()
    robot_client.usage = DeviceLLMUsage()
    robot_client.usage.rss_delta_mb_samples = [0.15, 0.08]
    robot_client.usage.memory_mb = 0.15

    device_llms = {
        "uav": uav_client,
        "vehicle": vehicle_client,
        "robot": robot_client,
    }

    total = aggregate_device_usage(device_llms)

    # Per-domain peak dictionary
    assert total.device_llm_memory_peak_mb["uav"] == pytest.approx(0.35)
    assert total.device_llm_memory_peak_mb["vehicle"] == pytest.approx(0.58)
    assert total.device_llm_memory_peak_mb["robot"] == pytest.approx(0.15)

    # Global maximum across all calls and domains
    assert total.memory_mb == pytest.approx(0.58)
    assert total.memory_mb == max(total.device_llm_memory_peak_mb.values())


def test_aggregate_device_usage_zero_calls():
    """Verify zero Device-LLM calls safely produces 0.0 without NaN or exception."""
    uav_client = MagicMock()
    uav_client.usage = DeviceLLMUsage()
    device_llms = {"uav": uav_client}

    total = aggregate_device_usage(device_llms)
    assert total.memory_mb == 0.0
    assert total.device_llm_memory_peak_mb["uav"] == 0.0


def test_evaluation_metrics_independence_and_no_fallback():
    """Verify memory_mb does not fall back to process_peak_rss_mb, and process_peak_rss_mb remains independent."""
    collector = MetricsCollector()
    metrics = collector.finalize(
        success_rate=0.85,
        steps=100,
        cloud_tokens=1000,
        cloud_api_calls=5,
        device_tokens=500,
        device_api_calls=10,
        device_memory_mb=0.45,
        computation_s=3.141,
        total_wall_clock_s=3.5,
        tfr_history=[0.9],
        cfr_history=[0.85],
        switch_count=0,
        config_name="B1",
        scenario="search_rescue",
        network_profile="stable",
        seed=2,
        process_peak_rss_mb=128.5,
        process_mean_rss_mb=110.0,
        device_llm_memory_peak_mb={"uav": 0.45},
    )

    d = metrics.to_dict()
    # memory_mb must reflect device_memory_mb, NOT process_peak_rss_mb
    assert d["memory_mb"] == pytest.approx(0.45)
    assert d["process_peak_rss_mb"] == pytest.approx(128.5)
    assert d["process_mean_rss_mb"] == pytest.approx(110.0)
    assert d["computation_s"] == pytest.approx(3.141)
    assert d["memory_mb"] != d["process_peak_rss_mb"]


def test_evaluation_metrics_zero_device_calls_no_cross_contamination():
    """Verify that when device_memory_mb is 0.0, it stays 0.0 and does NOT take process_peak_rss_mb."""
    collector = MetricsCollector()
    metrics = collector.finalize(
        success_rate=0.85,
        steps=100,
        cloud_tokens=1000,
        cloud_api_calls=5,
        device_tokens=0,
        device_api_calls=0,
        device_memory_mb=0.0,
        computation_s=2.5,
        total_wall_clock_s=2.8,
        tfr_history=[1.0],
        cfr_history=[1.0],
        switch_count=0,
        config_name="B1",
        scenario="search_rescue",
        network_profile="stable",
        seed=2,
        process_peak_rss_mb=250.0,
    )

    d = metrics.to_dict()
    assert d["memory_mb"] == 0.0
    assert d["process_peak_rss_mb"] == pytest.approx(250.0)


def test_orchestrator_passes_device_usage_memory():
    """Verify orchestrator correctly passes device_usage.memory_mb to finalize()."""
    orch = DACAOrchestrator(
        scenario="inspection",
        network_profile="stable",
        seed=0,
        config=CONFIGS["B1"],
        max_steps=2,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d in orch.device_llms.values():
        d.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()

    # Verify memory_mb matches the peak per-call device memory peak
    device_peaks = list(d.get("device_llm_memory_peak_mb", {}).values())
    expected_peak = round(max(device_peaks), 4) if device_peaks else 0.0
    assert d["memory_mb"] == pytest.approx(expected_peak, abs=1e-3)

    # Verify process_peak_rss_mb is distinct and tracks process RSS
    assert "process_peak_rss_mb" in d
    assert d["process_peak_rss_mb"] >= 0.0
