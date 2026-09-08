#!/usr/bin/env python3
"""Phase 14 Full Baseline Validation Runner for B1 and B2."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.exceptions import ExperimentFailed

SCENARIOS = ["logistics", "inspection", "search_rescue"]
CONFIGS_TO_TEST = ["B1", "B2"]
SEEDS = [1, 2, 3]

def run_matrix():
    results = []
    total = len(CONFIGS_TO_TEST) * len(SCENARIOS) * len(SEEDS)
    idx = 0
    start_t = time.time()
    
    print("=" * 60)
    print(f"STARTING PHASE 14 BASELINE VALIDATION ({total} runs)")
    print(f"Configurations : {CONFIGS_TO_TEST}")
    print(f"Scenarios      : {SCENARIOS}")
    print(f"Seeds          : {SEEDS}")
    print("=" * 60)

    for cfg_name in CONFIGS_TO_TEST:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                idx += 1
                orch = DACAOrchestrator(
                    scenario=scenario,
                    network_profile="oscillatory",
                    seed=seed,
                    config=CONFIGS[cfg_name],
                    max_steps=200,
                )
                orch.cloud_llm.config["use_mock"] = True
                for d in orch.device_llms.values():
                    d.config["use_mock"] = True

                t0 = time.time()
                try:
                    metrics = orch.run()
                    d = metrics.to_dict()
                    completed_tasks = len(orch.env.state.completed_subtasks)
                    total_tasks = len(orch.env.subtask_list)
                    success_rate = d.get("success_rate", 0.0)
                    cloud_calls = d.get("cloud_planning_calls", 0)
                    device_calls = d.get("device_planning_calls", 0)
                    steps = d.get("steps", 0)
                    err = None
                    valid = True
                except ExperimentFailed as e:
                    completed_tasks = len(orch.env.state.completed_subtasks)
                    total_tasks = len(orch.env.subtask_list)
                    success_rate = 0.0
                    cloud_calls = getattr(orch.cloud_llm.usage, "cloud_planning_calls", 0)
                    device_calls = sum(getattr(d.usage, "device_planning_calls", 0) for d in orch.device_llms.values())
                    steps = orch.env.state.step
                    err = str(e)
                    valid = False

                dur = time.time() - t0
                record = {
                    "config": cfg_name,
                    "scenario": scenario,
                    "seed": seed,
                    "valid": valid,
                    "success_rate": success_rate,
                    "completed_tasks": completed_tasks,
                    "total_tasks": total_tasks,
                    "steps": steps,
                    "cloud_calls": cloud_calls,
                    "device_calls": device_calls,
                    "parse_failures": 0,
                    "error": err,
                    "duration_s": round(dur, 2),
                }
                results.append(record)
                status_str = f"PASS ({success_rate:.1f}%)" if valid else f"FAIL ({err})"
                print(f"[{idx:2d}/{total}] {cfg_name:2s} | {scenario:13s} | seed={seed} | {status_str} | tasks={completed_tasks}/{total_tasks} | {dur:.1f}s", flush=True)

    elapsed = time.time() - start_t
    print("=" * 60)
    print(f"VALIDATION COMPLETED in {elapsed:.1f}s")
    print("=" * 60)

    # Summary table
    print("\n--- SUMMARY TABLE ---")
    print(f"{'Config':<6} | {'Scenario':<14} | {'Mean Success':<12} | {'Valid Runs':<10} | {'Total Tasks Done':<16}")
    print("-" * 65)
    for cfg_name in CONFIGS_TO_TEST:
        for scenario in SCENARIOS:
            matching = [r for r in results if r["config"] == cfg_name and r["scenario"] == scenario]
            mean_succ = sum(r["success_rate"] for r in matching) / len(matching)
            valid_cnt = sum(1 for r in matching if r["valid"])
            tasks_done = sum(r["completed_tasks"] for r in matching)
            total_tasks_all = sum(r["total_tasks"] for r in matching)
            print(f"{cfg_name:<6} | {scenario:<14} | {mean_succ:6.2f}%     | {valid_cnt}/{len(matching)}       | {tasks_done}/{total_tasks_all}")

    with open("phase14_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_matrix()
