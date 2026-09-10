#!/usr/bin/env python3
"""Issue 1 Before/After Comparison Experiment.

Runs a controlled comparison with identical seeds, scenarios, network
conditions, and model configuration.  Reports key metrics for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.exceptions import ExperimentFailed


def run_single(
    scenario: str,
    profile: str,
    seed: int,
    config_name: str = "A5",
    max_steps: int = 200,
    use_mock: bool = False,
) -> dict:
    """Run one experiment and return metrics dict."""
    config = CONFIGS[config_name]
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
        seed=seed,
        config=config,
        max_steps=max_steps,
    )
    if use_mock:
        orch.cloud_llm.config["use_mock"] = True
        for d in orch.device_llms.values():
            d.config["use_mock"] = True
    try:
        metrics = orch.run()
        return metrics.to_dict()
    except ExperimentFailed as e:
        print(f"[FAILED] {scenario}/{profile}/s{seed}: {e}")
        return {"success_rate": 0.0, "run_status": "FAILED"}


def main():
    parser = argparse.ArgumentParser(description="Issue 1 Comparison Experiment")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["logistics", "inspection", "search_rescue"],
        choices=["logistics", "inspection", "search_rescue"],
    )
    parser.add_argument("--profile", default="oscillatory", choices=["stable", "gradual", "sudden", "oscillatory"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default="A5", choices=list(CONFIGS.keys()))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--mock", action="store_true", default=False, help="Use mock LLM client for fast evaluation")
    parser.add_argument("--output", default="experiments/results/issue1_after_fix.json")

    args = parser.parse_args()

    scenarios = args.scenarios
    profile = args.profile
    seeds = args.seeds
    config_name = args.config
    max_steps = args.max_steps
    use_mock = args.mock

    results = {}

    for scenario in scenarios:
        results[scenario] = {}
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running: {scenario} / {profile} / seed={seed} / mock={use_mock}")
            print(f"{'='*60}\n")

            t0 = time.perf_counter()
            m = run_single(scenario, profile, seed, config_name, max_steps=max_steps, use_mock=use_mock)
            elapsed = time.perf_counter() - t0

            key = f"s{seed}"
            results[scenario][key] = {
                "success_rate": m.get("success_rate", 0.0),
                "paper_communication_steps": m.get("paper_communication_steps", 0),
                "cloud_api_calls": m.get("cloud_api_calls", 0),
                "cloud_tokens": m.get("cloud_total_tokens", m.get("cloud_tokens", 0)),
                "computation_s": m.get("computation_s", 0.0),
                "process_peak_rss_mb": m.get("process_peak_rss_mb", 0.0),
                "process_mean_rss_mb": m.get("process_mean_rss_mb", 0.0),
                "run_status": m.get("run_status", "UNKNOWN"),
                "wall_clock_s": elapsed,
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("ISSUE 1: AFTER-FIX RESULTS")
    print("=" * 80)
    for scenario in scenarios:
        print(f"\n--- {scenario} ---")
        print(f"{'Seed':<6} {'Success':>10} {'CommSteps':>12} {'CloudCalls':>12} {'Tokens':>10} {'Compute':>10} {'MemPeak':>10}")
        for seed in seeds:
            key = f"s{seed}"
            r = results[scenario][key]
            print(
                f"  s{seed:<4} "
                f"{r['success_rate']:>9.2f}% "
                f"{r['paper_communication_steps']:>12} "
                f"{r['cloud_api_calls']:>12} "
                f"{r['cloud_tokens']:>10} "
                f"{r['computation_s']:>9.2f}s "
                f"{r['process_peak_rss_mb']:>9.1f}MB"
            )

        # Averages
        avg_sr = sum(results[scenario][f"s{s}"]["success_rate"] for s in seeds) / len(seeds)
        avg_cs = sum(results[scenario][f"s{s}"]["paper_communication_steps"] for s in seeds) / len(seeds)
        avg_cc = sum(results[scenario][f"s{s}"]["cloud_api_calls"] for s in seeds) / len(seeds)
        avg_tk = sum(results[scenario][f"s{s}"]["cloud_tokens"] for s in seeds) / len(seeds)
        avg_cp = sum(results[scenario][f"s{s}"]["computation_s"] for s in seeds) / len(seeds)
        avg_mm = sum(results[scenario][f"s{s}"]["process_peak_rss_mb"] for s in seeds) / len(seeds)
        print(
            f"  {'AVG':<5} "
            f"{avg_sr:>9.2f}% "
            f"{avg_cs:>12.1f} "
            f"{avg_cc:>12.1f} "
            f"{avg_tk:>10.1f} "
            f"{avg_cp:>9.2f}s "
            f"{avg_mm:>9.1f}MB"
        )

    # Save full results
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
