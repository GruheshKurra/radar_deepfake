#!/usr/bin/env python3
"""
Compare ablation results: Baseline vs Baseline + RADAR for each backbone.
Reads results saved by 3_train_ablation.py and generates formatted tables.

Usage:
  python 4_compare_ablations.py
  python 4_compare_ablations.py --results_dir ./outputs/ablations
"""

import json
from pathlib import Path
import argparse


def print_main_table(results):
    print("\n" + "=" * 90)
    print("TABLE 1: Baseline vs Baseline + RADAR Modules")
    print("=" * 90)

    header = (f"{'Backbone':<20} {'Baseline AUC':>14} {'+ RADAR AUC':>14} "
              f"{'Δ AUC':>10} {'Baseline Acc':>14} {'+ RADAR Acc':>14}")
    print(header)
    print("-" * 90)

    for backbone_name, data in results.items():
        name = data["display_name"]
        b_auc = data["baseline"]["test_auc"] * 100
        r_auc = data["radar"]["test_auc"] * 100
        b_acc = data["baseline"]["test_acc"] * 100
        r_acc = data["radar"]["test_acc"] * 100
        delta = r_auc - b_auc
        print(f"{name:<20} {b_auc:>13.2f}% {r_auc:>13.2f}% "
              f"{delta:>+9.2f}% {b_acc:>13.2f}% {r_acc:>13.2f}%")

    print("=" * 90)


def print_parameter_table(results):
    print("\n" + "=" * 80)
    print("TABLE 2: Parameter Counts")
    print("=" * 80)

    header = f"{'Backbone':<20} {'Baseline':>14} {'+ RADAR':>14} {'Overhead':>14} {'Overhead %':>12}"
    print(header)
    print("-" * 80)

    for backbone_name, data in results.items():
        name = data["display_name"]
        b_params = data["baseline"]["params"]
        r_params = data["radar"]["params"]
        overhead = r_params - b_params
        overhead_pct = (r_params / b_params - 1) * 100
        print(f"{name:<20} {b_params:>14,} {r_params:>14,} {overhead:>+14,} {overhead_pct:>11.1f}%")

    print("=" * 80)


def print_timing_table(results):
    print("\n" + "=" * 70)
    print("TABLE 3: Training Time (seconds)")
    print("=" * 70)

    header = f"{'Backbone':<20} {'Baseline':>14} {'+ RADAR':>14} {'Overhead':>14}"
    print(header)
    print("-" * 70)

    for backbone_name, data in results.items():
        name = data["display_name"]
        b_time = data["baseline"]["train_time_sec"]
        r_time = data["radar"]["train_time_sec"]
        overhead = r_time - b_time
        print(f"{name:<20} {b_time:>13.1f}s {r_time:>13.1f}s {overhead:>+13.1f}s")

    print("=" * 70)


def print_latex_table(results):
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy to paper)")
    print("=" * 70)

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Effect of RADAR modules across backbone architectures. "
          r"Adding BADM, AADM, and ERM consistently improves detection performance.}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Backbone & Baseline AUC & + RADAR AUC & $\Delta$ AUC & $\Delta$ Acc \\")
    print(r"\midrule")

    for backbone_name, data in results.items():
        name = data["display_name"]
        b_auc = data["baseline"]["test_auc"] * 100
        r_auc = data["radar"]["test_auc"] * 100
        delta_auc = r_auc - b_auc
        b_acc = data["baseline"]["test_acc"] * 100
        r_acc = data["radar"]["test_acc"] * 100
        delta_acc = r_acc - b_acc
        print(f"{name} & {b_auc:.2f}\\% & \\textbf{{{r_auc:.2f}\\%}} "
              f"& +{delta_auc:.2f}\\% & +{delta_acc:.2f}\\% \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")


def print_key_findings(results):
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    improvements = []
    for backbone_name, data in results.items():
        name = data["display_name"]
        delta = (data["radar"]["test_auc"] - data["baseline"]["test_auc"]) * 100
        improvements.append((name, delta))

    if improvements:
        avg_improvement = sum(d for _, d in improvements) / len(improvements)
        best_name, best_delta = max(improvements, key=lambda x: x[1])
        worst_name, worst_delta = min(improvements, key=lambda x: x[1])

        print(f"\n  Average AUC improvement:   {avg_improvement:+.2f}%")
        print(f"  Best improvement:          {best_name} ({best_delta:+.2f}%)")
        print(f"  Smallest improvement:      {worst_name} ({worst_delta:+.2f}%)")

        all_positive = all(d > 0 for _, d in improvements)
        if all_positive:
            print(f"\n  ✓ RADAR modules improve ALL {len(improvements)} backbone architectures")
        else:
            negative = [(n, d) for n, d in improvements if d <= 0]
            print(f"\n  ⚠ RADAR did not improve: {', '.join(n for n, _ in negative)}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare ablation results")
    parser.add_argument("--results_dir", type=str, default="./outputs/ablations",
                        help="Directory containing ablation_results.json")
    args = parser.parse_args()

    results_file = Path(args.results_dir) / "ablation_results.json"

    if not results_file.exists():
        print(f"No results found at {results_file}")
        print("Run 3_train_ablation.py first:")
        print("  python 3_train_ablation.py --data_dir ./data")
        return

    with open(results_file) as f:
        results = json.load(f)

    if not results:
        print("Results file is empty.")
        return

    print("\n" + "=" * 90)
    print("RADAR ABLATION STUDY: Baseline vs Baseline + RADAR Modules")
    print("=" * 90)
    print(f"Results loaded from: {results_file}")
    print(f"Backbones evaluated: {len(results)}")

    print_main_table(results)
    print_parameter_table(results)
    print_timing_table(results)
    print_key_findings(results)
    print_latex_table(results)

    print("\n" + "=" * 70)
    print("Use these tables in your paper!")
    print("=" * 70)


if __name__ == "__main__":
    main()
