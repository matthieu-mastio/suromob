#!/usr/bin/env python3
"""
Trend and Sensitivity Analysis for AMoD Simulation Inputs and Outputs.

Answers the question:
"In which direction and with what magnitude do outputs evolve when inputs change?"

Analyzes:
1. Input-Output correlation matrix (Pearson & Spearman) per population.
2. Standardized regression coefficients (Beta gradients): dY/dXi.
3. Gradient consistency across sampling rates and territories (CUPUM gradient fidelity).
4. Generates publication figures: Sensitivity Tornado / Bar charts and Main Effects plots.

Usage:
    uv run python src/trend_analysis.py [--input res.csv] [--output-dir trend_results]
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration & Constants ────────────────────────────────────────────────

DEFAULT_INPUT = "res.csv"
DEFAULT_OUTPUT_DIR = "trend_results"

INPUT_VARS = [
    "capacity_density",
    "max_wait_time",
    "max_travel_time_alpha",
    "drt_constant"
]

INPUT_LABELS = {
    "capacity_density": "Fleet Capacity Density (seats / 5 000 pers)",
    "max_wait_time": "Max Wait Time (s)",
    "max_travel_time_alpha": "Max Travel Time Alpha (-)",
    "drt_constant": "DRT Constant / ASC (-)"
}

OUTPUT_VARS = [
    "rejection_rate",
    "wait_average",
    "drt_modal_share"
]

OUTPUT_LABELS = {
    "rejection_rate": "Rejection Rate (-)",
    "wait_average": "Mean Waiting Time (s)",
    "drt_modal_share": "DRT Modal Share (-)"
}

SAMPLING_ORDER = ["1pct", "5pct", "10pct", "20pct", "25pct", "50pct"]


# ── Core Analysis Functions ──────────────────────────────────────────────────

def filter_territories_if_incomplete(df: pd.DataFrame, territory_choice: str = "auto") -> pd.DataFrame:
    """Filter dataset to complete territories if Toulouse is missing high-fidelity runs."""
    if territory_choice in ["31000", "toulouse"]:
        print(f"[INFO] Restricting analysis to {territory_choice} (user specified).")
        return df[df["territory"] == territory_choice].copy()
    elif territory_choice == "all":
        return df.copy()

    # 'auto' mode: check if toulouse has 50pct and full sampling rates
    territories = sorted(df["territory"].unique())
    if "31000" in territories and "toulouse" in territories:
        s_31000 = set(df[df["territory"] == "31000"]["sampling_rate"].unique())
        s_toulouse = set(df[df["territory"] == "toulouse"]["sampling_rate"].unique())
        has_tou_50 = "50pct" in s_toulouse
        if not has_tou_50 or len(s_toulouse) < len(s_31000):
            print("[INFO] Toulouse results are incomplete (missing 50pct HF or fewer sampling rates).")
            print("       Restricting analysis to 31000.")
            return df[df["territory"] == "31000"].copy()

    return df.copy()


def compute_correlations_and_gradients(df: pd.DataFrame, populations: List[str]) -> pd.DataFrame:
    """
    Compute input-output correlation and standardized beta coefficients (gradients)
    for each population scenario.
    """
    records = []

    for pop in populations:
        sub = df[df["population"] == pop]
        if len(sub) < 5:
            continue

        territory = pop.split("_")[0]
        sampling_rate = pop.split("_")[1]

        X = sub[INPUT_VARS].copy()

        for out in OUTPUT_VARS:
            y = sub[out].copy()

            # Standardized regression (OLS on z-scores) -> Beta gradients
            X_std = (X - X.mean()) / (X.std(ddof=0) if (X.std(ddof=0) > 0).all() else 1.0)
            X_std = X_std.fillna(0)

            y_std = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) > 0 else 1.0)

            # Fit OLS
            betas, _, _, _ = np.linalg.lstsq(X_std.values, y_std.values, rcond=None)

            for feat, beta in zip(INPUT_VARS, betas):
                records.append({
                    "population": pop,
                    "territory": territory,
                    "sampling_rate": sampling_rate,
                    "output": out,
                    "input": feat,
                    "pearson_r": sub[feat].corr(y, method="pearson"),
                    "spearman_rho": sub[feat].corr(y, method="spearman"),
                    "beta_gradient": float(beta),
                    "direction": "POSITIVE (+)" if beta > 0.05 else ("NEGATIVE (-)" if beta < -0.05 else "NEUTRAL (~0)")
                })

    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.75,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def plot_sensitivity_bars(results_df: pd.DataFrame, output_dir: Path):
    """Plot standardized beta gradients for each input across sampling rates."""
    territories = sorted(results_df["territory"].unique())
    n_terr = len(territories)
    fig, axes = plt.subplots(len(OUTPUT_VARS), n_terr, figsize=(6.5 * n_terr, 11), squeeze=False)

    colors = {
        "capacity_density": "#1f77b4",
        "max_wait_time": "#ff7f0e",
        "max_travel_time_alpha": "#2ca02c",
        "drt_constant": "#d62728"
    }

    for row_idx, out in enumerate(OUTPUT_VARS):
        for col_idx, terr in enumerate(territories):
            ax = axes[row_idx, col_idx]
            sub = results_df[(results_df["output"] == out) & (results_df["territory"] == terr)]

            # Pivot to get inputs as columns and sampling_rate as index
            piv = sub.pivot_table(index="sampling_rate", columns="input", values="beta_gradient")
            piv = piv.reindex([s for s in SAMPLING_ORDER if s in piv.index])

            x = np.arange(len(piv))
            width = 0.18

            for i, feat in enumerate(INPUT_VARS):
                if feat in piv.columns:
                    label = INPUT_LABELS.get(feat, feat) if (row_idx == 0 and col_idx == 0) else ""
                    ax.bar(
                        x + (i - 1.5) * width,
                        piv[feat],
                        width=width,
                        color=colors[feat],
                        label=label
                    )

            terr_title = "Toulouse City" if terr == "toulouse" else "Hypercenter (31000)"
            ax.set_title(f"{OUTPUT_LABELS[out]} — {terr_title}")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace("pct", "%") for s in piv.index], fontsize=10)
            ax.grid(True, axis="y")
            ax.set_ylim(-1.05, 1.05)

            if col_idx == 0:
                ax.set_ylabel("Standardized Gradient (Beta)")
            if row_idx == len(OUTPUT_VARS) - 1:
                ax.set_xlabel("Population Sampling Rate")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "fig_input_sensitivities_by_fidelity.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {plot_path}")


def plot_main_effects(df: pd.DataFrame, output_dir: Path):
    """Plot main effect curves of inputs on rejection_rate across sampling rates."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), sharey=True)

    territories = sorted(df["territory"].unique())
    blue_shades = {
        "1pct": ("#9ecae1", ":", "31000 1%"),
        "5pct": ("#6baed6", ":", "31000 5%"),
        "10pct": ("#3182bd", "-.", "31000 10%"),
        "20pct": ("#08519c", "--", "31000 20%"),
        "25pct": ("#08306b", "--", "31000 25%"),
        "50pct": ("#021d49", "-", "31000 50% (HF)"),
    }
    red_shades = {
        "1pct": ("#fc9272", ":", "Toulouse 1%"),
        "5pct": ("#fb6a4a", ":", "Toulouse 5%"),
        "10pct": ("#de2d26", "-.", "Toulouse 10%"),
        "20pct": ("#a50f15", "--", "Toulouse 20%"),
        "25pct": ("#67000d", "--", "Toulouse 25%"),
        "50pct": ("#3a0007", "-", "Toulouse 50% (HF)"),
    }

    pops_to_plot = []
    for t in territories:
        shades = blue_shades if t == "31000" else red_shades
        for s in SAMPLING_ORDER:
            pop = f"{t}_{s}"
            if pop in df["population"].values:
                color, lstyle, label = shades.get(s, ("black", "-", pop))
                pops_to_plot.append((pop, color, lstyle, label))

    for idx, feat in enumerate(INPUT_VARS):
        ax = axes[idx]
        for pop, color, lstyle, label in pops_to_plot:
            sub = df[df["population"] == pop]
            if sub.empty:
                continue

            # Bin the input into quantiles or grouped values to observe the trend
            n_unique = sub[feat].nunique()
            grouped = sub.groupby(pd.qcut(sub[feat], q=min(5, n_unique), duplicates="drop"))["rejection_rate"].mean()
            x_vals = [interval.mid for interval in grouped.index]
            y_vals = grouped.values

            ax.plot(x_vals, y_vals, color=color, linestyle=lstyle, marker="o", markersize=4, label=label if idx == 0 else "")

        title = INPUT_LABELS[feat].split("(")[0].strip()
        ax.set_title(title)
        ax.set_xlabel(INPUT_LABELS[feat])
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel("Rejection Rate (-)")
            ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "fig_main_effects_rejection.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {plot_path}")


# ── Main Runner ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze input-to-output trends and sensitivity gradients"
    )
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT, help="Path to res.csv")
    parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--territory", "-t", type=str, default="auto",
        choices=["auto", "31000", "toulouse", "all"],
        help="Territory to analyze (default: 'auto', restricts to 31000 if Toulouse is incomplete)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INPUT-TO-OUTPUT TRENDS & SENSITIVITY GRADIENT ANALYSIS")
    print(f"  Input:  {input_path.resolve()}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 80)

    # 1. Load data
    df = pd.read_csv(input_path)
    df["territory"] = df["population"].apply(lambda x: x.split("_")[0])
    df["sampling_rate"] = df["population"].apply(lambda x: x.split("_")[1])

    # Compute capacity_density if not already present
    if "capacity_density" not in df.columns:
        df["capacity_density"] = df["nb_4"] * 4 + df["nb_6"] * 6 + df["nb_15"] * 15 + df["nb_20"] * 20

    # Filter incomplete territories
    df = filter_territories_if_incomplete(df, args.territory)

    ordered_pops = [f"{t}_{s}" for t in sorted(df["territory"].unique()) for s in SAMPLING_ORDER if f"{t}_{s}" in df["population"].values]

    # 2. Compute trends
    results_df = compute_correlations_and_gradients(df, ordered_pops)

    # Save complete table
    results_df.to_csv(output_dir / "sensitivity_gradients.csv", index=False)

    # 3. Print synthesis of trends
    print("\n" + "=" * 80)
    print("SYNTHESIS OF PHYSICAL TRENDS (Input -> Output Directions)")
    print("=" * 80)

    ref_candidates = ["31000_10pct", "31000_50pct", "toulouse_10pct"]
    ref_pop = next((p for p in ref_candidates if p in results_df["population"].values), results_df["population"].iloc[0])
    ref_sub = results_df[results_df["population"] == ref_pop]

    for out in OUTPUT_VARS:
        print(f"\n>>> SORTIE : {out.upper()} ({ref_pop})")
        sub_out = ref_sub[ref_sub["output"] == out][["input", "pearson_r", "beta_gradient", "direction"]]
        for _, row in sub_out.iterrows():
            sens = "HAUSSE ↗" if row["beta_gradient"] > 0.05 else ("BAISSE ↘" if row["beta_gradient"] < -0.05 else "STABLE →")
            print(f"  • Si on augmente {row['input']:22s} -> {out} : {sens:10s} (Beta = {row['beta_gradient']:+.2f}, r = {row['pearson_r']:+.2f})")

    # 4. Generate plots
    setup_style()
    print("\nGenerating figures...")
    plot_sensitivity_bars(results_df, output_dir)
    plot_main_effects(df, output_dir)

    print("\n[SUCCESS] Analysis complete. Saved tables and figures in:", output_dir.resolve())


if __name__ == "__main__":
    main()
