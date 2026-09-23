#!/usr/bin/env python3
"""
Correlation and Multi-Fidelity Analysis for AMoD Simulation Results.

Analyzes res.csv to assess:
1. Inter-fidelity correlations (Pearson & Spearman) against high-fidelity (50%).
2. Cross-territory correlations (31000 vs Toulouse).
3. Metric convergence and error quantification (RMSE, NRMSE, Bias).
4. Generates paper-ready figures and summary CSV tables.

Usage:
    uv run python src/correlation_analysis.py [--input res.csv] [--output-dir correlation_results]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration & Constants ────────────────────────────────────────────────

DEFAULT_INPUT = "res.csv"
DEFAULT_OUTPUT_DIR = "correlation_results"

INPUT_COLS = [
    "nb_4", "nb_6", "nb_15", "nb_20",
    "max_wait_time", "max_travel_time_alpha", "drt_constant"
]

DEFAULT_METRICS = [
    "rejection_rate",
    "wait_average",
    "drt_modal_share",
    "direct_distance_ratio",
    "avg_empty_seat_ratio"
]

METRIC_LABELS = {
    "rejection_rate": "Rejection Rate (-)",
    "wait_average": "Mean Waiting Time (s)",
    "drt_modal_share": "DRT Modal Share (-)",
    "direct_distance_ratio": "Direct Distance Ratio (-)",
    "avg_empty_seat_ratio": "Empty Seat Ratio (-)"
}

SAMPLING_ORDER = ["1pct", "5pct", "10pct", "20pct", "25pct", "50pct"]
SAMPLING_NUMERIC = {
    "1pct": 1,
    "5pct": 5,
    "10pct": 10,
    "20pct": 20,
    "25pct": 25,
    "50pct": 50
}


# ── Data Loading & Preparation ───────────────────────────────────────────────

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load res.csv and parse territory and sampling rate."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns
    required = set(INPUT_COLS + ["population"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Extract territory and sampling_rate from population (e.g., '31000_10pct' -> '31000', '10pct')
    df["territory"] = df["population"].apply(lambda x: x.split("_")[0])
    df["sampling_rate"] = df["population"].apply(lambda x: x.split("_")[1])

    return df


def get_ordered_populations(df: pd.DataFrame) -> List[str]:
    """Return population column names sorted logically by territory and sampling rate."""
    territories = sorted(df["territory"].unique())
    ordered = []
    for t in territories:
        for s in SAMPLING_ORDER:
            pop = f"{t}_{s}"
            if pop in df["population"].values:
                ordered.append(pop)
    return ordered


def build_pivot_table(df: pd.DataFrame, metric: str, ordered_pops: List[str]) -> pd.DataFrame:
    """Pivot data so rows are unique DoE points and columns are populations."""
    piv = df.pivot_table(index=INPUT_COLS, columns="population", values=metric)
    cols = [c for c in ordered_pops if c in piv.columns]
    return piv[cols]


# ── Correlation Computations ─────────────────────────────────────────────────

def analyze_hf_correlations(piv: pd.DataFrame, territories: List[str], metric: str) -> List[Dict]:
    """Compute Pearson/Spearman correlation and error metrics against 50% HF for each territory."""
    rows = []
    for t in territories:
        hf_col = f"{t}_50pct"
        if hf_col not in piv.columns:
            continue

        for s in ["1pct", "5pct", "10pct", "20pct", "25pct"]:
            lf_col = f"{t}_{s}"
            if lf_col not in piv.columns:
                continue

            sub = piv[[lf_col, hf_col]].dropna()
            n_pts = len(sub)
            if n_pts < 3:
                continue

            lf_vals = sub[lf_col].values
            hf_vals = sub[hf_col].values

            # Correlations
            r_pearson = float(np.corrcoef(lf_vals, hf_vals)[0, 1])
            r_spearman = float(sub[lf_col].corr(sub[hf_col], method="spearman"))

            # Errors & bias
            diff = lf_vals - hf_vals
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            hf_range = float(np.max(hf_vals) - np.min(hf_vals))
            nrmse = float(rmse / hf_range) if hf_range > 0 else np.nan
            mbe = float(np.mean(diff))  # Mean Bias Error (positive = LF overestimates)

            # Linear scaling factor rho from OLS (y_HF = rho * y_LF)
            rho = float(np.sum(lf_vals * hf_vals) / np.sum(lf_vals ** 2)) if np.sum(lf_vals ** 2) > 0 else np.nan

            rows.append({
                "metric": metric,
                "territory": t,
                "sampling_rate": s,
                "sampling_pct": SAMPLING_NUMERIC.get(s, np.nan),
                "n_points": n_pts,
                "pearson_r": r_pearson,
                "spearman_rho": r_spearman,
                "scaling_rho": rho,
                "rmse": rmse,
                "nrmse_pct": nrmse * 100.0,
                "mean_bias": mbe
            })
    return rows


def analyze_cross_territory(piv: pd.DataFrame, metric: str) -> List[Dict]:
    """Compute correlation between 31000 and Toulouse for identical sampling rates."""
    rows = []
    for s in SAMPLING_ORDER:
        col_31 = f"31000_{s}"
        col_tou = f"toulouse_{s}"
        if col_31 not in piv.columns or col_tou not in piv.columns:
            continue

        sub = piv[[col_31, col_tou]].dropna()
        if len(sub) < 3:
            continue

        r_p = float(sub[col_31].corr(sub[col_tou], method="pearson"))
        r_s = float(sub[col_31].corr(sub[col_tou], method="spearman"))
        diff = sub[col_tou].values - sub[col_31].values
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        rows.append({
            "metric": metric,
            "sampling_rate": s,
            "sampling_pct": SAMPLING_NUMERIC.get(s, np.nan),
            "n_points": len(sub),
            "pearson_r": r_p,
            "spearman_rho": r_s,
            "rmse": rmse
        })
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib style for scientific publication."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.75,
        "lines.markersize": 6,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def plot_convergence_curves(hf_df: pd.DataFrame, output_dir: Path):
    """Plot Pearson correlation r(LF, 50%) as a function of sampling rate."""
    key_metrics = ["rejection_rate", "wait_average", "drt_modal_share"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    colors = {"31000": "#1f77b4", "toulouse": "#d62728"}
    markers = {"31000": "o", "toulouse": "s"}

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        sub = hf_df[hf_df["metric"] == metric]

        for t in ["31000", "toulouse"]:
            t_sub = sub[sub["territory"] == t].sort_values("sampling_pct")
            if not t_sub.empty:
                label = f"Toulouse City ({t})" if t == "toulouse" else f"Hypercenter ({t})"
                ax.plot(
                    t_sub["sampling_pct"],
                    t_sub["pearson_r"],
                    marker=markers.get(t, "o"),
                    color=colors.get(t, "k"),
                    label=label
                )

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Population Sampling Rate (%)")
        ax.set_xticks([1, 5, 10, 20, 25])
        ax.grid(True)
        ax.axhline(0.95, color="gray", linestyle=":", label="Threshold r = 0.95" if idx == 0 else "")

        if idx == 0:
            ax.set_ylabel("Pearson Correlation with 50% HF ($r$)")
            ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = output_dir / "fig_correlation_convergence.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {plot_path}")


def plot_correlation_heatmaps(piv_dict: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot correlation matrix heatmaps for key metrics."""
    metrics_to_plot = ["rejection_rate", "drt_modal_share"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(13, 5.5))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        piv = piv_dict[metric]
        corr = piv.corr(method="pearson").values
        labels = [c.replace("_", " ") for c in piv.columns]

        im = ax.imshow(corr, cmap="viridis", vmin=0.6, vmax=1.0)
        ax.set_title(f"Pearson Correlation: {METRIC_LABELS.get(metric, metric)}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate text
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr[i, j]
                color = "white" if val < 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = output_dir / "fig_correlation_heatmaps.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {plot_path}")


def plot_scatter_lf_vs_hf(piv_dict: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot scatter plots of 1% vs 50% and 10% vs 50% for rejection_rate."""
    piv = piv_dict["rejection_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    comparisons = [("1pct", "50pct"), ("10pct", "50pct")]
    for idx, (lf_s, hf_s) in enumerate(comparisons):
        ax = axes[idx]
        for t, color, marker, label in [
            ("31000", "#1f77b4", "o", "Hypercenter (31000)"),
            ("toulouse", "#d62728", "s", "Toulouse City")
        ]:
            col_lf = f"{t}_{lf_s}"
            col_hf = f"{t}_{hf_s}"
            if col_lf in piv.columns and col_hf in piv.columns:
                sub = piv[[col_lf, col_hf]].dropna()
                ax.scatter(sub[col_lf], sub[col_hf], color=color, marker=marker, alpha=0.8, label=label)

                # Linear regression fit
                if len(sub) > 2:
                    p = np.polyfit(sub[col_lf], sub[col_hf], 1)
                    x_fit = np.linspace(sub[col_lf].min(), sub[col_lf].max(), 50)
                    ax.plot(x_fit, np.polyval(p, x_fit), color=color, linestyle="--", alpha=0.7)

        # Identity line
        lims = [0.45, 0.95]
        ax.plot(lims, lims, color="black", linestyle=":", alpha=0.5, label="Identity line (y = x)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"Rejection Rate: {lf_s.replace('pct', '%')} vs 50%")
        ax.set_xlabel(f"Low Fidelity ({lf_s.replace('pct', '%')})")
        ax.grid(True)
        if idx == 0:
            ax.set_ylabel("High Fidelity (50%)")
            ax.legend(loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "fig_scatter_lf_vs_hf.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {plot_path}")


# ── Main Runner ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive correlation and multi-fidelity analysis on res.csv"
    )
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT, help="Path to res.csv")
    parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-FIDELITY CORRELATION & SCALING ANALYSIS")
    print(f"  Input file:  {input_path.resolve()}")
    print(f"  Output dir:  {output_dir.resolve()}")
    print("=" * 80)

    # 1. Load data
    df = load_and_validate(input_path)
    ordered_pops = get_ordered_populations(df)
    territories = sorted(df["territory"].unique())

    print(f"\nLoaded {len(df)} total simulation runs across {len(ordered_pops)} population scenarios:")
    print("  Populations:", ", ".join(ordered_pops))

    # 2. Compute pivot tables and matrices
    all_hf_rows = []
    all_cross_rows = []
    piv_dict = {}

    for metric in DEFAULT_METRICS:
        piv = build_pivot_table(df, metric, ordered_pops)
        piv_dict[metric] = piv

        # Save raw pivot table
        piv.to_csv(output_dir / f"pivot_{metric}.csv")

        # Full correlation matrices
        piv.corr(method="pearson").to_csv(output_dir / f"corr_pearson_{metric}.csv")
        piv.corr(method="spearman").to_csv(output_dir / f"corr_spearman_{metric}.csv")

        # Fidelity & Cross analysis
        all_hf_rows.extend(analyze_hf_correlations(piv, territories, metric))
        all_cross_rows.extend(analyze_cross_territory(piv, metric))

    # 3. Build summary DataFrames
    hf_summary_df = pd.DataFrame(all_hf_rows)
    cross_summary_df = pd.DataFrame(all_cross_rows)

    hf_summary_df.to_csv(output_dir / "hf_correlation_summary.csv", index=False)
    cross_summary_df.to_csv(output_dir / "cross_territory_correlation.csv", index=False)

    # 4. Display tables in console
    print("\n" + "=" * 80)
    print("1. INTER-FIDELITY CORRELATION (WITH 50% HIGH-FIDELITY REFERENCE)")
    print("=" * 80)
    for metric in ["rejection_rate", "wait_average", "drt_modal_share"]:
        print(f"\n--- Metric: {metric.upper()} ---")
        sub = hf_summary_df[hf_summary_df["metric"] == metric][[
            "territory", "sampling_rate", "n_points",
            "pearson_r", "spearman_rho", "scaling_rho", "nrmse_pct", "mean_bias"
        ]]
        print(sub.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. CROSS-TERRITORY CORRELATION (31000 vs TOULOUSE AT SAME SAMPLING RATE)")
    print("=" * 80)
    for metric in ["rejection_rate", "wait_average", "drt_modal_share"]:
        print(f"\n--- Metric: {metric.upper()} ---")
        sub = cross_summary_df[cross_summary_df["metric"] == metric][[
            "sampling_rate", "n_points", "pearson_r", "spearman_rho", "rmse"
        ]]
        print(sub.to_string(index=False))

    # 5. Generate publication plots
    if not args.no_plots:
        print("\n" + "=" * 80)
        print("GENERATING FIGURES FOR PUBLICATION")
        print("=" * 80)
        setup_style()
        plot_convergence_curves(hf_summary_df, output_dir)
        plot_correlation_heatmaps(piv_dict, output_dir)
        plot_scatter_lf_vs_hf(piv_dict, output_dir)

    print("\n[SUCCESS] Analysis complete. All tables and plots are saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
