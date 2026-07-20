#!/usr/bin/env python3
"""
Scan sa_output/ directories and produce a consolidated CSV with input parameters
and simulation results for each experiment.

Usage:
    python src/get_outputs_AMoD.py [--sa-dir sa_output/] [--output results.csv]
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DRT_MODE = "drt"


def parse_tag(tag: str) -> dict:
    """
    Extract input parameters from the output_tag folder name.
    Format: <pop>_v<4>-<6>-<15>-<20>_w<wait>_a<alpha>_s<score>
    """
    m = re.match(
        r"^(.+?)_v(\d+)-(\d+)-(\d+)-(\d+)_w(\d+)_a([\d.]+)_s(-?[\d.]+)$",
        tag,
    )
    if not m:
        return {}
    return {
        "population": m.group(1),
        "nb_4": int(m.group(2)),
        "nb_6": int(m.group(3)),
        "nb_15": int(m.group(4)),
        "nb_20": int(m.group(5)),
        "max_wait_time": int(m.group(6)),
        "max_travel_time_alpha": float(m.group(7)),
        "drt_constant": float(m.group(8)),
    }


def read_fleet_capacity(exp_dir: Path) -> int:
    """Read total fleet capacity from drt_vehicles.xml."""
    vehicles_file = exp_dir / "drt_vehicles.xml"
    if not vehicles_file.exists():
        return 0

    tree = ET.parse(vehicles_file)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    total = 0
    for vehicle in root.findall(f"{ns}vehicle"):
        cap = vehicle.get("capacity")
        if cap is not None:
            total += int(cap)
    return total


def process_experiment(exp_dir: Path) -> dict:
    """Process a single experiment directory and return a flat dict of results."""
    tag = exp_dir.name
    params = parse_tag(tag)
    if not params:
        return None

    row = dict(params)

    # ── Fleet capacity ───────────────────────────────────────────────────
    total_capacity = read_fleet_capacity(exp_dir)
    row["total_capacity"] = total_capacity

    # ── Rejection rate & average waiting time (last iteration) ───────────
    f = exp_dir / f"drt_customer_stats_{DRT_MODE}.csv"
    if f.exists():
        df = pd.read_csv(f, sep=";")
        if not df.empty:
            last = df.loc[df["iteration"].idxmax()]
            row["rejection_rate"] = last["rejectionRate"] if pd.notna(last["rejectionRate"]) else None
            row["wait_average"] = last["wait_average"] if pd.notna(last["wait_average"]) else None

    # ── DRT modal share (last iteration) ─────────────────────────────────
    f = exp_dir / "modestats.csv"
    if f.exists():
        df = pd.read_csv(f, sep=";")
        if not df.empty and "drt" in df.columns:
            row["drt_modal_share"] = df.iloc[-1]["drt"]

    # ── Total distance & sum of direct distances ─────────────────────────
    f_legs = exp_dir / f"output_drt_legs_{DRT_MODE}.csv"
    f_veh = exp_dir / f"drt_vehicle_stats_{DRT_MODE}.csv"

    if f_legs.exists():
        df = pd.read_csv(f_legs, sep=";")
        if not df.empty and "directRideDistance" in df.columns:
            row["total_direct_distance"] = df["directRideDistance"].sum()

    if f_veh.exists():
        df = pd.read_csv(f_veh, sep=";")
        if not df.empty:
            last = df.loc[df["iteration"].idxmax()]
            row["total_distance"] = last.get("totalDistance")

    if row.get("total_direct_distance") and row.get("total_distance") and row["total_distance"] > 0:
        row["direct_distance_ratio"] = row["total_direct_distance"] / row["total_distance"]

    # ── Occupancy: average empty seats ───────────────────────────────────
    f = exp_dir / f"output_occupancy_time_profiles_{DRT_MODE}.txt"
    if f.exists() and total_capacity > 0:
        df = pd.read_csv(f, sep=";")
        if not df.empty:
            df = df[df["time"] <= "24:00:00"]
            occ_cols = [c for c in df.columns if re.match(r"\d+ pax", c)]
            occupancy_levels = [int(re.search(r"\d+", c).group()) for c in occ_cols]
            df["passengers"] = sum(
                df[col] * occ for col, occ in zip(occ_cols, occupancy_levels)
            )
            df["empty_capacity"] = total_capacity - df["passengers"]
            row["avg_empty_seats"] = df["empty_capacity"].mean()
            row["avg_empty_seat_ratio"] = row["avg_empty_seats"] / total_capacity

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Scan sensitivity analysis outputs and produce a consolidated CSV"
    )
    parser.add_argument(
        "--sa-dir", default="sa_output",
        help="Path to the sensitivity analysis output directory (default: sa_output)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: <sa-dir>/results_consolidated.csv)",
    )
    args = parser.parse_args()

    sa_dir = Path(args.sa_dir).resolve()
    if not sa_dir.is_dir():
        print(f"Error: {sa_dir} is not a directory")
        return

    output_path = Path(args.output) if args.output else sa_dir / "results_consolidated.csv"

    # Find all experiment directories
    exp_dirs = sorted([
        d for d in sa_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and parse_tag(d.name)
    ])

    print(f"Found {len(exp_dirs)} experiment directories in {sa_dir}")

    # Process each experiment
    rows = []
    errors = []
    for exp_dir in exp_dirs:
        try:
            row = process_experiment(exp_dir)
            if row:
                rows.append(row)
        except Exception as e:
            errors.append(f"{exp_dir.name}: {e}")

    if errors:
        print(f"\nWarnings ({len(errors)} experiments had issues):")
        for err in errors[:10]:
            print(f"  ⚠ {err}")

    if not rows:
        print("No results found!")
        return

    # Build DataFrame with fixed column order
    col_order = [
        "population",
        "nb_4", "nb_6", "nb_15", "nb_20",
        "max_wait_time", "max_travel_time_alpha", "drt_constant",
        "total_capacity",
        "rejection_rate", "wait_average",
        "drt_modal_share",
        "total_direct_distance", "total_distance", "direct_distance_ratio",
        "avg_empty_seats", "avg_empty_seat_ratio",
    ]
    df = pd.DataFrame(rows)[col_order]
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results consolidated: {len(rows)} experiments")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    print(f"\nPopulations: {sorted(df['population'].unique())}")
    for pop, count in df.groupby("population").size().items():
        print(f"  {pop}: {count}")


if __name__ == "__main__":
    main()
