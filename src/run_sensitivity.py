#!/usr/bin/env python3
"""
Sensitivity analysis runner for MATSim DRT simulations.

Usage:
    python run_sensitivity.py \
        --populations /path/to/pop1 /path/to/pop2 ... \
        --doe doe_HF_10.csv \
        --output-dir /path/to/results \
        [--max-parallel N] \
        [--java-jar /path/to/jar]

Workflow per (population, parameter_row) combination:
    1. Create an isolated working directory with symlinks to shared input files
    2. Call add_drt.py to generate config_drt.xml + drt_vehicles.xml in the workdir
    3. Launch MATSim simulation via Java
    4. Copy last iteration + root-level output files to output directory
    5. Clean up the working directory
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_JAR = (
    "/home/mmastio/popsynth/eqasim-java/ile_de_france/target/"
    "ile_de_france-2.2.0.jar"
)
MAIN_CLASS = "org.eqasim.ile_de_france.RunSimulation"

# Shared input files that are symlinked into each workdir (read-only by MATSim)
SHARED_INPUT_FILES = [
    "config.xml",
    "facilities.xml.gz",
    "households.xml.gz",
    "network.xml.gz",
    "population.xml.gz",
    "transit_schedule.xml.gz",
    "transit_vehicles.xml.gz",
    "vehicles.xml.gz",
]

# Memory estimation calibration
# population.xml.gz bytes → approximate heap GB
#   ~4.5 MB (1 pct)  → ~4 GB
#   ~44 MB  (10 pct) → ~12 GB
#   ~108 MB (25 pct) → ~25 GB
#   ~213 MB (50 pct) → ~40 GB
HEAP_FLOOR_GB = 4          # minimum heap per simulation
HEAP_CEILING_GB = 48       # maximum heap per simulation
RAM_RESERVE_GB = 6         # OS / buffer reserve
CPU_PER_SIM = 4            # MATSim uses ~4 threads with QSim

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sensitivity")


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DOERow:
    """One row from the Design of Experiments CSV."""
    index: int
    nb_4: int
    nb_6: int
    nb_15: int
    nb_20: int
    max_wait_time: float
    max_travel_time_alpha: float
    drt_constant: float

    @property
    def label(self) -> str:
        """Short label used in folder names."""
        return (
            f"v{self.nb_4}-{self.nb_6}-{self.nb_15}-{self.nb_20}"
            f"_w{int(self.max_wait_time)}"
            f"_a{self.max_travel_time_alpha:.2f}"
            f"_s{self.drt_constant:.2f}"
        )


@dataclass
class SimJob:
    """A single simulation to run."""
    pop_path: Path          # path to the original population directory
    pop_name: str           # human-readable population name
    doe_row: DOERow         # parameter set
    heap_gb: int            # estimated JVM heap (GB)
    output_tag: str         # subfolder name under --output-dir


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_doe(csv_path: str) -> List[DOERow]:
    """Parse the DOE CSV file into a list of DOERow objects."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Skip empty rows
            if not any(row.values()):
                continue
            rows.append(DOERow(
                index=i,
                nb_4=int(row["4_seats"]),
                nb_6=int(row["6_seats"]),
                nb_15=int(row["15_seats"]),
                nb_20=int(row["20_seats"]),
                max_wait_time=float(row["WaitTime"]),
                max_travel_time_alpha=float(row["Alpha"]),
                drt_constant=float(row["Score"]),
            ))
    return rows


def estimate_heap_gb(pop_path: Path) -> int:
    """
    Estimate JVM heap size from compressed population file size.
    Uses a piecewise-linear model calibrated on observed runs.
    """
    pop_file = pop_path / "population.xml.gz"
    if not pop_file.exists():
        log.warning("population.xml.gz not found in %s, using default heap %d GB",
                     pop_path, HEAP_FLOOR_GB)
        return HEAP_FLOOR_GB

    size_mb = pop_file.stat().st_size / (1024 * 1024)

    # Piecewise linear: ~0.2 GB per MB of compressed population, floor at 4 GB
    heap = max(HEAP_FLOOR_GB, int(size_mb * 0.2 + 2))
    heap = min(heap, HEAP_CEILING_GB)
    return heap


def _read_meminfo() -> dict:
    """Parse /proc/meminfo and return values in kB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])  # value in kB
    return info


def get_total_ram_gb() -> float:
    """Return total physical RAM in GB."""
    return _read_meminfo().get("MemTotal", 0) / (1024 * 1024)


def get_available_ram_gb() -> float:
    """Return currently available RAM in GB."""
    return _read_meminfo().get("MemAvailable", 0) / (1024 * 1024)


def build_jobs(pop_paths: List[Path], doe_rows: List[DOERow]) -> List[SimJob]:
    """Create the full job list: one per (population × DOE row)."""
    jobs = []
    for pop_path in pop_paths:
        pop_name = pop_path.name
        heap_gb = estimate_heap_gb(pop_path)
        for row in doe_rows:
            output_tag = f"{pop_name}_{row.label}"
            jobs.append(SimJob(
                pop_path=pop_path,
                pop_name=pop_name,
                doe_row=row,
                heap_gb=heap_gb,
                output_tag=output_tag,
            ))
    return jobs


# ── Workspace isolation ──────────────────────────────────────────────────────

def create_workdir(job: SimJob, base_workdir: Path) -> Path:
    """
    Create an isolated working directory for this job.
    Symlinks shared input files from the original population directory so that
    multiple jobs on the same population can run in parallel without conflicts.
    The job-specific files (config_drt.xml, drt_vehicles.xml, simulation_output)
    will be written directly into this workdir.
    """
    workdir = base_workdir / job.output_tag
    workdir.mkdir(parents=True, exist_ok=True)

    for filename in SHARED_INPUT_FILES:
        src = job.pop_path / filename
        dst = workdir / filename
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)

    return workdir


def cleanup_workdir(workdir: Path) -> None:
    """Remove the temporary working directory after results are collected."""
    if workdir.exists():
        shutil.rmtree(workdir)


# ── Core simulation steps ────────────────────────────────────────────────────

def run_add_drt(job: SimJob, workdir: Path) -> Path:
    """
    Run add_drt.py to generate config_drt.xml and drt_vehicles.xml in the workdir.
    Returns the path to the generated config_drt.xml.
    """
    add_drt_script = Path(__file__).parent / "add_drt.py"
    row = job.doe_row

    # Extract population percentage from folder name (e.g. '31000_10pct' -> 10)
    # The DoE fleet sizes are defined for the 1% population, so we scale them linearly.
    pop_pct_str = job.pop_path.name.split('_')[-1].replace('pct', '')
    pop_pct = int(pop_pct_str) if pop_pct_str.isdigit() else 1

    cmd = [
        sys.executable, str(add_drt_script),
        "--base-dir", str(workdir),
        "--nb-4", str(int(row.nb_4 * pop_pct)),
        "--nb-6", str(int(row.nb_6 * pop_pct)),
        "--nb-15", str(int(row.nb_15 * pop_pct)),
        "--nb-20", str(int(row.nb_20 * pop_pct)),
        "--max-wait-time", str(row.max_wait_time),
        "--max-travel-time-alpha", str(row.max_travel_time_alpha),
        "--drt-constant", str(row.drt_constant),
    ]

    log.info("[%s] Running add_drt in %s", job.output_tag, workdir)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"add_drt failed for {job.output_tag}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    log.info("[%s] add_drt completed: %s", job.output_tag, result.stdout.strip())
    return workdir / "config_drt.xml"


def run_simulation(job: SimJob, workdir: Path, jar_path: str) -> Path:
    """
    Launch the MATSim simulation and wait for it to complete.
    Returns the path to the simulation_output directory.
    """
    config_path = workdir / "config_drt.xml"
    sim_output = workdir / "simulation_output"

    cmd = [
        "java",
        f"-Xmx{job.heap_gb}G",
        "-cp", jar_path,
        MAIN_CLASS,
        "--config-path", str(config_path),
    ]

    log.info("[%s] Launching simulation (heap=%dG): %s",
             job.output_tag, job.heap_gb, " ".join(cmd))

    log_file = workdir / "sim.log"
    with open(log_file, "w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for {job.output_tag} (exit code {proc.returncode}). "
            f"See log: {log_file}"
        )

    log.info("[%s] Simulation completed successfully", job.output_tag)
    return sim_output


def find_last_iteration(sim_output: Path) -> Optional[Path]:
    """Find the highest-numbered iteration directory in ITERS/."""
    iters_dir = sim_output / "ITERS"
    if not iters_dir.exists():
        return None

    it_dirs = sorted(
        [d for d in iters_dir.iterdir() if d.is_dir() and d.name.startswith("it.")],
        key=lambda d: int(d.name.split(".")[1]),
    )
    return it_dirs[-1] if it_dirs else None


def collect_results(job: SimJob, workdir: Path, sim_output: Path,
                    output_dir: Path) -> None:
    """
    Copy the last iteration + root-level output files to the results directory.
    Structure:
        output_dir/
            <output_tag>/
                last_iteration/   ← contents of ITERS/it.N
                *.csv, *.png, ... ← root-level output files
                sim.log           ← simulation log
                config_drt.xml    ← config used for this run
                drt_vehicles.xml  ← vehicles used for this run
    """
    dest = output_dir / job.output_tag
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Copy last iteration
    last_iter = find_last_iteration(sim_output)
    if last_iter:
        dest_iter = dest / "last_iteration"
        if dest_iter.exists():
            shutil.rmtree(dest_iter)
        shutil.copytree(last_iter, dest_iter)
        log.info("[%s] Copied last iteration %s → %s",
                 job.output_tag, last_iter.name, dest_iter)
    else:
        log.warning("[%s] No iteration directories found in %s",
                    job.output_tag, sim_output)

    # 2. Copy root-level output files (not directories)
    for item in sim_output.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)

    # 3. Copy sim log and config files from workdir
    for fname in ["sim.log", "config_drt.xml", "drt_vehicles.xml"]:
        src = workdir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    log.info("[%s] Results collected in %s", job.output_tag, dest)


# ── Orchestrator ─────────────────────────────────────────────────────────────

# Global lock and counter for RAM allocation tracking
_ram_lock = threading.Lock()
_ram_allocated_gb = 0


def run_single_job(job: SimJob, jar_path: str, output_dir: Path,
                   workdir_base: Path) -> str:
    """Execute a single simulation job end-to-end. Returns a status message."""
    global _ram_allocated_gb
    tag = job.output_tag

    # Check if results already exist (resume support)
    dest = output_dir / tag
    if dest.exists() and any(dest.iterdir()):
        log.info("[%s] Results already exist, skipping.", tag)
        return f"SKIPPED: {tag} (already exists)"

    workdir = None
    try:
        # Wait until there's enough RAM available
        while True:
            with _ram_lock:
                total_usable = get_total_ram_gb() - RAM_RESERVE_GB
                if _ram_allocated_gb + job.heap_gb <= total_usable:
                    _ram_allocated_gb += job.heap_gb
                    break
            log.info("[%s] Waiting for RAM (need %dG, allocated %dG, available %.1fG)...",
                     tag, job.heap_gb, _ram_allocated_gb, get_available_ram_gb())
            time.sleep(30)

        # Step 0: create isolated workdir with symlinks
        workdir = create_workdir(job, workdir_base)

        # Step 1: generate DRT config + vehicles in the isolated workdir
        run_add_drt(job, workdir)

        # Step 2: run the simulation
        sim_output = run_simulation(job, workdir, jar_path)

        # Step 3: collect results
        collect_results(job, workdir, sim_output, output_dir)

        # Step 4: cleanup workdir to free disk space
        # cleanup_workdir(workdir)
        # workdir = None

        return f"OK: {tag}"

    except Exception as e:
        log.error("[%s] FAILED: %s", tag, e)
        # On failure, keep workdir for debugging but log its location
        if workdir and workdir.exists():
            log.error("[%s] Workdir preserved for debugging: %s", tag, workdir)
        return f"FAILED: {tag}: {e}"

    finally:
        with _ram_lock:
            _ram_allocated_gb -= job.heap_gb


def compute_max_parallel(jobs: List[SimJob], user_max: Optional[int]) -> int:
    """
    Compute the maximum number of parallel simulations based on RAM and CPUs.
    """
    total_ram_gb = get_total_ram_gb()
    usable_ram = total_ram_gb - RAM_RESERVE_GB
    cpu_count = os.cpu_count() or 4

    # Find the smallest heap among jobs to determine max theoretical parallelism
    if not jobs:
        return 1
    min_heap = min(j.heap_gb for j in jobs)
    max_by_ram = max(1, int(usable_ram / min_heap))
    max_by_cpu = max(1, cpu_count // CPU_PER_SIM)

    computed = min(max_by_ram, max_by_cpu)

    if user_max:
        computed = min(computed, user_max)

    log.info(
        "Parallelism: RAM=%.0fG (usable=%.0fG), CPUs=%d, "
        "min_heap=%dG → max_by_ram=%d, max_by_cpu=%d → effective=%d",
        total_ram_gb, usable_ram, cpu_count,
        min_heap, max_by_ram, max_by_cpu, computed,
    )
    return computed


def group_jobs_by_doe(jobs: List[SimJob]) -> dict:
    """
    Group jobs by DOE row index so that jobs with the same parameters
    (but different populations) run together.
    """
    groups = {}
    for job in jobs:
        groups.setdefault(job.doe_row.index, []).append(job)
    return groups


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MATSim DRT sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--populations", "-p", nargs="+", required=True,
        help="Paths to population directories (eqasim output)",
    )
    parser.add_argument(
        "--doe", "-d", required=True,
        help="Path to the DOE CSV file (Design of Experiments)",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to store results",
    )
    parser.add_argument(
        "--max-parallel", "-j", type=int, default=None,
        help="Max number of parallel simulations (auto-detected if not set)",
    )
    parser.add_argument(
        "--java-jar", default=DEFAULT_JAR,
        help=f"Path to the eqasim JAR (default: {DEFAULT_JAR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the job plan without running simulations",
    )

    args = parser.parse_args()

    # Validate inputs
    pop_paths = [Path(p).resolve() for p in args.populations]
    for p in pop_paths:
        if not p.is_dir():
            parser.error(f"Population directory does not exist: {p}")
        if not (p / "config.xml").exists():
            parser.error(f"No config.xml found in {p}")

    doe_path = Path(args.doe).resolve()
    if not doe_path.exists():
        parser.error(f"DOE file does not exist: {doe_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jar_path = args.java_jar
    if not Path(jar_path).exists():
        parser.error(f"JAR file does not exist: {jar_path}")

    # Parse DOE
    doe_rows = parse_doe(str(doe_path))
    log.info("Loaded %d parameter sets from %s", len(doe_rows), doe_path)

    # Build job list
    jobs = build_jobs(pop_paths, doe_rows)
    log.info("Total jobs: %d (%d populations × %d parameter sets)",
             len(jobs), len(pop_paths), len(doe_rows))

    # Compute parallelism
    max_parallel = compute_max_parallel(jobs, args.max_parallel)

    # Group jobs by DOE row for display
    groups = group_jobs_by_doe(jobs)

    # Print job plan
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS JOB PLAN")
    print("=" * 80)
    print(f"  Populations:      {[p.name for p in pop_paths]}")
    print(f"  DOE file:         {doe_path}")
    print(f"  Parameter sets:   {len(doe_rows)}")
    print(f"  Total jobs:       {len(jobs)}")
    print(f"  Max parallel:     {max_parallel}")
    print(f"  Output dir:       {output_dir}")
    print()

    for doe_idx, group in sorted(groups.items()):
        row = group[0].doe_row
        print(f"  DOE #{doe_idx}: {row.label}")
        for j in group:
            status = "EXISTS" if (output_dir / j.output_tag).exists() else "PENDING"
            print(f"    → {j.pop_name} (heap={j.heap_gb}G) [{status}]")
    print("=" * 80 + "\n")

    if args.dry_run:
        log.info("Dry run complete. No simulations were launched.")
        return

    # Create workdir base inside output_dir
    workdir_base = output_dir / ".workdirs"
    workdir_base.mkdir(parents=True, exist_ok=True)

    # Save job plan summary
    summary_path = output_dir / "job_plan.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "output_tag", "population", "heap_gb",
            "nb_4", "nb_6", "nb_15", "nb_20",
            "max_wait_time", "max_travel_time_alpha", "drt_constant",
        ])
        for j in jobs:
            r = j.doe_row
            writer.writerow([
                j.output_tag, j.pop_name, j.heap_gb,
                r.nb_4, r.nb_6, r.nb_15, r.nb_20,
                r.max_wait_time, r.max_travel_time_alpha, r.drt_constant,
            ])

    # Execute jobs with thread pool
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_job = {
            executor.submit(run_single_job, job, jar_path, output_dir,
                            workdir_base): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                log.info("Progress: %d/%d completed | %s",
                         len(results), len(jobs), result)
            except Exception as e:
                results.append(f"EXCEPTION: {job.output_tag}: {e}")
                log.error("Job %s raised exception: %s", job.output_tag, e)

    elapsed = time.time() - start_time

    # Clean up workdir base if empty
    try:
        workdir_base.rmdir()
    except OSError:
        pass  # Not empty (failed jobs preserved)

    # Final summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)
    ok = sum(1 for r in results if r.startswith("OK"))
    skipped = sum(1 for r in results if r.startswith("SKIPPED"))
    failed = sum(1 for r in results if r.startswith("FAILED") or r.startswith("EXCEPTION"))
    print(f"  Total:    {len(results)}")
    print(f"  OK:       {ok}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    print(f"  Elapsed:  {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Results:  {output_dir}")
    print()

    if failed > 0:
        print("FAILED JOBS:")
        for r in results:
            if r.startswith("FAILED") or r.startswith("EXCEPTION"):
                print(f"  ✗ {r}")
        print()

    # Save results summary
    with open(output_dir / "results_summary.txt", "w") as f:
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed: {elapsed / 3600:.1f} hours\n")
        f.write(f"OK: {ok}, Skipped: {skipped}, Failed: {failed}\n\n")
        for r in results:
            f.write(r + "\n")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
