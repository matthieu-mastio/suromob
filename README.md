# AMoD Surrogate Modeling

This project builds a **multi-fidelity surrogate model** of an Autonomous Mobility-on-Demand (AMoD / DRT) service in the Toulouse city center (postal code 31000), using agent-based simulations ([MATSim](https://www.matsim.org/) / [Eqasim](https://eqasim.org/)).

## Motivation

Agent-based transport simulations are computationally expensive: a single MATSim run can take hours to days depending on population size. This makes direct optimization or exhaustive sensitivity analysis impractical. Instead, we use a **surrogate modeling** approach:

1. **Design of Experiments (DoE)** — sample the parameter space efficiently using nested Latin Hypercube Sampling
2. **Multi-fidelity simulations** — run cheap low-resolution simulations (small populations) densely, and expensive high-resolution simulations (large populations) sparsely
3. **Multi-Fidelity Kriging (MFK)** — fit a surrogate that fuses all fidelity levels into one accurate predictor, using the [SMT](https://smt.readthedocs.io/) toolbox
4. **Exploit the surrogate** — use the fast surrogate for sensitivity analysis, optimization, or uncertainty quantification without running additional simulations

The multi-fidelity strategy leverages the correlation between small-population and large-population simulations: they share the same model structure but differ in statistical noise. By nesting the DOE plans (`HF ⊂ LF1 ⊂ LF0`), the MFK model can learn this correlation and predict high-fidelity outputs from a combination of many cheap runs and few expensive ones.

---

## Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  1. Population       │     │  2. Design Space     │     │  3. Batch              │     │  4. Surrogate     │
│     Synthesis        │────▶│     Sampling         │────▶│     Simulations        │────▶│     Modeling      │
│  (eqasim-france)     │     │  (design_space.py)   │     │  (run_sensitivity.py)  │     │     (MFK)         │
└─────────────────────┘     └─────────────────────┘     └──────────────────────┘     └──────────────────┘
        │                           │                            │                          │
   Populations at                Nested LHS               MATSim DRT sims           Multi-fidelity
   various sample rates          DOE CSVs                 per (pop × params)        Kriging surrogate
   (1%, 5%, 10%, …)         (HF ⊂ LF1 ⊂ LF0)           → results collection       → fast predictions
```

## Pipeline Steps

### 1. Data Download & Population Synthesis

Using [eqasim-france](https://github.com/eqasim-org/eqasim-java) (synpp pipeline):

```bash
# Download input data (INSEE, SIRENE, GTFS, OSM, …)
uv run scripts/download.py config_toulouse_mini.yml

# Run the synthetic population pipeline (e.g. Haute-Garonne at 5%)
uv run python -m synpp config_toulouse_mini.yml
```

This produces a complete MATSim scenario: `population.xml.gz`, `network.xml.gz`, `facilities.xml.gz`, transit files, etc.

### 2. Area Extraction — Toulouse City Center (31000)

The study area is restricted to the Toulouse city center (postal code 31000). A shapefile is generated and used to cut the regional scenario:

```bash
# Generate the 31000 shapefile (with optional buffer)
cd src/get_31000_shapefile/
uv run python get_31000_shapefile.py
```

Then, the Eqasim ScenarioCutter extracts the sub-scenario:

```bash
cd eqasim-java/
mvn install -pl core -am -DskipTests

mvn -pl ile_de_france exec:java \
    -Dexec.mainClass="org.eqasim.core.scenario.cutter.RunScenarioCutter" \
    -Dexec.args="--config-path ../generated_pops/haute_garonne_5pct/haute_garonne_5pct_config.xml \
                 --output-path ../generated_pops/31000_5pct/ \
                 --extent-path ../src/get_31000_shapefile/emprise_31000_buffer.shp \
                 --threads 8" \
    -Dexec.classpathScope=compile
```

This is repeated for each sample rate (1%, 5%, 10%, 25%, 50%) to create populations at different fidelity levels.

### 3. Design Space Definition & DOE Sampling

`src/design_space.py` defines the 7-dimensional parameter space for the AMoD service and generates nested Latin Hypercube Sampling (LHS) plans for multi-fidelity surrogate modeling (MFK):

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `4_seats` | Integer | 0–15 | Number of 4-seat shuttles |
| `6_seats` | Integer | 0–15 | Number of 6-seat shuttles |
| `15_seats` | Integer | 0–15 | Number of 15-seat shuttles |
| `20_seats` | Integer | 0–15 | Number of 20-seat shuttles |
| `WaitTime` | Integer | 300–1800 s | Maximum passenger wait time |
| `Alpha` | Float | 1.001–2.0 | Travel time detour factor (`maxTravelTimeAlpha`) |
| `Score` | Float | -0.5–0.5 | Mode utility constant (ASC) for DRT vs. car |

The nested DOE ensures that high-fidelity samples are a strict subset of lower-fidelity ones:

```bash
python src/design_space.py
```

Generates three CSV files:
- **`doe_LF0_80.csv`** — 80 points → run on small populations (1%, 5%, 10%)
- **`doe_LF1_20.csv`** — 20 points (⊂ LF0) → run on medium populations (20%, 25%)
- **`doe_HF_10.csv`** — 10 points (⊂ LF1) → run on large populations (50%)

### 4. DRT Configuration

`src/add_drt.py` takes a base MATSim scenario and injects the DRT (Demand-Responsive Transport) configuration:

- Generates a `drt_vehicles.xml` fleet file with the specified vehicle counts and capacities
- Patches `config.xml` → `config_drt.xml` with DRT modules (multiModeDrt, dvrp), scoring parameters, and mode availability

```bash
python src/add_drt.py --base-dir generated_pops/31000_1pct/ \
    --nb-4 8 --nb-6 11 --nb-15 9 --nb-20 15 \
    --max-wait-time 1516 --max-travel-time-alpha 1.12 --drt-constant 0.20
```

### 5. Automated Sensitivity Analysis

`src/run_sensitivity.py` orchestrates the full batch of simulations:

```bash
python src/run_sensitivity.py \
    --populations generated_pops/31000_1pct generated_pops/31000_5pct generated_pops/31000_10pct \
    --doe doe_LF0_80.csv \
    --output-dir sa_output/ \
    [--max-parallel 4] \
    [--dry-run]
```

**Features:**
- **Parallel execution** with dynamic memory management (heap sized per population)
- **Isolated workdirs** per job (symlinks to shared input files) — no conflicts between parallel runs
- **RAM guard** — waits for available memory before launching large simulations
- **Resume support** — skips already-completed jobs on restart
- Collects only the **last iteration** results + root-level output files to save disk space

### 6. Visualizing Results

Use [SimWrapper](https://simwrapper.github.io/) to explore simulation outputs:

```bash
simwrapper run --port 5000
# From client: ssh -L 5000:127.0.0.1:5000 user@server
```

### 6. Surrogate Model Fitting (upcoming)

Once simulation results are collected, the next step is to fit a **Multi-Fidelity Kriging (MFK)** surrogate using the [SMT toolbox](https://smt.readthedocs.io/). The surrogate fuses all fidelity levels:

| Fidelity level | DOE | Population sizes | Points | Cost per run |
|---|---|---|---|---|
| LF0 (low) | `doe_LF0_80.csv` | 1%, 5%, 10% | 80 | Minutes |
| LF1 (mid) | `doe_LF1_20.csv` | 20%, 25% | 20 | Hours |
| HF (high) | `doe_HF_10.csv` | 50% | 10 | Hours–days |

The nested structure (`HF ⊂ LF1 ⊂ LF0`) allows the MFK model to learn the auto-correlation between fidelity levels and produce accurate predictions at the high-fidelity level using mostly cheap simulations.

Target outputs to model include DRT mode share, average wait times, vehicle utilization, and passenger-km served.

---

## Project Structure

```
popsynth/
├── src/                         # Source code
│   ├── design_space.py          # Design space definition & nested DOE sampling (SMT)
│   ├── add_drt.py               # DRT config & fleet generator for MATSim
│   ├── run_sensitivity.py       # Automated batch simulation orchestrator
│   └── get_31000_shapefile/     # Shapefile extraction for Toulouse 31000
│       └── get_31000_shapefile.py
├── doe_HF_10.csv                # High-fidelity DOE (10 points)
├── doe_LF1_20.csv               # Mid-fidelity DOE (20 points, ⊃ HF)
├── doe_LF0_80.csv               # Low-fidelity DOE (80 points, ⊃ LF1)
├── generated_pops/              # Generated populations (not tracked in git)
│   ├── 31000_1pct/
│   ├── 31000_5pct/
│   ├── 31000_10pct/
│   └── ...
├── eqasim-france/               # Eqasim population synthesis (submodule/external)
├── eqasim-java/                 # Eqasim MATSim runner (submodule/external)
└── sa_output/                   # Simulation results (not tracked in git)
```

## Dependencies

- **Python 3.10+** with:
  - [SMT](https://smt.readthedocs.io/) (Surrogate Modeling Toolbox) — for design space & nested LHS
  - [pandas](https://pandas.pydata.org/) — DOE export
  - [osmnx](https://osmnx.readthedocs.io/) / [geopandas](https://geopandas.org/) — shapefile generation
- **Java 17+** — MATSim / Eqasim runtime
- **Maven** — to build eqasim-java
- **[uv](https://docs.astral.sh/uv/)** — Python project & dependency management (for eqasim-france)

## References

- [Eqasim](https://eqasim.org/) — Agent-based transport simulation framework for France
- [MATSim](https://www.matsim.org/) — Multi-Agent Transport Simulation
- [SMT](https://smt.readthedocs.io/) — Surrogate Modeling Toolbox (design space, MFK, nested LHS)
- [SimWrapper](https://simwrapper.github.io/) — MATSim output visualization
