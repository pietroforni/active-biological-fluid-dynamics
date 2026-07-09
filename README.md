# Particle Methods: Active Biological Fluid Dynamics

Numerical experiments for a Particle Methods course project on collective motion in active biological fluids. The simulations model self-propelled rods interacting through a repulsive Yukawa potential and examine both periodic domains and confined microchannels.

The complete project report is available at [docs/report.pdf](docs/report.pdf).

## Project overview

This project builds on Wensink et al., [*Meso-scale turbulence in living fluids*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3437854/) (PNAS, 2012). The reference paper combines experiments, self-propelled-rod simulations, and continuum theory to study active turbulence in dense bacterial suspensions. This repository focuses on the particle model: it first reproduces the paper's principal dynamical regimes qualitatively and then extends the model to confined microchannel geometries, including constrictions, internal obstacles, and sinusoidal walls.

The repository contains two related simulation implementations:

- **Periodic systems:** reproduce and inspect dilute, jamming, bionematic, turbulence, swarming, and laning regimes.
- **Microchannels:** study the M2 regime in straight channels and channels with constrictions, obstacles, or sinusoidal walls.

The scripts generate rod configurations, coarse-grained velocity and vorticity fields, velocity statistics, enstrophy measurements, animations, and parameter-sweep summaries. Experiment parameters are defined near the top of each runner.

### Computational performance

The performance-critical force, torque, and time-integration loops are compiled with Numba using cached `@njit` kernels and fast-math optimizations. Together with a cell-based neighbor search and a finite Yukawa interaction cutoff, this removes most Python-loop overhead and makes the large simulations and parameter sweeps substantially faster and more memory-efficient than a direct pure-Python implementation. The first execution includes Numba compilation time; subsequent runs can reuse the compiled cache.

## Repository layout

```text
.
├── docs/
│   └── report.pdf
├── experiments/
│   ├── periodic/
│   │   ├── sourcecode.py
│   │   ├── main_six_cases.py
│   │   └── main_inspectBT.py
│   └── channels/
│       ├── sourcecode_ex.py
│       ├── main_microchannel_m2.py
│       ├── main_microchannel_m2_constriction.py
│       ├── main_microchannel_m2_obstacles.py
│       └── main_microchannel_m2_sinusoidal.py
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

### Experiment runners

| Script | Purpose |
| --- | --- |
| `main_six_cases.py` | Simulates the six periodic-domain regimes described in the report. |
| `main_inspectBT.py` | Compares the bionematic, intermediate, and turbulence regimes. |
| `main_microchannel_m2.py` | Sweeps the aspect ratio of a straight microchannel. |
| `main_microchannel_m2_constriction.py` | Sweeps the throat size of a central constriction. |
| `main_microchannel_m2_obstacles.py` | Simulates a channel containing circular and square obstacles. |
| `main_microchannel_m2_sinusoidal.py` | Sweeps the amplitude of sinusoidal channel walls. |

`sourcecode.py` contains the periodic-domain simulation and diagnostics. `sourcecode_ex.py` extends the model with channel walls and internal geometries.

## Installation

Python 3.11 is recommended.

```bash
# From a clone of this repository:
cd active-biological-fluid-dynamics

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Running the experiments

The scripts use local imports, so run them from their respective experiment directories.

Periodic experiments:

```bash
cd experiments/periodic
python main_six_cases.py
python main_inspectBT.py
```

Channel experiments:

```bash
cd experiments/channels
python main_microchannel_m2.py
python main_microchannel_m2_constriction.py
python main_microchannel_m2_obstacles.py
python main_microchannel_m2_sinusoidal.py
```

Run one script at a time. The default configurations simulate 3,000 rods over 30,000–35,000 time steps and generate high-resolution figures and GIFs, so they can require substantial CPU time, memory, and disk space. Numba compilation also makes the first run slower.

All runners use a fixed random seed (`123`) for repeatable initialization. Generated results are written below the current experiment directory:

- `spr_output/`
- `spr_BT_inspection/`
- `spr_M2_microchannel/`
- `spr_M2_microchannel_obstacles/`
- `spr_M2_microchannel_sinusoidal/`

These directories are excluded from version control. Depending on the runner, they contain PNG and GIF visualizations, compressed NumPy (`.npz`) data, and parameter-sweep summaries.

## Academic context

This repository accompanies the report *Particle Methods — Project: Active Biological Fluid Dynamics*, completed by Pietro Bernardo Maria Forni for the Master in Artificial Intelligence at Università della Svizzera italiana, Spring 2026.

If you use this work, cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

The code is available under the [MIT License](LICENSE). The project report remains an academic work by its author.
