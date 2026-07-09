"""
Runner for the six SPR cases from the paper appendix.
Outputs for each case:
  - final rods PNG
  - rods GIF
  - final vorticity PNG
  - vorticity GIF
  - longitudinal velocity-increment PDF
  - normalized velocity-component histogram
Also outputs one a-phi map containing all six cases.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sourcecode import (SPRParams, run_simulation_numba, coarse_grained_velocity, vorticity_2d, normalized_velocity_components, velocity_increment_pdfs, 
                        rod_line_segments, periodic_visual_segments, make_rods_animation, make_vorticity_animation, enstrophy)

N_STEPS = 30000
RELAX_STEPS = 5000
SAMPLE_EVERY = 100
MAKE_GIFS = True
OUT = Path("spr_output")
OUT.mkdir(exist_ok=True)
DELTA_FACTOR = 1.31      # delta = 1.31 * ell
OVERLAP = 0.75
R_OVER_ELL = np.array([1.0, 2.0, 4.0])

CASES = [
    {"label": "D", "name": "dilute",     "N": 3000, "a": 4.0,  "phi": 0.063, "U0": 250.0},
    {"label": "J", "name": "jamming",    "N": 3000,  "a": 3.0,  "phi": 0.975, "U0": 250.0},
    {"label": "B", "name": "bionematic", "N": 3000,  "a": 9.0,  "phi": 0.55,  "U0": 250.0},
    {"label": "T", "name": "turbulence", "N": 3000,  "a": 5.0,  "phi": 0.84,  "U0": 250.0},
    {"label": "S", "name": "swarming",   "N": 3000,  "a": 16.0, "phi": 0.21,  "U0": 250.0},
    {"label": "L", "name": "laning",     "N": 3000,  "a": 16.0, "phi": 0.53,  "U0": 250.0},
]

##### HELPERS #####
def save_rods_snapshot(pos, theta, params, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    segments = rod_line_segments(pos, theta, params)
    segments = periodic_visual_segments(segments, params.L)
    ax.add_collection(LineCollection(segments, linewidths=1.1, alpha=0.85))
    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_title("Final rod snapshot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_vorticity(omega, params, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(omega.T, origin="lower", extent=(0, params.L, 0, params.L), interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title("Final vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$\omega$")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_velocity_histogram(vel, filename):
    vnorm = normalized_velocity_components(vel)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vnorm, bins=60, density=True)
    ax.set_title("Velocity-component histogram")
    ax.set_xlabel("normalized velocity component")
    ax.set_ylabel("probability density")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return vnorm

def save_longitudinal_increment(vgrid, h, params, filename):
    pdfs = velocity_increment_pdfs(vgrid=vgrid, h=h, R_values=R_OVER_ELL * params.ell, component="longitudinal", bins=80, normalize=True, max_vectors=30000)
    fig, ax = plt.subplots(figsize=(6, 4))
    for R, data in pdfs.items():
        ax.semilogy(data["bin_centers"], data["pdf"], label=fr"$R/\ell={R / params.ell:.1f}$")
    ax.set_title("Longitudinal velocity-increment PDF")
    ax.set_xlabel(r"normalized $\delta v_\parallel$")
    ax.set_ylabel("PDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return pdfs

def save_phi_map(filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    for case in CASES:
        ax.scatter(case["phi"], case["a"], s=90)
        ax.text(case["phi"] + 0.015, case["a"], case["label"], va="center")
    ax.set_title("Six cases in the a-phi plane")
    ax.set_xlabel(r"volume fraction $\phi$")
    ax.set_ylabel(r"aspect ratio $a$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_case(case):
    tag = f"{case['label']}_{case['name']}"
    params = SPRParams(N=case["N"], a=case["a"], phi=case["phi"], U0=case["U0"], rc=4.0, rmin=1e-4, seed=123, dt_scale=0.002)
    print(f"\nRunning {tag}")
    print(f"N={params.N}, a={params.a}, phi={params.phi}, U0={params.U0}")
    print(f"n_steps={N_STEPS}, relax_steps={RELAX_STEPS}, sample_every={SAMPLE_EVERY}")
    result = run_simulation_numba(params=params, n_steps=N_STEPS, relax_steps=RELAX_STEPS, sample_every=SAMPLE_EVERY)
    pos = result["positions"][-1]
    theta = result["theta"][-1]
    vel = result["velocities"][-1]
    delta = DELTA_FACTOR * params.ell
    vgrid, counts, h = coarse_grained_velocity(pos, vel, L=params.L, delta=delta, overlap=OVERLAP)
    omega = vorticity_2d(vgrid, h)
    ens_mean, ens_std = compute_time_averaged_enstrophy(result, params)
    save_rods_snapshot(pos, theta, params, OUT / f"{tag}_rods.png")
    save_vorticity(omega, params, OUT / f"{tag}_vorticity.png")
    vnorm = save_velocity_histogram(vel, OUT / f"{tag}_histogram.png")
    pdfs = save_longitudinal_increment(vgrid, h, params, OUT / f"{tag}_longitudinal_increment.png")

    if MAKE_GIFS:
        make_rods_animation(positions=result["positions"], theta=result["theta"], time=result["time"], params=params, 
                            filename=str(OUT / f"{tag}_rods.gif"), fps=8, dpi=100)
        make_vorticity_animation(positions=result["positions"], velocities=result["velocities"], time=result["time"], 
                                 params=params, filename=str(OUT / f"{tag}_vorticity.gif"), delta=delta, overlap=OVERLAP, 
                                 fps=8, dpi=100)
    np.savez_compressed(OUT / f"{tag}_data.npz", positions=result["positions"], theta=result["theta"], velocities=result["velocities"], 
                        time=result["time"], vgrid=vgrid, counts=counts, omega=omega, vnorm=vnorm, metadata=json.dumps(result["metadata"], indent=2),
                        increment_pdfs=json.dumps({str(R): {"bin_centers": data["bin_centers"].tolist(), "pdf": data["pdf"].tolist(),
                                                            "bin_edges": data["bin_edges"].tolist()}
                                                            for R, data in pdfs.items()}))
    return {"label": case["label"], "name": case["name"], "a": case["a"], "phi": case["phi"], "enstrophy_mean": ens_mean, "enstrophy_std": ens_std}

def save_enstrophy_plot(enstrophy_rows, filename):
    labels = [row["label"] for row in enstrophy_rows]
    means = [row["enstrophy_mean"] for row in enstrophy_rows]
    stds = [row["enstrophy_std"] for row in enstrophy_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_title("Time-averaged enstrophy for the six cases")
    ax.set_xlabel("case")
    ax.set_ylabel(r"$\Omega = \frac{1}{2}\langle \omega^2\rangle$")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def compute_time_averaged_enstrophy(result, params):
    delta = DELTA_FACTOR * params.ell
    ens_values = []
    for pos, vel in zip(result["positions"], result["velocities"]):
        vgrid, counts, h = coarse_grained_velocity(pos, vel, L=params.L, delta=delta, overlap=OVERLAP)
        omega = vorticity_2d(vgrid, h)
        ens_values.append(enstrophy(omega))
    return np.mean(ens_values), np.std(ens_values)

##### MAIN #####
def main():
    save_phi_map(OUT / "all_cases_phi_map.png")
    enstrophy_rows = []
    for case in CASES:
        row = save_case(case)
        enstrophy_rows.append(row)
    save_enstrophy_plot(enstrophy_rows, OUT / "all_cases_enstrophy.png")
    print(f"\nDone. Output folder: {OUT.resolve()}")

if __name__ == "__main__":
    main()