from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sourcecode import (SPRParams, run_simulation_numba, coarse_grained_velocity, vorticity_2d, enstrophy, normalized_velocity_components,
                        velocity_increment_pdfs, rod_line_segments, periodic_visual_segments, make_rods_animation, make_vorticity_animation)

N_STEPS = 35000
RELAX_STEPS = 5000
SAMPLE_EVERY = 100
MAKE_GIFS = True
OUT = Path("spr_BT_inspection")
OUT.mkdir(exist_ok=True)
DELTA_FACTOR = 1.31
OVERLAP = 0.75
R_OVER_ELL = np.array([1.0, 2.0, 4.0])

CASES = [
    {"label": "B",   "name": "bionematic",    "N": 3000, "a": 9.0, "phi": 0.55, "U0": 250.0},
    {"label": "M1",  "name": "middle_low",   "N": 3000, "a": 7.0, "phi": 0.65, "U0": 250.0},
    {"label": "M2",  "name": "middle_high",  "N": 3000, "a": 6.0, "phi": 0.75, "U0": 250.0},
    {"label": "T",   "name": "turbulence",   "N": 3000, "a": 5.0, "phi": 0.84, "U0": 250.0},
]

def save_rods_snapshot(pos, theta, params, filename, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    segments = rod_line_segments(pos, theta, params)
    segments = periodic_visual_segments(segments, params.L)
    ax.add_collection(LineCollection(segments, linewidths=1.1, alpha=0.85))
    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_vorticity_plot(omega, params, filename, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(omega.T, origin="lower", extent=(0, params.L, 0, params.L), interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$\omega$")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_histogram(vel, filename, title):
    vnorm = normalized_velocity_components(vel)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vnorm, bins=60, density=True)
    ax.set_title(title)
    ax.set_xlabel("normalized velocity component")
    ax.set_ylabel("PDF")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def compute_increment_pdfs(vgrid, h, params):
    return velocity_increment_pdfs(vgrid=vgrid, h=h, R_values=R_OVER_ELL * params.ell, component="longitudinal", bins=80, 
                                   normalize=True, max_vectors=30000)

def save_single_increment_plot(pdfs, params, filename, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for R, data in pdfs.items():
        ax.semilogy(data["bin_centers"], data["pdf"], label=fr"$R/\ell={R / params.ell:.1f}$")
    ax.set_title(title)
    ax.set_xlabel(r"normalized $\delta v_\parallel$")
    ax.set_ylabel("PDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_compare_increment_plot(all_rows, filename, R_over_ell=2.0):
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in all_rows:
        params = row["params"]
        target_R = R_over_ell * params.ell
        closest_R = min(row["pdfs"].keys(), key=lambda R: abs(R - target_R))
        data = row["pdfs"][closest_R]
        ax.semilogy(data["bin_centers"], data["pdf"], label=fr"{row['label']}: $a={params.a},\ \phi={params.phi}$")
    ax.set_title(fr"Comparison of longitudinal increment PDFs, $R/\ell={R_over_ell}$")
    ax.set_xlabel(r"normalized $\delta v_\parallel$")
    ax.set_ylabel("PDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_enstrophy_plot(rows, filename):
    labels = [row["label"] for row in rows]
    values = [row["enstrophy"] for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Final-frame enstrophy")
    ax.set_xlabel("case")
    ax.set_ylabel(r"$\Omega = \frac{1}{2}\langle \omega^2\rangle$")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def save_phi_map(filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    for case in CASES:
        ax.scatter(case["phi"], case["a"], s=90)
        ax.text(case["phi"] + 0.015, case["a"], case["label"], va="center")
    ax.set_title("B--T inspection cases")
    ax.set_xlabel(r"volume fraction $\phi$")
    ax.set_ylabel(r"aspect ratio $a$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def run_case(case):
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
    ens = enstrophy(omega)
    pdfs = compute_increment_pdfs(vgrid, h, params)
    save_rods_snapshot(pos, theta, params, OUT / f"{tag}_rods.png", title=f"{case['label']} rods: a={params.a}, phi={params.phi}")
    save_vorticity_plot(omega, params, OUT / f"{tag}_vorticity.png", title=f"{case['label']} vorticity: a={params.a}, phi={params.phi}")
    save_histogram(vel, OUT / f"{tag}_histogram.png", title=f"{case['label']} velocity histogram")
    save_single_increment_plot(pdfs, params, OUT / f"{tag}_longitudinal_increment.png", title=f"{case['label']} longitudinal velocity-increment PDF")
    if MAKE_GIFS:
        make_rods_animation(positions=result["positions"], theta=result["theta"], time=result["time"], params=params, 
                            filename=str(OUT / f"{tag}_rods.gif"), fps=12, dpi=100)
        make_vorticity_animation(positions=result["positions"], velocities=result["velocities"], time=result["time"], params=params, 
                                 filename=str(OUT / f"{tag}_vorticity.gif"), delta=delta, overlap=OVERLAP, fps=12, dpi=100)
    return {"label": case["label"], "name": case["name"], "params": params, "enstrophy": ens, "pdfs": pdfs}

def main():
    save_phi_map(OUT / "BT_inspection_phi_map.png")

    rows = []
    for case in CASES:
        rows.append(run_case(case))
    save_compare_increment_plot(rows, OUT / "BT_compare_longitudinal_increment_R2.png", R_over_ell=2.0)
    save_enstrophy_plot(rows, OUT / "BT_compare_enstrophy.png")
    print(f"\nDone. Output folder: {OUT.resolve()}")

if __name__ == "__main__":
    main()