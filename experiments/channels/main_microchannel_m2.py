from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sourcecode_ex import (SPRParams, run_simulation_numba, coarse_grained_velocity, vorticity_2d, enstrophy, normalized_velocity_components, 
                           rod_line_segments, periodic_visual_segments, make_rods_animation, make_vorticity_animation, compute_vortex_dimension_time_average)

N_STEPS = 30000
RELAX_STEPS = 0          
SAMPLE_EVERY = 100
CAR_VALUES = np.array([5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 25.0, 30.0, 40.0, 50.0]) 
MAKE_GIFS = True 
OUT = Path("spr_M2_microchannel")
OUT.mkdir(exist_ok=True)
DELTA_FACTOR = 1.31
OVERLAP = 0.75
R_OVER_ELL = np.array([1.0, 2.0, 4.0])

CASE = {"label": "M2", "name": "straight_channel", "N": 3000, "a": 6.0, "phi": 0.75, "U0": 250.0}

def save_rods_snapshot(pos, theta, wall_mask, params, filename, title):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    segments = rod_line_segments(pos, theta, params)
    segments = periodic_visual_segments(segments, params.Lx)
    wall_segments = periodic_visual_segments(rod_line_segments(pos[wall_mask], theta[wall_mask], params), params.Lx)
    mobile_segments = periodic_visual_segments(rod_line_segments(pos[~wall_mask], theta[~wall_mask], params), params.Lx)
    ax.add_collection(LineCollection(mobile_segments, linewidths=1.0, alpha=0.80, label="mobile rods"))
    ax.add_collection(LineCollection(wall_segments, linewidths=1.5, alpha=0.95, label="fixed wall rods"))
    ax.set_xlim(0.0, params.Lx)
    ax.set_ylim(0.0, params.Ly)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_vorticity_plot(omega, params, filename, title):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(omega.T, origin="lower", extent=(0.0, params.Lx, 0.0, params.Ly), interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=r"$\omega$")
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_histogram(vel, wall_mask, filename, title):
    vnorm = normalized_velocity_components(vel[~wall_mask])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vnorm, bins=60, density=True)
    ax.set_title(title)
    ax.set_xlabel("normalized mobile-rod velocity component")
    ax.set_ylabel("PDF")
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_channel_profiles(pos, vel, wall_mask, params, filename, title, n_bins=40):
    mobile_pos = pos[~wall_mask]
    mobile_vel = vel[~wall_mask]
    edges = np.linspace(0.0, params.Ly, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.digitize(mobile_pos[:, 1], edges) - 1
    mean_vx = np.full(n_bins, np.nan)
    mean_vy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        mask = bin_id == k
        counts[k] = np.count_nonzero(mask)
        if counts[k] > 0:
            mean_vx[k] = np.mean(mobile_vel[mask, 0])
            mean_vy[k] = np.mean(mobile_vel[mask, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_vx, centers, label=r"$\langle v_x \rangle_y$")
    ax.plot(mean_vy, centers, label=r"$\langle v_y \rangle_y$")
    ax.set_title(title)
    ax.set_xlabel("mean velocity")
    ax.set_ylabel("y")
    ax.set_ylim(0.0, params.Ly)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    return {"y_centers": centers, "mean_vx": mean_vx, "mean_vy": mean_vy, "counts": counts}

def save_time_averaged_channel_profiles(result, params, filename, title, n_bins=20, start_fraction=0.5):
    wall_mask = result["wall_mask"]
    positions = result["positions"]
    velocities = result["velocities"]
    start = int(start_fraction * positions.shape[0])
    edges = np.linspace(0.0, params.Ly, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sum_vx = np.zeros(n_bins)
    sum_vy = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for frame in range(start, positions.shape[0]):
        pos = positions[frame, ~wall_mask]
        vel = velocities[frame, ~wall_mask]
        bin_id = np.digitize(pos[:, 1], edges) - 1
        for k in range(n_bins):
            mask = bin_id == k
            if np.any(mask):
                sum_vx[k] += np.sum(vel[mask, 0])
                sum_vy[k] += np.sum(vel[mask, 1])
                counts[k] += np.count_nonzero(mask)
    mean_vx = np.full(n_bins, np.nan)
    mean_vy = np.full(n_bins, np.nan)
    valid = counts > 0
    mean_vx[valid] = sum_vx[valid] / counts[valid]
    mean_vy[valid] = sum_vy[valid] / counts[valid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_vx, centers, label=r"time-avg $\langle v_x \rangle_y$")
    ax.plot(mean_vy, centers, label=r"time-avg $\langle v_y \rangle_y$")
    ax.set_title(title)
    ax.set_xlabel("mean velocity")
    ax.set_ylabel("y")
    ax.set_ylim(0.0, params.Ly)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    return {"y_centers": centers, "mean_vx": mean_vx, "mean_vy": mean_vy, "counts": counts}

def _normalize_samples(samples):
    samples = np.asarray(samples, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return samples
    std = np.std(samples)
    if std > 0.0:
        return (samples - np.mean(samples)) / std
    return samples * 0.0

def compute_x_increment_pdfs(vgrid, counts, h, params):
    if np.isscalar(h):
        hx = float(h)
    else:
        hx = float(h[0])
    results = {}
    valid = counts > 0
    for R in R_OVER_ELL * params.ell:
        shift = max(1, int(round(float(R) / hx)))
        effective_R = shift * hx
        shifted_vx = np.roll(vgrid[..., 0], shift=-shift, axis=0)
        shifted_valid = np.roll(valid, shift=-shift, axis=0)
        mask = valid & shifted_valid
        increments = shifted_vx[mask] - vgrid[..., 0][mask]
        increments = _normalize_samples(increments)
        pdf, edges = np.histogram(increments, bins=80, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        results[float(effective_R)] = {"increments": increments, "bin_centers": bin_centers, "pdf": pdf,
                                       "bin_edges": edges, "component": "x-periodic longitudinal", "normalized": True}
    return results

def save_x_increment_plot(pdfs, params, filename, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for R, data in pdfs.items():
        ax.semilogy(data["bin_centers"], data["pdf"], label=fr"$R/\ell={R / params.ell:.2f}$")
    ax.set_title(title)
    ax.set_xlabel(r"normalized $\delta v_x(x+R,y)$")
    ax.set_ylabel("PDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_vorticity_y_profile(omega, params, filename, title):
    if np.isscalar(params.Ly):
        y = np.linspace(0.0, params.Ly, omega.shape[1])
    mean_abs_omega = np.mean(np.abs(omega), axis=0)
    mean_omega2 = np.mean(omega * omega, axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_abs_omega, y, label=r"$\langle |\omega| \rangle_x$")
    ax.plot(mean_omega2, y, label=r"$\langle \omega^2 \rangle_x$")
    ax.set_title(title)
    ax.set_xlabel("vorticity intensity")
    ax.set_ylabel("y")
    ax.set_ylim(0.0, params.Ly)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_density_profile(pos, wall_mask, params, filename, title, n_bins=20):
    mobile_y = pos[~wall_mask, 1]
    counts, edges = np.histogram(mobile_y, bins=n_bins, range=(0.0, params.Ly))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(counts, centers)
    ax.set_title(title)
    ax.set_xlabel("mobile rod count")
    ax.set_ylabel("y")
    ax.set_ylim(0.0, params.Ly)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def compute_enstrophy_time_series(result, params, delta, overlap):
    wall_mask = result["wall_mask"]
    ens_t = np.zeros(result["positions"].shape[0])
    for k in range(result["positions"].shape[0]):
        pos = result["positions"][k]
        vel = result["velocities"][k]
        vgrid, counts, h = coarse_grained_velocity(pos[~wall_mask], vel[~wall_mask],
                                                   L=params.Lx, Ly=params.Ly,
                                                   delta=delta, overlap=overlap)
        omega = vorticity_2d(vgrid, h)
        ens_t[k] = enstrophy(omega)
    return ens_t

def save_enstrophy_vs_aspect_ratio(summary, filename):
    car_values = np.array([row["channel_aspect_ratio"] for row in summary])
    final_ens = np.array([row["final_enstrophy"] for row in summary])
    mean_ens = np.array([row["mean_enstrophy"] for row in summary])
    std_ens = np.array([row["std_enstrophy"] for row in summary])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(car_values, final_ens, "o-", label="final enstrophy")
    ax.errorbar(car_values, mean_ens, yerr=std_ens, marker="s", capsize=4,
                label="time-avg enstrophy")
    ax.set_xlabel("channel aspect ratio (Lx/Ly)")
    ax.set_ylabel(r"enstrophy $\Omega = \frac{1}{2}\langle \omega^2 \rangle$")
    ax.set_title(f"Enstrophy vs channel aspect ratio (rod a={CASE['a']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def save_vortex_dimension_vs_aspect_ratio(summary, filename):
    car_values = np.array([row["channel_aspect_ratio"] for row in summary])
    mean_dim = np.array([row["time_mean_mean_vortex_dimension_over_Ly"] for row in summary])
    std_dim = np.array([row["time_std_mean_vortex_dimension_over_Ly"] for row in summary])
    n_vortices = np.array([row["time_mean_n_vortices"] for row in summary])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(car_values, mean_dim, yerr=std_dim, marker="o", capsize=4, label=r"mean vortex dimension / $L_y$")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label=r"$L_y$")
    ax.set_xlabel("channel aspect ratio (Lx/Ly)")
    ax.set_ylabel(r"mean vortex dimension / $L_y$")
    ax.set_title(f"Mean vortex dimension vs channel aspect ratio (rod a={CASE['a']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)
    count_file = filename.with_name("vortex_count_vs_channel_aspect_ratio.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(car_values, n_vortices, "o-", label="mean number of vortices")
    ax.set_xlabel("channel aspect ratio (Lx/Ly)")
    ax.set_ylabel("number of detected vortices")
    ax.set_title(f"Detected vortices vs channel aspect ratio (rod a={CASE['a']})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(count_file, dpi=400)
    plt.close(fig)

def run_m2_channel(car_value):
    case = CASE.copy()
    # 'a' is now fixed from CASE, car_value is dynamic
    tag = f"{case['label']}_{case['name']}_car{car_value:.1f}".replace(".", "p")
    params = SPRParams(N=case["N"], a=case["a"], phi=case["phi"], U0=case["U0"], rc=4.0, rmin=1e-4, seed=123, dt_scale=0.002,
                       channel_aspect_ratio=float(car_value), n_wall_layers=1, initial_impulse_force=50.0, initial_impulse_steps=100, wall_angle=0.0)
    print(f"\nRunning {tag}")
    print(f"N={params.N}, a={params.a}, phi={params.phi}, U0={params.U0}")
    print(f"Lx={params.Lx:.3f}, Ly={params.Ly:.3f}, Lx/Ly={params.channel_aspect_ratio}")
    result = run_simulation_numba(params=params, n_steps=N_STEPS, relax_steps=RELAX_STEPS, sample_every=SAMPLE_EVERY)
    wall_mask = result["wall_mask"]
    pos = result["positions"][-1]
    theta = result["theta"][-1]
    vel = result["velocities"][-1]
    delta = DELTA_FACTOR * params.ell
    vgrid, counts, h = coarse_grained_velocity(pos[~wall_mask], vel[~wall_mask], L=params.Lx, Ly=params.Ly, delta=delta, overlap=OVERLAP)
    omega = vorticity_2d(vgrid, h)
    ens = enstrophy(omega)
    ens_t = compute_enstrophy_time_series(result, params, delta, OVERLAP)
    half = ens_t.shape[0] // 2
    mean_ens = np.mean(ens_t[half:])
    std_ens = np.std(ens_t[half:])
    pdfs = compute_x_increment_pdfs(vgrid, counts, h, params)
    vortex_dim = compute_vortex_dimension_time_average(result, params, delta=delta, overlap=OVERLAP, start_fraction=0.5, threshold_factor=0.5, min_cells=4)
    print(f"mean vortex dimension/Ly="f"{vortex_dim['time_mean_mean_vortex_dimension_over_Ly']:.3f}, "f"n vortices={vortex_dim['time_mean_n_vortices']:.1f}")

    # savings
    save_rods_snapshot(pos, theta, wall_mask, params, OUT / f"{tag}_rods.png",
                       title=f"M2 rods: Lx/Ly={params.channel_aspect_ratio:.1f}, a={params.a}")
    save_vorticity_plot(omega, params, OUT / f"{tag}_vorticity.png",
                        title=f"M2 vorticity: Lx/Ly={params.channel_aspect_ratio:.1f}, a={params.a}")
    save_vorticity_y_profile(omega, params, OUT / f"{tag}_vorticity_y_profile.png",
                             title=f"M2 vorticity profile: Lx/Ly={params.channel_aspect_ratio:.1f}")
    save_density_profile(pos, wall_mask, params, OUT / f"{tag}_density_profile.png",
                         title=f"M2 density profile: Lx/Ly={params.channel_aspect_ratio:.1f}")
    save_histogram(vel, wall_mask, OUT / f"{tag}_histogram.png",
                   title=f"M2 velocity histogram: Lx/Ly={params.channel_aspect_ratio:.1f}")
    save_channel_profiles(pos, vel, wall_mask, params, OUT / f"{tag}_velocity_profile_final.png",
                                    title=f"M2 velocity profile: Lx/Ly={params.channel_aspect_ratio:.1f}", n_bins=20)
    save_x_increment_plot(pdfs, params, OUT / f"{tag}_x_increment.png",
                          title=f"M2 velocity-increment PDF: Lx/Ly={params.channel_aspect_ratio:.1f}")

    if MAKE_GIFS:
        make_rods_animation(positions=result["positions"], theta=result["theta"], time=result["time"], params=params,
                            filename=str(OUT / f"{tag}_rods.gif"), fps=12, dpi=400)
        make_vorticity_animation(positions=result["positions"], velocities=result["velocities"], time=result["time"], params=params,
                                 filename=str(OUT / f"{tag}_vorticity.gif"), delta=delta, overlap=OVERLAP, fps=12, dpi=400)
    
    row = {"channel_aspect_ratio": params.channel_aspect_ratio, "a": params.a, "final_enstrophy": ens, "mean_enstrophy": mean_ens, 
           "std_enstrophy": std_ens, "Lx": params.Lx, "Ly": params.Ly}
    row.update(vortex_dim)
    return row

def main():
    summary = []
    for car_value in CAR_VALUES:
        row = run_m2_channel(car_value)
        summary.append(row)
    
    save_enstrophy_vs_aspect_ratio(summary, OUT / "enstrophy_vs_channel_aspect_ratio.png")
    save_vortex_dimension_vs_aspect_ratio(summary, OUT / "vortex_dimension_vs_channel_aspect_ratio.png")

    np.savez_compressed(
        OUT / "channel_aspect_ratio_sweep_summary.npz",
        car=np.array([row["channel_aspect_ratio"] for row in summary]),
        Lx=np.array([row["Lx"] for row in summary]),
        Ly=np.array([row["Ly"] for row in summary]),
        final_enstrophy=np.array([row["final_enstrophy"] for row in summary]),
        mean_enstrophy=np.array([row["mean_enstrophy"] for row in summary]),
        std_enstrophy=np.array([row["std_enstrophy"] for row in summary]),
        mean_vortex_dimension_over_Ly=np.array([row["time_mean_mean_vortex_dimension_over_Ly"] for row in summary]),
        std_vortex_dimension_over_Ly=np.array([row["time_std_mean_vortex_dimension_over_Ly"] for row in summary]),
        mean_n_vortices=np.array([row["time_mean_n_vortices"] for row in summary]),
    )
    print("\nChannel aspect-ratio sweep complete.")

if __name__ == "__main__":
    main()