from __future__ import annotations
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from numba import njit
import math
import numpy as np

Array = np.ndarray

@njit(cache=True, fastmath=True)
def _build_segments_and_cells(pos, theta, segment_offsets, L, rc):
    N = pos.shape[0]
    nseg = segment_offsets.shape[0]
    M = N * nseg
    ncell = int(math.floor(L / rc))
    if ncell < 1:
        ncell = 1
    cell_size = L / ncell
    n_cells_total = ncell * ncell

    seg_x = np.empty(M, dtype=np.float64)
    seg_y = np.empty(M, dtype=np.float64)
    ux = np.empty(N, dtype=np.float64)
    uy = np.empty(N, dtype=np.float64)
    cell_id = np.empty(M, dtype=np.int64)
    counts = np.zeros(n_cells_total, dtype=np.int64)
    sidx = 0
    for i in range(N):
        c = math.cos(theta[i])
        s = math.sin(theta[i])
        ux[i] = c
        uy[i] = s
        for k in range(nseg):
            x = pos[i, 0] + segment_offsets[k] * c
            y = pos[i, 1] + segment_offsets[k] * s
            x = x - L * math.floor(x / L)
            y = y - L * math.floor(y / L)
            seg_x[sidx] = x
            seg_y[sidx] = y
            cx = int(math.floor(x / cell_size))
            cy = int(math.floor(y / cell_size))
            if cx >= ncell:
                cx = ncell - 1
            if cy >= ncell:
                cy = ncell - 1
            cid = cx + ncell * cy
            cell_id[sidx] = cid
            counts[cid] += 1
            sidx += 1
    starts = np.empty(n_cells_total + 1, dtype=np.int64)
    starts[0] = 0
    for cid in range(n_cells_total):
        starts[cid + 1] = starts[cid] + counts[cid]
    cursor = starts[:-1].copy()
    order = np.empty(M, dtype=np.int64)
    for sidx in range(M):
        cid = cell_id[sidx]
        p = cursor[cid]
        order[p] = sidx
        cursor[cid] += 1
    return seg_x, seg_y, ux, uy, order, starts, ncell

@njit(cache=True, fastmath=True)
def _compute_forces_numba(pos, theta, segment_offsets, L, rc, rmin, U0):
    N = pos.shape[0]
    nseg = segment_offsets.shape[0]
    seg_x, seg_y, ux, uy, order, starts, ncell = _build_segments_and_cells(pos, theta, segment_offsets, L, rc)
    n_cells_total = ncell * ncell
    forces = np.zeros((N, 2), dtype=np.float64)
    torques = np.zeros(N, dtype=np.float64)
    rc2 = rc * rc
    rmin2 = rmin * rmin
    half_L = 0.5 * L
    prefactor = U0 / float(nseg * nseg)
    # Used to avoid duplicate periodic neighbor cells when ncell is 1 or 2
    seen = np.empty(9, dtype=np.int64)

    for cid in range(n_cells_total):
        cx = cid % ncell
        cy = cid // ncell
        nseen = 0
        for oy in range(-1, 2):
            ncy = (cy + oy) % ncell
            for ox in range(-1, 2):
                ncx = (cx + ox) % ncell
                ncid = ncx + ncell * ncy
                duplicate = False
                for q in range(nseen):
                    if seen[q] == ncid:
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen[nseen] = ncid
                nseen += 1
                for pa in range(starts[cid], starts[cid + 1]):
                    sa = order[pa]
                    ra = sa // nseg
                    ka = sa - ra * nseg
                    la = segment_offsets[ka]
                    for pb in range(starts[ncid], starts[ncid + 1]):
                        sb = order[pb]
                        # Unique unordered segment pairs only.
                        if sb <= sa:
                            continue
                        rb = sb // nseg
                        if ra == rb:
                            continue
                        dx = seg_x[sa] - seg_x[sb]
                        dy = seg_y[sa] - seg_y[sb]
                        if dx > half_L:
                            dx -= L
                        elif dx < -half_L:
                            dx += L
                        if dy > half_L:
                            dy -= L
                        elif dy < -half_L:
                            dy += L
                        r2 = dx * dx + dy * dy
                        if r2 >= rc2:
                            continue
                        if r2 < rmin2:
                            r = rmin
                        else:
                            r = math.sqrt(r2)
                        inv_r = 1.0 / r
                        coeff = prefactor * math.exp(-r) * (inv_r * inv_r + inv_r * inv_r * inv_r)
                        fx = coeff * dx
                        fy = coeff * dy
                        forces[ra, 0] += fx
                        forces[ra, 1] += fy
                        forces[rb, 0] -= fx
                        forces[rb, 1] -= fy
                        lb = segment_offsets[sb - rb * nseg]
                        torques[ra] += la * (ux[ra] * fy - uy[ra] * fx)
                        torques[rb] += lb * (ux[rb] * (-fy) - uy[rb] * (-fx))
    return forces, torques

@njit(cache=True, fastmath=True)
def _advance_one_step_inplace(pos, theta, vel, segment_offsets, L, dt, V, f_parallel, f_perp, f_rot, rc, rmin, U0):
    forces, torques = _compute_forces_numba(pos, theta, segment_offsets, L, rc, rmin, U0)
    N = pos.shape[0]
    two_pi = 2.0 * math.pi
    for i in range(N):
        c = math.cos(theta[i])
        s = math.sin(theta[i])
        f_parallel_scalar = forces[i, 0] * c + forces[i, 1] * s
        fpar_x = f_parallel_scalar * c
        fpar_y = f_parallel_scalar * s
        fperp_x = forces[i, 0] - fpar_x
        fperp_y = forces[i, 1] - fpar_y
        vx = V * c + fpar_x / f_parallel + fperp_x / f_perp
        vy = V * s + fpar_y / f_parallel + fperp_y / f_perp
        vel[i, 0] = vx
        vel[i, 1] = vy
        x = pos[i, 0] + dt * vx
        y = pos[i, 1] + dt * vy
        pos[i, 0] = x - L * math.floor(x / L)
        pos[i, 1] = y - L * math.floor(y / L)
        theta_i = theta[i] + dt * torques[i] / f_rot
        theta[i] = (theta_i + math.pi) % two_pi - math.pi

@njit(cache=True, fastmath=True)
def _run_simulation_numba(pos0, theta0, segment_offsets, L, dt, V, f_parallel, f_perp, f_rot, rc, rmin, U0, n_steps, sample_every, relax_steps):
    N = pos0.shape[0]
    n_saved = (n_steps + sample_every - 1) // sample_every
    saved_pos = np.empty((n_saved, N, 2), dtype=np.float64)
    saved_theta = np.empty((n_saved, N), dtype=np.float64)
    saved_vel = np.empty((n_saved, N, 2), dtype=np.float64)
    saved_time = np.empty(n_saved, dtype=np.float64)
    pos = pos0.copy()
    theta = theta0.copy()
    vel = np.empty((N, 2), dtype=np.float64)
    save_idx = 0
    total_steps = relax_steps + n_steps
    for step in range(total_steps):
        _advance_one_step_inplace(pos, theta, vel, segment_offsets, L, dt, V, f_parallel, f_perp, f_rot, rc, rmin, U0)
        if step >= relax_steps and (step - relax_steps) % sample_every == 0:
            for i in range(N):
                saved_pos[save_idx, i, 0] = pos[i, 0]
                saved_pos[save_idx, i, 1] = pos[i, 1]
                saved_theta[save_idx, i] = theta[i]
                saved_vel[save_idx, i, 0] = vel[i, 0]
                saved_vel[save_idx, i, 1] = vel[i, 1]
            saved_time[save_idx] = step * dt
            save_idx += 1
    return saved_pos, saved_theta, saved_vel, saved_time

def run_simulation_numba(params: SPRParams, n_steps: int, sample_every: int, relax_steps: int = 0) -> dict:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")
    if relax_steps < 0:
        raise ValueError("relax_steps must be >= 0.")
    pos, theta = initialize_lattice(params)
    saved_pos, saved_theta, saved_vel, saved_time = _run_simulation_numba(np.asarray(pos, dtype=np.float64), np.asarray(theta, dtype=np.float64), 
                                                                          np.asarray(params.segment_offsets, dtype=np.float64), float(params.L), 
                                                                          float(params.dt), float(params.V), float(params.f_parallel), float(params.f_perp), 
                                                                          float(params.f_rot), float(params.rc), float(params.rmin), float(params.U0), 
                                                                          int(n_steps), int(sample_every), int(relax_steps))
    metadata = params.metadata()
    metadata.update(n_steps=n_steps, sample_every=sample_every, relax_steps=relax_steps, integrator="explicit Euler, Numba core")
    return {"positions": saved_pos, "theta": saved_theta, "velocities": saved_vel, "time": saved_time, "metadata": metadata}

def n_segments_for_aspect(a: float) -> int:
    if np.isclose(a, 1.0): # n = 1 for a = 1
        return 1
    if 1.0 < a <= 3.0: # n = 3 for 1 < a <= 3
        return 3
    if a > 3.0: # n = nearest integer to 9a/8 for a > 3
        return int(np.floor(9.0 * a / 8.0 + 0.5))
    raise ValueError("Aspect ratio a must be >= 1.")

def friction_coefficients(a: float) -> tuple[float, float, float]: # in nondimensional units lambda=1, F=1, f0=1
    if np.isclose(a, 1.0):
        return 1.0, 1.0, 1.0
    loga = math.log(a)
    f_parallel = 2.0 * math.pi / (loga - 0.207 + 0.980 / a - 0.133 / (a * a))
    f_perp = 4.0 * math.pi / (loga + 0.839 + 0.185 / a + 0.233 / (a * a))
    f_rot = math.pi * a * a / (3.0 * (loga - 0.662 + 0.917 / a - 0.050 / (a * a)))
    return f_parallel, f_perp, f_rot

@dataclass
class SPRParams: # sim params in nondimensional units lambda=1, F=1, f0=1
    N: int
    a: float
    phi: float
    U0: float
    rc: float = 3.0          # numerical Yukawa cutoff; pipeline recommends 3-5
    rmin: float = 1.0e-6     # protects division by zero
    seed: int = 12345
    propulsion_force: float = 1.0
    dt_scale: float = 0.002  # paper: dt = 0.002 rho^(-1/2)

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.a < 1.0:
            raise ValueError("a must be >= 1.")
        if self.phi <= 0.0:
            raise ValueError("phi must be positive.")
        if self.U0 < 0.0:
            raise ValueError("U0 must be nonnegative.")
        if self.rc <= 0.0:
            raise ValueError("rc must be positive.")
        if self.rmin <= 0.0:
            raise ValueError("rmin must be positive.")

        self.lambda_yukawa = 1.0
        self.ell = self.a
        self.n_segments = n_segments_for_aspect(self.a)
        self.segment_offsets = make_segment_offsets(self.a, self.n_segments)
        self.segment_rod_id = np.repeat(np.arange(self.N, dtype=np.int64), self.n_segments)
        self.segment_l_flat = np.tile(self.segment_offsets, self.N)
        particle_area = self.a - 1.0 + math.pi / 4.0
        self.area = self.N * particle_area / self.phi
        self.L = math.sqrt(self.area)
        self.rho = self.N / self.area
        self.dt = self.dt_scale * self.rho ** (-0.5)
        self.f_parallel, self.f_perp, self.f_rot = friction_coefficients(self.a)
        self.V = self.propulsion_force / self.f_parallel

    def metadata(self) -> dict:
        return {"N": self.N, "a": self.a, "phi": self.phi, "U0": self.U0, "L": self.L, "rho": self.rho, 
                "n_segments": self.n_segments, "rc": self.rc, "dt": self.dt, "rmin": self.rmin, 
                "seed": self.seed, "propulsion_force": self.propulsion_force, "dt_scale": self.dt_scale,
                "integrator": "explicit Euler", "units": "lambda=1, F=1, f0=1"}

def make_segment_offsets(a: float, n_segments: int) -> Array: # segment offsets along the rod axis for Yukawa interaction
    if n_segments == 1:
        return np.array([0.0], dtype=np.float64)
    half = 0.5 * (a - 1.0) # lambda = 1, so skeleton length is a - 1
    return np.linspace(-half, half, n_segments, dtype=np.float64)

def unit_vectors(theta: Array) -> Array: # u = (cos theta, sin theta) for all rods
    return np.column_stack((np.cos(theta), np.sin(theta)))

def wrap_positions(pos: Array, L: float) -> Array: # in [0, L) with periodic boundaries
    return np.mod(pos, L)

def initialize_lattice(params: SPRParams) -> tuple[Array, Array]: 
    # rods with orientations randomly 0 or pi on a rectangular grid adapted to the rod length
    rng = np.random.default_rng(params.seed)
    # Enough x-spacing for rods that initially lie horizontally.
    nx = max(1, min(params.N, int(np.floor(params.L / max(params.ell, 1.0)))))
    ny = int(np.ceil(params.N / nx))
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * params.L / nx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * params.L / ny
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pos = np.column_stack((xx.ravel(), yy.ravel()))[: params.N].copy()
    theta = rng.choice(np.array([0.0, math.pi, math.pi / 18]), size=params.N).astype(np.float64) # add a little random noise to break perfect symmetry
    return wrap_positions(pos, params.L), theta

##### DIAGNOSTICS #####

# vorticity + esentrophy
def coarse_grained_velocity(pos: Array, vel: Array, L: float, delta: float, overlap: float = 0.75) -> tuple[Array, Array, float]:
    # coarse-grained velocity field on an overlapping square grid with periodic boundaries
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    target_h = delta * (1.0 - overlap)
    G = max(4, int(np.floor(L / target_h)))
    h = L / G
    centers = (np.arange(G, dtype=np.float64) + 0.5) * h
    sum_v = np.zeros((G, G, 2), dtype=np.float64)
    counts = np.zeros((G, G), dtype=np.int64)
    base_x = np.floor(pos[:, 0] / h).astype(np.int64) % G
    base_y = np.floor(pos[:, 1] / h).astype(np.int64) % G
    m = int(np.ceil(0.5 * delta / h)) + 1
    for ox in range(-m, m + 1):
        ix = (base_x + ox) % G
        dx = pos[:, 0] - centers[ix]
        dx = dx - L * np.round(dx / L)
        mask_x = np.abs(dx) <= 0.5 * delta
        for oy in range(-m, m + 1):
            iy = (base_y + oy) % G
            dy = pos[:, 1] - centers[iy]
            dy = dy - L * np.round(dy / L)
            mask = mask_x & (np.abs(dy) <= 0.5 * delta)
            if not np.any(mask):
                continue
            np.add.at(sum_v[..., 0], (ix[mask], iy[mask]), vel[mask, 0])
            np.add.at(sum_v[..., 1], (ix[mask], iy[mask]), vel[mask, 1])
            np.add.at(counts, (ix[mask], iy[mask]), 1)
    vgrid = np.zeros_like(sum_v)
    nonempty = counts > 0
    vgrid[nonempty] = sum_v[nonempty] / counts[nonempty, None]
    return vgrid, counts, h

def vorticity_2d(vgrid: Array, h: float) -> Array:
    # omega = d_x v_y - d_y v_x using periodic centered differences on the grid
    dvydx = (np.roll(vgrid[..., 1], -1, axis=0) - np.roll(vgrid[..., 1], 1, axis=0)) / (2.0 * h)
    dvxdy = (np.roll(vgrid[..., 0], -1, axis=1) - np.roll(vgrid[..., 0], 1, axis=1)) / (2.0 * h)
    return dvydx - dvxdy

def enstrophy(omega: Array) -> float: # Omega = 1/2 <omega^2>
    return 0.5 * float(np.mean(omega * omega))

# velocity histogram
def normalized_velocity_components(vel: Array) -> Array:
    # flattened, normalized Cartesian velocity components: this combines vx and vy into one 
    # sample, subtracts the mean, and divides by the standard deviation
    comps = vel.reshape(-1)
    mean = np.mean(comps)
    std = np.std(comps)
    if std == 0.0:
        return comps * 0.0
    return (comps - mean) / std

# longitudinal velocity increment PDF
def _periodic_shift_vectors_for_radius(R: float, h: float, G: int, tolerance: float | None = None, max_vectors: int | None = None) -> tuple[Array, Array, Array]:
    # Return integer grid shifts with physical length close to R for a GxG periodic grid with spacing h:
    # The velocity field is assumed to live on a square periodic G x G grid with spacing h. A displacement 
    # vector is represented by integer shifts (sx, sy), corresponding to physical vector (sx*h, sy*h). 
    # Only the minimum-image range -G/2 <= sx, sy <= G/2 is considered.
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if G < 2:
        raise ValueError("G must be at least 2.")
    if tolerance is None:
        tolerance = 0.5 * h
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative.")
    max_shift = G // 2
    shifts = []
    lengths = []
    for sx in range(-max_shift, max_shift + 1):
        for sy in range(-max_shift, max_shift + 1):
            if sx == 0 and sy == 0:
                continue
            # Avoid double-counting the exactly periodic Nyquist displacement
            # for even G: +G/2 and -G/2 are the same periodic shift.
            if G % 2 == 0 and (sx == max_shift or sy == max_shift):
                continue
            length = h * math.sqrt(float(sx * sx + sy * sy))
            shifts.append((sx, sy))
            lengths.append(length)
    shifts_arr = np.asarray(shifts, dtype=np.int64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    err = np.abs(lengths_arr - R)
    selected = err <= tolerance
    if not np.any(selected):
        # Fall back to the nearest discrete shell if the requested R does not
        # exactly match the grid spacing.
        nearest = np.min(err)
        selected = np.isclose(err, nearest)
    shifts_arr = shifts_arr[selected]
    lengths_arr = lengths_arr[selected]
    err = err[selected]
    # Stable ordering: closest vectors first, then by shift components. This is
    # useful when max_vectors is used for cheaper sampling on large grids.
    order = np.lexsort((shifts_arr[:, 1], shifts_arr[:, 0], err))
    shifts_arr = shifts_arr[order]
    lengths_arr = lengths_arr[order]
    if max_vectors is not None:
        if max_vectors <= 0:
            raise ValueError("max_vectors must be positive or None.")
        shifts_arr = shifts_arr[:max_vectors]
        lengths_arr = lengths_arr[:max_vectors]
    directions = shifts_arr.astype(np.float64)
    norms = np.sqrt(np.sum(directions * directions, axis=1))
    directions /= norms[:, None]
    return shifts_arr, directions, lengths_arr

def velocity_increments_grid(vgrid: Array, h: float, R: float, component: str = "longitudinal", tolerance: float | None = None, 
                             normalize: bool = False, max_vectors: int | None = None) -> Array:
    # compute velocity increments on a periodic coarse-grained grid for a given separation R and component choice
    vgrid = np.asarray(vgrid, dtype=np.float64)
    if vgrid.ndim != 3 or vgrid.shape[2] != 2:
        raise ValueError("vgrid must have shape (G, G, 2).")
    Gx, Gy, _ = vgrid.shape
    if Gx != Gy:
        raise ValueError("vgrid must be square.")
    component = component.lower()
    allowed = {"longitudinal", "transverse", "magnitude", "x", "y"}
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}.")
    shifts, directions, _ = _periodic_shift_vectors_for_radius(R=R, h=h, G=Gx, tolerance=tolerance, max_vectors=max_vectors)
    samples = []
    for (sx, sy), rhat in zip(shifts, directions):
        # np.roll with negative shifts places v[i+sx, j+sy] at index [i, j].
        shifted = np.roll(vgrid, shift=(-int(sx), -int(sy)), axis=(0, 1))
        dv = shifted - vgrid
        if component == "longitudinal":
            vals = dv[..., 0] * rhat[0] + dv[..., 1] * rhat[1]
        elif component == "transverse":
            # Rotate R_hat by +90 degrees.
            vals = -dv[..., 0] * rhat[1] + dv[..., 1] * rhat[0]
        elif component == "magnitude":
            vals = np.sqrt(dv[..., 0] * dv[..., 0] + dv[..., 1] * dv[..., 1])
        elif component == "x":
            vals = dv[..., 0]
        else:  # component == "y"
            vals = dv[..., 1]
        samples.append(vals.ravel())
    increments = np.concatenate(samples)
    if normalize:
        std = np.std(increments)
        if std > 0.0:
            increments = (increments - np.mean(increments)) / std
        else:
            increments = increments * 0.0
    return increments

def velocity_increment_pdf(increments: Array, bins: int | Array = 80, density: bool = True, symmetric: bool = True) -> tuple[Array, Array, Array]:
    # convert velocity-increment samples into a PDF/histogram. With density=True the area under the histogram is approximately one
    increments = np.asarray(increments, dtype=np.float64).ravel()
    increments = increments[np.isfinite(increments)]
    if increments.size == 0:
        raise ValueError("increments contains no finite samples.")
    hist_range = None
    if symmetric and np.isscalar(bins):
        vmax = float(np.max(np.abs(increments)))
        if vmax == 0.0:
            vmax = 1.0
        hist_range = (-vmax, vmax)
    pdf, edges = np.histogram(increments, bins=bins, range=hist_range, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf, edges

def velocity_increment_pdfs(vgrid: Array, h: float, R_values: Array, component: str = "longitudinal", bins: int | Array = 80, tolerance: float | None = None, 
                            normalize: bool = True, symmetric: bool = True, max_vectors: int | None = None) -> dict:
    # compute velocity-increment PDFs for several separations. v_R = v(x + R) - v(x), especially the longitudinal component v_parallel = v_R dot R_hat. 
    results = {}
    for R in np.asarray(R_values, dtype=np.float64):
        increments = velocity_increments_grid(vgrid=vgrid, h=h, R=float(R), component=component, tolerance=tolerance, normalize=normalize, max_vectors=max_vectors)
        centers, pdf, edges = velocity_increment_pdf(increments=increments, bins=bins, density=True, symmetric=symmetric)
        results[float(R)] = {"increments": increments, "bin_centers": centers, "pdf": pdf, "bin_edges": edges, "component": component, "normalized": normalize}
    return results

# rod snapshots/gif
def rod_line_segments(pos: Array, theta: Array, params: SPRParams) -> Array:
    # for visualization: return line segments for rods based on their positions and orientations
    u = unit_vectors(theta)
    half_skeleton = 0.5 * (params.ell - params.lambda_yukawa)
    p0 = pos - half_skeleton * u
    p1 = pos + half_skeleton * u
    return np.stack((p0, p1), axis=1)

def make_rods_animation(positions: Array, theta: Array, time: Array, params: SPRParams, 
                        filename: str = "spr_rods_animation.gif", fps: int = 12, dpi: int = 120, 
                        linewidth: float = 1.3) -> None:
    T = positions.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    seg0 = rod_line_segments(positions[0], theta[0], params)
    seg0 = periodic_visual_segments(seg0, params.L)
    lc = LineCollection(seg0, linewidths=linewidth, alpha=0.85)
    ax.add_collection(lc)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")
    _setup_equal_axis(ax, "SPR rods", params)
    def update(frame: int):
        seg = rod_line_segments(positions[frame], theta[frame], params)
        seg = periodic_visual_segments(seg, params.L)
        lc.set_segments(seg)
        time_text.set_text(f"frame {frame + 1}/{T}, t = {time[frame]:.2f}")
        return lc, time_text
    _save_animation(fig, update, T, filename, fps, dpi, blit=True)

def periodic_visual_segments(segments: Array, L: float) -> Array:
    return np.concatenate([segments + np.array([sx, sy]) for sx in (-L, 0.0, L) for sy in (-L, 0.0, L)], axis=0)

# vorticity gif
def _setup_equal_axis(ax: plt.Axes, title: str, params: SPRParams, xlabel: str = "x", ylabel: str = "y") -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0.0, params.L)
    ax.set_ylim(0.0, params.L)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def _save_animation(fig: plt.Figure, update, frames: int, filename: str, fps: int, dpi: int, blit: bool) -> None:
    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps, blit=blit)
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

def make_vorticity_animation(positions: Array, velocities: Array, time: Array, params: SPRParams, 
                             filename: str = "spr_vorticity_animation.gif", delta: float | None = None, 
                             overlap: float = 0.75, fps: int = 8, dpi: int = 120) -> None:
    if delta is None:
        delta = 1.31 * params.ell
    T = positions.shape[0]
    omegas = []
    for frame in range(T):
        vgrid, counts, h = coarse_grained_velocity(positions[frame], velocities[frame], L=params.L, delta=delta, overlap=overlap)
        omegas.append(vorticity_2d(vgrid, h))
    omegas = np.asarray(omegas)
    vmax = np.percentile(np.abs(omegas), 99.0)
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(omegas[0].T, origin="lower", extent=(0.0, params.L, 0.0, params.L), interpolation="nearest", vmin=-vmax, vmax=vmax)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", color="white")
    _setup_equal_axis(ax, "Coarse-grained vorticity", params)
    fig.colorbar(im, ax=ax, label="omega")
    def update(frame: int):
        im.set_data(omegas[frame].T)
        time_text.set_text(f"frame {frame + 1}/{T}, t = {time[frame]:.2f}")
        return im, time_text
    _save_animation(fig, update, T, filename, fps, dpi, blit=False)
