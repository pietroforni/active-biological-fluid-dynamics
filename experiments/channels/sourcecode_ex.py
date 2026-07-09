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
def _build_segments_and_cells(pos, theta, segment_offsets, Lx, Ly, rc):
    N = pos.shape[0]
    nseg = segment_offsets.shape[0]
    M = N * nseg
    ncell_x = int(math.floor(Lx / rc))
    ncell_y = int(math.floor(Ly / rc))
    if ncell_x < 1:
        ncell_x = 1
    if ncell_y < 1:
        ncell_y = 1
    cell_size_x = Lx / ncell_x
    cell_size_y = Ly / ncell_y
    n_cells_total = ncell_x * ncell_y

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
            x = x - Lx * math.floor(x / Lx)
            seg_x[sidx] = x
            seg_y[sidx] = y
            cx = int(math.floor(x / cell_size_x))
            cy = int(math.floor(y / cell_size_y))
            if cx >= ncell_x:
                cx = ncell_x - 1
            if cy < 0:
                cy = 0
            if cy >= ncell_y:
                cy = ncell_y - 1
            cid = cx + ncell_x * cy
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
    return seg_x, seg_y, ux, uy, order, starts, ncell_x, ncell_y

@njit(cache=True, fastmath=True)
def _compute_forces_numba(pos, theta, segment_offsets, Lx, Ly, rc, rmin, U0):
    N = pos.shape[0]
    nseg = segment_offsets.shape[0]
    seg_x, seg_y, ux, uy, order, starts, ncell_x, ncell_y = _build_segments_and_cells(pos, theta, segment_offsets, Lx, Ly, rc)
    n_cells_total = ncell_x * ncell_y
    forces = np.zeros((N, 2), dtype=np.float64)
    torques = np.zeros(N, dtype=np.float64)
    rc2 = rc * rc
    rmin2 = rmin * rmin
    half_Lx = 0.5 * Lx
    prefactor = U0 / float(nseg * nseg)
    # Used to avoid duplicate periodic neighbor cells when ncell_x is 1 or 2
    seen = np.empty(9, dtype=np.int64)

    for cid in range(n_cells_total):
        cx = cid % ncell_x
        cy = cid // ncell_x
        nseen = 0
        for oy in range(-1, 2):
            ncy = cy + oy
            if ncy < 0 or ncy >= ncell_y:
                continue
            for ox in range(-1, 2):
                ncx = (cx + ox) % ncell_x
                ncid = ncx + ncell_x * ncy
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
                        if dx > half_Lx:
                            dx -= Lx
                        elif dx < -half_Lx:
                            dx += Lx
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

# NEW GEOMETRY
GEOMETRY_STRAIGHT = 0
GEOMETRY_MIDDLE_CONSTRICTION = 1
GEOMETRY_STRAIGHT_OBSTACLES = 2
GEOMETRY_SINUSOIDAL_WALLS = 3

@njit(cache=True, fastmath=True)
def _local_channel_bounds_numba(x, Lx, Ly, geometry_code, constriction_length, constriction_throat_fraction, 
                                sinusoidal_wall_amplitude_fraction, sinusoidal_wall_periods, sinusoidal_wall_phase):
    # straight channel: y in [0, Ly]
    # middle constriction: symmetric rectangular constriction centered at x = Lx/2, with specified length and throat fraction
    if geometry_code == GEOMETRY_MIDDLE_CONSTRICTION:
        center = 0.5 * Lx
        dx = x - center
        # x is periodic, so we use minimum-image distance from the constriction center
        if dx > 0.5 * Lx:
            dx -= Lx
        elif dx < -0.5 * Lx:
            dx += Lx
        if abs(dx) <= 0.5 * constriction_length:
            throat_height = constriction_throat_fraction * Ly
            indent = 0.5 * (Ly - throat_height)
            return indent, Ly - indent
    if geometry_code == GEOMETRY_SINUSOIDAL_WALLS:
        phase = 2.0 * math.pi * sinusoidal_wall_periods * x / Lx + sinusoidal_wall_phase
        wave_lower = 0.5 * (1.0 + math.sin(phase))
        wave_upper = 0.5 * (1.0 + math.sin(phase + math.pi))
        lower = sinusoidal_wall_amplitude_fraction * Ly * wave_lower
        upper = Ly - sinusoidal_wall_amplitude_fraction * Ly * wave_upper
        return lower, upper
    return 0.0, Ly

@njit(cache=True, fastmath=True)
def _advance_one_step_inplace(pos, theta, vel, wall_mask, segment_offsets, Lx, Ly, dt, V, f_parallel, f_perp, 
                              f_rot, rc, rmin, U0, initial_impulse_force, impulse_is_active, geometry_code, 
                              constriction_length, constriction_throat_fraction, sinusoidal_wall_amplitude_fraction, 
                              sinusoidal_wall_periods, sinusoidal_wall_phase):
    forces, torques = _compute_forces_numba(pos, theta, segment_offsets, Lx, Ly, rc, rmin, U0)
    N = pos.shape[0]
    two_pi = 2.0 * math.pi
    y_upper = Ly - 1.0e-12
    for i in range(N):
        if wall_mask[i]:
            vel[i, 0] = 0.0
            vel[i, 1] = 0.0
            theta[i] = 0.0 # for robustness
            continue
        c = math.cos(theta[i])
        s = math.sin(theta[i])
        force_x = forces[i, 0]
        force_y = forces[i, 1]
        if impulse_is_active:
            force_x += initial_impulse_force
        f_parallel_scalar = force_x * c + force_y * s
        fpar_x = f_parallel_scalar * c
        fpar_y = f_parallel_scalar * s
        fperp_x = force_x - fpar_x
        fperp_y = force_y - fpar_y
        vx = V * c + fpar_x / f_parallel + fperp_x / f_perp
        vy = V * s + fpar_y / f_parallel + fperp_y / f_perp
        vel[i, 0] = vx
        vel[i, 1] = vy
        x = pos[i, 0] + dt * vx
        y = pos[i, 1] + dt * vy
        x = x - Lx * math.floor(x / Lx)
        y_min, y_max = _local_channel_bounds_numba(x, Lx, Ly, geometry_code, constriction_length, constriction_throat_fraction, 
                                                   sinusoidal_wall_amplitude_fraction, sinusoidal_wall_periods, sinusoidal_wall_phase)
        y_upper = y_max - 1.0e-12
        pos[i, 0] = x
        if y < y_min:
            y = y_min
        elif y >= y_max:
            y = y_upper
        pos[i, 1] = y
        theta_i = theta[i] + dt * torques[i] / f_rot
        theta[i] = (theta_i + math.pi) % two_pi - math.pi

@njit(cache=True, fastmath=True)
def _run_simulation_numba(pos0, theta0, wall_mask, segment_offsets, Lx, Ly, dt, V, f_parallel, f_perp, f_rot, rc, rmin, 
                          U0, initial_impulse_force, initial_impulse_steps, n_steps, sample_every, relax_steps, geometry_code, 
                          constriction_length, constriction_throat_fraction, sinusoidal_wall_amplitude_fraction,
                          sinusoidal_wall_periods, sinusoidal_wall_phase):
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
        observed_step = step - relax_steps
        impulse_is_active = observed_step >= 0 and observed_step < initial_impulse_steps
        _advance_one_step_inplace(pos, theta, vel, wall_mask, segment_offsets, Lx, Ly, dt, V, f_parallel, f_perp, f_rot, rc, rmin, 
                                  U0, initial_impulse_force, impulse_is_active, geometry_code, constriction_length, constriction_throat_fraction,
                                  sinusoidal_wall_amplitude_fraction, sinusoidal_wall_periods, sinusoidal_wall_phase)
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
    pos, theta, wall_mask = initialize_channel(params)
    geometry_code = geometry_code_from_params(params)
    constriction_length = float(params.constriction_length_fraction * params.Lx)
    saved_pos, saved_theta, saved_vel, saved_time = _run_simulation_numba(np.asarray(pos, dtype=np.float64), np.asarray(theta, dtype=np.float64),
                                                                          np.asarray(wall_mask, dtype=np.bool_), np.asarray(params.segment_offsets, dtype=np.float64),
                                                                          float(params.Lx), float(params.Ly), float(params.dt), float(params.V),
                                                                          float(params.f_parallel), float(params.f_perp), float(params.f_rot),
                                                                          float(params.rc), float(params.rmin), float(params.U0),
                                                                          float(params.initial_impulse_force), int(params.initial_impulse_steps),
                                                                          int(n_steps), int(sample_every), int(relax_steps), int(geometry_code), 
                                                                          float(constriction_length), float(params.constriction_throat_fraction), 
                                                                          float(params.sinusoidal_wall_amplitude_fraction), float(params.sinusoidal_wall_periods), 
                                                                          float(params.sinusoidal_wall_phase))
    metadata = params.metadata()
    metadata.update(n_steps=n_steps, sample_every=sample_every, relax_steps=relax_steps, integrator="explicit Euler, Numba core")
    metadata.update(n_wall=int(np.count_nonzero(wall_mask)), n_mobile=int(params.N - np.count_nonzero(wall_mask)))
    return {"positions": saved_pos, "theta": saved_theta, "velocities": saved_vel, "time": saved_time, "wall_mask": wall_mask, "metadata": metadata}

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
    channel_aspect_ratio: float = 4.0
    n_wall_layers: int = 1
    initial_impulse_force: float = 1.0
    initial_impulse_steps: int = 1
    wall_angle: float = 0.0
    # NEW GEOMETRY: set "straight_channel" to keep old behavior,
    # or "middle_constriction" to add a symmetric constriction at x=Lx/2.
    geometry: str = "straight_channel"
    constriction_length_fraction: float = 0.20   # constriction length / Lx
    constriction_throat_fraction: float = 0.50   # open throat height / Ly
    obstacle_circle_center_fraction: tuple[float, float] = (0.40, 0.50)  # (x/Lx, y/Ly)
    obstacle_circle_radius_fraction: float = 0.20                        # radius / Ly
    obstacle_square_center_fraction: tuple[float, float] = (0.62, 0.50)  # (x/Lx, y/Ly)
    obstacle_square_side_fraction: float = 0.35                          # side / Ly
    sinusoidal_wall_amplitude_fraction: float = 0.15  # max single-wall indentation / Ly; must be < 0.5
    sinusoidal_wall_periods: float = 3.0              # number of sinusoidal periods along Lx
    sinusoidal_wall_phase: float = 0.0                # phase shift in radians

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
        if self.channel_aspect_ratio <= 0.0:
            raise ValueError("channel_aspect_ratio must be positive.")
        if self.n_wall_layers < 1:
            raise ValueError("n_wall_layers must be >= 1.")
        if self.initial_impulse_steps < 0:
            raise ValueError("initial_impulse_steps must be >= 0.")
        self.geometry = self.geometry.lower()
        if self.geometry not in {"straight", "straight_channel", "middle_constriction", "constriction", "straight_channel_obstacles", "straight_obstacles", "obstacles", "sinusoidal_walls", "sinusoidal_channel", "sine_walls"}:
            raise ValueError("geometry name error.")
        if not (0.0 < self.constriction_length_fraction <= 1.0):
            raise ValueError("constriction_length_fraction must satisfy 0 < value <= 1.")
        if not (0.0 < self.constriction_throat_fraction <= 1.0):
            raise ValueError("constriction_throat_fraction must satisfy 0 < value <= 1.")
        if not (0.0 <= self.sinusoidal_wall_amplitude_fraction < 0.5):
            raise ValueError("sinusoidal_wall_amplitude_fraction must satisfy 0 <= value < 0.5.")
        if self.sinusoidal_wall_periods <= 0.0:
            raise ValueError("sinusoidal_wall_periods must be positive.")
        if len(self.obstacle_circle_center_fraction) != 2:
            raise ValueError("obstacle_circle_center_fraction must be a pair (x_fraction, y_fraction).")
        if len(self.obstacle_square_center_fraction) != 2:
            raise ValueError("obstacle_square_center_fraction must be a pair (x_fraction, y_fraction).")

        self.obstacle_circle_center_fraction = tuple(float(v) for v in self.obstacle_circle_center_fraction)
        self.obstacle_square_center_fraction = tuple(float(v) for v in self.obstacle_square_center_fraction)

        for name, pair in (("obstacle_circle_center_fraction", self.obstacle_circle_center_fraction), 
                           ("obstacle_square_center_fraction", self.obstacle_square_center_fraction)):
            if not (0.0 <= pair[0] <= 1.0 and 0.0 <= pair[1] <= 1.0):
                raise ValueError(f"{name} entries must lie between 0 and 1.")
        if self.obstacle_circle_radius_fraction <= 0.0:
            raise ValueError("obstacle_circle_radius_fraction must be positive.")
        if self.obstacle_square_side_fraction <= 0.0:
            raise ValueError("obstacle_square_side_fraction must be positive.")

        self.lambda_yukawa = 1.0
        self.ell = self.a
        self.n_segments = n_segments_for_aspect(self.a)
        self.segment_offsets = make_segment_offsets(self.a, self.n_segments)
        self.segment_rod_id = np.repeat(np.arange(self.N, dtype=np.int64), self.n_segments)
        self.segment_l_flat = np.tile(self.segment_offsets, self.N)
        particle_area = self.a - 1.0 + math.pi / 4.0
        self.area = self.N * particle_area / self.phi
        self.Ly = math.sqrt(self.area / self.channel_aspect_ratio)
        self.Lx = self.channel_aspect_ratio * self.Ly
        self.L = self.Lx
        self.rho = self.N / self.area
        self.dt = self.dt_scale * self.rho ** (-0.5)
        self.f_parallel, self.f_perp, self.f_rot = friction_coefficients(self.a)
        self.V = self.propulsion_force / self.f_parallel

    def metadata(self) -> dict:
        return {"N": self.N, "a": self.a, "phi": self.phi, "U0": self.U0, "Lx": self.Lx, "Ly": self.Ly,
                "channel_aspect_ratio": self.channel_aspect_ratio, "rho": self.rho, "n_segments": self.n_segments,
                "rc": self.rc, "dt": self.dt, "rmin": self.rmin, "seed": self.seed,
                "propulsion_force": self.propulsion_force, "dt_scale": self.dt_scale,
                "n_wall_layers": self.n_wall_layers, "initial_impulse_force": self.initial_impulse_force,
                "initial_impulse_steps": self.initial_impulse_steps, "wall_angle": self.wall_angle,
                "integrator": "explicit Euler", "units": "lambda=1, F=1, f0=1", "geometry": self.geometry,
                "constriction_length_fraction": self.constriction_length_fraction, "constriction_throat_fraction": self.constriction_throat_fraction,
                "obstacle_circle_center_fraction": self.obstacle_circle_center_fraction, "obstacle_circle_radius_fraction": self.obstacle_circle_radius_fraction, 
                "obstacle_square_center_fraction": self.obstacle_square_center_fraction, "obstacle_square_side_fraction": self.obstacle_square_side_fraction,
                "sinusoidal_wall_amplitude_fraction": self.sinusoidal_wall_amplitude_fraction, "sinusoidal_wall_periods": self.sinusoidal_wall_periods, 
                "sinusoidal_wall_phase": self.sinusoidal_wall_phase}

def make_segment_offsets(a: float, n_segments: int) -> Array: # segment offsets along the rod axis for Yukawa interaction
    if n_segments == 1:
        return np.array([0.0], dtype=np.float64)
    half = 0.5 * (a - 1.0) # lambda = 1, so skeleton length is a - 1
    return np.linspace(-half, half, n_segments, dtype=np.float64)

def unit_vectors(theta: Array) -> Array: # u = (cos theta, sin theta) for all rods
    return np.column_stack((np.cos(theta), np.sin(theta)))

def wrap_positions(pos: Array, L: float, Ly: float | None = None) -> Array: # x-periodic; y is periodic only when Ly is omitted
    wrapped = np.asarray(pos, dtype=np.float64).copy()
    wrapped[:, 0] = np.mod(wrapped[:, 0], L)
    if Ly is None:
        wrapped[:, 1] = np.mod(wrapped[:, 1], L)
    else:
        wrapped[:, 1] = np.clip(wrapped[:, 1], 0.0, np.nextafter(float(Ly), 0.0))
    return wrapped

def _channel_lattice_positions(params: SPRParams) -> Array:
    # rods initialized on a rectangular grid over the whole channel area
    nx = max(1, min(params.N, int(np.floor(params.Lx / max(params.ell, 1.0)))))
    ny = int(np.ceil(params.N / nx))
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * params.Lx / nx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * params.Ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pos = np.column_stack((xx.ravel(), yy.ravel()))[: params.N].copy()
    return pos

# NEW GEOMETRY
def geometry_code_from_params(params: SPRParams) -> int:
    geometry = params.geometry.lower()
    if geometry in {"straight", "straight_channel"}:
        return GEOMETRY_STRAIGHT
    if geometry in {"middle_constriction", "constriction"}:
        return GEOMETRY_MIDDLE_CONSTRICTION
    if geometry in {"straight_channel_obstacles", "straight_obstacles", "obstacles"}:
        return GEOMETRY_STRAIGHT_OBSTACLES
    if geometry in {"sinusoidal_walls", "sinusoidal_channel", "sine_walls"}:
        return GEOMETRY_SINUSOIDAL_WALLS
    raise ValueError(f"Unknown geometry: {params.geometry}")

def sinusoidal_channel_bounds(x: Array | float, params: SPRParams) -> tuple[Array, Array] | tuple[float, float]:
    # sinusoidal walls: the lower wall moves up while the upper wall moves down
    # The open height is smallest where the wave is one and largest where the wave is zero
    scalar_input = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    theta = 2.0 * np.pi * params.sinusoidal_wall_periods * x_arr / params.Lx
    delta_phase = np.pi  # upper wall is half a period out of phase with lower wall
    wave_lower = 0.5 * (1.0 + np.sin(theta + params.sinusoidal_wall_phase))
    wave_upper = 0.5 * (1.0 + np.sin(theta + params.sinusoidal_wall_phase + delta_phase))
    lower = params.sinusoidal_wall_amplitude_fraction * params.Ly * wave_lower
    upper = params.Ly - params.sinusoidal_wall_amplitude_fraction * params.Ly * wave_upper
    if scalar_input:
        return float(lower[0]), float(upper[0])
    return lower, upper

def channel_wall_thickness(params: SPRParams) -> float:
    return (params.n_wall_layers * params.Ly / max(1, int(np.ceil(math.sqrt(params.N / params.channel_aspect_ratio)))))

def middle_constriction_bounds(x: Array | float, params: SPRParams) -> tuple[Array, Array] | tuple[float, float]:
    scalar_input = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lower = np.zeros_like(x_arr, dtype=np.float64)
    upper = np.full_like(x_arr, params.Ly, dtype=np.float64)
    length = params.constriction_length_fraction * params.Lx
    dx = x_arr - 0.5 * params.Lx
    dx = dx - params.Lx * np.round(dx / params.Lx)
    in_constriction = np.abs(dx) <= 0.5 * length
    throat_height = params.constriction_throat_fraction * params.Ly
    indent = 0.5 * (params.Ly - throat_height)
    lower[in_constriction] = indent
    upper[in_constriction] = params.Ly - indent
    if scalar_input:
        return float(lower[0]), float(upper[0])
    return lower, upper

def local_channel_bounds(x: Array | float, params: SPRParams) -> tuple[Array, Array] | tuple[float, float]:
    geometry_code = geometry_code_from_params(params)

    if geometry_code == GEOMETRY_MIDDLE_CONSTRICTION:
        return middle_constriction_bounds(x, params)

    if geometry_code == GEOMETRY_SINUSOIDAL_WALLS:
        return sinusoidal_channel_bounds(x, params)

    scalar_input = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lower = np.zeros_like(x_arr, dtype=np.float64)
    upper = np.full_like(x_arr, params.Ly, dtype=np.float64)

    if scalar_input:
        return float(lower[0]), float(upper[0])

    return lower, upper

def local_channel_height(x: Array | float, params: SPRParams) -> Array | float:
    lower, upper = local_channel_bounds(x, params)
    return np.asarray(upper) - np.asarray(lower)

def make_constricted_channel_wall_mask(pos: Array, params: SPRParams) -> Array:
    wall_thickness = channel_wall_thickness(params)
    lower, upper = local_channel_bounds(pos[:, 0], params)
    open_lower = lower + wall_thickness
    open_upper = upper - wall_thickness
    open_exists = open_upper > open_lower
    mobile_region = (open_exists & (pos[:, 1] > open_lower) & (pos[:, 1] < open_upper))
    return ~mobile_region

def make_sinusoidal_channel_wall_mask(pos: Array, params: SPRParams) -> Array:
    wall_thickness = channel_wall_thickness(params)
    lower, upper = local_channel_bounds(pos[:, 0], params)
    open_lower = lower + wall_thickness
    open_upper = upper - wall_thickness
    open_exists = open_upper > open_lower
    mobile_region = open_exists & (pos[:, 1] > open_lower) & (pos[:, 1] < open_upper)
    return ~mobile_region

def make_channel_wall_mask(pos: Array, params: SPRParams) -> Array:
    geometry_code = geometry_code_from_params(params)
    if geometry_code == GEOMETRY_MIDDLE_CONSTRICTION:
        return make_constricted_channel_wall_mask(pos, params)
    if geometry_code == GEOMETRY_STRAIGHT_OBSTACLES:
        return make_obstacles_channel_wall_mask(pos, params)
    if geometry_code == GEOMETRY_SINUSOIDAL_WALLS:
        return make_sinusoidal_channel_wall_mask(pos, params)
    return make_straight_channel_wall_mask(pos, params)

def initialize_obstacles_channel(params: SPRParams) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(params.seed)
    pos = _channel_lattice_positions(params)
    pos = wrap_positions(pos, params.Lx, params.Ly)
    wall_mask = make_obstacles_channel_wall_mask(pos, params).astype(np.bool_)
    if np.all(wall_mask):
        raise ValueError("The obstacle channel contains no mobile rods. Reduce obstacle sizes or n_wall_layers.")
    theta = rng.choice(np.array([0.0, math.pi]), size=params.N).astype(np.float64)
    theta += rng.uniform(-math.pi / 18.0, math.pi / 18.0, size=params.N)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    theta[wall_mask] = 0.0
    return pos, theta, wall_mask

def _periodic_dx_array(x: Array, center_x: float, Lx: float) -> Array:
    dx = np.asarray(x, dtype=np.float64) - center_x
    return dx - Lx * np.round(dx / Lx)

def obstacle_circle_mask(pos: Array, params: SPRParams) -> Array:
    cx = params.obstacle_circle_center_fraction[0] * params.Lx
    cy = params.obstacle_circle_center_fraction[1] * params.Ly
    r = params.obstacle_circle_radius_fraction * params.Ly

    dx = _periodic_dx_array(pos[:, 0], cx, params.Lx)
    dy = pos[:, 1] - cy
    return dx * dx + dy * dy <= r * r

def obstacle_square_mask(pos: Array, params: SPRParams) -> Array:
    cx = params.obstacle_square_center_fraction[0] * params.Lx
    cy = params.obstacle_square_center_fraction[1] * params.Ly
    half_side = 0.5 * params.obstacle_square_side_fraction * params.Ly

    dx = np.abs(_periodic_dx_array(pos[:, 0], cx, params.Lx))
    dy = np.abs(pos[:, 1] - cy)
    return (dx <= half_side) & (dy <= half_side)

def make_obstacles_channel_wall_mask(pos: Array, params: SPRParams) -> Array:
    wall_mask = make_straight_channel_wall_mask(pos, params).copy()
    wall_mask |= obstacle_circle_mask(pos, params)
    wall_mask |= obstacle_square_mask(pos, params)
    return wall_mask

def initialize_sinusoidal_channel(params: SPRParams) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(params.seed)
    pos = _channel_lattice_positions(params)
    pos = wrap_positions(pos, params.Lx, params.Ly)

    wall_mask = make_sinusoidal_channel_wall_mask(pos, params).astype(np.bool_)

    if np.all(wall_mask):
        raise ValueError(
            "The sinusoidal channel contains no mobile rods. "
            "Reduce sinusoidal_wall_amplitude_fraction or n_wall_layers."
        )

    theta = rng.choice(np.array([0.0, math.pi]), size=params.N).astype(np.float64)
    theta += rng.uniform(-math.pi / 18.0, math.pi / 18.0, size=params.N)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    theta[wall_mask] = 0.0

    return pos, theta, wall_mask

def initialize_constricted_channel(params: SPRParams) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(params.seed)
    pos = _channel_lattice_positions(params)
    pos = wrap_positions(pos, params.Lx, params.Ly)
    wall_mask = make_constricted_channel_wall_mask(pos, params).astype(np.bool_)
    if np.all(wall_mask):
        raise ValueError("The constricted channel contains no mobile rods. ""Increase constriction_throat_fraction or reduce n_wall_layers.")
    theta = rng.choice(np.array([0.0, math.pi]), size=params.N).astype(np.float64)
    theta += rng.uniform(-math.pi / 18.0, math.pi / 18.0, size=params.N)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    theta[wall_mask] = 0.0
    return pos, theta, wall_mask

def initialize_channel(params: SPRParams) -> tuple[Array, Array, Array]:
    geometry_code = geometry_code_from_params(params)

    if geometry_code == GEOMETRY_MIDDLE_CONSTRICTION:
        return initialize_constricted_channel(params)

    if geometry_code == GEOMETRY_STRAIGHT_OBSTACLES:
        return initialize_obstacles_channel(params)

    if geometry_code == GEOMETRY_SINUSOIDAL_WALLS:
        return initialize_sinusoidal_channel(params)

    return initialize_straight_channel(params)


def make_straight_channel_wall_mask(pos: Array, params: SPRParams) -> Array:
    # Freeze rods whose centers are close to the bottom or top boundary.
    wall_thickness = channel_wall_thickness(params)
    return (pos[:, 1] <= wall_thickness) | (pos[:, 1] >= params.Ly - wall_thickness)

def initialize_lattice(params: SPRParams) -> tuple[Array, Array]:
    # rods with symmetric nearly-horizontal orientations on a rectangular grid
    rng = np.random.default_rng(params.seed)
    pos = _channel_lattice_positions(params)
    theta = rng.choice(np.array([0.0, math.pi]), size=params.N).astype(np.float64)
    theta += rng.uniform(-math.pi / 18.0, math.pi / 18.0, size=params.N)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return wrap_positions(pos, params.Lx, params.Ly), theta

def initialize_straight_channel(params: SPRParams) -> tuple[Array, Array, Array]:
    # initialize all rods first, then freeze the ones close to top/bottom walls
    rng = np.random.default_rng(params.seed)
    pos = _channel_lattice_positions(params)
    pos = wrap_positions(pos, params.Lx, params.Ly)
    wall_mask = make_straight_channel_wall_mask(pos, params).astype(np.bool_)
    if np.all(wall_mask):
        raise ValueError("The straight channel contains no mobile rods. Reduce n_wall_layers or wall thickness.")
    theta = rng.choice(np.array([0.0, math.pi]), size=params.N).astype(np.float64)
    theta += rng.uniform(-math.pi / 18.0, math.pi / 18.0, size=params.N)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    theta[wall_mask] = 0.0
    return pos, theta, wall_mask

##### DIAGNOSTICS #####

# vorticity + esentrophy
def coarse_grained_velocity(pos: Array, vel: Array, L: float, delta: float, overlap: float = 0.75, Ly: float | None = None) -> tuple[Array, Array, float | tuple[float, float]]:
    # coarse-grained velocity field on an overlapping grid; x is periodic, y is periodic only when Ly is omitted
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    periodic_y = Ly is None
    Lx = float(L)
    if Ly is None:
        Ly = Lx
    Ly = float(Ly)
    target_h = delta * (1.0 - overlap)
    Gx = max(4, int(np.floor(Lx / target_h)))
    Gy = max(4, int(np.floor(Ly / target_h)))
    hx = Lx / Gx
    hy = Ly / Gy
    centers_x = (np.arange(Gx, dtype=np.float64) + 0.5) * hx
    centers_y = (np.arange(Gy, dtype=np.float64) + 0.5) * hy
    sum_v = np.zeros((Gx, Gy, 2), dtype=np.float64)
    counts = np.zeros((Gx, Gy), dtype=np.int64)
    base_x = np.floor(pos[:, 0] / hx).astype(np.int64) % Gx
    base_y = np.floor(pos[:, 1] / hy).astype(np.int64)
    base_y = np.clip(base_y, 0, Gy - 1)
    mx = int(np.ceil(0.5 * delta / hx)) + 1
    my = int(np.ceil(0.5 * delta / hy)) + 1
    for ox in range(-mx, mx + 1):
        ix = (base_x + ox) % Gx
        dx = pos[:, 0] - centers_x[ix]
        dx = dx - Lx * np.round(dx / Lx)
        mask_x = np.abs(dx) <= 0.5 * delta
        for oy in range(-my, my + 1):
            if periodic_y:
                iy = (base_y + oy) % Gy
                dy = pos[:, 1] - centers_y[iy]
                dy = dy - Ly * np.round(dy / Ly)
                mask = mask_x & (np.abs(dy) <= 0.5 * delta)
            else:
                iy_raw = base_y + oy
                valid_y = (iy_raw >= 0) & (iy_raw < Gy)
                if not np.any(valid_y):
                    continue
                iy = np.clip(iy_raw, 0, Gy - 1)
                dy = pos[:, 1] - centers_y[iy]
                mask = mask_x & valid_y & (np.abs(dy) <= 0.5 * delta)
            if not np.any(mask):
                continue
            np.add.at(sum_v[..., 0], (ix[mask], iy[mask]), vel[mask, 0])
            np.add.at(sum_v[..., 1], (ix[mask], iy[mask]), vel[mask, 1])
            np.add.at(counts, (ix[mask], iy[mask]), 1)
    vgrid = np.zeros_like(sum_v)
    nonempty = counts > 0
    vgrid[nonempty] = sum_v[nonempty] / counts[nonempty, None]
    if periodic_y:
        return vgrid, counts, hx
    return vgrid, counts, (hx, hy)

def vorticity_2d(vgrid: Array, h: float | tuple[float, float]) -> Array:
    # omega = d_x v_y - d_y v_x; x is periodic, y is non-periodic when h is a tuple
    if np.isscalar(h):
        hx = float(h)
        hy = float(h)
        periodic_y = True
    else:
        hx = float(h[0])
        hy = float(h[1])
        periodic_y = False
    dvydx = (np.roll(vgrid[..., 1], -1, axis=0) - np.roll(vgrid[..., 1], 1, axis=0)) / (2.0 * hx)
    if periodic_y:
        dvxdy = (np.roll(vgrid[..., 0], -1, axis=1) - np.roll(vgrid[..., 0], 1, axis=1)) / (2.0 * hy)
    else:
        dvxdy = np.empty(vgrid.shape[:2], dtype=np.float64)
        if vgrid.shape[1] == 1:
            dvxdy[:, 0] = 0.0
        else:
            dvxdy[:, 1:-1] = (vgrid[:, 2:, 0] - vgrid[:, :-2, 0]) / (2.0 * hy)
            dvxdy[:, 0] = (vgrid[:, 1, 0] - vgrid[:, 0, 0]) / hy
            dvxdy[:, -1] = (vgrid[:, -1, 0] - vgrid[:, -2, 0]) / hy
    return dvydx - dvxdy

def enstrophy(omega: Array) -> float: # Omega = 1/2 <omega^2>
    return 0.5 * float(np.mean(omega * omega))

def mean_vortex_dimension_over_height(omega: Array, h: float | tuple[float, float], params: SPRParams, threshold_factor: float = 0.5, min_cells: int = 4) -> dict:
    omega = np.asarray(omega, dtype=np.float64)
    if np.isscalar(h):
        hx = hy = float(h)
    else:
        hx = float(h[0])
        hy = float(h[1])
    Gx, Gy = omega.shape
    rms_omega = np.sqrt(np.mean(omega * omega))
    if rms_omega == 0.0 or not np.isfinite(rms_omega):
        return {"n_vortices": 0, "mean_vortex_dimension": np.nan, "mean_vortex_dimension_over_Ly": np.nan, "std_vortex_dimension_over_Ly": np.nan}
    threshold = threshold_factor * rms_omega
    vortex_dimensions = []
    vortex_dimensions_over_local_height = []
    vortex_local_heights = []
    x_centers = (np.arange(Gx, dtype=np.float64) + 0.5) * params.Lx / Gx
    for sign in (+1, -1):
        if sign > 0:
            mask = omega > threshold
        else:
            mask = omega < -threshold
        visited = np.zeros_like(mask, dtype=bool)
        for i0, j0 in np.argwhere(mask):
            if visited[i0, j0]:
                continue
            stack = [(int(i0), int(j0))]
            visited[i0, j0] = True
            n_cells = 0
            cols = []
            while stack:
                i, j = stack.pop()
                n_cells += 1
                cols.append(i)
                # x is periodic, y is not periodic
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni = (i + di) % Gx
                    nj = j + dj
                    if nj < 0 or nj >= Gy:
                        continue
                    if mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
            if n_cells < min_cells:
                continue
            area = n_cells * hx * hy
            equivalent_diameter = 2.0 * np.sqrt(area / np.pi)
            unique_cols = np.unique(np.asarray(cols, dtype=np.int64))
            local_heights = local_channel_height(x_centers[unique_cols], params)
            mean_local_height = float(np.mean(local_heights))
            vortex_local_heights.append(mean_local_height)
            vortex_dimensions_over_local_height.append(equivalent_diameter / mean_local_height)
            vortex_dimensions.append(equivalent_diameter)
    if len(vortex_dimensions) == 0:
        return {"n_vortices": 0, "mean_vortex_dimension": np.nan, "mean_vortex_dimension_over_Ly": np.nan, "std_vortex_dimension_over_Ly": np.nan, "mean_vortex_dimension_over_local_height": np.nan,
                "std_vortex_dimension_over_local_height": np.nan, "mean_local_channel_height_for_vortices": np.nan}
    vortex_dimensions = np.asarray(vortex_dimensions, dtype=np.float64)
    vortex_dimensions_over_Ly = vortex_dimensions / params.Ly
    return {"n_vortices": int(len(vortex_dimensions)), "mean_vortex_dimension": float(np.mean(vortex_dimensions)), 
            "mean_vortex_dimension_over_Ly": float(np.mean(vortex_dimensions_over_Ly)), "std_vortex_dimension_over_Ly": float(np.std(vortex_dimensions_over_Ly)),
            "mean_vortex_dimension_over_local_height": float(np.mean(vortex_dimensions_over_local_height)), "std_vortex_dimension_over_local_height": float(np.std(vortex_dimensions_over_local_height)),
            "mean_local_channel_height_for_vortices": float(np.mean(vortex_local_heights))}

def compute_vortex_dimension_time_average(result: dict, params: SPRParams, delta: float, overlap: float, start_fraction: float = 0.5, threshold_factor: float = 0.5, min_cells: int = 4) -> dict:
    wall_mask = result["wall_mask"]
    positions = result["positions"]
    velocities = result["velocities"]
    start = int(start_fraction * positions.shape[0])
    rows = []
    for frame in range(start, positions.shape[0]):
        pos = positions[frame]
        vel = velocities[frame]
        vgrid, counts, h = coarse_grained_velocity(pos[~wall_mask], vel[~wall_mask], L=params.Lx, Ly=params.Ly, delta=delta, overlap=overlap)
        omega = vorticity_2d(vgrid, h)
        row = mean_vortex_dimension_over_height(omega, h, params, threshold_factor=threshold_factor, min_cells=min_cells)
        rows.append(row)
    keys = rows[0].keys()
    out = {}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        finite = np.isfinite(vals)
        if np.any(finite):
            out[f"time_mean_{key}"] = float(np.mean(vals[finite]))
            out[f"time_std_{key}"] = float(np.std(vals[finite]))
        else:
            out[f"time_mean_{key}"] = np.nan
            out[f"time_std_{key}"] = np.nan
    return out

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
    fig, ax = plt.subplots(figsize=(8, 3))
    seg0 = rod_line_segments(positions[0], theta[0], params)
    seg0 = periodic_visual_segments(seg0, params.Lx)
    lc = LineCollection(seg0, linewidths=linewidth, alpha=0.85)
    ax.add_collection(lc)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")
    plot_channel_boundaries(ax, params, linewidth=1.0, linestyle="--", alpha=0.85)
    _setup_equal_axis(ax, "SPR rods", params)
    def update(frame: int):
        seg = rod_line_segments(positions[frame], theta[frame], params)
        seg = periodic_visual_segments(seg, params.Lx)
        lc.set_segments(seg)
        time_text.set_text(f"frame {frame + 1}/{T}, t = {time[frame]:.2f}")
        return lc, time_text
    _save_animation(fig, update, T, filename, fps, dpi, blit=True)

def periodic_visual_segments(segments: Array, L: float) -> Array:
    return np.concatenate([segments + np.array([sx, 0.0]) for sx in (-L, 0.0, L)], axis=0)

# vorticity gif
def _setup_equal_axis(ax: plt.Axes, title: str, params: SPRParams, xlabel: str = "x", ylabel: str = "y") -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0.0, params.Lx)
    ax.set_ylim(0.0, params.Ly)
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
        vgrid, counts, h = coarse_grained_velocity(positions[frame], velocities[frame], L=params.Lx, Ly=params.Ly, delta=delta, overlap=overlap)
        omegas.append(vorticity_2d(vgrid, h))
    omegas = np.asarray(omegas)
    vmax = np.percentile(np.abs(omegas), 99.0)
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(omegas[0].T, origin="lower", extent=(0.0, params.Lx, 0.0, params.Ly), interpolation="nearest", vmin=-vmax, vmax=vmax)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", color="white")
    _setup_equal_axis(ax, "Coarse-grained vorticity", params)
    fig.colorbar(im, ax=ax, label="omega")
    def update(frame: int):
        im.set_data(omegas[frame].T)
        time_text.set_text(f"frame {frame + 1}/{T}, t = {time[frame]:.2f}")
        return im, time_text
    _save_animation(fig, update, T, filename, fps, dpi, blit=False)

def plot_obstacle_boundaries(ax: plt.Axes, params: SPRParams, n_points: int = 240, **kwargs) -> None:
    if geometry_code_from_params(params) != GEOMETRY_STRAIGHT_OBSTACLES:
        return
    circle_cx = params.obstacle_circle_center_fraction[0] * params.Lx
    circle_cy = params.obstacle_circle_center_fraction[1] * params.Ly
    circle_r = params.obstacle_circle_radius_fraction * params.Ly
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    square_cx = params.obstacle_square_center_fraction[0] * params.Lx
    square_cy = params.obstacle_square_center_fraction[1] * params.Ly
    half_side = 0.5 * params.obstacle_square_side_fraction * params.Ly
    square_x = np.array([-half_side, half_side, half_side, -half_side, -half_side])
    square_y = np.array([-half_side, -half_side, half_side, half_side, -half_side])
    for shift in (-params.Lx, 0.0, params.Lx):
        ax.plot(circle_cx + shift + circle_r * np.cos(t),
                circle_cy + circle_r * np.sin(t), **kwargs)
        ax.plot(square_cx + shift + square_x,
                square_cy + square_y, **kwargs)
        
def plot_channel_boundaries(ax: plt.Axes, params: SPRParams, n_points: int = 600, **kwargs) -> None:
    x = np.linspace(0.0, params.Lx, n_points)
    lower, upper = local_channel_bounds(x, params)
    ax.plot(x, lower, **kwargs)
    ax.plot(x, upper, **kwargs)
    plot_obstacle_boundaries(ax, params, **kwargs)