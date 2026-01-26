from __future__ import annotations

import numpy as np


def stable_dt_3d(dx: float, D_xy: float, D_z: float, safety: float = 0.2) -> float:
    """Return a stable time step for explicit 3D diffusion with anisotropy."""
    if D_xy <= 0 or D_z <= 0:
        raise ValueError("D_xy and D_z must be positive")
    return safety * dx * dx / (2 * (2 * D_xy + D_z))


def initial_point_3d(shape: tuple[int, int, int], center: tuple[int, int, int], total_particles: float) -> np.ndarray:
    """Place all particles at a single grid cell (discrete point source)."""
    n = np.zeros(shape, dtype=float)
    n[center] = total_particles
    return n


def initial_gaussian_3d(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    sigma: float,
    total_particles: float,
    dx: float = 1.0,
) -> np.ndarray:
    """Return a 3D Gaussian normalized to total_particles."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    nx, ny, nz = shape
    x = (np.arange(nx) - center[0]) * dx
    y = (np.arange(ny) - center[1]) * dx
    z = (np.arange(nz) - center[2]) * dx
    x2 = x[:, None, None] ** 2
    y2 = y[None, :, None] ** 2
    z2 = z[None, None, :] ** 2
    n = np.exp(-(x2 + y2 + z2) / (2 * sigma * sigma))
    n *= total_particles / (n.sum() * dx**3)
    return n


def _fill_neumann_pad(n: np.ndarray, n_pad: np.ndarray) -> None:
    """Fill padded array with edge-replicated values (zero-flux boundaries)."""
    n_pad[1:-1, 1:-1, 1:-1] = n

    n_pad[0, 1:-1, 1:-1] = n[0, :, :]
    n_pad[-1, 1:-1, 1:-1] = n[-1, :, :]
    n_pad[1:-1, 0, 1:-1] = n[:, 0, :]
    n_pad[1:-1, -1, 1:-1] = n[:, -1, :]
    n_pad[1:-1, 1:-1, 0] = n[:, :, 0]
    n_pad[1:-1, 1:-1, -1] = n[:, :, -1]

    n_pad[0, 0, 1:-1] = n[0, 0, :]
    n_pad[0, -1, 1:-1] = n[0, -1, :]
    n_pad[-1, 0, 1:-1] = n[-1, 0, :]
    n_pad[-1, -1, 1:-1] = n[-1, -1, :]

    n_pad[0, 1:-1, 0] = n[0, :, 0]
    n_pad[0, 1:-1, -1] = n[0, :, -1]
    n_pad[-1, 1:-1, 0] = n[-1, :, 0]
    n_pad[-1, 1:-1, -1] = n[-1, :, -1]

    n_pad[1:-1, 0, 0] = n[:, 0, 0]
    n_pad[1:-1, 0, -1] = n[:, 0, -1]
    n_pad[1:-1, -1, 0] = n[:, -1, 0]
    n_pad[1:-1, -1, -1] = n[:, -1, -1]

    n_pad[0, 0, 0] = n[0, 0, 0]
    n_pad[0, 0, -1] = n[0, 0, -1]
    n_pad[0, -1, 0] = n[0, -1, 0]
    n_pad[0, -1, -1] = n[0, -1, -1]
    n_pad[-1, 0, 0] = n[-1, 0, 0]
    n_pad[-1, 0, -1] = n[-1, 0, -1]
    n_pad[-1, -1, 0] = n[-1, -1, 0]
    n_pad[-1, -1, -1] = n[-1, -1, -1]


def step_explicit_3d(
    n: np.ndarray,
    D_xy: float,
    D_z: float,
    dx: float,
    dt: float,
    drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_pad: np.ndarray | None = None,
) -> np.ndarray:
    """Single explicit Euler step with anisotropic diffusion and optional drift."""
    if n_pad is None:
        n_pad = np.empty((n.shape[0] + 2, n.shape[1] + 2, n.shape[2] + 2), dtype=n.dtype)
    _fill_neumann_pad(n, n_pad)

    d2x = (n_pad[2:, 1:-1, 1:-1] - 2 * n + n_pad[:-2, 1:-1, 1:-1]) / (dx * dx)
    d2y = (n_pad[1:-1, 2:, 1:-1] - 2 * n + n_pad[1:-1, :-2, 1:-1]) / (dx * dx)
    d2z = (n_pad[1:-1, 1:-1, 2:] - 2 * n + n_pad[1:-1, 1:-1, :-2]) / (dx * dx)

    lap = D_xy * (d2x + d2y) + D_z * d2z

    vx, vy, vz = drift
    if vx != 0.0 or vy != 0.0 or vz != 0.0:
        dn_dx = (n_pad[2:, 1:-1, 1:-1] - n_pad[:-2, 1:-1, 1:-1]) / (2 * dx)
        dn_dy = (n_pad[1:-1, 2:, 1:-1] - n_pad[1:-1, :-2, 1:-1]) / (2 * dx)
        dn_dz = (n_pad[1:-1, 1:-1, 2:] - n_pad[1:-1, 1:-1, :-2]) / (2 * dx)
        lap = lap - (vx * dn_dx + vy * dn_dy + vz * dn_dz)

    return n + dt * lap


def simulate_3d(
    nx: int,
    ny: int,
    nz: int,
    D_xy: float,
    D_z: float,
    t_end: float,
    total_particles: float,
    dx: float = 1.0,
    dt: float | None = None,
    sigma0: float = 1.5,
    drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial: str = "gaussian",
) -> np.ndarray:
    """Simulate 3D diffusion of a source until t_end."""
    if dt is None:
        dt = stable_dt_3d(dx, D_xy, D_z)

    center = (nx // 2, ny // 2, nz // 2)
    if initial == "point":
        n = initial_point_3d((nx, ny, nz), center, total_particles)
    elif initial == "gaussian":
        n = initial_gaussian_3d((nx, ny, nz), center, sigma0, total_particles, dx)
    else:
        raise ValueError("initial must be 'point' or 'gaussian'")

    n_pad = np.empty((nx + 2, ny + 2, nz + 2), dtype=float)

    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        n = step_explicit_3d(n, D_xy, D_z, dx, dt, drift=drift, n_pad=n_pad)
        t += dt

    return n


def line_of_sight_integral(n: np.ndarray, axis: int = 2, dx: float = 1.0) -> np.ndarray:
    """Integrate density along one axis to simulate camera observation."""
    return np.sum(n, axis=axis) * dx


def analytic_los_gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D_xy: float,
    total_particles: float,
) -> np.ndarray:
    """Analytical 2D LOS projection of a 3D point-source diffusion cloud."""
    if t <= 0:
        raise ValueError("t must be positive for analytic solution")
    r2 = x * x + y * y
    return (total_particles / (4 * np.pi * D_xy * t)) * np.exp(-r2 / (4 * D_xy * t))
