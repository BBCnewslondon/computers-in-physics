from __future__ import annotations

import numpy as np


def stable_dt_3d(dx: float, D: float, safety: float = 0.2) -> float:
    """Return a stable time step for explicit 3D diffusion."""
    if D <= 0:
        raise ValueError("D must be positive")
    return safety * dx * dx / (6 * D)


def initial_point_3d(shape: tuple[int, int, int], center: tuple[int, int, int], total_particles: float) -> np.ndarray:
    """Place all particles at a single grid cell (discrete point source)."""
    n = np.zeros(shape, dtype=float)
    n[center] = total_particles
    return n


def step_explicit_3d(n: np.ndarray, D: float, dx: float, dt: float) -> np.ndarray:
    """Single explicit Euler step with zero-flux boundaries using edge padding."""
    n_pad = np.pad(n, 1, mode="edge")
    lap = (
        n_pad[2:, 1:-1, 1:-1]
        + n_pad[:-2, 1:-1, 1:-1]
        + n_pad[1:-1, 2:, 1:-1]
        + n_pad[1:-1, :-2, 1:-1]
        + n_pad[1:-1, 1:-1, 2:]
        + n_pad[1:-1, 1:-1, :-2]
        - 6.0 * n
    ) / (dx * dx)
    return n + D * dt * lap


def simulate_3d(
    nx: int,
    ny: int,
    nz: int,
    D: float,
    t_end: float,
    total_particles: float,
    dx: float = 1.0,
    dt: float | None = None,
) -> np.ndarray:
    """Simulate 3D diffusion of a point source until t_end."""
    if dt is None:
        dt = stable_dt_3d(dx, D)

    center = (nx // 2, ny // 2, nz // 2)
    n = initial_point_3d((nx, ny, nz), center, total_particles)

    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        n = step_explicit_3d(n, D, dx, dt)
        t += dt

    return n


def line_of_sight_integral(n: np.ndarray, axis: int = 2, dx: float = 1.0) -> np.ndarray:
    """Integrate density along one axis to simulate camera observation."""
    return np.sum(n, axis=axis) * dx
