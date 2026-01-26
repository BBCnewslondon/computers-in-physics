from __future__ import annotations

import numpy as np


def initial_gaussian_1d(x: np.ndarray, x0: float, sigma: float, total_particles: float) -> np.ndarray:
    """Return a normalized 1D Gaussian with total integral equal to total_particles."""
    prefactor = total_particles / (np.sqrt(2 * np.pi) * sigma)
    return prefactor * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def analytic_point_source_1d(x: np.ndarray, t: float, D: float, total_particles: float) -> np.ndarray:
    """Analytical 1D diffusion solution for a point source at the origin."""
    if t <= 0:
        raise ValueError("t must be positive for analytic solution")
    return (total_particles / np.sqrt(4 * np.pi * D * t)) * np.exp(-(x**2) / (4 * D * t))


def stable_dt_1d(dx: float, D: float, safety: float = 0.45) -> float:
    """Return a stable time step for explicit 1D diffusion."""
    if D <= 0:
        raise ValueError("D must be positive")
    return safety * dx * dx / (2 * D)


def step_explicit_1d(n: np.ndarray, D: float, dx: float, dt: float) -> np.ndarray:
    """Single explicit Euler step with zero-flux boundaries."""
    n_pad = np.pad(n, 1, mode="edge")
    lap = (n_pad[2:] - 2 * n_pad[1:-1] + n_pad[:-2]) / (dx * dx)
    return n + D * dt * lap


def simulate_1d(
    x_min: float,
    x_max: float,
    nx: int,
    D: float,
    t_end: float,
    total_particles: float,
    x0: float = 0.0,
    sigma0: float = 0.1,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 1D diffusion of a Gaussian bump until t_end."""
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    if dt is None:
        dt = stable_dt_1d(dx, D)

    n = initial_gaussian_1d(x, x0, sigma0, total_particles)
    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        n = step_explicit_1d(n, D, dx, dt)
        t += dt

    return x, n
