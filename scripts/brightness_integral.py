from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import quad
from scipy.special import iv


def brightness_bessel(rho: float, R: float, D: float, t: float) -> float:
    """Compute B(rho) using numerical integration over theta."""
    if D <= 0 or t <= 0 or R <= 0:
        raise ValueError("R, D, t must be positive")

    prefactor = (R**3) / (D * t) * np.exp(-(rho**2) / (4 * D * t))

    def integrand(theta: float) -> float:
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        exp_term = np.exp(-(R**2) * (sin_t**2) / (4 * D * t))
        arg = (R * rho * sin_t) / (2 * D * t)
        return exp_term * iv(0, arg) * sin_t * (cos_t**2)

    integral, _ = quad(integrand, 0.0, np.pi / 2, limit=200)
    return prefactor * integral


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically integrate B(rho) with I0.")
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.2)
    parser.add_argument("--t", type=float, nargs="+", default=[0.5, 2.0, 5.0])
    parser.add_argument("--rho", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 3.0])
    args = parser.parse_args()

    for t in args.t:
        for r in args.rho:
            val = brightness_bessel(r, args.R, args.D, t)
            print(f"t={t:.4f} rho={r:.4f} -> B={val:.6e}")


if __name__ == "__main__":
    main()
