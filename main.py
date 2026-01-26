from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.diffusion_1d import analytic_point_source_1d, simulate_1d
from src.diffusion_3d import line_of_sight_integral, simulate_3d


def run_1d_demo(output_dir: Path) -> None:
    D = 0.5
    total_particles = 1.0
    t_end = 0.5
    x_min, x_max = -5.0, 5.0
    nx = 401

    x, n_num = simulate_1d(
        x_min=x_min,
        x_max=x_max,
        nx=nx,
        D=D,
        t_end=t_end,
        total_particles=total_particles,
        x0=0.0,
        sigma0=0.2,
    )
    n_exact = analytic_point_source_1d(x, t_end, D, total_particles)

    plt.figure()
    plt.plot(x, n_num, label="Numerical")
    plt.plot(x, n_exact, "--", label="Analytical")
    plt.xlabel("x")
    plt.ylabel("n(x, t)")
    plt.title("1D Diffusion: Numerical vs Analytical")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "diffusion_1d.png", dpi=150)
    np.save(output_dir / "diffusion_1d_x.npy", x)
    np.save(output_dir / "diffusion_1d_numerical.npy", n_num)
    np.save(output_dir / "diffusion_1d_analytical.npy", n_exact)
    plt.show()


def run_3d_demo(output_dir: Path) -> None:
    D = 0.2
    total_particles = 1.0
    t_end = 0.5

    nx, ny, nz = 41, 41, 41
    n = simulate_3d(nx=nx, ny=ny, nz=nz, D=D, t_end=t_end, total_particles=total_particles, dx=1.0)
    image = line_of_sight_integral(n, axis=2, dx=1.0)

    plt.figure()
    plt.imshow(image, origin="lower", cmap="viridis")
    plt.colorbar(label="Integrated density")
    plt.title("Line-of-Sight Integrated Density (z-axis)")
    plt.tight_layout()
    plt.savefig(output_dir / "diffusion_3d_los.png", dpi=150)
    np.save(output_dir / "diffusion_3d_los.npy", image)
    plt.show()


if __name__ == "__main__":
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    run_1d_demo(output_dir)
    run_3d_demo(output_dir)
