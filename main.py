from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.diffusion_1d import analytic_point_source_1d, simulate_1d
from src.diffusion_3d import analytic_los_gaussian_2d, line_of_sight_integral, simulate_3d


def run_1d_demo(out_dir: Path) -> None:
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
    plt.xlabel("x (km)")
    plt.ylabel("n(x, t)")
    plt.title("1D Diffusion: Numerical vs Analytical")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_1d.png", dpi=150)
    np.save(out_dir / "diffusion_1d_x.npy", x)
    np.save(out_dir / "diffusion_1d_numerical.npy", n_num)
    np.save(out_dir / "diffusion_1d_analytical.npy", n_exact)
    plt.show()


def run_3d_demo(out_dir: Path) -> None:
    total_particles = 1.0
    t_end = 0.5
    dx = 1.0
    sigma0 = 1.5

    nx, ny, nz = 41, 41, 41

    D_xy_iso, D_z_iso = 0.2, 0.2
    n_iso = simulate_3d(
        nx=nx,
        ny=ny,
        nz=nz,
        D_xy=D_xy_iso,
        D_z=D_z_iso,
        t_end=t_end,
        total_particles=total_particles,
        dx=dx,
        sigma0=sigma0,
        initial="gaussian",
    )
    image_iso = line_of_sight_integral(n_iso, axis=2, dx=dx)

    D_xy_aniso, D_z_aniso = 0.2, 0.05
    n_aniso = simulate_3d(
        nx=nx,
        ny=ny,
        nz=nz,
        D_xy=D_xy_aniso,
        D_z=D_z_aniso,
        t_end=t_end,
        total_particles=total_particles,
        dx=dx,
        sigma0=sigma0,
        initial="gaussian",
    )
    image_aniso = line_of_sight_integral(n_aniso, axis=2, dx=dx)

    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    analytic = analytic_los_gaussian_2d(X, Y, t_end, D_xy_iso, total_particles)

    err_iso = image_iso - analytic
    err_aniso = image_aniso - analytic

    np.save(out_dir / "diffusion_3d_los_isotropic.npy", image_iso)
    np.save(out_dir / "diffusion_3d_los_anisotropic.npy", image_aniso)
    np.save(out_dir / "diffusion_3d_los_analytic.npy", analytic)
    np.save(out_dir / "diffusion_3d_los_error_isotropic.npy", err_iso)
    np.save(out_dir / "diffusion_3d_los_error_anisotropic.npy", err_aniso)

    vmin = min(image_iso.min(), image_aniso.min())
    vmax = max(image_iso.max(), image_aniso.max())

    extent = [x.min(), x.max(), y.min(), y.max()]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_iso, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    plt.title("Isotropic LOS")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.subplot(1, 2, 2)
    plt.imshow(image_aniso, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    plt.title("Anisotropic LOS")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_3d_los_compare.png", dpi=150)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(err_iso, origin="lower", cmap="coolwarm", extent=extent)
    plt.title("Isotropic Error")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.subplot(1, 2, 2)
    plt.imshow(err_aniso, origin="lower", cmap="coolwarm", extent=extent)
    plt.title("Anisotropic Error")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_3d_los_error.png", dpi=150)

    plt.figure()
    plt.imshow(analytic, origin="lower", cmap="viridis", extent=extent)
    plt.colorbar(label="Integrated density")
    plt.title("Analytical LOS (2D Gaussian)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_3d_los_analytic.png", dpi=150)
    plt.show()


def run_parameter_study(out_dir: Path) -> None:
    total_particles = 1.0
    t_end = 0.5
    dx = 1.0
    sigma0 = 1.5
    nx, ny, nz = 41, 41, 41

    # NOTE: Replace these placeholder values with measured/derived diffusion
    # coefficients for Ba at 150-250 km. Units should match the model (dx, t).
    altitude_cases = [
        {"alt_km": 150, "D_xy": 0.15, "D_z": 0.05, "drift": (0.0, 0.0, 0.0)},
        {"alt_km": 200, "D_xy": 0.22, "D_z": 0.08, "drift": (0.0, 0.0, 0.0)},
        {"alt_km": 250, "D_xy": 0.30, "D_z": 0.12, "drift": (0.0, 0.0, 0.0)},
    ]

    for case in altitude_cases:
        alt_km = case["alt_km"]
        D_xy = case["D_xy"]
        D_z = case["D_z"]
        drift = case["drift"]

        n = simulate_3d(
            nx=nx,
            ny=ny,
            nz=nz,
            D_xy=D_xy,
            D_z=D_z,
            t_end=t_end,
            total_particles=total_particles,
            dx=dx,
            sigma0=sigma0,
            initial="gaussian",
            drift=drift,
        )
        image = line_of_sight_integral(n, axis=2, dx=dx)

        np.save(out_dir / f"diffusion_3d_los_alt_{alt_km}km.npy", image)

        plt.figure()
        plt.imshow(image, origin="lower", cmap="viridis", extent=extent)
        plt.colorbar(label="Integrated density")
        plt.title(f"LOS Density (Alt {alt_km} km)")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.tight_layout()
        plt.savefig(out_dir / f"diffusion_3d_los_alt_{alt_km}km.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    run_1d_demo(output_path)
    run_3d_demo(output_path)
    run_parameter_study(output_path)
