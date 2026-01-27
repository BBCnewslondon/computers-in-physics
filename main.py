from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from scipy.integrate import quad
from scipy.special import iv
from src.diffusion_1d import analytic_point_source_1d, simulate_1d, simulate_1d_center_series, stable_dt_1d
from src.diffusion_3d import (
    analytic_los_gaussian_2d,
    analytic_sphere_density,
    line_of_sight_integral,
    simulate_3d
)

def run_1d_demo(out_dir: Path) -> None:
    """Run 1D diffusion demo comparing numerical vs analytical solutions."""
    D = 0.5
    total_particles = 1.0
    t_end = 0.5
    x_min, x_max = -5.0, 5.0
    nx = 401

    x, n_num = simulate_1d(
        x_min=x_min, x_max=x_max, nx=nx, D=D, t_end=t_end,
        total_particles=total_particles, x0=0.0, sigma0=0.2,
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
    # Saving data for potential later plotting
    np.save(out_dir / "diffusion_1d_x.npy", x)
    np.save(out_dir / "diffusion_1d_numerical.npy", n_num)
    np.save(out_dir / "diffusion_1d_analytical.npy", n_exact)
    plt.show()

def run_3d_visualization_demo(out_dir: Path) -> None:
    """
    From 'main' branch:
    Visualizes 3D Isosurfaces and compares Isotropic vs Anisotropic diffusion.
    """
    total_particles = 1.0
    t_end = 5.0
    dx = 1.0
    sigma0 = 1.5
    nx, ny, nz = 31, 31, 81

    # Case 1: Isotropic
    D_xy_iso, D_z_iso = 0.2, 0.2
    n_iso = cast(np.ndarray, simulate_3d(
        nx=nx, ny=ny, nz=nz, D_xy=D_xy_iso, D_z=D_z_iso, t_end=t_end,
        total_particles=total_particles, dx=dx, sigma0=sigma0, initial="gaussian",
    ))
    image_iso = line_of_sight_integral(n_iso, axis=2, dx=dx)

    # Case 2: Anisotropic
    D_xy_aniso, D_z_aniso = 0.2, 1.0
    n_aniso = cast(np.ndarray, simulate_3d(
        nx=nx, ny=ny, nz=nz, D_xy=D_xy_aniso, D_z=D_z_aniso, t_end=t_end,
        total_particles=total_particles, dx=dx, sigma0=sigma0, initial="gaussian",
    ))
    image_aniso = line_of_sight_integral(n_aniso, axis=2, dx=dx)

    # 3D isosurface visualization 
    iso_level = float(n_aniso.max() * 0.2)
    verts, faces, _, _ = marching_cubes(n_aniso, level=iso_level, spacing=(dx, dx, dx))
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    linewidth=0.1, alpha=0.8, color="steelblue")
    ax.set_title("3D Isosurface (Anisotropic)", pad=12)
    ax.set_box_aspect((nx, ny, nz))
    plt.savefig(out_dir / "diffusion_3d_isosurface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Error analysis vs Analytical Gaussian
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    analytic = analytic_los_gaussian_2d(X, Y, t_end, D_xy_iso, total_particles)

    err_iso = image_iso - analytic
    
    # Plot comparisons
    vmin = min(image_iso.min(), image_aniso.min())
    vmax = max(image_iso.max(), image_aniso.max())
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_iso, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    plt.title("Isotropic LOS")
    plt.subplot(1, 2, 2)
    plt.imshow(image_aniso, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    plt.title("Anisotropic LOS")
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_3d_los_compare.png", dpi=150)
    plt.close()

def run_project_c2_simulation(out_dir: Path) -> None:
    """
    From 'Alternative' branch:
    Simulates the specific 'Sphere' initial condition requested in Project C2.
    """
    R = 4.0
    dx = 0.5
    D = 1.0
    nx, ny, nz = 31, 31, 31
    dt = (dx**2) / (6 * D)
    t_end = 2.0

    # Note: explicit initial=sphere
    n_numerical = simulate_3d(
        nx=nx, ny=ny, nz=nz, D_xy=D, D_z=D, t_end=t_end,
        total_particles=1.0, dx=dx, dt=dt, initial="sphere", radius=R,
    )

    center_idx = nx // 2
    c_radial_numerical = n_numerical[:, ny // 2, nz // 2]
    r_axis = np.abs((np.arange(nx) - center_idx) * dx)

    c_radial_analytic = analytic_sphere_density(r_axis, t_end, D, R)

    plt.figure()
    plt.plot(r_axis, c_radial_numerical, "o", label="Numerical (Finite Diff)")
    plt.plot(r_axis, c_radial_analytic, "-", label="Analytical (Theory)")
    plt.xlabel("Radius r")
    plt.ylabel("Density C")
    plt.title(f"Radial Density Profile (t={t_end})")
    plt.legend()
    plt.savefig(out_dir / "project_c2_radial_density.png")
    plt.show()

def brightness_bessel(rho: float, R: float, D: float, t: float) -> float:
    """Compute B(rho) using Bessel function integral (Project C2 Page 3)."""
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

def run_brightness_verification(out_dir: Path) -> None:
    """
    From 'Alternative' branch:
    Verifies the brightness integral which uses Bessel functions.
    """
    R = 4.0
    D = 1.0
    t = 2.0

    rho = np.linspace(0.0, 2.5 * R, 80)
    brightness = np.array([brightness_bessel(r, R, D, t) for r in rho])

    plt.figure()
    plt.plot(rho, brightness, "-", label="Brightness integral")
    plt.xlabel("Impact parameter $\\rho$")
    plt.ylabel("Brightness B($\\rho$)")
    plt.title(f"Brightness Verification (R={R}, D={D}, t={t})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "brightness_verification.png", dpi=150)
    plt.show()
def run_sensitivity_analysis(out_dir: Path) -> None:
    """
    From 'main' branch:
    Monte Carlo sensitivity analysis (perturbing D by +/- 5%).
    """
    D_base = 0.5
    total_particles = 1.0
    t_end = 0.5
    x_min, x_max = -5.0, 5.0
    nx = 401
    sigma0 = 0.2
    
    # Re-calculate stable dt
    dx = (x_max - x_min) / (nx - 1)
    dt_fixed = stable_dt_1d(dx, D_base)

    rng = np.random.default_rng(42)
    n_runs = 5
    perturb = 0.2

    series = []
    times_ref = None
    
    print(f"Running Sensitivity Analysis ({n_runs} runs)...")
    for _ in range(n_runs):
        factor = 1.0 + rng.uniform(-perturb, perturb)
        D = D_base * factor
        # Note: We use the 1D simulation for speed here
        times, values = simulate_1d_center_series(
            x_min=x_min, x_max=x_max, nx=nx, D=D, t_end=t_end,
            total_particles=total_particles, x0=0.0, sigma0=sigma0, dt=dt_fixed,
        )
        if times_ref is None:
            times_ref = times
        series.append(values)

    data = np.vstack(series)
    mean = data.mean(axis=0)
    low = data.min(axis=0)
    high = data.max(axis=0)

    plt.figure()
    plt.plot(times_ref, mean, label="Mean")
    plt.fill_between(times_ref, low, high, alpha=0.3, label="±20% D range")
    plt.xlabel("time")
    plt.ylabel("center density")
    plt.title("Sensitivity Analysis (1D, D ±20%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_1d.png", dpi=150)
    plt.close()
    
def run_parameter_study(out_dir: Path) -> None:
    """Run parameter study for different altitudes (used in both branches)."""
    total_particles = 1.0
    t_end = 0.5
    dx = 1.0
    sigma0 = 1.5
    nx, ny, nz = 41, 41, 41
    
    # Measured diffusion coefficients for Ba at 150-250 km
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

        n = cast(np.ndarray, simulate_3d(
            nx=nx, ny=ny, nz=nz, D_xy=D_xy, D_z=D_z, t_end=t_end,
            total_particles=total_particles, dx=dx, sigma0=sigma0, 
            initial="gaussian", drift=drift,
        ))
        image = line_of_sight_integral(n, axis=2, dx=dx)
        
        plt.figure()
        plt.imshow(image, origin="lower", cmap="viridis")
        plt.colorbar(label="Integrated density")
        plt.title(f"LOS Density (Alt {alt_km} km)")
        plt.tight_layout()
        plt.savefig(out_dir / f"diffusion_3d_los_alt_{alt_km}km.png", dpi=150)
        plt.close()
        
def run_mass_conservation_check(out_dir: Path) -> None:
    """
    From 'main' branch:
    Checks if total mass is conserved during a simulation with drift.
    """
    nx, ny, nz = 31, 31, 81
    dx = 1.0
    sigma0 = 1.5
    t_end = 5.0
    total_particles = 1.0
    
    # Drift velocity (vx, vy, vz)
    drift = (0.25, 0.0, 0.0)
    
    # Run simulation with mass tracking enabled
    # Note: We use D=0.2 (isotropic) for this test
    _, times, masses = simulate_3d(
        nx=nx, ny=ny, nz=nz,
        D_xy=0.2, D_z=0.2,
        t_end=t_end, total_particles=total_particles,
        dx=dx, sigma0=sigma0,
        initial="gaussian",
        drift=drift,
        track_mass=True,
        mass_interval=1,
    )

    np.save(out_dir / "diffusion_3d_mass_time.npy", times)
    np.save(out_dir / "diffusion_3d_mass.npy", masses)

    plt.figure()
    plt.plot(times, masses)
    plt.xlabel("time")
    plt.ylabel("total mass")
    plt.title("Mass vs Time (Drift with Outflow)")
    plt.ylim(0.95 * total_particles, 1.05 * total_particles)  # Zoom in to see errors
    plt.tight_layout()
    plt.savefig(out_dir / "diffusion_3d_mass.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    
    print("Running 1D Demo...")
    run_1d_demo(output_path)
    
    print("Running 3D Visualization Demo (Isosurfaces)...")
    run_3d_visualization_demo(output_path)
    
    print("Running Project C2 Simulation (Sphere Initial Condition)...")
    run_project_c2_simulation(output_path)
    
    print("Running Brightness Verification (Bessel)...")
    run_brightness_verification(output_path)
    
    print("Running Parameter Study...")
    run_parameter_study(output_path)
    
    print("Done! Check 'outputs/' directory.")
    run_sensitivity_analysis(output_path)
    run_mass_conservation_check(output_path)
    
    