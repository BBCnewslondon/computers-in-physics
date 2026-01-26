from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import iv

from src.diffusion_1d import analytic_point_source_1d, simulate_1d, simulate_1d_center_series, stable_dt_1d
from src.diffusion_3d import analytic_sphere_density, line_of_sight_integral, simulate_3d


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


def run_project_c2_simulation(out_dir: Path) -> None:
    R = 4.0
    dx = 0.5
    D = 1.0
    nx, ny, nz = 31, 31, 31

    dt = (dx**2) / (6 * D)
    t_end = 2.0

    n_numerical = simulate_3d(
        nx=nx,
        ny=ny,
        nz=nz,
        D_xy=D,
        D_z=D,
        t_end=t_end,
        total_particles=1.0,
        dx=dx,
        dt=dt,
        initial="sphere",
        radius=R,
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


def run_brightness_verification(out_dir: Path) -> None:
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


def run_parameter_study(out_dir: Path) -> None:
    total_particles = 1.0
    t_end = 0.5
    dx = 1.0
    sigma0 = 1.5
    nx, ny, nz = 41, 41, 41
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))

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


def run_sensitivity_analysis(out_dir: Path) -> None:
    D_base = 0.5
    total_particles = 1.0
    t_end = 0.5
    x_min, x_max = -5.0, 5.0
    nx = 401
    sigma0 = 0.2

    dx = (x_max - x_min) / (nx - 1)
    dt_fixed = stable_dt_1d(dx, D_base)

    rng = np.random.default_rng(42)
    n_runs = 5
    perturb = 0.05

    series = []
    times_ref = None
    for _ in range(n_runs):
        factor = 1.0 + rng.uniform(-perturb, perturb)
        D = D_base * factor
        times, values = simulate_1d_center_series(
            x_min=x_min,
            x_max=x_max,
            nx=nx,
            D=D,
            t_end=t_end,
            total_particles=total_particles,
            x0=0.0,
            sigma0=sigma0,
            dt=dt_fixed,
        )
        if times_ref is None:
            times_ref = times
        series.append(values)

    data = np.vstack(series)
    mean = data.mean(axis=0)
    low = data.min(axis=0)
    high = data.max(axis=0)

    np.save(out_dir / "sensitivity_time.npy", times_ref)
    np.save(out_dir / "sensitivity_center_mean.npy", mean)
    np.save(out_dir / "sensitivity_center_low.npy", low)
    np.save(out_dir / "sensitivity_center_high.npy", high)
    np.save(out_dir / "sensitivity_center_ensemble.npy", data)

    plt.figure()
    plt.plot(times_ref, mean, label="Mean")
    plt.fill_between(times_ref, low, high, alpha=0.3, label="±5% D range")
    plt.xlabel("time")
    plt.ylabel("center density")
    plt.title("Sensitivity Analysis (1D, D ±5%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_1d.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    run_1d_demo(output_path)
    run_project_c2_simulation(output_path)
    run_brightness_verification(output_path)
    run_parameter_study(output_path)
    run_sensitivity_analysis(output_path)
