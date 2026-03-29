# Diffusion Equation Model

This project simulates diffusion processes using explicit finite-difference methods in one and three dimensions. It is designed for educational and research purposes in computational physics, particularly for modeling particle diffusion in atmospheric or plasma environments.

## Features

- **1D Diffusion Simulation**: Compares numerical solutions against analytical Gaussian profiles for point-source diffusion.
- **3D Diffusion Simulation**: Supports isotropic and anisotropic diffusion coefficients ($D_{xy}$, $D_z$), with optional drift terms ($\mathbf{v} \cdot \nabla n$). Includes line-of-sight (LOS) integration for 2D projections.
- **Initial Conditions**: Gaussian, point source, and uniform sphere.
- **Analytical Comparisons**: Includes analytical solutions for 1D point sources, 3D Gaussian LOS projections, and sphere density profiles.
- **Brightness Integral**: Numerical computation of brightness profiles using Bessel functions for sphere diffusion.
- **Sensitivity Analysis**: Monte Carlo perturbations on diffusion coefficients to assess variability.
- **Parameter Studies**: Altitude-dependent diffusion coefficients (e.g., for Ba at 150, 300, and 600 km).
- **Mass Conservation**: Tracks total mass over time, including with drift and outflow boundaries.
- **Visualization**: Generates isosurfaces, 2D images, and plots using Matplotlib and scikit-image.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- scikit-image
- SciPy

Install dependencies:

```bash
pip install -r [`requirements.txt`](requirements.txt )
```

## Usage

Run the main script to execute all demos and simulations:

```bash
python [`main.py`](main.py )
```

This will generate outputs in the `outputs/` directory, including plots and data arrays.

### Key Functions

- [`simulate_1d`](src/diffusion_1d.py): 1D diffusion simulation.
- [`simulate_3d`](src/diffusion_3d.py): 3D diffusion with anisotropy and drift.
- [`brightness_bessel`](scripts/brightness_integral.py): Brightness calculation for sphere diffusion.

## Outputs

Running the script saves the following files under `outputs/`:

- `diffusion_1d.png`, `diffusion_1d_x.npy`, `diffusion_1d_numerical.npy`, `diffusion_1d_analytical.npy`: 1D comparison plots and data.
- `diffusion_3d_isosurface.png`, `diffusion_3d_los_compare.png`: 3D visualizations.
- `project_c2_radial_density.png`, `brightness_verification.png`: Sphere and brightness demos.
- `diffusion_3d_los_alt_150km.png`, `diffusion_3d_los_alt_300km.png`, `diffusion_3d_los_alt_600km.png`: Altitude parameter studies.
- `diffusion_3d_los_alt_150_300_600km_subplot.png`, `diffusion_3d_altitude_comparison.png`: Shared-axis subplot comparison across altitudes.
- `sensitivity_1d.png`: Sensitivity analysis plot.
- `diffusion_3d_mass.png`, `diffusion_3d_mass_time.npy`, `diffusion_3d_mass.npy`: Mass conservation data.

## Notes

- The explicit finite-difference method is conditionally stable; the code automatically selects stable time steps based on diffusion constraints.
- For 3D simulations, Neumann (zero-flux) boundaries are used by default, with outflow boundaries for drift terms.
- Analytical solutions are provided for validation and comparison.

## Contributing

Ensure changes maintain numerical stability and include appropriate tests. Update this README for new features.
