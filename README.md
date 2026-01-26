# Diffusion Equation Model

This project simulates diffusion using an explicit finite-difference method. It includes:

- 1D diffusion with a numerical solution compared against the analytical Gaussian solution.
- 3D diffusion with Gaussian initialization, anisotropic diffusion ($D_{xy}$, $D_z$), and line-of-sight integration.
- Optional drift support ($\mathbf{v} \cdot \nabla n$) in the 3D solver with outflow boundaries.
- Parameter study hooks for 150–250 km diffusion coefficients.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- scikit-image

Install dependencies:

- pip install -r requirements.txt

## Run

- python main.py

## Outputs

Running the script saves figures and arrays under the outputs/ folder:

- diffusion_1d.png
- diffusion_1d_x.npy
- diffusion_1d_numerical.npy
- diffusion_1d_analytical.npy
- diffusion_3d_los_compare.png
- diffusion_3d_los_analytic.png
- diffusion_3d_los_error.png
- diffusion_3d_los_isotropic.npy
- diffusion_3d_los_anisotropic.npy
- diffusion_3d_los_analytic.npy
- diffusion_3d_los_error_isotropic.npy
- diffusion_3d_los_error_anisotropic.npy
- diffusion_3d_los_alt_150km.png
- diffusion_3d_los_alt_200km.png
- diffusion_3d_los_alt_250km.png
- diffusion_3d_los_alt_150km.npy
- diffusion_3d_los_alt_200km.npy
- diffusion_3d_los_alt_250km.npy
- diffusion_3d_mass.png
- diffusion_3d_mass_time.npy
- diffusion_3d_mass.npy
- diffusion_3d_isosurface.png
- sensitivity_1d.png
- sensitivity_time.npy
- sensitivity_center_mean.npy
- sensitivity_center_low.npy
- sensitivity_center_high.npy
- sensitivity_center_ensemble.npy

## Notes

The explicit method is conditionally stable. The code selects a stable time step using standard diffusion constraints.
