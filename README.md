# Diffusion Equation Model

This project simulates diffusion using an explicit finite-difference method. It includes:

- 1D diffusion with a numerical solution compared against the analytical Gaussian solution.
- 3D diffusion with line-of-sight integration to simulate a camera observation.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

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
- diffusion_3d_los.png
- diffusion_3d_los.npy

## Notes

The explicit method is conditionally stable. The code selects a stable time step using standard diffusion constraints.
