# Cagayan Valley 2D Shallow Water Flood Simulation

## Overview

This repository contains a Python-based 2D flood simulation model. It simulates the propagation of floodwater over a given Digital Elevation Model (DEM) of the Cagayan Valley by solving the Shallow Water Equations (SWE). The model incorporates Numba JIT compilation for significant performance enhancement of the core numerical computations.

The simulation handles irregular domain boundaries defined by NoData values (NaNs) in the input DEM and includes features for preprocessing GIS data, setting boundary conditions, adding water sources, and visualizing results.

## Features

*   **2D Shallow Water Equations Solver:** Core hydrodynamics based on SWE.
*   **Numerical Scheme:**
    *   Runge-Kutta 2 (Heun's method) for time integration.
    *   Upwind scheme for advection terms.
    *   Finite difference method on a regular grid.
*   **Performance:** Core computational step (`compute_step`) optimized using Numba's Just-In-Time (JIT) compilation.
*   **Physics:**
    *   Manning's equation for bottom friction (spatially constant for now).
    *   Optional constant infiltration rate.
*   **Timestepping:** Adaptive timestepping based on the Courant-Friedrichs-Lewy (CFL) condition for stability.
*   **Boundary Conditions:**
    *   Handles irregular domains defined by NoData/NaN values in the input DEM.
    *   Automatically generates boundary mask (`flood_tools.generate_boundary_mask`):
        *   Identifies internal domain vs. outside NoData areas.
        *   Defines boundary cells along the valid data edge.
        *   Distinguishes between closed 'Wall' boundaries and potential 'Open Outlet' boundaries based on elevation percentile.
    *   Applies appropriate boundary conditions (zero-gradient or velocity reflection) using a ghost cell approach within the `compute_step` function.
*   **Input Sources:**
    *   Supports adding one or more point water sources with specified rates and durations.
    *   Supports adding spatially distributed rainfall with specified rates and durations (simple spatial/temporal patterns included).
*   **GIS Tools (`flood_tools.py`):**
    *   Loads DEMs from common formats (GeoTIFF, NumPy `.npy`, ASCII `.asc`/`.txt`) using `rasterio` (recommended) or fallback methods.
    *   Handles NoData value identification.
    *   Optional preprocessing: Gaussian smoothing, efficient Priority-Flood sink filling.
    *   Extracts basic metadata (CRS, transform) from GeoTIFFs.
*   **Driver Script (`flood_simulation_driver.py`):**
    *   Provides a command-line interface (CLI) for easy configuration and execution.
    *   Parses arguments for file paths, simulation parameters, source/rain inputs, and visualization options.
*   **Visualization:**
    *   Generates animations (GIF/MP4) of flood depth propagation over the terrain.
    *   Optionally displays terrain with hillshading and contour lines.
    *   Handles plotting for irregular domains, masking invalid areas.
    *   Saves the final frame of the simulation as a PNG image.

## Requirements

*   **Python:** 3.8 or higher recommended.
*   **Core Libraries:**
    *   `numpy`: For numerical array operations.
    *   `scipy`: For Gaussian filter and potentially interpolation.
    *   `matplotlib`: For plotting and animation.
    *   `numba`: For JIT compilation and performance.
*   **Optional Libraries:**
    *   `rasterio`: **Highly recommended** for robust GeoTIFF I/O and metadata handling. Install via conda or pip (may require GDAL).
    *   `pillow`: Required for saving GIF animations.
    *   `ffmpeg`: Required for saving MP4 animations (system installation usually needed).

You can install the required libraries using pip:
```bash
pip install numpy scipy matplotlib numba
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Louis-Miguel/cag-flood-sim
    cd cag-flood-sim
    ```

2.  **Install dependencies:** (Create a `requirements.txt` file first)
    ```bash
    pip install -r requirements.txt
    ```

**`requirements.txt` example:**
```
numpy>=1.18
scipy>=1.5
matplotlib>=3.3
numba>=0.50
# Optional but recommended
rasterio>=1.2
pillow>=8.0
```

## Usage

The primary way to run the simulation is through the command-line driver script `flood_simulation_driver.py`.

**Basic Syntax:**

```bash
python flood_simulation_driver.py <path_to_dem> [options]
```

**Get Help:**

To see all available command-line options and their descriptions:
```bash
python flood_simulation_driver.py -h
```

**Examples:**

1.  **Run simulation on a GeoTIFF, specify grid size, add rain, save as MP4:**
    ```bash
    python flood_simulation_driver.py ./data/my_dem.tif --dx 10 --dy 10 --rain-rate 40 --rain-duration 500 --manning 0.045 --steps 1000 -o output/flood_rain.mp4
    ```
## File Structure

```
.
├── flood_sim_class.py          # Contains the main FloodSimulation class
├── compute_flood_step.py       # Contains the Numba-optimized core compute_step function
├── flood_tools.py              # Utility functions for DEM loading, preprocessing, boundary masks
├── flood_simulation_driver.py  # Command-line interface script to run simulations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Algorithm Overview
*   **Governing Equations:** 2D Shallow Water Equations (SWE).
*   **Spatial Discretization:** Finite Difference Method on a regular grid.
*   **Time Integration:** Runge-Kutta 2nd Order (Heun's Method).
*   **Advection:** First-order Upwind Scheme.
*   **Boundary Conditions:** Ghost cell approach informed by a pre-calculated boundary mask (`-1`: Outside, `0`: Internal, `1`: Wall, `2`: Open Outlet). Wall BCs use velocity reflection; Open BCs use zero-gradient.
*   **Sink Filling:** Priority-Flood algorithm (`flood_tools.fill_sinks_priority_flood`).
*   **Optimization:** Numba JIT compilation (`@numba.jit(nopython=True)`).
