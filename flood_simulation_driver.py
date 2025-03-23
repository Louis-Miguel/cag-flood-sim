import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource
import os
import time

# Import your flood tools functions
from flood_tools import fill_sinks, load_dem_from_file
from flood_sim_class import FloodSimulation

def run_simulation_on_dem(dem_path=None, dem_array=None, 
                          water_source_location=None, 
                          simulation_steps=200,
                          dx=10, dy=10, manning=0.03, 
                          water_rate=50, water_duration=50,
                          output_freq=5, 
                          output_folder="flood_results", 
                          show_animation=True,
                          adaptive_timestep=True, 
                          max_velocity=10.0, 
                          stability_factor=0.25):
    """
    Run a flood simulation on a Digital Elevation Model (DEM) with optional water source.
    
    Parameters:
    -----------
    dem_path : str or None
        Path to the DEM file. If None, dem_array must be provided
    dem_array : numpy.ndarray or None
        DEM as a numpy array. If None, dem_path must be provided
    water_source_location : tuple or None
        (row, col) location for water source. If None, uses rainfall
    simulation_steps : int
        Number of simulation steps to run
    dx, dy : float
        Grid cell size in x and y directions (meters)
    manning : float
        Manning's roughness coefficient
    water_rate : float
        Water inflow rate (m³/s) or rainfall intensity (m/s)
    water_duration : int or None
        Duration of source in simulation steps (None for unlimited)
    output_freq : int
        Frequency of storing results for visualization
    output_folder : str
        Folder to save all output files
    show_animation : bool
        Whether to display the animation
    adaptive_timestep : bool
        Whether to use adaptive timestep to maintain stability
    max_velocity : float
        Maximum allowed velocity (m/s) for stability control
    stability_factor : float
        Factor to reduce timestep (0-1) for stability control
        
    Returns:
    --------
    sim : FloodSimulation
        The simulation object
    results : list
        List of water depth arrays at specified steps
    time_steps : list
        List of time values corresponding to results
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load DEM if path is provided
    if dem_path is not None:
        dem = load_dem_from_file(dem_path, preprocess=True)
    elif dem_array is not None:
        dem = dem_array
    else:
        raise ValueError("Either dem_path or dem_array must be provided")
    
    # Preprocess DEM to fill sinks
    print("Preprocessing DEM to fill sinks...")
    dem_processed = fill_sinks(dem)
    
    # Create simulation object
    print("Initializing flood simulation...")
    sim = FloodSimulation(dem_processed, dx=dx, dy=dy, manning=manning, 
                          adaptive_timestep=adaptive_timestep,
                          stability_factor=stability_factor,
                          max_velocity=max_velocity)
    
    # Set up water source or rainfall pattern
    if water_source_location is not None:
        row, col = water_source_location
        print(f"Adding water source at ({row}, {col}) with rate {water_rate} m³/s")
        sim.add_water_source(row, col, water_rate, water_duration)
    else:
        # Default to simple rainfall
        print(f"Adding default uniform rainfall with rate {water_rate} m/s")
        sim.add_rainfall(water_rate, water_duration)
    
    # Record start time
    start_time = time.time()
    
    # Run simulation
    print(f"Running simulation for {simulation_steps} steps...")
    results, time_steps = sim.run_simulation(simulation_steps, output_freq)
    
    # Report simulation time
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    
    # Visualize results
    animation_path = os.path.join(output_folder, "flood_animation.gif")
    print(f"Creating flood animation at {animation_path}")
    sim.visualize_results(results, time_steps, 
                          output_path=animation_path, 
                          show_animation=show_animation)
    
    print("Flood simulation and analysis complete!")
    return sim, results, time_steps


# Example usage as a script
if __name__ == '__main__':
    print("Running flood simulation...")
    
    sim2, results2, time_steps2 = run_simulation_on_dem(
        dem_path="output_NASADEM.tif",
        simulation_steps=10000,
        dx=1,
        dy=1,
        manning=0.04,
        water_rate=0.005,  # m/s of rainfall (about 360 mm/hour)
        water_duration=None,
        output_freq=100,
        output_folder="flood_results_rainfall",
        show_animation=True
    )
    
    print("Simulation complete!")
