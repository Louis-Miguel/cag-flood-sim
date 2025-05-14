import argparse
import os
import sys
import time
import numpy as np
from flood_sim_class import FloodSimulation, run_twophase_river_rain_test
from flood_tools import load_dem_from_file

def flood_simulation_driver(args):
    """
    Loads data, sets up, runs, and visualizes the flood simulation
    based on command-line arguments.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """

    print("--- Starting Flood Simulation Driver ---")

    if not args.dem_path:
        print("ERROR: dem_path argument is required for a standard simulation.")
        print("Use --run-twophase-test for the synthetic river test.")
        sys.exit(1)

    print(f"Loading DEM from: {args.dem_path}")

    # Load DEM 
    try:
        # Pass the explicit nodata value if provided
        dem, valid_mask, metadata = load_dem_from_file(
            args.dem_path,
            preprocess=True, 
            smooth_sigma=args.smooth_sigma,
            fill_sinks_flag=args.fill_sinks,
            no_data_value=args.nodata_value 
        )
        print(f"DEM Loaded. Shape: {dem.shape}. Valid cells: {np.sum(valid_mask)}")
        if dem.size == 0 or np.sum(valid_mask) == 0:
            raise ValueError("Loaded DEM is empty or has no valid cells.")
    except FileNotFoundError:
        print(f"ERROR: DEM file not found at '{args.dem_path}'")
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: Missing dependency for reading DEM file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load or process DEM '{args.dem_path}'. {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not np.any(valid_mask):
        print("ERROR: No valid data cells found in the loaded DEM after processing.")
        sys.exit(1)

    # Determine Grid Spacing (dx, dy)
    dx = args.dx
    dy = args.dy
    if dx is None or dy is None:
        print("Attempting to get dx/dy from metadata...")
        if metadata and 'transform' in metadata and metadata['transform']:
            try:
                # Check for Affine transform object or GDAL list/tuple
                transform = metadata['transform']
                if hasattr(transform, 'a'): # Affine object (rasterio default)
                    dx = abs(transform.a)
                    dy = abs(transform.e)
                elif isinstance(transform, (list, tuple)) and len(transform) >= 6: # GDAL geotransform
                    dx = abs(transform[1])
                    dy = abs(transform[5])
                else:
                    raise ValueError("Unsupported transform format")
                print(f"Using dx={dx:.3f}, dy={dy:.3f} from GeoTIFF metadata.")
            except Exception as e:
                print(f"Warning: Could not parse dx/dy from metadata transform ({metadata.get('transform', 'N/A')}). Error: {e}")
        else:
            print("Warning: dx/dy not provided and no suitable metadata found.")

    # If still None, use a default or raise error
    if dx is None or dy is None:
        if args.dx_default and args.dy_default:
            dx = args.dx_default
            dy = args.dy_default
            print(f"Warning: Using default dx={dx}, dy={dy}. SPECIFY VIA --dx/--dy if incorrect!")
        else:
            print("ERROR: dx and dy could not be determined. Please provide them via --dx and --dy arguments or ensure GeoTIFF has valid metadata.")
            sys.exit(1)
    elif dx <= 0 or dy <= 0:
        print(f"ERROR: Invalid dx ({dx}) or dy ({dy}). Must be positive.")
        sys.exit(1)

    # Convert Units
    infiltration_rate_ms = args.infiltration_rate / (1000.0 * 3600.0) if args.infiltration_rate else 0.0
    rain_rate_ms = args.rain_rate / (1000.0 * 3600.0) if args.rain_rate else 0.0

    # Initialize Simulation
    print("Initializing simulation...")
    try:
        sim = FloodSimulation(
            dem=dem,
            valid_mask=valid_mask, # Pass the valid mask
            dx=dx,
            dy=dy,
            manning=args.manning,
            infiltration_rate=infiltration_rate_ms,
            stability_factor=args.stability_factor,
            adaptive_timestep=not args.fixed_timestep, 
            max_velocity=args.max_velocity,
            min_depth=args.min_depth,
            outlet_threshold_percentile=args.outlet_percentile,
            target_time=args.target_time 
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize FloodSimulation. {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Add Sources/Rainfall 
    # Point Source
    if args.source_row is not None and args.source_col is not None:
        try:
            r, c = args.source_row, args.source_col
            # Check bounds
            if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
                print(f"Warning: Source location ({r}, {c}) is out of DEM bounds ({dem.shape}). Ignoring source.")
            # Check if source is in valid area
            elif not valid_mask[r, c]:
                print(f"Warning: Source location ({r}, {c}) is in a NoData area of the DEM! Ignoring source.")
            else:
                sim.add_water_source(
                    row=r,
                    col=c,
                    rate=args.source_rate,
                    duration_steps=args.source_duration
                )
        except Exception as e:
            print(f"ERROR configuring source: {e}")
            # Continue without source? Or exit? For now, just print error.
            # sys.exit(1)
    elif args.source_row is not None or args.source_col is not None:
        print("Warning: Both --source_row and --source_col must be provided to add a point source. Ignoring source.")

    # Rainfall
    if rain_rate_ms > 0:
        print(f"Adding rainfall rate: {args.rain_rate} mm/hr")
        sim.add_rainfall(
            rate=rain_rate_ms,
            duration_steps=args.rain_duration
        )

    # Run Simulation 
    print(f"\n--- Running Simulation for {args.steps} steps ---")
    start_time = time.time()
    try:
        results, time_steps = sim.run_simulation(
            num_steps=args.steps,
            output_freq=args.output_freq
        )
    except Exception as e:
        print(f"\n--- ERROR occurred during simulation run ---")
        print(e)
        import traceback
        traceback.print_exc()
        # Optionally save partial state here if desired
        sys.exit(1) # Exit if simulation crashes
        
    end_time = time.time()
    print(f"Simulation compute time: {end_time - start_time:.2f} seconds")

    # Visualize and Save Results
    if not results:
        print("Simulation finished, but no results were generated (possibly 0 steps or immediate instability).")
        sys.exit(0) 

    print("\n--- Visualizing and Saving Results ---")
    try:
        # Check if output path is specified for saving
        save_anim = args.output_path is not None
        save_frame = save_anim and args.save_last_frame # Only save frame if anim is saved

        _ = sim.visualize_results(
            results,
            time_steps,
            output_path=args.output_path if save_anim else None,
            show_animation=not args.hide_animation,
            save_last_frame=save_frame,
            hillshade=not args.no_hillshade,
            contour_levels=args.contour_levels,
            show_boundary_mask=args.show_boundary_mask, # Use argument
            steady_state_viz_active=False # Standard run never uses custom viz
        )

    except ImportError as e:
        print(f"\nERROR during visualization/saving: Missing dependency. {e}")
        print("Ensure matplotlib and optionally pillow/ffmpeg are installed.")
    except Exception as e:
        print(f"\nERROR during visualization or saving: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Flood Simulation Driver Finished ---")

# Main function to run the simulation from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flood Simulation.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input/Output Arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("dem_path", type=str, nargs='?', default=None,
                        help="Path to the DEM file (e.g., .tif, .asc, .npy). Required unless a test case is run.")
    io_group.add_argument("-o", "--output-path", type=str, default=None, # Default None -> don't save animation
                        help="Path to save the output animation (e.g., output.gif or output.mp4). If provided, enables saving.")
    io_group.add_argument("--save-last-frame", action='store_true',
                        help="Save the last frame as a PNG image (only if --output-path is specified).")

    # DEM Arguments
    dem_group = parser.add_argument_group('DEM Processing')
    dem_group.add_argument("--dx", type=float, default=None, help="Grid cell size in X direction (meters). Overrides metadata.")
    dem_group.add_argument("--dy", type=float, default=None, help="Grid cell size in Y direction (meters). Overrides metadata.")
    dem_group.add_argument("--dx_default", type=float, default=5.0, help="Default dx if not in metadata or args.") # Provide default
    dem_group.add_argument("--dy_default", type=float, default=5.0, help="Default dy if not in metadata or args.") # Provide default
    dem_group.add_argument("--nodata-value", type=float, default=None, help="Specify the NoData value in the DEM file if non-standard.")
    dem_group.add_argument("--smooth-sigma", type=float, default=0.5, help="Sigma for Gaussian smoothing pre-processing (0 to disable).")
    dem_group.add_argument("--fill-sinks", action='store_true', help="Apply sink filling algorithm pre-processing.")
    dem_group.add_argument("--outlet-percentile", type=float, default=5.0,
                        help="Boundary elevation percentile (0-100) to identify open outlets. 100 = all walls.")

    # Simulation Core Parameters
    sim_core_group = parser.add_argument_group('Simulation Core Parameters')
    sim_core_group.add_argument("--manning", type=float, default=0.04, help="Manning's roughness coefficient.")
    sim_core_group.add_argument("--infiltration-rate", type=float, default=0.0, help="Constant infiltration rate (mm/hour).")
    sim_core_group.add_argument("--stability-factor", type=float, default=0.35, help="CFL factor (0-1) for adaptive timestep.")
    sim_core_group.add_argument("--max-velocity", type=float, default=8.0, help="Max allowed velocity (m/s) for clipping.")
    sim_core_group.add_argument("--min-depth", type=float, default=0.01, help="Min water depth threshold for calculations (meters).")
    sim_core_group.add_argument("--fixed-timestep", action='store_true', help="Disable adaptive timestepping (potentially unstable).")

    # Simulation Run Arguments
    sim_run_group = parser.add_argument_group('Simulation Run Control')
    sim_run_group.add_argument("--steps", type=int, default=1000, help="Number of simulation steps.")
    sim_run_group.add_argument("--output-freq", type=int, default=50, help="Frequency (steps) for saving results for animation.")
    sim_run_group.add_argument("--target-time", type=float, default=3600.0, help="Target simulation time (seconds) for adaptive timestep stop.")

    # Source Arguments
    source_group = parser.add_argument_group('Water Sources')
    source_group.add_argument("--source-row", type=int, default=None, help="Row index (0-based) for point water source.")
    source_group.add_argument("--source-col", type=int, default=None, help="Column index (0-based) for point water source.")
    source_group.add_argument("--source-rate", type=float, default=10.0, help="Inflow rate for point source (m³/s).")
    source_group.add_argument("--source-duration", type=int, default=None, help="Duration (steps) for point source (None=continuous).")

    # Rainfall Arguments
    rain_group = parser.add_argument_group('Rainfall')
    rain_group.add_argument("--rain-rate", type=float, default=0.0, help="Base rainfall rate (mm/hour).")
    rain_group.add_argument("--rain-duration", type=int, default=None, help="Duration (steps) for rainfall (None=continuous).")

    # Visualization Arguments
    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument("--hide-animation", action='store_true', help="Do not display the animation window interactively.")
    viz_group.add_argument("--no-hillshade", action='store_true', help="Do not use hillshading for DEM background.")
    viz_group.add_argument("--contour-levels", type=int, default=10, help="Number of contour lines on DEM (0=disable).")
    viz_group.add_argument("--show-boundary-mask", action='store_true', help="Show the boundary mask plot alongside the simulation.")


    # Two-Phase Test Arguments
    twophase_group = parser.add_argument_group('Two-Phase River Test (Synthetic DEM)')
    twophase_group.add_argument("--run-twophase-test", action='store_true',
                                help="Run two-phase river (steady flow) then rainfall test using a synthetic DEM.")
    twophase_group.add_argument("--phase1-steps", type=int, default=4000, help="Steps for phase 1 (river flow establish).") # Adjusted default
    twophase_group.add_argument("--phase2-steps", type=int, default=3000, help="Steps for phase 2 (e.g., rainfall).") # Adjusted default
    # Parameters for create_natural_river_dem
    twophase_group.add_argument("--rows", type=int, default=100, help="DEM rows for two-phase test.")
    twophase_group.add_argument("--cols", type=int, default=300, help="DEM columns for two-phase test.") # Made slightly shorter
    twophase_group.add_argument("--base-elev", type=float, default=20.0, help="Base elevation for synthetic DEM.")
    twophase_group.add_argument("--main-slope", type=float, default=0.005, help="Main channel slope for synthetic DEM.") # Slightly steeper
    twophase_group.add_argument("--cross-slope", type=float, default=0.001, help="Cross-valley slope for synthetic DEM.")
    twophase_group.add_argument("--channel-width", type=float, default=12.0, help="Channel width (meters) for synthetic DEM.")
    twophase_group.add_argument("--channel-depth", type=float, default=2.5, help="Channel depth (meters) for synthetic DEM.")
    twophase_group.add_argument("--meander-amplitude", type=float, default=15.0, help="Meander amplitude (rows) for synthetic DEM.")
    twophase_group.add_argument("--meander-freq", type=float, default=2.0, help="Meander frequency (cycles per DEM length) for synthetic DEM.")
    twophase_group.add_argument("--noise-sigma", type=float, default=2.0, help="Sigma for noise generation smoothing.")
    twophase_group.add_argument("--noise-strength", type=float, default=0.4, help="Strength of random noise added.")
    twophase_group.add_argument("--final-smooth-sigma", type=float, default=1.0, help="Sigma for final DEM smoothing.")

    # Parse Arguments 
    args = parser.parse_args()

    # Set dx/dy defaults if not provided 
    if args.dx is None: 
        args.dx = args.dx_default
    if args.dy is None: 
        args.dy = args.dy_default

    # Execute appropriate function 
    if args.run_twophase_test:
        # Ensure output path is set if not provided by user for the test
        if args.output_path is None:
            args.output_path = "two_phase_river_test_output.gif"
            print(f"Output path not specified for test, defaulting to: {args.output_path}")
        # Run the two-phase test function (now defined in flood_sim_class.py)
        run_twophase_river_rain_test(args)
    else:
        # Run the standard simulation driver
        flood_simulation_driver(args)