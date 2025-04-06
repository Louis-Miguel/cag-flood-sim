import argparse
import os
import sys
import time
import numpy as np
from flood_sim_class import FloodSimulation
from flood_tools import load_dem_from_file

def flood_simulation_driver(args):
    """
    Loads data, sets up, runs, and visualizes the flood simulation
    based on command-line arguments.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """

    print("--- Starting Flood Simulation Driver ---")
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
    except FileNotFoundError:
        print(f"ERROR: DEM file not found at '{args.dem_path}'")
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: Missing dependency for reading DEM file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load or process DEM '{args.dem_path}'. {e}")
        sys.exit(1)

    if not np.any(valid_mask):
        print("ERROR: No valid data cells found in the loaded DEM after processing.")
        sys.exit(1)

    # Determine Grid Spacing (dx, dy)
    dx = args.dx
    dy = args.dy
    if dx is None or dy is None:
        print("Attempting to get dx/dy from metadata...")
        if metadata and 'transform' in metadata:
            try:
                # Check for Affine transform object or GDAL list/tuple
                if hasattr(metadata['transform'], 'a'): # Affine object
                    dx = abs(metadata['transform'].a)
                    dy = abs(metadata['transform'].e)
                else: 
                    dx = abs(metadata['transform'][1])
                    dy = abs(metadata['transform'][5])
                print(f"Using dx={dx:.3f}, dy={dy:.3f} from GeoTIFF metadata.")
            except Exception as e:
                print(f"Warning: Could not parse dx/dy from metadata transform ({metadata.get('transform', 'N/A')}). Error: {e}")
        else:
            print("Warning: dx/dy not provided and no metadata found.")

    # If still None, use a default or raise error
    if dx is None or dy is None:
        if args.dx_default and args.dy_default:
            dx = args.dx_default
            dy = args.dy_default
            print(f"Warning: Using default dx={dx}, dy={dy}. SPECIFY VIA --dx/--dy if incorrect!")
        else:
            print("ERROR: dx and dy could not be determined. Please provide them via --dx and --dy arguments.")
            sys.exit(1)
    elif dx <= 0 or dy <= 0:
        print(f"ERROR: Invalid dx ({dx}) or dy ({dy}). Must be positive.")
        sys.exit(1)

    #  Convert Units 
    # Infiltration rate from mm/hr to m/s
    infiltration_rate_ms = args.infiltration_rate / (1000.0 * 3600.0) if args.infiltration_rate else 0.0
    # Rainfall rate from mm/hr to m/s
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
            adaptive_timestep=not args.fixed_timestep, # Use adaptive unless fixed is flagged
            max_velocity=args.max_velocity,
            min_depth=args.min_depth,
            outlet_threshold_percentile=args.outlet_percentile,
            target_time=args.target_time # For adaptive timestep
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize FloodSimulation. {e}")
        sys.exit(1)

    # Add Sources/Rainfall 
    # Point Source
    if args.source_row is not None and args.source_col is not None:
        try:
            # Check for bounds 
            if not (0 <= args.source_row < dem.shape[0] and 0 <= args.source_col < dem.shape[1]):
                print(f"Warning: Source location ({args.source_row}, {args.source_col}) might be out of DEM bounds ({dem.shape}).")
            # Check if source is in valid area
            if not valid_mask[args.source_row, args.source_col]:
                print(f"Warning: Source location ({args.source_row}, {args.source_col}) is in a NoData area of the DEM!")

            sim.add_water_source(
                row=args.source_row,
                col=args.source_col,
                rate=args.source_rate,
                duration_steps=args.source_duration # Pass None if not specified
            )
        except IndexError as e:
            print(f"ERROR adding source: {e}. Check row/col indices.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR configuring source: {e}")
            sys.exit(1)
    elif args.source_row is not None or args.source_col is not None:
        print("Warning: Both --source_row and --source_col must be provided to add a point source. Ignoring source.")

    # Rainfall
    if rain_rate_ms > 0:
        print(f"Adding rainfall rate: {args.rain_rate_mmhr} mm/hr")
        sim.add_rainfall(
            rate=rain_rate_ms,
            duration_steps=args.rain_duration # Pass None if not specified
            # Add spatial/temporal patterns here if implemented and controllable via args
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
        # Optionally save partial state here if desired
        sys.exit(1) 
        
    end_time = time.time()
    print(f"Simulation compute time: {end_time - start_time:.2f} seconds")

    # Visualize and Save Results
    if not results:
        print("Simulation finished, but no results were generated (possibly 0 steps or immediate instability).")
        sys.exit(0)

    print("\n--- Visualizing and Saving Results ---")
    try:
        # Derive final frame path from output path
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            base_name = os.path.basename(args.output_path)
            name_part, _ = os.path.splitext(base_name)
            final_frame_path = os.path.join(output_dir, f"{name_part}_final_frame.png")
        else:
            # Don't save if animation path not given
            final_frame_path = None 

        _ = sim.visualize_results(
            results,
            time_steps,
            output_path=args.output_path, # Pass the animation path (e.g., .gif, .mp4)
            show_animation=not args.hide_animation,
            save_last_frame= (final_frame_path is not None), # Only save if path makes sense
            hillshade=not args.no_hillshade,
            contour_levels=args.contour_levels
        )

    except ImportError as e:
        print(f"\nERROR during visualization/saving: Missing dependency. {e}")
        print("Ensure matplotlib and optionally pillow/ffmpeg are installed.")
    except Exception as e:
        print(f"\nERROR during visualization or saving: {e}")
    print("\n--- Flood Simulation Driver Finished ---")

# Main function to run the simulation from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flood Simulation on a DEM file.")

    # Input/Output Arguments 
    parser.add_argument("dem_path", type=str, help="Path to the Digital Elevation Model file (e.g., .tif, .npy, .asc).")
    parser.add_argument("-o", "--output-path", type=str, default="flood_simulation_output.gif",
                        help="Path to save the output animation (e.g., output.gif or output.mp4). Extension determines format.")

    # DEM Arguments 
    parser.add_argument("--dx", type=float, default=None, help="Grid cell size in X direction (meters). Overrides metadata if provided.")
    parser.add_argument("--dy", type=float, default=None, help="Grid cell size in Y direction (meters). Overrides metadata if provided.")
    parser.add_argument("--dx_default", type=float, default=None, help="Default dx if not in metadata or args.")
    parser.add_argument("--dy_default", type=float, default=None, help="Default dy if not in metadata or args.")
    parser.add_argument("--nodata-value", type=float, default=None, help="Specify the NoData value used in the DEM file if not standard.")
    parser.add_argument("--smooth-sigma", type=float, default=0.5, help="Sigma for Gaussian smoothing during DEM preprocessing (0 to disable).")
    parser.add_argument("--fill-sinks", action='store_true', help="Apply sink filling algorithm during DEM preprocessing.")
    parser.add_argument("--outlet-percentile", type=float, default=5.0,
                        help="Elevation percentile threshold (0-100) on boundary cells to identify open outlets. 100 disables outlets.")

    #  Simulation Core Parameters 
    parser.add_argument("--manning", type=float, default=0.04, help="Manning's roughness coefficient.")
    parser.add_argument("--infiltration-rate", type=float, default=0.0, help="Constant infiltration rate (mm/hour).")
    parser.add_argument("--stability-factor", type=float, default=0.35, help="CFL stability factor (0 to 1) for adaptive timestep.")
    parser.add_argument("--max-velocity", type=float, default=8.0, help="Maximum allowed velocity magnitude (m/s) for clipping.")
    parser.add_argument("--min-depth", type=float, default=0.01, help="Minimum water depth threshold for friction/velocity (meters).")
    parser.add_argument("--fixed-timestep", action='store_true', help="Disable adaptive timestepping (use initial guess - potentially unstable).")

    # Simulation Run Arguments 
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run.")
    parser.add_argument("--output-freq", type=int, default=10, help="Frequency (in steps) for saving results for animation.")
    parser.add_argument("--target-time", type=float, default=3600.0, help="Target simulation time in seconds (for adaptive timestep).")

    # Source Arguments 
    parser.add_argument("--source-row", type=int, default=None, help="Row index (0-based) for point water source.")
    parser.add_argument("--source-col", type=int, default=None, help="Column index (0-based) for point water source.")
    parser.add_argument("--source-rate", type=float, default=10.0, help="Inflow rate for point source (m^3/s).")
    parser.add_argument("--source-duration", type=int, default=None, help="Duration (in simulation steps) for point source (None for continuous).")

    # Rainfall Arguments
    parser.add_argument("--rain-rate", type=float, default=0.0, help="Base rainfall rate (mm/hour).")
    parser.add_argument("--rain-duration", type=int, default=None, help="Duration (in simulation steps) for rainfall (None for continuous).")

    # Visualization Arguments 
    parser.add_argument("--hide-animation", action='store_true', help="Do not display the animation window interactively.")
    parser.add_argument("--no-hillshade", action='store_true', help="Do not use hillshading for DEM background.")
    parser.add_argument("--contour-levels", type=int, default=10, help="Number of contour lines to draw on DEM background (0 to disable).")

    # Parse Arguments 
    args = parser.parse_args()

    # Run the driver function 
    flood_simulation_driver(args)