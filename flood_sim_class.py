import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors as mcolors, gridspec
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource, Normalize
from flood_tools import generate_boundary_mask, create_natural_river_dem
from compute_flood_step import compute_step

class FloodSimulation:
    def __init__(self, dem, valid_mask, 
                dx=1.0, dy=1.0, g=9.81, 
                manning=0.03, infiltration_rate=0.0, adaptive_timestep=True,
                stability_factor=0.4, max_velocity=10.0, min_depth=0.001,
                outlet_threshold_percentile=5.0, target_time=3600.0):
        """
        Initialize the flood simulation using the DEM data 

        Parameters:
        -----------
        dem : numpy.ndarray : Digital Elevation Model (2D array of elevation values)
        valid_mask : numpy.ndarray : Mask indicating valid areas for simulation 
        dx, dy : float : Grid cell size in x and y directions
        g : float : Acceleration due to gravity
        manning : float : Manning's roughness coefficient
        infiltration_rate : float : Rate of infiltration (m/s)
        adaptive_timestep : bool : Whether to use adaptive timestep to maintain stability
        stability_factor : float : Factor to reduce timestep (0-1) for stability control 
        max_velocity : float : Maximum allowed velocity (m/s) for stability control 
        outlet_threshold_percentile : float : Percentile for outlet detection (0-100)
        target_time : float : Target simulation time (seconds)
        """

        # Store DEM and the valid mask
        self.dem = dem.astype(np.float64)
        self.valid_mask = valid_mask.astype(bool)

        #Initialize parameters
        self.dx = float(dx)
        self.dy = float(dy)
        self.g = float(g)
        self.manning = float(manning)
        self.infiltration_rate = float(infiltration_rate)
        self.adaptive_timestep = adaptive_timestep
        self.stability_factor = float(stability_factor)
        self.max_velocity = float(max_velocity)
        self.min_depth = float(min_depth)
        self.target_time = float(target_time)
        self.steady_state_h = None

        # Generate boundary mask based on DEM and valid_mask
        self.boundary_mask = generate_boundary_mask(self.dem, self.valid_mask, outlet_threshold_percentile)

        # Initialize SWE variable arrays 
        self.h = np.zeros_like(self.dem, dtype=np.float64)
        self.u = np.zeros_like(self.dem, dtype=np.float64)
        self.v = np.zeros_like(self.dem, dtype=np.float64)
        self.nx, self.ny = self.dem.shape

        # Zeroing the water depth in invalid areas since the velecities are already zeroed
        self.h[~self.valid_mask] = 0.0

        # Initial timestep guess (use stats only from valid areas)
        if np.any(self.valid_mask):
            dem_range = np.nanmax(dem[self.valid_mask]) - np.nanmin(dem[self.valid_mask])
            max_h_guess = max(1.0, dem_range if dem_range > 0 else 1.0)
        else:
            max_h_guess = 1.0 # Default if no valid cells

        # Compute timestep based on grid size (CFL step initialization)
        sqrt_term = np.sqrt(g * max_h_guess + 1e-9) # Add epsilon
        if sqrt_term < 1e-9:
             self.dt = 0.1 * min(dx, dy) # Fallback if wave speed is near zero
        else:
             self.dt = 0.1 * min(dx, dy) / sqrt_term
        self.original_dt_guess = self.dt

        # For debugging purposes, print the initial dt guess
        print(f"Initial dt guess: {self.dt:.4f} s")

        self.sources = []
        self.rainfall = None
        self.current_step = 0

        self.max_flood_extent = np.zeros_like(self.dem, dtype=bool)
        self.max_flood_depth = np.zeros_like(self.dem)
        
        # Ensure max trackers not set outside valid area
        self.max_flood_depth[~self.valid_mask] = np.nan

    def add_water_source(self, row, col, rate, duration_steps=None):
        """
        Add a water source at a specific location

        Parameters:
        -----------
        row : int : Row index of the source
        col : int : Column index of the source
        rate : float : Rate of water source (m³/s)
        duration_steps : int or None : Duration of source in simulation steps (None for unlimited)
        """
        
        self.sources.append({'row': row, 'col': col, 'rate': rate, 'duration': duration_steps, 'steps_active': 0})
        print(f"Added source at ({row}, {col}), rate={rate} m³/s, duration={duration_steps} steps")
        
    def add_rainfall(self, rate, duration_steps=None, spatial_distribution=None, time_pattern=None):
        """
        Add rainfall with spatial and temporal distribution
        
        Parameters:
        -----------
        rate : float : Rainfall rate (m/s)
        duration_steps : int or None : Duration of rainfall in simulation steps (None for unlimited)
        spatial_distribution : numpy.ndarray or None : Spatial distribution of rainfall (2D array)
        time_pattern : function or None : Temporal pattern function (step, duration) -> intensity
        """
        # Spatial distribution 
        if spatial_distribution is None: 
            # Instead of uniform rainfall, create a smoother pattern
            dist = np.ones_like(self.dem)
            # Apply slight randomness to break uniformity
            noise = np.random.normal(1.0, 0.1, self.dem.shape)
            dist *= noise
            # Smooth the distribution
            dist = gaussian_filter(dist, sigma=2.0)
            # Normalize
            if max_dist > 1e-9:
                dist /= max_dist
            else:
                dist = np.ones_like(self.dem) 
        else:
            # Ensure correct shape and smooth input distribution
            assert spatial_distribution.shape == self.dem.shape, "Spatial distribution must match DEM shape"
            dist = np.maximum(0, spatial_distribution)
            # Smooth the distribution
            dist = gaussian_filter(dist, sigma=2.0)
            # Normalize
            if np.max(dist) > 1e-9:
                dist /= np.max(dist)
            else:
                dist = np.zeros_like(self.dem)

        # Temporal pattern
        if time_pattern is None: 
            pattern = lambda step, duration: 1.0
        else: 
            pattern = time_pattern

        self.rainfall = {'rate': rate, 'duration': duration_steps, 'steps_active': 0, 'distribution': dist, 'time_pattern': pattern}
        print(f"Added rainfall: rate={rate} m/s, duration={duration_steps} steps")

    def apply_sources_or_rainfall(self, h_current):
        """
        Apply water sources or rainfall to the current state
        """
        h_new = h_current.copy()
        # Apply water sources
        for source in self.sources:
            if source['duration'] is None or source['steps_active'] < source['duration']:
                volume_per_step = source['rate'] * self.dt
                depth_increase = volume_per_step / (self.dx * self.dy)
                if 0 <= source['row'] < self.nx and 0 <= source['col'] < self.ny:
                    if self.valid_mask[source['row'], source['col']]:
                        h_new[source['row'], source['col']] = max(0.0, h_new[source['row'], source['col']] + depth_increase)
                source['steps_active'] += 1
        # Apply rainfall
        if self.rainfall:
            if self.rainfall['duration'] is None or self.rainfall['steps_active'] < self.rainfall['duration']:
                time_factor = self.rainfall['time_pattern'](self.rainfall['steps_active'], self.rainfall['duration'])
                effective_rate = self.rainfall['rate'] * time_factor
                rainfall_depth = effective_rate * self.dt * self.rainfall['distribution']
                h_new += np.maximum(0.0, rainfall_depth) 
                self.rainfall['steps_active'] += 1
        return h_new

    def compute_fluxes_and_update(self):
        """
        Compute fluxes and update state variables
        """
        if self.adaptive_timestep:
            # Calculate stats only on valid cells 
            h_valid = self.h[self.valid_mask]
            u_valid = self.u[self.valid_mask]
            v_valid = self.v[self.valid_mask]

            with np.errstate(invalid='ignore', divide='ignore'):
                # Use valid subsets for max calculations
                max_vel_sq = np.max(u_valid**2 + v_valid**2) if u_valid.size > 0 else 0.0
                max_vel = np.sqrt(max_vel_sq) if max_vel_sq > 0 else 0.0
                max_depth = np.max(h_valid) if h_valid.size > 0 else 0.0

            if np.isnan(max_vel): 
                max_vel = 0.0
            if np.isnan(max_depth): 
                max_depth = 0.0

            # Adjust timestep based on CFL condition (Courant-Friedrichs-Lewy (CFL) condition)
            wave_speed = np.sqrt(self.g * max(max_depth, self.min_depth))
            denominator = wave_speed + max_vel + 1e-9 
            dt_cfl = self.stability_factor * min(self.dx, self.dy) / denominator
            
            # Constraint: Don't increase dt too rapidly from initial guess or previous step
            max_dt_allowed = max(self.original_dt_guess, self.dt * 1.5)

            self.dt = min(dt_cfl, max_dt_allowed)

            self.dt = np.clip(self.dt, 1e-9, self.original_dt_guess * 10)

            if self.dt < 1e-9: 
                print("Warning: Timestep extremely small...") 
                self.dt = 1e-9

        # Call numba function
        h_new, u_new, v_new = compute_step(
            self.h, self.u, self.v, self.dem, self.boundary_mask, # Pass mask
            self.g, self.manning, self.dx, self.dy, self.dt,
            self.infiltration_rate, self.min_depth, self.max_velocity
        )

        # # Post-computation checks/updates (apply only to valid area)

        # # Velocity Clipping
        # vel_new_sq = u_new**2 + v_new**2
        # scale = np.ones_like(vel_new_sq)
        
        # mask_to_scale = self.valid_mask & (vel_new_sq > self.max_velocity**2)
        # if np.any(mask_to_scale):
        #     # Add epsilon to prevent division by zero if vel_new_sq is exactly max_velocity**2
        #     scale[mask_to_scale] = self.max_velocity / (np.sqrt(vel_new_sq[mask_to_scale]) + 1e-9)

        #     u_new[mask_to_scale] *= scale[mask_to_scale]
        #     v_new[mask_to_scale] *= scale[mask_to_scale]

        # Apply Sources/Rainfall
        h_new_after_sources = self.apply_sources_or_rainfall(h_new)

        # Update State Variables
        self.h = h_new_after_sources 
        self.u = u_new
        self.v = v_new

        # Ensure state variables are zero outside the valid mask
        self.h[~self.valid_mask] = 0.0 
        self.u[~self.valid_mask] = 0.0 
        self.v[~self.valid_mask] = 0.0 

        # Ensure velocities are zero where depth is below min_depth within the valid mask
        zero_vel_mask = self.valid_mask & (self.h < self.min_depth)
        self.u[zero_vel_mask] = 0.0
        self.v[zero_vel_mask] = 0.0

        # Update Max Flood Trackers
        is_flooded = self.valid_mask & (self.h > self.min_depth)
        self.max_flood_extent = np.logical_or(self.max_flood_extent, is_flooded)

        # Initialize max_flood_depth with NaN outside valid area
        if not hasattr(self, 'max_flood_depth_initialized'):
            self.max_flood_depth = np.full_like(self.dem, np.nan)
            self.max_flood_depth[self.valid_mask] = 0.0 # Init valid area to 0
            self.max_flood_depth_initialized = True

        # Update max depth only where flooded
        self.max_flood_depth = np.fmax(self.max_flood_depth, np.where(is_flooded, self.h, np.nan))
        # Ensure max depth outside valid area remains NaN 
        self.max_flood_depth[~self.valid_mask] = np.nan


    def run_simulation(self, num_steps, output_freq=10):
        """
        Run the flood simulation for a specified number of steps
        """
        # Initialize results
        results = []
        time_steps_list = []
        total_time = 0.0
        start_run_time = time.time()

        initial_sim_time_offset = 0.0 

        # Run simulation loop
        for step in range(num_steps):
            self.current_step = step

            # Compute fluxes and update state variables and check for errors
            try: 
                self.compute_fluxes_and_update()
            except Exception as e: 
                print(f"\n--- ERROR occurred at step {step}, time {total_time:.2f}s ---\n{e}") 
                break 

            current_dt = self.dt # Get timestep used for this step
            total_time += current_dt

            # Display progress and show results at the end
            if step % output_freq == 0 or step == num_steps - 1:
                results.append(self.h.copy())
                time_steps_list.append(total_time)

                h_valid = self.h[self.valid_mask]
                u_valid = self.u[self.valid_mask]
                v_valid = self.v[self.valid_mask]
                # Compute max depth and velocity
                with np.errstate(invalid='ignore'): 
                    max_depth = np.nanmax(h_valid) if h_valid.size > 0 else 0.0
                    max_vel_sq = np.nanmax(u_valid**2 + v_valid**2) if u_valid.size > 0 else 0.0
                    max_vel = np.sqrt(max_vel_sq) if max_vel_sq > 0 else 0.0
                
                # Handle NaN values
                max_depth = 0.0 if np.isnan(max_depth) else max_depth
                max_vel = 0.0 if np.isnan(max_vel) else max_vel

                # Compute flood area and volume (within valid mask)
                flooded_cells_mask = self.valid_mask & (self.h > self.min_depth)
                flood_area = np.sum(flooded_cells_mask) * self.dx * self.dy
                # Sum only positive depths within the valid mask
                valid_positive_h = self.h[self.valid_mask & (self.h > 0)]
                flood_volume = np.sum(valid_positive_h) * self.dx * self.dy
                
                # Print progress and stop if unstable
                print(
                    f"Step {step}/{num_steps - 1}, "
                    f"Sim Time: {total_time + initial_sim_time_offset:.3f}s (+{current_dt:.3e}s), "
                    f"Max Depth: {max_depth:.10f}m, "
                    f"Max Vel: {max_vel:.10f}m/s, "
                    f"Area: {flood_area:.1f}m², "
                    f"Vol: {flood_volume:.1f}m³"
                )
                # Check for instability (very large depth or non-finite values)
                if not np.isfinite(max_depth) or max_depth > 1000: # Increased threshold
                    print("\n--- WARNING: Simulation appears unstable (large/infinite depth). Stopping. ---")
                    # Add last valid state before potential crash if needed
                    if len(results) > 1 and step % output_freq != 0: # Avoid duplicate if last step saved
                        results.append(results[-1]) # Append previous state as best guess
                        time_steps_list.append(time_steps_list[-1])
                    break

                # Check if target simulation time is reached (relative to this run's start)
                if total_time >= self.target_time:
                    print(f" Simulation time reached target time {self.target_time}. Stopping.")
                    # Ensure the very last state is saved if stopping due to time
                    if step % output_freq != 0:
                        results.append(self.h.copy())
                        time_steps_list.append(total_time + initial_sim_time_offset)
                    break


        end_run_time = time.time()

        print(f"\nSimulation finished {step+1} steps in {end_run_time - start_run_time:.2f} seconds (wall clock).")
        final_flood_area = np.sum(self.max_flood_extent) * self.dx * self.dy

        max_recorded_depth = np.nanmax(self.max_flood_depth)
        max_recorded_depth = 0.0 if np.isnan(max_recorded_depth) else max_recorded_depth 

        print(f"Final Maximum Flood Extent Area: {final_flood_area:.2f} m²")
        print(f"Peak Recorded Water Depth Anywhere: {max_recorded_depth:.3f} m")

        return results, time_steps_list

    def visualize_results(self, results, time_steps, output_path=None,show_animation=True, save_last_frame=True,
                            hillshade=True, contour_levels=10,
                            show_boundary_mask=True, steady_state_viz_active=False):
        """
        Visualize the simulation results with enhanced terrain visualization.
        Handles potential issues with empty or NaN results and large DEMs.

        Parameters:
        -----------
        results : list : List of simulation results (2D arrays)
        time_steps : list : List of time steps corresponding to results
        output_path : str : Path to save the animation (optional)
        show_animation : bool : Whether to display the animation
        save_last_frame : bool : Whether to save the last frame as an image
        hillshade : bool : Whether to include hillshade effect on terrain
        contour_levels : int : Number of contour levels to display
        steady_state_viz_active : bool : If True and self.steady_state_h exists, use custom colormap for river + overflow.
        """

        if not results:
            print("No results to visualize.")
            return None

        num_subplots = 2 if show_boundary_mask else 1
        fig = plt.figure(figsize=(8 * num_subplots, 8)) # Adjust figsize

        # Use GridSpec for better layout control
        if show_boundary_mask:
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) # 1 row, 2 columns
            ax_sim = plt.subplot(gs[0]) # Simulation plot axes
            ax_mask = plt.subplot(gs[1]) # Boundary mask axes
        else:
            gs = gridspec.GridSpec(1, 1)
            ax_sim = plt.subplot(gs[0]) # Only simulation plot axes
            ax_mask = None # No mask axes needed

        # Plot Boundary Mask
        if ax_mask:
            print("Plotting Boundary Mask...")
            ax_mask.set_title("Boundary Mask (-1:Out, 0:In, 1:Wall, 2:Open)")
            # Define colors and normalization for the mask codes
            cmap_mask = mcolors.ListedColormap(['lightgrey', 'white', 'black', 'blue'])
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5] # Define bin edges for discrete values
            norm_mask = mcolors.BoundaryNorm(bounds, cmap_mask.N)

            mask_plot = ax_mask.imshow(self.boundary_mask, cmap=cmap_mask, norm=norm_mask,
                                    interpolation='none', origin='upper',
                                    extent=(0, self.ny*self.dx, 0, self.nx*self.dy))

            # Create a colorbar for the boundary mask
            cbar_mask = plt.colorbar(mask_plot, ax=ax_mask, shrink=0.7, ticks=[-1, 0, 1, 2])
            cbar_mask.set_ticklabels(['Outside', 'Internal', 'Wall', 'Outlet'])
            ax_mask.set_xlabel("X distance (m)")
            ax_mask.set_ylabel("Y distance (m)")
            print("Boundary Mask plotted.")


        # Simulation Plot Setup (on ax_sim) 
        ax_sim.set_title(f'Flood Simulation (Frame 0)') # Initial title

        # Determine Color Limits for Water Depth (Robust calculation)
        valid_maxes = []
        print("Calculating max water depth for color bar...")
        for idx, h_frame in enumerate(results):
            if h_frame is not None and h_frame.size > 0 and self.valid_mask.shape == h_frame.shape:
                # Only consider valid cells for max depth calculation
                h_frame_valid = h_frame[self.valid_mask]
                if h_frame_valid.size > 0:
                    try:
                        max_val = np.nanmax(h_frame_valid)
                        if np.isfinite(max_val):
                                valid_maxes.append(max_val)
                    except ValueError: # Should not happen if h_frame_valid.size > 0
                        continue

        if valid_maxes:
            overall_vmax = max(valid_maxes)
        else:
            overall_vmax = self.min_depth * 5 # Default if no water found
        overall_vmax = max(overall_vmax, self.min_depth * 1.1)

        if np.isnan(overall_vmax): 
            overall_vmax = 1.0 
        
        print(f"Overall water depth color bar max (vmax): {overall_vmax:.3f}")

        # Prepare DEM for Display
        dem_display = self.dem.copy()
        dem_display[~self.valid_mask] = np.nan

        print("Plotting DEM background on simulation axes...")
        if hillshade:
            try:
                ls = LightSource(azdeg=315, altdeg=45)
                dem_min_valid = np.nanmin(dem_display)
                dem_max_valid = np.nanmax(dem_display)
                
                if np.isnan(dem_min_valid) or np.isnan(dem_max_valid) or dem_max_valid <= dem_min_valid:
                    # Handle cases with no valid DEM data or flat DEM
                    print("Warning: DEM data is missing, flat, or invalid. Cannot generate hillshade accurately.")
                    # Create a dummy shaded background (e.g., grey)
                    rgb = np.full((self.nx, self.ny, 3), 0.7) # Grey background
                    hillshade = True 
                
                else:
                    # Normalize valid DEM data for shading
                    dem_norm_val = (dem_display - dem_min_valid) / (dem_max_valid - dem_min_valid)
                    # Fill NaN values *in the normalized array* for shading function
                    # Use a neutral value (0.5) for NaNs
                    dem_for_shade = np.nan_to_num(dem_norm_val, nan=0.5)

                    print("Performing hillshading calculation...")
                    rgb = ls.shade(dem_for_shade, cmap=plt.cm.terrain, vert_exag=1.0, blend_mode='soft')
                    print("Hillshading calculation complete.")



                # Convert shaded RGB to RGBA uint8 for imshow, applying transparency mask
                if rgb.ndim == 3 and rgb.shape[2] >= 3:
                    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
                    rgba[:, :, :3] = np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8)
                    rgba[:, :, 3] = 255 # Start fully opaque
                    # Apply transparency where the original DEM was invalid (NaN)
                    rgba[~self.valid_mask, 3] = 0
                else: # Fallback if shade output is unexpected
                    print("Warning: Hillshade output unexpected format. Falling back to non-hillshade.")
                    hillshade = False # Force non-hillshade path below

                if hillshade:
                    dem_plot = ax_sim.imshow(rgba, extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                            origin='upper', interpolation='nearest')

            except Exception as e_shade:
                print(f"ERROR during hillshading: {e_shade}. Falling back.")
                hillshade = False

        if not hillshade: # If hillshade failed or was disabled
            terrain_cmap = plt.cm.terrain.copy()
            terrain_cmap.set_bad(color='white', alpha=0) # Make NaNs transparent
            terrain_vmin = np.nanmin(dem_display)
            terrain_vmax = np.nanmax(dem_display)
            # Handle case where DEM is flat or all NaN
            if np.isnan(terrain_vmin) or np.isnan(terrain_vmax) or terrain_vmax <= terrain_vmin:
                terrain_vmin, terrain_vmax = 0, 1 # Default range
            dem_plot = ax_sim.imshow(dem_display, cmap=terrain_cmap,
                                    vmin=terrain_vmin, vmax=terrain_vmax,
                                    extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                    origin='upper', interpolation='nearest')

        print("DEM background plotted.")

        # Add Contour Lines on ax_sim 
        if contour_levels > 0:
            print(f"Adding {contour_levels} contour lines...")
            try:
                # Use nanmin/max for contour levels to avoid issues with NaNs
                cont_min = np.nanmin(dem_display)
                cont_max = np.nanmax(dem_display)
                if not (np.isnan(cont_min) or np.isnan(cont_max) or cont_max <= cont_min):
                    levels = np.linspace(cont_min, cont_max, contour_levels)
                    # Contour requires finite values, fill NaNs temporarily
                    dem_for_contour = np.nan_to_num(dem_display, nan=cont_min) # Fill with min
                    ax_sim.contour(dem_for_contour, colors='black', alpha=0.3, levels=levels, linewidths=0.5,
                                extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper')
                else:
                    print("Warning: Cannot draw contours due to invalid DEM range.")
            except Exception as e_contour:
                print(f"Warning: Could not draw contours. {e_contour}")
            print("Contours added.")

        #Water Visualization Setup on ax_sim 
        print("Setting up water layer...")

        custom_cmap_active = steady_state_viz_active and (self.steady_state_h is not None)

        if custom_cmap_active:
            print("Using custom steady-state + overflow colormap.")
            
            blue_gradient_map = plt.cm.get_cmap('Blues', 256) 
            overflow_colors = blue_gradient_map(np.linspace(0.1, 0.9, 200))
            black = np.array([0, 0, 0, 1]) # Black for steady flow (RGBA)

            # Define how many "levels" or color slots represent the steady state (black)
            num_black_levels = 10 # Adjust thickness of black band if needed

            # Combine: stack black levels then the blue gradient
            combined_colors = np.vstack((np.tile(black, (num_black_levels, 1)), overflow_colors))
            water_cmap = mcolors.LinearSegmentedColormap.from_list("RiverOverflow", combined_colors)
            water_cmap.set_bad('none') # Transparent for NaN/masked values

            # Normalization for custom map:
            # Range [min_depth, h_steady_marker] -> Black
            # Range (h_steady_marker, overall_vmax] -> Blue gradient
            steady_h = self.steady_state_h
            if hasattr(steady_h, 'get'): 
                steady_h = steady_h.get()
                
            relevant_steady_depths = steady_h[self.valid_mask & (steady_h > self.min_depth)]

            if relevant_steady_depths.size > 0:
                # Use a high percentile (e.g., 95th) to represent the main flow depth
                h_steady_marker = np.percentile(relevant_steady_depths, 95)
                # Ensure marker is slightly above min_depth for distinct boundary
                h_steady_marker = max(h_steady_marker, self.min_depth * 1.05)
            else:
                # Fallback if no water in steady state (unlikely for river scenario)
                h_steady_marker = self.min_depth + 1e-3 # Slightly above min_depth

            print(f"Steady-state marker depth for black color: {h_steady_marker:.3f}m")

            # Define the boundaries for the normalization
            # Ensure boundaries are distinct and ordered: min_depth < h_steady_marker < overall_vmax

            if h_steady_marker >= overall_vmax:
                # If steady marker is too high, adjust it down slightly below vmax
                # This might happen if vmax itself is very low.
                h_steady_marker = overall_vmax - (overall_vmax - self.min_depth) * 0.1
                # Ensure it's still above min_depth
                h_steady_marker = max(h_steady_marker, self.min_depth + 1e-4)
                print(f"Adjusted steady marker due to vmax: {h_steady_marker:.3f}m")

            # Final check for distinct bounds
            color_bounds = [0, h_steady_marker, overall_vmax] # Start bound at 0 for clarity

            if not (color_bounds[0] < color_bounds[1] < color_bounds[2]):
                print("Warning: Cannot create distinct bounds for custom colormap. Falling back to standard Blues.")
                custom_cmap_active = False # Disable custom map
                # Fallback to standard Blues
                water_cmap = plt.cm.Blues.copy()
                water_cmap.set_bad('none')
                norm_water = Normalize(vmin=0, vmax=overall_vmax) # Standard norm
            else:
                # Create the BoundaryNorm
                norm_water = mcolors.BoundaryNorm(color_bounds, water_cmap.N)

        else:
            if steady_state_viz_active: # Print message only if it was requested but failed
                print("Steady state visualization requested but steady state data missing. Using standard Blues.")
            water_cmap = plt.cm.Blues.copy()
            water_cmap.set_bad('none') # Make values below vmin/NaNs transparent
            norm_water = Normalize(vmin=0, vmax=overall_vmax) # Use vmin=0 for standard

        water_cmap = plt.cm.Blues.copy(); water_cmap.set_bad('none')

        first_frame = results[0]
        if first_frame is not None and first_frame.size > 0:
            # Mask values below min_depth and outside valid area
            mask = ~self.valid_mask | (first_frame <= self.min_depth)
            masked_water = np.ma.masked_where(mask, first_frame)
        else:
            # Create an empty masked array if first frame is invalid
            masked_water = np.ma.masked_array(np.zeros_like(self.dem), mask=True)


        # Plot water on ax_sim
        water_plot = ax_sim.imshow(masked_water, cmap=water_cmap, norm=norm_water, alpha=0.7,
                                extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                origin='upper', interpolation='nearest', zorder=10)

        # Add colorbar for water depth, associated with ax_sim
        cbar_water = plt.colorbar(water_plot, ax=ax_sim, pad=0.01, label='Water Depth (m)', shrink=0.7)

        if custom_cmap_active:
            # Set ticks at the boundaries
            cbar_water.set_ticks(color_bounds)
            # Set labels explaining the sections
            cbar_water.set_ticklabels([f'{color_bounds[0]:.1f}', f'River ({color_bounds[1]:.1f})', f'{color_bounds[2]:.1f} (Overflow)'])



        # Add Water Source Markers on ax_sim 
        source_labels = []
        for i, source in enumerate(self.sources):
            # Check if source location is within valid DEM bounds and mask
            if 0 <= source['row'] < self.nx and 0 <= source['col'] < self.ny:
                if self.valid_mask[source['row'], source['col']]:
                    # Calculate plot coordinates (origin='upper')
                    x = (source['col'] + 0.5) * self.dx
                    y = self.nx * self.dy - (source['row'] + 0.5 ) * self.dy # Invert row for 'upper' origin
                    handle = ax_sim.plot(x, y, 'ro', markersize=8, markeredgecolor='k', label=f'Source {i+1}', zorder=20)
                    source_labels.append(handle[0])
            else:
                print(f"Warning: Source {i+1} at ({source['row']}, {source['col']}) is outside DEM bounds.")

                
        if source_labels:
            ax_sim.legend(handles=source_labels, loc='upper right')


        # Add Title and Labels for ax_sim 
        sim_title = ax_sim.set_title(f'Flood Simulation (Time: {time_steps[0]:.2f} s)') # Store title handle
        ax_sim.set_xlabel("X distance (m)")
        ax_sim.set_ylabel("Y distance (m)")

        print("Initial frame plotted.")

        # Animation Update Function 
        # Needs to only update artists related to ax_sim
        def update(frame_idx):
            frame_data = results[frame_idx]
            current_h_frame = frame_data if frame_data is not None else np.zeros_like(self.dem) # Handle None frame

            # Update water layer data
            # Mask values below min_depth and outside valid area
            mask_update = ~self.valid_mask | (current_h_frame <= self.min_depth)
            masked_water_update = np.ma.masked_where(mask_update, current_h_frame)
            water_plot.set_array(masked_water_update)

            # Update title
            sim_title.set_text(f'Flood Simulation (Time: {time_steps[frame_idx]:.2f} s)')

            changed_artists = [water_plot, sim_title]
            return changed_artists

        # Create Animation 
        print("Creating animation object...")
        use_blit = False 
        anim = animation.FuncAnimation(
            fig, update, frames=len(results),
            interval=150, blit=use_blit, repeat=False
        )
        print("Animation object created.")

        # Adjust Layout 
        # Use tight_layout or constrained_layout after creating all elements
        try:
            fig.tight_layout(pad=1.5) 
        except Exception as e:
            print(f"Warning: Could not apply tight_layout. {e}")


        # Save Animation 
        if output_path:
            writer_name = None
            if output_path.endswith('.gif'):
                writer_name = 'pillow'
            elif output_path.endswith('.mp4'):
                writer_name = 'ffmpeg'
            else:
                print("Warning: Unknown animation format. Defaulting to GIF.")
                output_path += ".gif"
                writer_name = 'pillow'

            if writer_name:
                print(f"Saving animation to {output_path} (using {writer_name})... this may take a while...")
                save_dpi = 120 # Lower DPI for faster saving
                try:
                    anim.save(output_path, writer=writer_name, fps=10, dpi=save_dpi)
                    print("Animation saved.")
                except Exception as e_save:
                    print(f"ERROR saving animation: {e_save}")
                    print("Ensure required libraries (pillow for GIF, ffmpeg for MP4) are installed.")

        # Save Last Frame
        if save_last_frame and output_path:
            last_valid_frame_idx = -1
            for i in range(len(results) - 1, -1, -1):
                if results[i] is not None and results[i].size > 0:
                    last_valid_frame_idx = i
                    break

            if last_valid_frame_idx >= 0:
                print("Generating final frame...")
                # Ensure the plot is updated to the last valid frame state
                update(last_valid_frame_idx) # Call update to set the plot correctly
                last_frame_path = output_path.rsplit('.', 1)[0] + '_final_frame.png'
                try:
                    plt.savefig(last_frame_path, dpi=200, bbox_inches='tight')
                    print(f"Final frame saved to {last_frame_path}")
                except Exception as e_frame:
                    print(f"Error saving final frame: {e_frame}")
            else:
                print("Could not save last frame: No valid result frames found.")

        # Show Animation
        if show_animation:
            print("Displaying animation window...")
            try:
                plt.show()
            except Exception as e_show:
                print(f"Could not display animation interactively: {e_show}")
                # Closes the figure if plt.show failed mid-way
                try: plt.close(fig)
                except Exception: pass
        else:
            # Close the figure immediately if not showing interactively
            plt.close(fig)

        print("Visualization finished.")
        return anim


def run_natural_river_phase1_for_twophase(args_test):
    """
    Runs Phase 1 (river flow from initial patch) for the two-phase test.
    Returns: sim object (for phase 2 init), phase 1 results, phase 1 time_steps.
    """
    print("\n--- Running Phase 1 for Two-Phase Test: Establishing River Flow ---")

    # Use parameters from args_test for DEM generation and simulation
    print("Generating DEM for Phase 1...")
    dem, valid_mask, start_center_row = create_natural_river_dem(
        rows=args_test.rows, cols=args_test.cols,
        base_elev=args_test.base_elev, main_slope=args_test.main_slope,
        cross_slope=args_test.cross_slope, channel_width=args_test.channel_width,
        channel_depth=args_test.channel_depth, meander_amplitude=args_test.meander_amplitude,
        meander_freq=args_test.meander_freq, noise_sigma=args_test.noise_sigma,
        noise_strength=args_test.noise_strength, final_smooth_sigma=args_test.final_smooth_sigma
    )
    nx, ny = dem.shape

    print(f"Initializing Phase 1 sim with outlet_percentile={args_test.outlet_percentile}.")
    try:
        sim_p1 = FloodSimulation(
            dem=dem, valid_mask=valid_mask, dx=args_test.dx, dy=args_test.dy,
            manning=args_test.manning,
            infiltration_rate=0.0, # No infiltration for phase 1 river
            stability_factor=args_test.stability_factor, adaptive_timestep=True,
            max_velocity=args_test.max_velocity, min_depth=args_test.min_depth,
            outlet_threshold_percentile=args_test.outlet_percentile, 
            target_time=9999999.0 # Let steps control phase 1 duration
        )
    except Exception as e:
        print(f"ERROR initializing Phase 1 sim: {e}")
        return None, None, None


    sim_p1.add_water_source(
        row=start_center_row, col=0, 
        rate=args_test.source_rate, duration_steps=1,
    )

    print(f"\n--- Running Phase 1 Simulation for {args_test.phase1_steps} steps ---")
    results1, time_steps1 = sim_p1.run_simulation(
        num_steps=args_test.phase1_steps,
        output_freq=args_test.output_freq
    )
    if not results1:
        print("Phase 1 produced no results.")
        return None, None, None

    print(f"Phase 1 simulation complete. Final sim time: {time_steps1[-1]:.2f}s")
    return sim_p1, results1, time_steps1

def run_twophase_river_rain_test(args):
    """
    Runs the two-phase simulation: river flow then rainfall (combined result).
    Uses custom visualization.
    """
    print("\n--- Starting Two-Phase River + Rainfall Test (Combined Output, Custom Viz) ---")

    # Run Phase 1 
    sim_phase1, results1, time_steps1 = run_natural_river_phase1_for_twophase(args)

    if sim_phase1 is None or not results1:
        print("Phase 1 failed. Stopping.")
        return

    last_time_phase1 = time_steps1[-1] if time_steps1 else 0.0
    print(f"End of Phase 1 at Sim Time: {last_time_phase1:.2f} s")
    final_h_phase1 = sim_phase1.h
    h_valid_p1 = final_h_phase1[sim_phase1.valid_mask]
    max_depth_p1 = np.nanmax(h_valid_p1) if h_valid_p1.size > 0 else 0.0
    print(f"Phase 1 Final Max Depth: {max_depth_p1:.3f} m")

    # Setup for Phase 2, continuing from Phase 1 state 
    sim_phase1.steady_state_h = final_h_phase1.copy() # Capture h at end of phase 1
    print("Stored final Phase 1 depth as steady_state_h for visualization.")

    print(f"\n--- Starting Phase 2: Adding Rainfall ({args.phase2_steps} steps) ---")
    rain_rate_mmhr = args.rain_rate
    if rain_rate_mmhr <= 0:
        print("No rainfall specified for Phase 2 (rain_rate <= 0).")
        rain_rate_ms = 0.0
    else:
        rain_rate_ms = rain_rate_mmhr / (1000.0 * 3600.0)
        print(f"Adding rainfall rate: {rain_rate_mmhr} mm/hr ({rain_rate_ms:.2e} m/s)")
        # Add rainfall to the existing sim_phase1 object
        sim_phase1.add_rainfall(rate=rain_rate_ms, duration_steps=args.phase2_steps)

    # Also set the infiltration rate for phase 2 from args
    infiltration_rate_mmhr = args.infiltration_rate
    if infiltration_rate_mmhr > 0:
        sim_phase1.infiltration_rate = infiltration_rate_mmhr / (1000.0*3600.0)
        print(f"Setting infiltration rate for Phase 2: {infiltration_rate_mmhr} mm/hr ({sim_phase1.infiltration_rate:.2e} m/s)")
    else:
        sim_phase1.infiltration_rate = 0.0
        print("Infiltration rate for Phase 2 is 0.")


    # Run Phase 2 
    sim_phase1.target_time = 9999999.0 # Let steps control this phase
    sim_phase1.original_dt_guess = sim_phase1.dt # Reset dt guess based on current state? Optional.
    print(f"Running Phase 2 for {args.phase2_steps} additional steps...")
    results2, time_steps2_relative = sim_phase1.run_simulation(
        num_steps=args.phase2_steps,
        output_freq=args.output_freq
    )

    if not results2:
        print("Phase 2 did not produce results.")
        # Still visualize phase 1 results?
        if results1:
            print("Visualizing Phase 1 results only.")
            try:
                sim_phase1.visualize_results(
                    results1, time_steps1,
                    output_path=args.output_path.replace(".","_phase1_only.") if args.output_path else 'river_phase1_only.gif',
                    show_animation=not args.hide_animation, save_last_frame=args.save_last_frame,
                    hillshade=not args.no_hillshade, contour_levels=args.contour_levels,
                    show_boundary_mask=False, steady_state_viz_active=False # No custom viz for phase 1 only
                )
            except Exception as e: print(f"Error visualizing phase 1 results: {e}")
        return


    # Combine Results for Final Visualization 
    print("Combining results for final visualization...")
    # Adjust phase 2 time steps to be absolute (start after phase 1 ended)
    adjusted_time_steps2 = [t + last_time_phase1 for t in time_steps2_relative]


    if len(time_steps2_relative) > 0 and time_steps2_relative[0] < sim_phase1.dt * 1.1: # 
        all_time_steps = time_steps1 + adjusted_time_steps2
        print(f"Concatenated results: {len(results1)} (P1) + {len(results2)} (P2) = {len(all_results)} frames.")

    else: # Step 0 of phase 2 was likely not saved by output_freq
        all_results = results1 + results2
        all_time_steps = time_steps1 + adjusted_time_steps2
        print(f"Concatenated results: {len(results1)} (P1) + {len(results2)} (P2) = {len(all_results)} frames.")


    # Visualize Combined Results with Custom Colormap 
    if not all_results:
        print("No combined results to visualize.")
        return

    print("\n--- Visualizing Combined River + Rainfall Results (Custom Colormap) ---")
    try:
        # Call visualize_results on the 'sim_phase1' object which now contains steady_state_h
        sim_phase1.visualize_results(
            all_results,
            all_time_steps,
            output_path=args.output_path if args.output_path else 'river_then_rain_custom_viz.gif',
            show_animation=not args.hide_animation,
            save_last_frame=args.save_last_frame,
            hillshade=not args.no_hillshade,
            contour_levels=args.contour_levels,
            show_boundary_mask=False, # Hide mask for final combined plot
            steady_state_viz_active=True # <<< ACTIVATE CUSTOM VISUALIZATION
        )
    except Exception as e:
        print(f"\nERROR during final visualization: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

    print("\n--- Two-Phase River + Rainfall Test (Custom Viz) Finished ---")