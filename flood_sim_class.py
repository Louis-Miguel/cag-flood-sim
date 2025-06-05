import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors as mcolors, gridspec
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource, Normalize
from flood_tools import generate_boundary_mask, create_natural_river_dem
from compute_flood_step import compute_step


np.random.seed(24) 

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
        self.dem_contour_thresholds_viz = [] # For phased viz

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
            if np.max(dist) > 1e-9:
                dist /= np.max(dist)
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
            wave_speed = np.sqrt(self.g * max(max_depth, self.min_depth)) # Add epsilon to avoid division by zero
            denominator = wave_speed + max_vel + 1e-9 
            dt_cfl = self.stability_factor * min(self.dx, self.dy) / denominator
            
            # Constraint: Don't increase dt too rapidly from initial guess or previous step
            max_dt_allowed = max(self.original_dt_guess, self.dt * 1.5)

            self.dt = min(dt_cfl, max_dt_allowed)

            if self.dt < 1e-9: 
                print("Warning: Timestep extremely small...") 
                self.dt = 1e-9

        # Call numba function
        h_new, u_new, v_new = compute_step(
            self.h, self.u, self.v, self.dem, self.boundary_mask, # Pass mask
            self.g, self.manning, self.dx, self.dy, self.dt,
            self.infiltration_rate, self.min_depth
        )

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
                            show_boundary_mask=True, steady_state_viz_active=True, num_phase1_frames=0):
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
        fig = plt.figure(figsize=(10 * num_subplots, 8))

        if show_boundary_mask:
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            ax_sim = plt.subplot(gs[0])
            ax_mask = plt.subplot(gs[1])
        else:
            gs = gridspec.GridSpec(1, 1)
            ax_sim = plt.subplot(gs[0])
            ax_mask = None

        if ax_mask:
            ax_mask.set_title("Boundary Mask (-1:Out, 0:In, 1:Wall, 2:Open)")
            cmap_mask_viz = mcolors.ListedColormap(['lightgrey', 'white', 'black', 'blue'])
            bounds_mask_viz = [-1.5, -0.5, 0.5, 1.5, 2.5]
            norm_mask_viz = mcolors.BoundaryNorm(bounds_mask_viz, cmap_mask_viz.N)
            mask_plot_obj = ax_mask.imshow(self.boundary_mask, cmap=cmap_mask_viz, norm=norm_mask_viz,
                                    interpolation='none', origin='upper',
                                    extent=(0, self.ny*self.dx, 0, self.nx*self.dy))
            cbar_mask_plot = plt.colorbar(mask_plot_obj, ax=ax_mask, shrink=0.7, ticks=[-1, 0, 1, 2])
            cbar_mask_plot.set_ticklabels(['Outside', 'Internal', 'Wall', 'Outlet'])
            ax_mask.set_xlabel("X distance (m)")
            ax_mask.set_ylabel("Y distance (m)")

        ax_sim.set_title(f'Flood Simulation (Frame 0)')

        dem_display_viz = self.dem.copy()
        dem_display_viz[~self.valid_mask] = np.nan

        self.dem_contour_thresholds_viz = []
        if contour_levels > 0:
            # Use nanmin/nanmax safely on potentially all-NaN masked DEM
            valid_dem_cells = dem_display_viz[self.valid_mask]
            cont_min_dem_viz = np.nanmin(valid_dem_cells) if valid_dem_cells.size > 0 else 0
            cont_max_dem_viz = np.nanmax(valid_dem_cells) if valid_dem_cells.size > 0 else 1
            
            if not (np.isnan(cont_min_dem_viz) or np.isnan(cont_max_dem_viz) or cont_max_dem_viz <= cont_min_dem_viz):
                num_actual_levels_viz = max(2, contour_levels) 
                self.dem_contour_thresholds_viz = sorted(list(np.linspace(cont_min_dem_viz, cont_max_dem_viz, num_actual_levels_viz)))
                if contour_levels == 1 and len(self.dem_contour_thresholds_viz) > 1: 
                    self.dem_contour_thresholds_viz = [self.dem_contour_thresholds_viz[len(self.dem_contour_thresholds_viz)//2]]
            else: 
                default_elev_viz = np.nanmedian(valid_dem_cells) if valid_dem_cells.size > 0 else 10.0
                self.dem_contour_thresholds_viz = [default_elev_viz] if not np.isnan(default_elev_viz) else [10.0]
            print(f"Using DEM contour thresholds for WSE viz: {self.dem_contour_thresholds_viz}")


        if hillshade:
            try:
                ls_viz = LightSource(azdeg=315, altdeg=45)
                dem_min_valid_viz = np.nanmin(dem_display_viz) # Use full dem_display for min/max
                dem_max_valid_viz = np.nanmax(dem_display_viz)
                if np.isnan(dem_min_valid_viz) or np.isnan(dem_max_valid_viz) or dem_max_valid_viz <= dem_min_valid_viz:
                    rgb_hillshade = np.full((self.nx, self.ny, 3), 0.7) # Default grey
                else:
                    dem_norm_val_viz = (dem_display_viz - dem_min_valid_viz) / (dem_max_valid_viz - dem_min_valid_viz)
                    dem_for_shade_viz = np.nan_to_num(dem_norm_val_viz, nan=0.5) # Fill NaNs for shading
                    rgb_hillshade = ls_viz.shade(dem_for_shade_viz, cmap=plt.cm.terrain, vert_exag=1.0, blend_mode='soft')
                
                if rgb_hillshade.ndim == 3 and rgb_hillshade.shape[2] >= 3:
                    rgba_hillshade = np.zeros((rgb_hillshade.shape[0], rgb_hillshade.shape[1], 4), dtype=np.uint8)
                    rgba_hillshade[:, :, :3] = np.clip(rgb_hillshade[:, :, :3] * 255, 0, 255).astype(np.uint8)
                    rgba_hillshade[:, :, 3] = 255 # Opaque
                    rgba_hillshade[~self.valid_mask, 3] = 0 # Transparent where DEM is invalid
                    ax_sim.imshow(rgba_hillshade, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper', interpolation='nearest', zorder=1)
                else: 
                    hillshade = False
            except Exception: 
                hillshade = False 
        if not hillshade: 
            terrain_cmap_viz = plt.cm.terrain.copy()
            terrain_cmap_viz.set_bad(color='white', alpha=0) # NaNs in DEM transparent
            terrain_vmin_viz = np.nanmin(dem_display_viz)
            terrain_vmax_viz = np.nanmax(dem_display_viz)
            if np.isnan(terrain_vmin_viz) or np.isnan(terrain_vmax_viz) or terrain_vmax_viz <= terrain_vmin_viz: # Flat or all NaN
                terrain_vmin_viz, terrain_vmax_viz = 0, 1 # Default range
            ax_sim.imshow(dem_display_viz, cmap=terrain_cmap_viz,
                                    vmin=terrain_vmin_viz, vmax=terrain_vmax_viz,
                                    extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                    origin='upper', interpolation='nearest', zorder=1)

        if contour_levels > 0 and self.dem_contour_thresholds_viz:
            try:
                # Fill NaNs for contouring, e.g., with min valid DEM value
                dem_for_contour_viz = dem_display_viz.copy()
                fill_val_contour = np.nanmin(dem_for_contour_viz[self.valid_mask]) if np.any(self.valid_mask) else 0
                dem_for_contour_viz[~self.valid_mask] = fill_val_contour 
                dem_for_contour_viz = np.nan_to_num(dem_for_contour_viz, nan=fill_val_contour)

                ax_sim.contour(dem_for_contour_viz, colors='black', alpha=0.3, levels=self.dem_contour_thresholds_viz,
                            linewidths=0.5, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper', zorder=2)
            except Exception as e_contour_viz:
                print(f"Warning: Could not draw DEM contours. {e_contour_viz}")

        use_global_wse_layered_viz = steady_state_viz_active and num_phase1_frames > 0 and self.steady_state_h is not None

        black_cmap = mcolors.ListedColormap(['black'])
        black_cmap.set_bad('none')

        max_h_for_norm = np.nanmax(results) if results and np.any([r is not None for r in results]) else self.min_depth * 2

        if np.isnan(max_h_for_norm) or max_h_for_norm <= self.min_depth : 
            max_h_for_norm = self.min_depth * 2

        norm_black = Normalize(vmin=self.min_depth, vmax=max_h_for_norm)
        
        blue_cmap = plt.cm.Blues.copy()
        blue_cmap.set_bad('none')
        norm_blue = Normalize(vmin=0, vmax=max_h_for_norm)

        self.layered_fill_cmap = None
        self.layered_fill_norm = None
        if use_global_wse_layered_viz and self.dem_contour_thresholds_viz:
            num_shades = len(self.dem_contour_thresholds_viz)
            if num_shades > 0:
                grey_vals = np.linspace(1, 0.50, num_shades) # Darker grey to lighter grey
                self.layered_fill_cmap = mcolors.ListedColormap([plt.cm.Greys(v) for v in grey_vals])
                self.layered_fill_cmap.set_bad(alpha=0) 
                self.layered_fill_norm = mcolors.BoundaryNorm(np.arange(-0.5, num_shades), self.layered_fill_cmap.N)
            else: 
                use_global_wse_layered_viz = False

        first_h = results[0] if results else np.zeros_like(self.dem)
        mask_init = ~self.valid_mask | (first_h <= self.min_depth)
        init_display = np.ma.masked_where(mask_init, first_h)
        init_cmap, init_norm, init_cbar_label = (black_cmap, norm_black, 'River Depth (m) - P1') if use_global_wse_layered_viz else (blue_cmap, norm_blue, 'Water Depth (m)')

        main_water_plot = ax_sim.imshow(init_display, cmap=init_cmap, norm=init_norm, alpha=1.0, origin='upper', extent=(0, self.ny*self.dx, 0, self.nx*self.dy), zorder=10)
        fixed_river_underlay = ax_sim.imshow(np.ma.masked_array(np.zeros_like(self.dem), True), cmap=black_cmap, norm=norm_black, alpha=1.0, origin='upper', extent=(0, self.ny*self.dx, 0, self.nx*self.dy), zorder=9, visible=False)

        cbar = plt.colorbar(main_water_plot, ax=ax_sim, label=init_cbar_label, shrink=0.7)

        source_labels_viz = []
        for i_src, source_item_viz in enumerate(self.sources):
            if 0 <= source_item_viz['row'] < self.nx and 0 <= source_item_viz['col'] < self.ny and self.valid_mask[source_item_viz['row'], source_item_viz['col']]:
                x_coord_src_viz = (source_item_viz['col'] + 0.5) * self.dx
                y_coord_src_viz = self.nx * self.dy - (source_item_viz['row'] + 0.5 ) * self.dy
                handle_src_viz = ax_sim.plot(x_coord_src_viz, y_coord_src_viz, 'ro', markersize=8, markeredgecolor='k', label=f'Source {i_src+1}', zorder=20)
                source_labels_viz.append(handle_src_viz[0])
        if source_labels_viz: 
            ax_sim.legend(handles=source_labels_viz, loc='upper right')

        sim_title = ax_sim.set_title(f'Flood Simulation (Time: {time_steps[0]:.2f} s)')
        ax_sim.set_xlabel("X distance (m)")
        ax_sim.set_ylabel("Y distance (m)")

        def update(frame_idx):
            current_h = results[frame_idx] if results[frame_idx] is not None else np.zeros_like(self.dem)
            title = f'Flood Sim (Time: {time_steps[frame_idx]:.2f}s)'
            cbar_lbl = 'Water Depth (m)'
            artists = [sim_title]
            active_cmap, active_norm = blue_cmap, norm_blue
            is_p1 = frame_idx < num_phase1_frames

            if use_global_wse_layered_viz:
                if is_p1:
                    fixed_river_underlay.set_visible(False)
                    mask = ~self.valid_mask | (current_h <= self.min_depth)
                    
                    main_water_plot.set_array(np.ma.masked_where(mask, current_h))
                    main_water_plot.set_cmap(blue_cmap)
                    main_water_plot.set_norm(norm_blue)
                    main_water_plot.set_alpha(1.0)
                    
                    active_cmap, active_norm = blue_cmap, norm_blue
                    cbar_lbl = 'River Depth (m) - P1'
                    title += ' - P1: River Flow Sim'
                    artists.append(main_water_plot)
                else: # Phase 2: Flooded Visualization
                    fixed_river_underlay.set_array(np.ma.masked_where(~self.valid_mask | (self.steady_state_h <= self.min_depth), self.steady_state_h))
                    fixed_river_underlay.set_visible(True)
                    artists.append(fixed_river_underlay)

                    idx_max_h = (-1,-1)
                    h_for_max_f = np.where(self.valid_mask, current_h, -np.inf)
                    if np.any(current_h[self.valid_mask] > self.min_depth):
                        idx_max_h = np.unravel_index(np.argmax(h_for_max_f), self.dem.shape)
                    
                    idx_highest_globally_breached = -1
                    if idx_max_h!= (-1,-1) and self.dem_contour_thresholds_viz and self.layered_fill_cmap:
                        wse_peak = self.dem[idx_max_h] + current_h[idx_max_h]
                        wse_peak -= 2.5
                        for i, thresh in enumerate(self.dem_contour_thresholds_viz):
                            if wse_peak >= thresh: 
                                idx_highest_globally_breached = i
                            else: 
                                break
                    
                    layered_indices = np.full(self.dem.shape, -1, dtype=np.int8)
                    if idx_highest_globally_breached != -1:
                        for band_idx in range(idx_highest_globally_breached + 1):
                            upper_dem = self.dem_contour_thresholds_viz[band_idx]
                            lower_dem = self.dem_contour_thresholds_viz[band_idx-1] if band_idx > 0 else -np.inf
                            
                            # Fill DEM band if it's below the globally breached WSE level
                            band_mask = (
                                self.valid_mask &
                                (self.dem > lower_dem) &
                                (self.dem <= upper_dem) &
                                (~(self.steady_state_h > self.min_depth))
                            )
                            layered_indices[band_mask] = band_idx 
                    
                    main_water_plot.set_array(np.ma.masked_where(layered_indices == -1, layered_indices))
                    main_water_plot.set_cmap(self.layered_fill_cmap)
                    main_water_plot.set_norm(self.layered_fill_norm)
                    main_water_plot.set_alpha(0.80) 
                    
                    active_cmap, active_norm = main_water_plot.cmap, main_water_plot.norm
                    artists.append(main_water_plot)
                    cbar_lbl = 'DEM Contour Band'; title += ' - P2: Rainfall Flood Visualization'
            else: 
                fixed_river_underlay.set_visible(False)
                mask_default = ~self.valid_mask | (current_h <= self.min_depth)
                masked_data_default = np.ma.masked_where(mask_default, current_h)
                
                main_water_plot.set_array(masked_data_default)
                main_water_plot.set_cmap(blue_cmap)
                main_water_plot.set_norm(norm_blue)
                main_water_plot.set_alpha(0.7); main_water_plot.set_visible(True)
                
                active_cmap, active_norm = blue_cmap, norm_blue
                cbar_lbl = 'Water Depth (m)'
                artists.append(main_water_plot)


            sim_title.set_text(title)
            if cbar:
                cbar.mappable.set_cmap(active_cmap)
                cbar.mappable.set_norm(active_norm)
                cbar.set_label(cbar_lbl)
                
                if use_global_wse_layered_viz and not is_p1 and self.dem_contour_thresholds_viz and self.layered_fill_cmap:
                    num_t = len(self.dem_contour_thresholds_viz)
                    if num_t > 0:
                        t_pos = np.arange(num_t)
                        t_lab = [f'{v:.1f}m' for v in self.dem_contour_thresholds_viz]
                        if num_t > 8: 
                            step_t = max(1, num_t//8)
                            t_pos,t_lab = t_pos[::step_t],t_lab[::step_t]
                        cbar.set_ticks(t_pos)
                        cbar.set_ticklabels(t_lab)
                else: 
                    cbar.update_ticks()
            return artists


        anim = animation.FuncAnimation(fig, update, frames=len(results), interval=150, blit=False, repeat=False)

        try:
            fig.tight_layout(pad=1.5)
        except Exception as e_layout_viz:
            print(f"Warning: Could not apply tight_layout. {e_layout_viz}")

        if output_path:
            writer_name_save_viz = 'pillow' if output_path.endswith('.gif') else 'ffmpeg' if output_path.endswith('.mp4') else None
            if not writer_name_save_viz: 
                output_path += ".gif"
                writer_name_save_viz = 'pillow'
            print(f"Saving animation to {output_path} (using {writer_name_save_viz})...")
            try:
                anim.save(output_path, writer=writer_name_save_viz, fps=10, dpi=120)
                print("Animation saved.")
            except Exception as e_save_anim_viz:
                print(f"ERROR saving animation: {e_save_anim_viz}. Ensure {writer_name_save_viz} is installed.")

        if save_last_frame and output_path:
            last_idx_save_viz = next((i_save_viz for i_save_viz in range(len(results)-1, -1, -1) if results[i_save_viz] is not None), -1)
            if last_idx_save_viz >= 0:
                update(last_idx_save_viz)
                last_frame_path_save_viz = output_path.rsplit('.', 1)[0] + '_final_frame.png'
                try:
                    plt.savefig(last_frame_path_save_viz, dpi=200, bbox_inches='tight')
                    print(f"Final frame saved to {last_frame_path_save_viz}")
                except Exception as e_frame_save_viz: 
                    print(f"Error saving final frame: {e_frame_save_viz}")

        if show_animation:
            try: 
                plt.show()
            except Exception as e_show_anim_viz: 
                print(f"Could not display animation: {e_show_anim_viz}")
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
    
    rows, cols = 100, 400
    dx, dy = 5.0, 5.0
    dem, valid_mask, start_center_row = create_natural_river_dem( 
        rows=rows, cols=cols, base_elev=25.0, main_slope=0.001 * dx,
        cross_slope=0.0, channel_width=10 * dx, channel_depth=3.0,
        meander_amplitude=10.0, meander_freq=1.5, noise_sigma=2.0,
        noise_strength=0.3, final_smooth_sigma=1.0 )
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
        rate=50, duration_steps=1,
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
    sim_phase1.steady_state_h = sim_phase1.h.copy()  # Capture h at end of phase 1
    print(f"End of Phase 1 at Sim Time: {last_time_phase1:.2f}s. Stored steady_state_h.")

    sim_phase1.sources = [] # Clear Phase 1 sources/patches

    if args.source_row is not None and args.source_col is not None and args.source_rate > 0:
        r, c = args.source_row, args.source_col
        if 0 <= r < sim_phase1.dem.shape[0] and 0 <= c < sim_phase1.dem.shape[1] and sim_phase1.valid_mask[r,c]:
            sim_phase1.add_water_source(row=r, col=c, rate=args.source_rate,
                                        duration_steps=args.source_duration if args.source_duration is not None else args.phase2_steps)
        else: print(f"Warning (P2): Source at ({r},{c}) invalid. Ignored.")

    print(f"\n--- Starting Phase 2: Rainfall & WSE Monitoring ({args.phase2_steps} steps) ---")
    if args.rain_rate > 0:
        rain_rate_ms = args.rain_rate / (1000.0 * 3600.0)
        sim_phase1.add_rainfall(rate=rain_rate_ms,
                                duration_steps=args.rain_duration if args.rain_duration is not None else args.phase2_steps)
        print(f"Added rainfall for Phase 2: {args.rain_rate} mm/hr")
    else:
        print("No rainfall for Phase 2.")

    if args.infiltration_rate > 0:
        sim_phase1.infiltration_rate = args.infiltration_rate / (1000.0 * 3600.0)
        print(f"Set infiltration for Phase 2: {args.infiltration_rate} mm/hr")
    else: 
        sim_phase1.infiltration_rate = 0.0

    sim_phase1.target_time = float('inf') # Let steps control P2 duration
    results2, time_steps2_relative = sim_phase1.run_simulation(
        num_steps=args.phase2_steps,
        output_freq=args.output_freq
    )

    if not results2:
        print("Phase 2 did not produce results."); # Optionally visualize P1 only
        return

    total_time_rain = time_steps2_relative[-1] if time_steps2_relative else 0.0
    max_depth_rain = np.nanmax(sim_phase1.h.copy())
    steady_state_h_max = np.nanmax(sim_phase1.steady_state_h)

    rate_of_increase = (max_depth_rain - steady_state_h_max) / total_time_rain 
    print(f"Phase 2 max depth: {max_depth_rain:.3f} m, steady state max depth: {steady_state_h_max:.3f} m, rate of increase: {rate_of_increase:.6f} m/s")       

    adjusted_time_steps2 = [t for t in time_steps2_relative]
    all_results = results1 + results2
    all_time_steps = time_steps1 + adjusted_time_steps2
    num_total_phase1_frames = len(results1)

    if not all_results: 
        print("No combined results to visualize.")
        return

    print("\n--- Visualizing Combined River (P1) + WSE Contour Breach (P2) Results ---")
    try:
        sim_phase1.visualize_results(
            all_results,
            all_time_steps,
            output_path=args.output_path if args.output_path else 'river_wse_contours.gif',
            show_animation=not args.hide_animation,
            save_last_frame=args.save_last_frame,
            hillshade=not args.no_hillshade,
            contour_levels=args.contour_levels, # This is important for the WSE viz logic
            show_boundary_mask=args.show_boundary_mask,
            steady_state_viz_active=True, # ACTIVATE WSE VISUALIZATION MODES
            num_phase1_frames=num_total_phase1_frames
        )
    except Exception as e:
        print(f"\nERROR during final visualization: {e}"); import traceback; traceback.print_exc()

    print("\n--- Two-Phase River + WSE Contour Breach Test Finished ---")