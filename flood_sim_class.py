import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource, Normalize
from flood_tools import generate_boundary_mask 

class FloodSimulation:
    def __init__(self, dem, valid_mask, 
                dx=1.0, dy=1.0, g=9.81, 
                manning=0.03, infiltration_rate=0.0, adaptive_timestep=True,
                stability_factor=0.4, max_velocity=10.0, min_depth=0.001,
                outlet_threshold_percentile=5.0):
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
        self.min_depth = float(min_depth)

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
            max_h_guess = max(1.0, np.nanmax(dem[self.valid_mask]) - np.nanmin(dem[self.valid_mask]))
        else: 
            max_h_guess = 1.0 # Default if no valid cells

        # Compute timestep based on grid size (CFL step initialization)
        self.dt = 0.1 * min(dx, dy) / np.sqrt (g * max_h_guess + 1e-6)
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
            dist /= np.max(dist)
        else:
            # Ensure correct shape and smooth input distribution
            assert spatial_distribution.shape == self.dem.shape, "Spatial distribution must match DEM shape"
            dist = np.maximum(0, spatial_distribution)
            # Smooth the distribution
            dist = gaussian_filter(dist, sigma=2.0)
            # Normalize
            if np.max(dist) > 0:
                dist /= np.max(dist)

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
                h_new[source['row'], source['col']] = max(0.0, h_new[source['row'], source['col']] + depth_increase) 
                source['steps_active'] += 1
        # Apply rainfall
        if self.rainfall:
            if self.rainfall['duration'] is None or self.rainfall['steps_active'] < self.rainfall['duration']:
                time_factor = self.rainfall['time_pattern'](self.rainfall['steps_active'], self.rainfall['duration'])
                effective_rate = self.rainfall['base_rate'] * time_factor
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

            with np.errstate(invalid='ignore'):
                # Use valid subsets for max calculations
                max_vel_sq = np.max(u_valid**2 + v_valid**2) if u_valid.size > 0 else 0.0
                max_vel = np.sqrt(max_vel_sq)
                max_depth = np.max(h_valid) if h_valid.size > 0 else 0.0

            if np.isnan(max_vel): max_vel = 0.0
            if np.isnan(max_depth): max_depth = 0.0

            # Adjust timestep based on CFL condition (Courant-Friedrichs-Lewy (CFL) condition)
            wave_speed = np.sqrt(self.g * max(max_depth, self.min_depth))
            dt_cfl = self.stability_factor * min(self.dx, self.dy) / (wave_speed + max_vel + 1e-6)
            
            # More conservative approach
            max_dt_allowed = max(self.original_dt_guess, self.dt * 1.5)
            self.dt = min(dt_cfl, max_dt_allowed)

            if self.dt < 1e-9: 
                print("Warning: Timestep extremely small...") 
                self.dt = 1e-9

        # Call numba function
        h_new, u_new, v_new = compute_step_jit(
            self.h, self.u, self.v, self.dem, self.boundary_mask, # Pass mask
            self.g, self.manning, self.dx, self.dy, self.dt,
            self.infiltration_rate, self.min_depth
        )

        # Post-computation checks/updates (apply only to valid area)

        # Velocity Clipping
        vel_new_sq = u_new**2 + v_new**2
        scale = np.ones_like(vel_new_sq)
        mask_to_scale = self.valid_mask & (vel_new_sq > self.max_velocity**2)
        if np.any(mask_to_scale):
            scale[mask_to_scale] = self.max_velocity / (np.sqrt(vel_new_sq[mask_to_scale]) + 1e-9)
        u_new *= scale
        v_new *= scale

        # Apply Sources/Rainfall
        h_new = self.apply_sources_or_rainfall(h_new)

        # Update State Variables
        self.h = h_new
        self.u = u_new
        self.v = v_new
        self.h[~self.valid_mask] = 0.0 # Ensure h is 0 outside
        self.u[~self.valid_mask] = 0.0 # Ensure u is 0 outside
        self.v[~self.valid_mask] = 0.0 # Ensure v is 0 outside

        # Update Max Flood Trackers
        is_flooded = self.valid_mask & (self.h > self.min_depth)
        self.max_flood_extent = np.logical_or(self.max_flood_extent, is_flooded)
        # Update max depth only where flooded
        self.max_flood_depth = np.maximum(self.max_flood_depth, np.where(is_flooded, self.h, 0))
        # Ensure max depth outside valid area remains NaN (or 0 if preferred)
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

        # Run simulation loop
        for step in range(num_steps):
            self.current_step = step

            # Compute fluxes and update state variables and check for errors
            try: 
                self.compute_fluxes_and_update()
            except Exception as e: 
                print(f"\n--- ERROR occurred at step {step}, time {total_time:.2f}s ---\n{e}") 
                break 

            total_time += self.dt

            # Display progress and show results at the end
            if step % output_freq == 0 or step == num_steps - 1:
                results.append(self.h.copy())
                time_steps_list.append(total_time)

                # Compute max depth and velocity
                with np.errstate(invalid='ignore'): 
                    max_depth = np.nanmax(self.h)
                    max_vel = np.sqrt(np.nanmax(self.u**2 + self.v**2))
                
                # Handle NaN values
                max_depth = 0.0 if np.isnan(max_depth) else max_depth
                max_vel = 0.0 if np.isnan(max_vel) else max_vel

                # Compute flood area and volume
                flood_area = np.sum(self.h > self.min_depth) * self.dx * self.dy
                flood_volume = np.sum(self.h[self.h > 0]) * self.dx * self.dy
                
                # Print progress and stop if unstable
                print(
                    f"Step {step}/{num_steps - 1}, "
                    f"Sim Time: {total_time:.3f}s, "
                    f"dt: {self.dt:.4f}s, "
                    f"Max Depth: {max_depth:.3f}m, "
                    f"Max Vel: {max_vel:.3f}m/s, "
                    f"Area: {flood_area:.1f}m², "
                    f"Vol: {flood_volume:.1f}m³"
                )
                if not np.isfinite(max_depth) or max_depth > 1000:
                    print(" Warning: Simulation may be unstable. Stopping.")
                    break

        end_run_time = time.time()
        print(f"\nSimulation finished {step+1} steps in {end_run_time - start_run_time:.2f} seconds (wall clock).")
        final_flood_area = np.sum(self.max_flood_extent) * self.dx * self.dy
        max_recorded_depth = np.max(self.max_flood_depth) 
        print(f"Final Maximum Flood Extent Area: {final_flood_area:.2f} m²")
        print(f"Peak Recorded Water Depth Anywhere: {max_recorded_depth:.3f} m")

        return results, time_steps_list

    def visualize_results(self, results, time_steps, output_path=None,
                        show_animation=True, save_last_frame=True,
                        hillshade=True, contour_levels=10):
        """
        Visualize the simulation results with enhanced terrain visualization.
        Handles potential issues with empty or NaN results.

        Parameters:
        -----------
        results : list : List of simulation results (2D arrays)
        time_steps : list : List of time steps corresponding to results
        output_path : str : Path to save the animation (optional)
        show_animation : bool : Whether to display the animation
        save_last_frame : bool : Whether to save the last frame as an image
        hillshade : bool : Whether to include hillshade effect on terrain
        contour_levels : int : Number of contour levels to display
        """

        if not results:
            print("No results to visualize.")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine color limits for water depth (Robust calculation) 
        valid_maxes = []
        for h in results:
            # Consider only valid area for max depth calculation
            if h is not None and h.size > 0 and self.valid_mask.shape == h.shape:
                try:
                    max_val = np.nanmax(h[self.valid_mask]) 
                    if np.isfinite(max_val): valid_maxes.append(max_val)
                except Exception: pass 


        vmax = max(valid_maxes) if valid_maxes else 1.0
        if vmax < self.min_depth: vmax = self.min_depth * 10
        if np.isnan(vmax): vmax = 1.0
        norm = Normalize(vmin=0, vmax=vmax)
        
        # Plotting DEM background (mask invalid areas if desired)
        dem_display = self.dem.copy()
        dem_display[~self.valid_mask] = np.nan # Mask DEM for display

        if hillshade:
            ls = LightSource(azdeg=315, altdeg=45)
            
            # Calculate normalization only on valid DEM parts
            dem_min_valid = np.nanmin(dem_display)
            dem_max_valid = np.nanmax(dem_display)

            if dem_max_valid > dem_min_valid:
                # Create norm array, keeping NaNs
                dem_norm_val = (dem_display - dem_min_valid) / (dem_max_valid - dem_min_valid)
            else: # Handle flat valid terrain
                dem_norm_val = np.full_like(dem_display, 0.5)
                dem_norm_val[~self.valid_mask] = np.nan # Keep NaNs

            # Shade
            rgb = ls.shade(dem_norm_val, cmap=plt.cm.terrain, vert_exag=1.0, blend_mode='soft')
            
            # Set NaN areas in RGB to background color (e.g., transparent or specific color)
            dem_plot = ax.imshow(rgb, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper')
        else:
            # Use a colormap that handles NaN 
            terrain_cmap = plt.cm.terrain
            terrain_cmap.set_bad(color='none') 

            dem_plot = ax.imshow(dem_display, cmap=terrain_cmap, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper')
            

        # Add contour lines 
        if contour_levels > 0:
            ax.contour(dem_display, colors='black', alpha=0.3, levels=contour_levels,
                    linewidths=0.5, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper')

        # Water visualization 
        water_cmap = plt.cm.Blues
        water_cmap.set_bad('none') 

        first_frame = results[0]
        if first_frame is not None and first_frame.size > 0:
            # Mask outside area OR where depth is below minimum
            mask = ~self.valid_mask | (first_frame <= self.min_depth)
            masked_water = np.ma.masked_where(mask, first_frame)
        else:
            masked_water = np.ma.masked_array(np.zeros_like(self.dem), mask=True) # Empty

        water_plot = ax.imshow(masked_water, cmap=water_cmap, norm=norm, alpha=0.7, 
                            extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper', interpolation='none')

        # Add colorbar for water depth
        cbar = plt.colorbar(water_plot, ax=ax, pad=0.01, label='Water Depth (m)', shrink=0.7)

        # Add water source markers
        source_labels = []
        for i, source in enumerate(self.sources):
            x = (source['col'] + 0.5) * self.dx
            y = (self.nx - 1 - source['row'] + 0.5) * self.dy
            handle = ax.plot(x, y, 'ro', markersize=8, label=f'Source {i+1}')
            source_labels.append(handle[0])
        if source_labels:
            ax.legend(handles=source_labels, loc='upper right')

        # Add title and labels
        title = ax.set_title(f'Flood Simulation (Time: {time_steps[0]:.2f} s)')
        ax.set_xlabel("X distance (m)")
        ax.set_ylabel("Y distance (m)")
        ax.invert_yaxis()

        # Animation update function
        def update(frame_idx):
            frame_data = results[frame_idx]
            if frame_data is not None and frame_data.size > 0:
                mask_update = ~self.valid_mask | (frame_data <= self.min_depth)
                masked_water_update = np.ma.masked_where(mask_update, frame_data)
                water_plot.set_array(masked_water_update)
            else:
                water_plot.set_array(np.ma.masked_array(np.zeros_like(self.dem), mask=True)) # Empty

            title.set_text(f'Flood Simulation (Time: {time_steps[frame_idx]:.2f} s)')
            return [water_plot, title]

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(results),
            interval=150, blit=True, repeat=False
        )

        # Save animation if requested
        if output_path:
            try:
                writer_name = None
                if output_path.endswith('.gif'):
                    writer_name = 'pillow'
                elif output_path.endswith('.mp4'):
                    writer_name = 'ffmpeg'
                else:
                    print("Warning: Unknown animation format. Attempting to save as GIF.")
                    output_path = output_path.rsplit('.', 1)[0] + '.gif'
                    writer_name = 'pillow'

                if writer_name:
                    print(f"Saving animation to {output_path} (using {writer_name})... this may take a while.")
                    anim.save(output_path, writer=writer_name, fps=10, dpi=150)
                    print("Animation saved.")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Ensure required libraries (e.g., pillow, ffmpeg) are installed and accessible.")

        # Save last frame if requested
        if save_last_frame and output_path:
            try:
                # Update plot to the last valid frame before saving
                last_valid_frame_idx = len(results) - 1 
                
                if last_valid_frame_idx >= 0:
                    update(last_valid_frame_idx) 
                    last_frame_path = output_path.rsplit('.', 1)[0] + '_final_frame.png'
                    plt.savefig(last_frame_path, dpi=300, bbox_inches='tight')
                    print(f"Final frame saved to {last_frame_path}")
                else:
                    print("Could not save last frame: No valid result frames found.")
            except Exception as e:
                print(f"Error saving final frame: {e}")

        # Show animation if requested
        if show_animation:
            plt.tight_layout()
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display animation interactively: {e}")
                plt.close(fig) 
        else:
            plt.close(fig) 

        return anim
