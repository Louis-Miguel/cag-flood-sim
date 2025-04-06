import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource, Normalize
from flood_tools import generate_boundary_mask 
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
        h_new, u_new, v_new = compute_step(
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
                    f"dt: {self.dt:.5f}s, "
                    f"Max Depth: {max_depth:.10f}m, "
                    f"Max Vel: {max_vel:.10f}m/s, "
                    f"Area: {flood_area:.10f}m², "
                    f"Vol: {flood_volume:.1f}m³"
                )
                if not np.isfinite(max_depth) or max_depth > 1000:
                    print(" Warning: Simulation may be unstable. Stopping.")
                    break
                if total_time > self.target_time:
                    print(f" Simulation time reached target time {self.target_time}. Stopping.")
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
        """

        if not results:
            print("No results to visualize.")
            return None

        # Increase figure size slightly for potentially large aspect ratios
        fig, ax = plt.subplots(figsize=(12, 10)) # Consider adjusting figsize based on aspect ratio?

        # Valid Mask Check (Keep for debugging if needed, comment out for speed) 
        # plt.figure("Valid Mask Check")
        # plt.imshow(self.valid_mask, interpolation='none', origin='upper') 
        # plt.title("Loaded Valid Mask")
        # plt.show(block=False)
        # 

        # Determine Color Limits for Water Depth 
        valid_maxes = []
        print("Calculating max water depth for color bar...")
        for idx, h in enumerate(results):
            if h is not None and h.size > 0 and self.valid_mask.shape == h.shape:
                if np.any(self.valid_mask): 
                    try:
                        max_val = np.nanmax(h[self.valid_mask])
                        if np.isfinite(max_val):
                            valid_maxes.append(max_val)
                    except ValueError: # 
                        print(f"Warning: Could not process frame {idx} for vmax calculation.")
                        pass 
                # If no valid mask, max depth is effectively 0, handled by vmax defaulting below

        if valid_maxes:
            vmax = max(valid_maxes)
        else:
            print("Warning: No valid finite water depths found. Using default vmax=1.0.")
            vmax = 1.0

        # Ensure vmax is slightly above min_depth for visual clarity if depths are very low
        if vmax <= self.min_depth:
             vmax = self.min_depth * 5 

        # Final check if something went wrong
        if np.isnan(vmax): 
            print("Warning: Calculated vmax is NaN. Defaulting to 1.0.")
            vmax = 1.0
        print(f"Water depth color bar max (vmax): {vmax:.3f}")
        # Define normalization for water
        norm = Normalize(vmin=0, vmax=vmax) 

        # Prepare DEM for Display 
        dem_display = self.dem.copy()
        dem_display[~self.valid_mask] = np.nan

        # Plot DEM Background 
        print("Plotting DEM background...")
        if hillshade:
            try:
                ls = LightSource(azdeg=315, altdeg=45)
                # Calculate normalization ONLY on valid DEM parts
                dem_min_valid = np.nanmin(dem_display)
                dem_max_valid = np.nanmax(dem_display)

                # Handle cases: all NaN, flat terrain, normal terrain
                if np.isnan(dem_min_valid) or np.isnan(dem_max_valid):
                    print("Warning: DEM display has no valid data for hillshade normalization.")
                    # Create a fully NaN normalized array if no valid data
                    dem_norm_val = np.full_like(dem_display, np.nan)
                elif dem_max_valid <= dem_min_valid: # Handle flat valid terrain
                    print("Warning: DEM display has flat valid terrain for hillshade.")
                    dem_norm_val = np.full_like(dem_display, 0.5) 
                    dem_norm_val[~self.valid_mask] = np.nan      
                else:
                    # Normalize valid data, NaNs remain NaN
                    dem_norm_val = (dem_display - dem_min_valid) / (dem_max_valid - dem_min_valid)

                # Fast NaN Filling for Shading 
                # Create a temporary array JUST for the shading function
                dem_for_shade = dem_norm_val 
                fill_value = 0.5 

                # Check if NaNs actually exist *before* creating the mask and filling
                nan_mask = ~np.isfinite(dem_for_shade)
                if np.any(nan_mask):
                    print(f"Filling ~{np.sum(nan_mask)} NaN values with {fill_value} for hillshading...")
                    dem_for_shade[nan_mask] = fill_value 
                

                # Perform shading 
                print("Performing hillshading calculation...")
                rgb = ls.shade(dem_for_shade, cmap=plt.cm.terrain, vert_exag=1.0, blend_mode='soft')
                print("Hillshading calculation complete.")

                # Convert shaded result (likely 0-1 float) to RGBA uint8 image
                if rgb.ndim == 3 and rgb.shape[2] >= 3:
                    # Create RGBA array, ensure correct dtype (uint8 for imshow RGBA)
                    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
                    # Convert float RGB [0,1] to uint8 [0,255]
                    rgba[:, :, :3] = np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8)
                    rgba[:, :, 3] = 255 

                    # Apply transparency using the ORIGINAL valid_mask
                    rgba[~self.valid_mask, 3] = 0 # Set alpha to 0 (transparent) outside
                else:
                    print("Warning: Hillshade output was not in expected RGB format. Skipping terrain plot.")
                    # Fallback to non-hillshade path
                    hillshade = False 

                # Display the RGBA image
                if hillshade: # Check if we are still on hillshade path
                    dem_plot = ax.imshow(rgba, extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                    origin='upper', interpolation='nearest')

            except Exception as e_shade:
                print(f"ERROR during hillshading: {e_shade}")
                print("Falling back to non-hillshaded terrain display.")
                hillshade = False 

        # Non-Hillshade Path (or fallback from failed hillshade)
        if not hillshade:
            terrain_cmap = plt.cm.terrain.copy() 
            terrain_cmap.set_bad(color='white', alpha=0) 

            terrain_vmin = np.nanmin(dem_display)
            terrain_vmax = np.nanmax(dem_display)
            if np.isnan(terrain_vmin) or np.isnan(terrain_vmax): terrain_vmin, terrain_vmax = 0, 1

            dem_plot = ax.imshow(dem_display, cmap=terrain_cmap,
                                vmin=terrain_vmin, vmax=terrain_vmax,
                                extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                                origin='upper', interpolation='nearest') 
            # Optional colorbar for elevation when not hillshading
            plt.colorbar(dem_plot, ax=ax, label='Elevation (m)', shrink=0.7)

        print("DEM background plotted.")

        # Add Contour Lines 
        if contour_levels > 0:
            print(f"Adding {contour_levels} contour lines...")
            try:
                ax.contour(dem_display, colors='black', alpha=0.3, levels=contour_levels,
                        linewidths=0.5, extent=(0, self.ny*self.dx, 0, self.nx*self.dy), origin='upper')
            except Exception as e_contour:
                print(f"Warning: Could not draw contours. {e_contour}")
            print("Contours added.")

        # Water Visualization Setup 
        print("Setting up water layer...")
        water_cmap = plt.cm.Blues.copy() 
        water_cmap.set_bad('none') 

        # Prepare the initial frame
        first_frame = results[0]
        if first_frame is not None and first_frame.size > 0:
            mask = ~self.valid_mask | (first_frame <= self.min_depth)
            masked_water = np.ma.masked_where(mask, first_frame)
        else:
            # Create an empty masked array if first frame is bad
            masked_water = np.ma.masked_array(np.zeros_like(self.dem), mask=True)

        water_plot = ax.imshow(masked_water, cmap=water_cmap, norm=norm, alpha=0.7,
                            extent=(0, self.ny*self.dx, 0, self.nx*self.dy),
                            origin='upper', interpolation='nearest') 

        # Add colorbar for water depth
        cbar = plt.colorbar(water_plot, ax=ax, pad=0.01, label='Water Depth (m)', shrink=0.7)

        # Add water source markers 
        source_labels = []
        for i, source in enumerate(self.sources):
            if self.valid_mask[source['row'], source['col']]:
                x = (source['col'] + 0.5) * self.dx
                # Correct y coordinate calculation for origin='upper'
                y = (source['row'] + 0.5) * self.dy 
                handle = ax.plot(x, y, 'ro', markersize=8, markeredgecolor='k', label=f'Source {i+1}')
                source_labels.append(handle[0])
        if source_labels:
            ax.legend(handles=source_labels, loc='upper right')

        # Add Title and Labels 
        title = ax.set_title(f'Flood Simulation (Time: {time_steps[0]:.2f} s)')
        ax.set_xlabel("X distance (m)")
        ax.set_ylabel("Y distance (m)")

        print("Initial frame plotted.")

        # Animation Update Function 
        def update(frame_idx):
            frame_data = results[frame_idx]
            if frame_data is not None and frame_data.size > 0:
                mask_update = ~self.valid_mask | (frame_data <= self.min_depth)
                masked_water_update = np.ma.masked_where(mask_update, frame_data)
                water_plot.set_array(masked_water_update)
            else:
                water_plot.set_array(np.ma.masked_array(np.zeros_like(self.dem), mask=True))

            title.set_text(f'Flood Simulation (Time: {time_steps[frame_idx]:.2f} s)')
            # Return list of artists changed for blitting
            # If blit=False, returning isn't strictly necessary but good practice
            return [water_plot, title]

        # Create Animation 
        print("Creating animation object...")
        # blit=False first for a better initial frame rendering
        use_blit = False
        anim = animation.FuncAnimation(
            fig, update, frames=len(results),
            interval=150, blit=use_blit, repeat=False
        )
        print("Animation object created.")

        # Save Animation 
        if output_path:
            writer_name = None
            if output_path.endswith('.gif'): writer_name = 'pillow'
            elif output_path.endswith('.mp4'): writer_name = 'ffmpeg'
            else: print("Warning: Unknown animation format. Defaulting to GIF."); output_path += ".gif"; writer_name = 'pillow'

            if writer_name:
                print(f"Saving animation to {output_path} (using {writer_name})... this may take a while...")
                # Reduce DPI slightly for potentially faster saving on large figs
                save_dpi = 120
                try:
                    anim.save(output_path, writer=writer_name, fps=10, dpi=save_dpi)
                    print("Animation saved.")
                except Exception as e_save:
                    print(f"ERROR saving animation: {e_save}")
                    print("Ensure required libraries (pillow for GIF, ffmpeg for MP4) are installed.")

        # Save Last Frame 
        if save_last_frame and output_path:
            last_valid_frame_idx = -1
            # Find last valid frame index (simple linear scan from end)
            for i in range(len(results) - 1, -1, -1):
                if results[i] is not None and results[i].size > 0:
                    last_valid_frame_idx = i
                    break

            if last_valid_frame_idx >= 0:
                print("Generating final frame...")
                update(last_valid_frame_idx) 
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
            plt.tight_layout()
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