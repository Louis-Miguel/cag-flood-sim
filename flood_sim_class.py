import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource
import numba  # For performance optimization

@numba.jit(nopython=True)
def compute_fluxes_numba(h, u, v, dem, dx, dy, dt, g, manning, max_velocity):
    """Optimized flux computation using Numba with improved flow routing"""
    # Output arrays
    h_new = np.zeros_like(h)
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    
    # Constants
    nx, ny = h.shape
    min_depth = 0.005  # Increased minimum depth for better stability
    max_grad = 0.5     # Reduced maximum gradient to prevent unrealistic flow
    max_flux = 50.0    # Reduced max flux for stability
    
    # Water surface elevation (terrain + water)
    eta = dem + h
    
    # Process each cell
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Compute water surface gradients in all 4 directions
            # This improves flow direction accuracy
            deta_e = eta[i, j+1] - eta[i, j]     # East
            deta_w = eta[i, j] - eta[i, j-1]     # West
            deta_n = eta[i+1, j] - eta[i, j]     # North
            deta_s = eta[i, j] - eta[i-1, j]     # South
            
            # Use the steepest descent direction for flow
            # This helps water flow naturally toward lower elevations
            detadx = 0.0
            if deta_e < 0 and abs(deta_e) > abs(deta_w):
                detadx = deta_e / dx
            elif deta_w < 0 and abs(deta_w) >= abs(deta_e):
                detadx = -deta_w / dx
                
            detady = 0.0
            if deta_n < 0 and abs(deta_n) > abs(deta_s):
                detady = deta_n / dy
            elif deta_s < 0 and abs(deta_s) >= abs(deta_n):
                detady = -deta_s / dy
            
            # Limit gradient for stability
            detadx = max(-max_grad, min(detadx, max_grad))
            detady = max(-max_grad, min(detady, max_grad))

            # Only compute velocity if there's water and a slope
            if h[i, j] > min_depth:
                # Compute momentum terms (Central Finite Difference d'x = (f(x+dx) - f(x-dx)) / 2dx)
                dudx = (u[i, j+1] - u[i, j-1]) / (2 * dx)
                dudy = (u[i+1, j] - u[i-1, j]) / (2 * dy)
                dvdx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
                dvdy = (v[i+1, j] - v[i-1, j]) / (2 * dy)
                
                # Calculate friction (Manning equation)
                h_safe = max(h[i, j], min_depth)
                cf = g * manning**2 / (h_safe**(1/3))
                cf = min(cf, 10.0)  # Limit friction coefficient
                
                vel_mag = np.sqrt(u[i, j]**2 + v[i, j]**2 + 1e-6)
                friction_u = cf * u[i, j] * vel_mag
                friction_v = cf * v[i, j] * vel_mag
                
                # Update velocities with better momentum conservation (Shallow Water Momntum Eqn)
                du_dt = -g * detadx - friction_u
                dv_dt = -g * detady - friction_v
                
                # Add inertia terms with damping for stability
                du_dt -= 0.8 * (u[i, j] * dudx + v[i, j] * dudy)
                dv_dt -= 0.8 * (u[i, j] * dvdx + v[i, j] * dvdy)
                
                # Forward Euler Method
                u_new[i, j] = u[i, j] + dt * du_dt
                v_new[i, j] = v[i, j] + dt * dv_dt
                
                # Apply velocity limits for stability
                vel_mag_new = np.sqrt(u_new[i, j]**2 + v_new[i, j]**2)
                if vel_mag_new > max_velocity:
                    u_new[i, j] = u_new[i, j] * max_velocity / vel_mag_new
                    v_new[i, j] = v_new[i, j] * max_velocity / vel_mag_new
            else:
                # No water or too shallow - zero velocity
                u_new[i, j] = 0.0
                v_new[i, j] = 0.0
    
    # Mass conservation with improved stability
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Calculate fluxes using upwind scheme for better stability
            qx_east = max(0, -u_new[i, j]) * h[i, j+1] + max(0, u_new[i, j]) * h[i, j]
            qx_west = max(0, u_new[i, j-1]) * h[i, j-1] + max(0, -u_new[i, j-1]) * h[i, j]
            
            qy_north = max(0, -v_new[i, j]) * h[i+1, j] + max(0, v_new[i, j]) * h[i, j]
            qy_south = max(0, v_new[i-1, j]) * h[i-1, j] + max(0, -v_new[i-1, j]) * h[i, j]
            
            # Limit fluxes
            qx_east = max(min(qx_east, max_flux), -max_flux)
            qx_west = max(min(qx_west, max_flux), -max_flux)
            qy_north = max(min(qy_north, max_flux), -max_flux)
            qy_south = max(min(qy_south, max_flux), -max_flux)
            
            # Calculate divergence
            dqx = (qx_east - qx_west) / dx
            dqy = (qy_north - qy_south) / dy
            
            # Update water depth (Continuity Eqn and Forward Euler)
            dh_dt = -(dqx + dqy)
            h_new[i, j] = max(h[i, j] + dt * dh_dt, 0.0)
            
            # Apply a small diffusion to prevent grid artifacts
            if i > 1 and i < nx-2 and j > 1 and j < ny-2:
                h_avg = (h[i-1, j] + h[i+1, j] + h[i, j-1] + h[i, j+1]) / 4.0
                h_new[i, j] = 0.95 * h_new[i, j] + 0.05 * h_avg
    
    return h_new, u_new, v_new


class FloodSimulation:
    def __init__(self, dem, dx=1.0, dy=1.0, g=9.81, manning=0.03, 
                 adaptive_timestep=True, stability_factor=0.15, max_velocity=5.0,
                 use_numba=True):
        """
        Initialize the improved flood simulation with better stability parameters
        
        Parameters:
        -----------
        dem : numpy.ndarray
            Digital Elevation Model (2D array of elevation values)
        dx, dy : float
            Grid cell size in x and y directions
        g : float
            Acceleration due to gravity
        manning : float
            Manning's roughness coefficient
        adaptive_timestep : bool
            Whether to use adaptive timestep to maintain stability
        stability_factor : float
            Factor to reduce timestep (0-1) for stability control (reduced from original)
        max_velocity : float
            Maximum allowed velocity (m/s) for stability control (reduced from original)
        """
        self.dem = dem
        self.dx = dx
        self.dy = dy
        self.g = g
        self.manning = manning
        self.adaptive_timestep = adaptive_timestep
        self.stability_factor = stability_factor
        self.max_velocity = max_velocity
        
        # Initialize water depth, velocities
        self.h = np.zeros_like(dem)
        self.u = np.zeros_like(dem)  # x-velocity
        self.v = np.zeros_like(dem)  # y-velocity
        
        # Calculate grid dimensions
        self.nx, self.ny = dem.shape
        
        # Compute timestep based on grid size (more conservative CFL condition)
        self.dt = 0.05 * min(dx, dy) / np.sqrt(g * np.max(dem) + 1e-6)
        self.original_dt = self.dt  # Store the original timestep for adaptive control

        # Track maximum flood extent and depth
        self.max_flood_extent = np.zeros_like(dem, dtype=bool)
        self.max_flood_depth = np.zeros_like(dem)
        
        # Performance tracking
        self.computation_times = []

    def add_water_source(self, row, col, rate, duration=None):
        """Add a water source at a specific location"""
        self.source_row = row
        self.source_col = col
        self.source_rate = rate
        self.source_duration = duration
        self.source_steps = 0
        
    def add_rainfall(self, rate, duration=None, spatial_distribution=None, time_pattern=None):
        """Add rainfall with spatial and temporal distribution"""
        self.rainfall = True
        self.base_rain_rate = rate
        self.rain_duration = duration
        self.rain_steps = 0
        
        # Set spatial distribution (default is uniform)
        if spatial_distribution is None:
            # Instead of uniform rainfall, create a smoother pattern
            self.rain_distribution = np.ones_like(self.dem)
            # Apply slight randomness to break uniformity
            noise = np.random.normal(1.0, 0.1, self.dem.shape)
            self.rain_distribution *= noise
            # Smooth the distribution
            self.rain_distribution = gaussian_filter(self.rain_distribution, sigma=2.0)
            # Normalize
            self.rain_distribution /= np.max(self.rain_distribution)
        else:
            # Ensure correct shape and smooth input distribution
            assert spatial_distribution.shape == self.dem.shape, "Spatial distribution must match DEM shape"
            self.rain_distribution = np.maximum(0, spatial_distribution)
            # Smooth the distribution
            self.rain_distribution = gaussian_filter(self.rain_distribution, sigma=2.0)
            # Normalize
            if np.max(self.rain_distribution) > 0:
                self.rain_distribution /= np.max(self.rain_distribution)
            
        # Set temporal pattern (default is constant)
        if time_pattern is None:
            self.rain_time_pattern = lambda step, duration: 1.0
        else:
            self.rain_time_pattern = time_pattern

    def compute_fluxes(self):
        """Compute fluxes using the improved shallow water equations"""
        # Adaptive timestep if enabled - more conservative approach
        if self.adaptive_timestep:
            # Compute maximum velocity
            max_vel = np.sqrt(np.nanmax(self.u**2 + self.v**2) + 1e-6)
            max_depth = np.nanmax(self.h)
            
            # Adjust timestep based on CFL condition with lower factor (Courant-Friedrichs-Lewy (CFL) condition)
            wave_speed = np.sqrt(self.g * max_depth + 1e-6)
            if wave_speed > 0:
                dt_cfl = self.stability_factor * min(self.dx, self.dy) / (wave_speed + max_vel + 1e-6)
                self.dt = min(self.original_dt, dt_cfl)
        
        # Use numba version of compute flux (faster) (need pip install numba)
        h_new, u_new, v_new = compute_fluxes_numba(
            self.h, self.u, self.v, self.dem, 
            self.dx, self.dy, self.dt, 
            self.g, self.manning, self.max_velocity
        )
        
        # Add water source if defined 
        if hasattr(self, 'source_row') and (self.source_duration is None or self.source_steps < self.source_duration):
            volume_per_step = self.source_rate * self.dt
            # Distribute source over a small area for stability
            r, c = self.source_row, self.source_col
            radius = 2
            for i in range(max(1, r-radius), min(self.nx-1, r+radius+1)):
                for j in range(max(1, c-radius), min(self.ny-1, c+radius+1)):
                    dist = np.sqrt((i-r)**2 + (j-c)**2)
                    if dist <= radius:
                        weight = (1.0 - dist/radius)**2
                        h_new[i, j] += weight * volume_per_step / (self.dx * self.dy * np.pi * radius**2)
            self.source_steps += 1

        # Add rainfallon
        if hasattr(self, 'rainfall') and (self.rain_duration is None or self.rain_steps < self.rain_duration):
            # Calculate rain intensity with temporal pattern
            time_factor = self.rain_time_pattern(self.rain_steps, self.rain_duration)
            effective_rate = self.base_rain_rate * time_factor
            
            # Apply spatial distribution with smoothness
            rainfall_volume = effective_rate * self.dt * self.rain_distribution
            
            # Add rainfall to water depth
            h_new += rainfall_volume
            
            self.rain_steps += 1
        
        # Update max flood extent and depth trackers
        self.max_flood_extent = np.logical_or(self.max_flood_extent, h_new > 0.01)
        self.max_flood_depth = np.maximum(self.max_flood_depth, h_new)

        # Update state variables
        self.h = h_new
        self.u = u_new
        self.v = v_new

    def run_simulation(self, num_steps, output_freq=10):
        """Run simulation with performance monitoring"""
        import time
        
        # Initialize results list
        results = []
        time_steps = []
        total_time = 0.0
        
        # Run simulation
        for step in range(num_steps):
            step_start_time = time.time()
            
            if step % output_freq == 0:
                # Store results
                results.append(self.h.copy())
                time_steps.append(total_time)
                
                # Report stats
                max_depth = np.nanmax(self.h)
                max_vel = np.sqrt(np.nanmax(self.u**2 + self.v**2) + 1e-6)
                print(f"Step {step}/{num_steps}, Time: {total_time:.2f}s, Max depth: {max_depth:.2f} m")
                
                # Report performance if available
                if len(self.computation_times) > 0:
                    avg_step_time = np.mean(self.computation_times[-output_freq:])
                    print(f"Avg computation time per step: {avg_step_time:.4f}s")
            
            # Compute the next step
            self.compute_fluxes()
            
            # Measure step time
            step_time = time.time() - step_start_time
            self.computation_times.append(step_time)
            
            total_time += self.dt
        
        # Add final state if not already added
        if len(results) == 0 or not np.array_equal(results[-1], self.h):
            results.append(self.h.copy())
            time_steps.append(total_time)
        
        # Print final flood statistics
        flood_area = np.sum(self.max_flood_extent) * self.dx * self.dy
        flood_volume = np.sum(self.max_flood_depth) * self.dx * self.dy
        print(f"Final maximum flood area: {flood_area:.2f} m², Maximum flood volume: {flood_volume:.2f} m³")
        
        # Performance summary
        if len(self.computation_times) > 0:
            print(f"Average computation time per step: {np.mean(self.computation_times):.4f}s")
            print(f"Total computation time: {np.sum(self.computation_times):.2f}s")
            
        return results, time_steps

    def visualize_results(self, results, time_steps, output_path=None, show_animation=True, save_last_frame=True, include_rainfall=True):
        """visualization of the flood simulation results"""
        # Create a figure with larger size
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a hillshade for better terrain visualization
        ls = LightSource(azdeg=315, altdeg=45)
        dem_norm = (self.dem - np.min(self.dem)) / (np.max(self.dem) - np.min(self.dem) + 1e-6)
        rgb = ls.shade(dem_norm, cmap=plt.cm.terrain, vert_exag=2, blend_mode='soft')
        dem_plot = ax.imshow(rgb)
        
        # Adding improved contour lines - more levels, better visibility
        contour_levels = np.linspace(np.min(self.dem), np.max(self.dem), 15)
        contour = ax.contour(self.dem, colors='black', alpha=0.3, levels=contour_levels, linewidths=0.5)
        
        # Water visualization - better colormap and transparency
        water_cmap = plt.cm.Blues_r.copy()
        water_cmap.set_bad('none')
        
        # Increase water visibility threshold for clearer display
        masked_water = np.ma.masked_where(results[0] < 0.02, results[0])
        water_plot = ax.imshow(masked_water, cmap=water_cmap, vmin=0, 
                            vmax=max(0.1, np.max([np.max(h) for h in results]) * 0.8),
                            alpha=0.8)
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(water_plot, ax=ax, pad=0.01, label='Water Depth (m)')
        cbar.ax.tick_params(labelsize=10) 
        
        # Rainfall visualization if needed
        if include_rainfall and hasattr(self, 'rainfall'):
            # Create a more natural rainfall effect
            rain_dots = ax.scatter([], [], marker='.', color='royalblue', alpha=0.4, s=2, label='Rainfall')
        
        # Add water source marker if defined - larger and more visible
        if hasattr(self, 'source_row') and hasattr(self, 'source_col'):
            ax.plot(self.source_col, self.source_row, 'ro', markersize=10, label='Water Source')
            # Add a circular marker around the source
            circle = plt.Circle((self.source_col, self.source_row), 2, color='r', fill=False, linewidth=2)
            ax.add_patch(circle)
        
        # Add legend with better formatting
        if include_rainfall and hasattr(self, 'rainfall'):
            ax.legend(loc='upper right', fontsize=10)
        
        # Add title with time information
        title = ax.set_title(f'Flood Simulation (Time: {time_steps[0]:.2f} s)', fontsize=14)
        
        # Add grid lines for reference
        ax.grid(False)
        
        def update(frame):
            # Update water depth display with better masking
            masked_water = np.ma.masked_where(results[frame] < 0.02, results[frame])
            water_plot.set_array(masked_water)
            
            # Rainfall visualization with more natural scatter
            if include_rainfall and hasattr(self, 'rainfall') and frame < (self.rain_duration or np.inf):
                # Generate random rain dot positions based on rainfall distribution
                positions = []
                intensities = []
                n_dots = 2000  
                
                # Find areas with rainfall
                y_indices, x_indices = np.where(self.rain_distribution > 0.05)
                if len(y_indices) > 0:
                    # Randomly select positions weighted by rainfall intensity
                    idx = np.random.choice(len(y_indices), size=min(n_dots, len(y_indices)), 
                                        p=self.rain_distribution[y_indices, x_indices]/np.sum(self.rain_distribution[y_indices, x_indices]),
                                        replace=True)
                    
                    
                    jitter = 0.5
                    x_with_jitter = x_indices[idx] + np.random.uniform(-jitter, jitter, size=len(idx))
                    y_with_jitter = y_indices[idx] + np.random.uniform(-jitter, jitter, size=len(idx))
                    
                    rain_dots.set_offsets(np.column_stack([x_with_jitter, y_with_jitter]))
                    
                    # Adjust dot size based on intensity
                    sizes = 2 + 3 * self.rain_distribution[y_indices[idx], x_indices[idx]]
                    rain_dots.set_sizes(sizes)
                else:
                    rain_dots.set_offsets(np.empty((0, 2)))
            else:
                rain_dots.set_offsets(np.empty((0, 2)))
            
            # Update title with current time
            title.set_text(f'Flood Simulation (Time: {time_steps[frame]:.2f} s)')
            
            return water_plot, rain_dots, title
        
        # Create animation with improved performance
        anim = animation.FuncAnimation(
            fig, update, frames=len(results), 
            interval=200, blit=True
        )
        
        # Save animation if requested with better resolution
        if output_path:
            try:
                writer = animation.PillowWriter(fps=10)
                anim.save(output_path, writer=writer, dpi=120)
                print(f"Animation saved to {output_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
                print("Trying alternative writer...")
                try:
                    writer = animation.FFMpegWriter(fps=10)
                    anim.save(output_path.replace('.gif', '.mp4'), writer=writer, dpi=120)
                    print(f"Animation saved with alternative writer")
                except:
                    print("Could not save animation. Try installing ffmpeg or pillow.")
        
        # Save last frame with high resolution if requested
        if save_last_frame and output_path:
            # Update to show the last frame
            update(len(results) - 1)
            # Generate the last frame filename
            last_frame_path = output_path.replace('.gif', '_final_frame.png')
            last_frame_path = last_frame_path.replace('.mp4', '_final_frame.png')
            
            # Save the figure with high resolution
            plt.savefig(last_frame_path, dpi=300, bbox_inches='tight')
            print(f"Final frame saved to {last_frame_path}")
        
        # Show animation if requested
        if show_animation:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
            
        return anim
