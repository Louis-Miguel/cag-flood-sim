import numba # Required: pip install numba
import numpy as np

@numba.jit(nopython=True, cache=True) # Numba decorator for JIT compilation (Faster Python)
def compute_step(h, u, v, dem, boundary_mask, 
                    g, manning, dx, dy, dt,
                    infiltration_rate, min_depth=0.001):
    """
    Performs one step of the shallow water simulation using RK2, Upwinding,
    and boundary conditions applied via ghost cells based on boundary_mask.
    (Numba optimized)

    Parameters:
    ----------
    h : 2D array (depth) : Current depth at each grid cell (interior only)
    u : 2D array (velocity x) : Current x-velocity at each grid cell (interior only)   
    v : 2D array (velocity y) : Current y-velocity at each grid cell (interior only)
    dem : 2D array (digital elevation model) : Elevation at each grid cell (interior only)
    boundary_mask : 2D array (boundary conditions) : Boundary condition type at each grid cell 
        Types: 0=Internal, 1=Wall, 2=Open
    g : float : Gravitational acceleration (m/s^2)
    manning : float : Manning's n (friction coefficient)
    dx : float : Grid cell size in x-direction (m)
    dy : float : Grid cell size in y-direction (m)
    dt : float : Time step size (s)
    infiltration_rate : float : Rate of infiltration (m/s)
    min_depth : float : Minimum depth for velocity calculation (m)
    """
    # Initialization of dimensions
    nx_int, ny_int = dem.shape 
    nx_pad, ny_pad = nx_int + 2, ny_int + 2 

    # Create Padded Arrays 
    h_pad = np.zeros((nx_pad, ny_pad), dtype=h.dtype)
    u_pad = np.zeros((nx_pad, ny_pad), dtype=u.dtype)
    v_pad = np.zeros((nx_pad, ny_pad), dtype=v.dtype)
    dem_pad = np.zeros((nx_pad, ny_pad), dtype=dem.dtype) # Pad DEM too
    eta_pad = np.zeros((nx_pad, ny_pad), dtype=dem.dtype)

    #  Fill Padded Arrays - Stage 1: Interior and Ghost Cell Filling (Current State

    # Copy interior values
    h_pad[1:-1, 1:-1] = h
    u_pad[1:-1, 1:-1] = u
    v_pad[1:-1, 1:-1] = v
    
    # Pad DEM using edge mode (simplest way to handle elevation near boundary)
    dem_pad[1:-1, 1:-1] = dem
    dem_pad[0, :] = dem_pad[1, :]
    dem_pad[-1, :] = dem_pad[-2, :]
    dem_pad[:, 0] = dem_pad[:, 1]    
    dem_pad[:, -1] = dem_pad[:, -2]
    dem_pad[0, 0] = dem_pad[1, 1]
    dem_pad[0, -1] = dem_pad[1, -2]
    dem_pad[-1, 0] = dem_pad[-2, 1]
    dem_pad[-1, -1] = dem_pad[-2, -2]

    # Fill ghost cells based on boundary_mask and *local orientation*
    for i in range(nx_pad):
        for j in range(ny_pad):
            # Skip if it's an interior cell (no need to fill ghost values)
            if 1 <= i < nx_pad - 1 and 1 <= j < ny_pad - 1:
                continue

            # Determine coordinates of the adjacent interior cell
            i_int_neighbor = min(max(0, i - 1), nx_int - 1) # Row index of the valid neighbor
            j_int_neighbor = min(max(0, j - 1), ny_int - 1) # Col index of the valid neighbor

            # Get the boundary condition type FROM the adjacent interior cell
            bc_type = boundary_mask[i_int_neighbor, j_int_neighbor]

            # Get state values from the adjacent interior cell
            h_neighbor = h[i_int_neighbor, j_int_neighbor]
            u_neighbor = u[i_int_neighbor, j_int_neighbor]
            v_neighbor = v[i_int_neighbor, j_int_neighbor]

            # Apply Boundary Conditions based on bc_type and RELATIVE POSITION
            if bc_type == 1: # Closed Wall
                h_pad[i, j] = h_neighbor 

                # Determine relative position to apply correct reflection
                # i_int_pad, j_int_pad are the coordinates of the neighbor in the padded array
                i_int_pad = i_int_neighbor + 1
                j_int_pad = j_int_neighbor + 1

                if i == i_int_pad and (j == j_int_pad - 1 or j == j_int_pad + 1):
                    # Reflect U (normal velocity), copy V (tangential)
                    u_pad[i, j] = -u_neighbor
                    v_pad[i, j] = v_neighbor
                elif j == j_int_pad and (i == i_int_pad - 1 or i == i_int_pad + 1):
                    # Reflect V (normal velocity), copy U (tangential)
                    u_pad[i, j] = u_neighbor
                    v_pad[i, j] = -v_neighbor
                elif abs(i - i_int_pad) == 1 and abs(j - j_int_pad) == 1:
                    # Reflect both (simple corner handling, prevents flow)
                    u_pad[i, j] = -u_neighbor
                    v_pad[i, j] = -v_neighbor
                else:
                    u_pad[i, j] = u_neighbor
                    v_pad[i, j] = v_neighbor

            elif bc_type == 2: # Open Outlet (Zero Gradient)
                h_pad[i, j] = h_neighbor
                u_pad[i, j] = u_neighbor
                v_pad[i, j] = v_neighbor

            else: 
                h_pad[i, j] = h_neighbor
                u_pad[i, j] = u_neighbor
                v_pad[i, j] = v_neighbor


    # Calculate padded eta using the filled h_pad and padded dem_pad
    eta_pad = dem_pad + h_pad

    # RK2 Step 1: Predictor
    # Calculations use the correctly filled padded arrays

    # Use interior h for safe depth
    h_safe = np.maximum(h, min_depth) 

    # Gradients (Central Difference on padded eta)
    detadx = (eta_pad[1:-1, 2:] - eta_pad[1:-1, :-2]) / (2 * dx)
    detady = (eta_pad[2:, 1:-1] - eta_pad[:-2, 1:-1]) / (2 * dy)


    # Advection (Upwind using padded u/v for gradients, interior u/v for direction)
    u_int = u
    v_int = v 

    # forward and backward differences for upwinding for u and v    
    dudx_fwd = (u_pad[1:-1, 2:] - u_pad[1:-1, 1:-1]) / dx
    dudx_bwd = (u_pad[1:-1, 1:-1] - u_pad[1:-1, :-2]) / dx
    adv_u_dudx = np.where(u_int > 0, u_int * dudx_bwd, u_int * dudx_fwd)
    
    dudy_fwd = (u_pad[2:, 1:-1] - u_pad[1:-1, 1:-1]) / dy
    dudy_bwd = (u_pad[1:-1, 1:-1] - u_pad[:-2, 1:-1]) / dy
    adv_v_dudy = np.where(v_int > 0, v_int * dudy_bwd, v_int * dudy_fwd)

    dvdx_fwd = (v_pad[1:-1, 2:] - v_pad[1:-1, 1:-1]) / dx
    dvdx_bwd = (v_pad[1:-1, 1:-1] - v_pad[1:-1, :-2]) / dx
    adv_u_dvdx = np.where(u_int > 0, u_int * dvdx_bwd, u_int * dvdx_fwd)

    dvdy_fwd = (v_pad[2:, 1:-1] - v_pad[1:-1, 1:-1]) / dy
    dvdy_bwd = (v_pad[1:-1, 1:-1] - v_pad[:-2, 1:-1]) / dy
    adv_v_dvdy = np.where(v_int > 0, v_int * dvdy_bwd, v_int * dvdy_fwd)


    # Friction (using interior h, u, v) 
    vel_mag_sq = u**2 + v**2
    vel_mag = np.sqrt(vel_mag_sq + 1e-9)
    friction_term = g * manning**2 / ((h_safe + 1e-9)**(4./3.))
    friction_u = friction_term * u * vel_mag
    friction_v = friction_term * v * vel_mag

    # Rates of Change (Interior state using interior h, u, v)
    du_dt = -adv_u_dudx - adv_v_dudy - g * detadx - friction_u
    dv_dt = -adv_u_dvdx - adv_v_dvdy - g * detady - friction_v

    # Continuity (Flux divergence using interior h, u, v for fluxes)
    qx = h * u 
    qy = h * v
    
    # Pad fluxes for continuity calculation (using padded arrays) (Same scheme as above)
    qx_pad_flux = np.zeros((nx_pad, ny_pad), dtype=qx.dtype)
    qx_pad_flux[1:-1, 1:-1] = qx
    qx_pad_flux[0, :] = qx_pad_flux[1, :]
    qx_pad_flux[-1, :] = qx_pad_flux[-2, :]
    qx_pad_flux[:, 0] = qx_pad_flux[:, 1]
    qx_pad_flux[:, -1] = qx_pad_flux[:, -2]
    
    qy_pad_flux = np.zeros((nx_pad, ny_pad), dtype=qy.dtype)
    qy_pad_flux[1:-1, 1:-1] = qy
    qy_pad_flux[0, :] = qy_pad_flux[1, :]
    qy_pad_flux[-1, :] = qy_pad_flux[-2, :]
    qy_pad_flux[:, 0] = qy_pad_flux[:, 1]
    qy_pad_flux[:, -1] = qy_pad_flux[:, -2]

    dqx_dx = (qx_pad_flux[1:-1, 2:] - qx_pad_flux[1:-1, :-2]) / (2 * dx)
    dqy_dy = (qy_pad_flux[2:, 1:-1] - qy_pad_flux[:-2, 1:-1]) / (2 * dy)
    dh_dt = -(dqx_dx + dqy_dy)

    # Runge Kutta Predictor Step (updates interior state)
    h_pred_int = h + 0.5 * dt * dh_dt
    u_pred_int = u + 0.5 * dt * du_dt
    v_pred_int = v + 0.5 * dt * dv_dt

    h_pred_int = np.maximum(0.0, h_pred_int)
    u_pred_int = np.where(h_pred_int >= min_depth, u_pred_int, 0.0)
    v_pred_int = np.where(h_pred_int >= min_depth, v_pred_int, 0.0)


    # RK2 Step 2: Corrector 
    
    # Re-fill ghost cells using the *predicted* state and re-calculate
    # Create and fill padded arrays for the predicted state
    h_pred_pad = np.zeros((nx_pad, ny_pad), dtype=h_pred_int.dtype)
    u_pred_pad = np.zeros((nx_pad, ny_pad), dtype=u_pred_int.dtype)
    v_pred_pad = np.zeros((nx_pad, ny_pad), dtype=v_pred_int.dtype)
    
    # eta_pred_pad will be calculated after filling h_pred_pad
    h_pred_pad[1:-1, 1:-1] = h_pred_int
    u_pred_pad[1:-1, 1:-1] = u_pred_int
    v_pred_pad[1:-1, 1:-1] = v_pred_int

    # Fill ghost cells for predicted state (using SAME LOCAL LOGIC as above)
    for i in range(nx_pad):
        for j in range(ny_pad):
            if 1 <= i < nx_pad - 1 and 1 <= j < ny_pad - 1: continue
            i_int_neighbor = min(max(0, i - 1), nx_int - 1)
            j_int_neighbor = min(max(0, j - 1), ny_int - 1)
            bc_type = boundary_mask[i_int_neighbor, j_int_neighbor]

            # Use PREDICTED neighbor values
            h_neighbor = h_pred_int[i_int_neighbor, j_int_neighbor]
            u_neighbor = u_pred_int[i_int_neighbor, j_int_neighbor]
            v_neighbor = v_pred_int[i_int_neighbor, j_int_neighbor]

            if bc_type == 1: #
                h_pred_pad[i, j] = h_neighbor
                i_int_pad = i_int_neighbor + 1; j_int_pad = j_int_neighbor + 1
                if i == i_int_pad and (j == j_int_pad - 1 or j == j_int_pad + 1):
                    u_pred_pad[i, j] = -u_neighbor
                    v_pred_pad[i, j] = v_neighbor 
                elif j == j_int_pad and (i == i_int_pad - 1 or i == i_int_pad + 1):
                    u_pred_pad[i, j] = u_neighbor
                    v_pred_pad[i, j] = -v_neighbor 
                elif abs(i - i_int_pad) == 1 and abs(j - j_int_pad) == 1:
                    u_pred_pad[i, j] = -u_neighbor
                    v_pred_pad[i, j] = -v_neighbor 
                else:
                    u_pred_pad[i, j] = u_neighbor
                    v_pred_pad[i, j] = v_neighbor
            elif bc_type == 2: 
                h_pred_pad[i, j] = h_neighbor
                u_pred_pad[i, j] = u_neighbor
                v_pred_pad[i, j] = v_neighbor
            else: 
                h_pred_pad[i, j] = h_neighbor
                u_pred_pad[i, j] = u_neighbor
                v_pred_pad[i, j] = v_neighbor

    # Calculate predicted eta_pad
    eta_pred_pad = dem_pad + h_pred_pad

    # Recalculate gradients, friction, rates using PREDICTED padded state 
    h_pred_safe = np.maximum(h_pred_int, min_depth) # Use interior predicted h

    # Predicted Gradients (Central Difference on padded eta_pred)
    detadx_pred = (eta_pred_pad[1:-1, 2:] - eta_pred_pad[1:-1, :-2]) / (2 * dx)
    detady_pred = (eta_pred_pad[2:, 1:-1] - eta_pred_pad[:-2, 1:-1]) / (2 * dy)

    # Predicted Advection (using interior predicted u/v for direction)
    u_pred_int_dir = u_pred_int; v_pred_int_dir = v_pred_int 

    # Same upwinding logic as above but using predicted u/v for gradients
    dudx_fwd_pred = (u_pred_pad[1:-1, 2:] - u_pred_pad[1:-1, 1:-1]) / dx
    dudx_bwd_pred = (u_pred_pad[1:-1, 1:-1] - u_pred_pad[1:-1, :-2]) / dx
    adv_u_dudx_pred = np.where(u_pred_int_dir > 0, u_pred_int_dir * dudx_bwd_pred, u_pred_int_dir * dudx_fwd_pred)

    dudy_fwd_pred = (u_pred_pad[2:, 1:-1] - u_pred_pad[1:-1, 1:-1]) / dy
    dudy_bwd_pred = (u_pred_pad[1:-1, 1:-1] - u_pred_pad[:-2, 1:-1]) / dy
    adv_v_dudy_pred = np.where(v_pred_int_dir > 0, v_pred_int_dir * dudy_bwd_pred, v_pred_int_dir * dudy_fwd_pred)

    dvdx_fwd_pred = (v_pred_pad[1:-1, 2:] - v_pred_pad[1:-1, 1:-1]) / dx
    dvdx_bwd_pred = (v_pred_pad[1:-1, 1:-1] - v_pred_pad[1:-1, :-2]) / dx
    adv_u_dvdx_pred = np.where(u_pred_int_dir > 0, u_pred_int_dir * dvdx_bwd_pred, u_pred_int_dir * dvdx_fwd_pred)

    dvdy_fwd_pred = (v_pred_pad[2:, 1:-1] - v_pred_pad[1:-1, 1:-1]) / dy
    dvdy_bwd_pred = (v_pred_pad[1:-1, 1:-1] - v_pred_pad[:-2, 1:-1]) / dy
    adv_v_dvdy_pred = np.where(v_pred_int_dir > 0, v_pred_int_dir * dvdy_bwd_pred, v_pred_int_dir * dvdy_fwd_pred)

    # Predicted Friction (using interior predicted h,u,v)
    # .
    vel_mag_sq_pred = u_pred_int**2 + v_pred_int**2; vel_mag_pred = np.sqrt(vel_mag_sq_pred + 1e-9)
    friction_term_pred = g * manning**2 / ((h_pred_safe + 1e-9)**(4./3.))
    friction_u_pred = friction_term_pred * u_pred_int * vel_mag_pred
    friction_v_pred = friction_term_pred * v_pred_int * vel_mag_pred

    # Predicted Rates
    du_dt_pred = -adv_u_dudx_pred - adv_v_dudy_pred - g * detadx_pred - friction_u_pred
    dv_dt_pred = -adv_u_dvdx_pred - adv_v_dvdy_pred - g * detady_pred - friction_v_pred

    # Predicted Continuity (Flux divergence using interior predicted h,u,v for fluxes)
    qx_pred = h_pred_int * u_pred_int
    qy_pred = h_pred_int * v_pred_int

    qx_pred_pad_flux = np.zeros((nx_pad, ny_pad), dtype=qx_pred.dtype)
    qx_pred_pad_flux[1:-1, 1:-1] = qx_pred
    qx_pred_pad_flux[0, :] = qx_pred_pad_flux[1, :]
    qx_pred_pad_flux[-1, :] = qx_pred_pad_flux[-2, :]
    qx_pred_pad_flux[:, 0] = qx_pred_pad_flux[:, 1]
    qx_pred_pad_flux[:, -1] = qx_pred_pad_flux[:, -2]

    qy_pred_pad_flux = np.zeros((nx_pad, ny_pad), dtype=qy_pred.dtype)
    qy_pred_pad_flux[1:-1, 1:-1] = qy_pred
    qy_pred_pad_flux[0, :] = qy_pred_pad_flux[1, :]
    qy_pred_pad_flux[-1, :] = qy_pred_pad_flux[-2, :]
    qy_pred_pad_flux[:, 0] = qy_pred_pad_flux[:, 1]
    qy_pred_pad_flux[:, -1] = qy_pred_pad_flux[:, -2]

    dqx_dx_pred = (qx_pred_pad_flux[1:-1, 2:] - qx_pred_pad_flux[1:-1, :-2]) / (2 * dx)
    dqy_dy_pred = (qy_pred_pad_flux[2:, 1:-1] - qy_pred_pad_flux[:-2, 1:-1]) / (2 * dy)
    dh_dt_pred = -(dqx_dx_pred + dqy_dy_pred)

    # Runge Kutta Corrector Step (Update interior state using predicted rates) 
    h_new = h + dt * dh_dt_pred
    u_new = u + dt * du_dt_pred
    v_new = v + dt * dv_dt_pred

    # Post-Timestep Adjustments (applied to interior results) 
    h_new = np.maximum(0.0, h_new)
    if infiltration_rate > 0:
        potential_infiltration = infiltration_rate * dt
        actual_infiltration = np.minimum(h_new, potential_infiltration)
        h_new -= actual_infiltration
        h_new = np.maximum(0.0, h_new)
    u_new = np.where(h_new >= min_depth, u_new, 0.0)
    v_new = np.where(h_new >= min_depth, v_new, 0.0)
    h_new = np.nan_to_num(h_new); u_new = np.nan_to_num(u_new); v_new = np.nan_to_num(v_new)

    return h_new, u_new, v_new 