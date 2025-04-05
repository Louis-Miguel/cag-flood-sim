import os
import numpy as np
from scipy.ndimage import gaussian_filter

def fill_sinks(dem, slope=0.01, max_iter=100):
    """
    Fill depressions or sinks in a Digital Elevation Model (DEM) to prevent water accumulation.

    Parameters:
    -----------
    dem : numpy.ndarray : The input DEM (2D array of elevation values)
    slope : float : A small value to add to the minimum neighbor elevation to ensure a slope
    max_iter : int : Maximum number of iterations to prevent infinite loops

    Returns:
    --------
    dem_filled : numpy.ndarray : The DEM with sinks filled
    """
    dem_filled = dem.copy()

    # Handling extreme values
    dem_filled = np.clip(dem_filled, np.percentile(dem_filled, 1), np.percentile(dem_filled, 99))
    
    # Smooth the DEM to reduce numerical instability
    dem_filled = gaussian_filter(dem_filled, sigma=1.0)
    changed = True
    iter_count = 0
    ny, nx = dem_filled.shape

    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        
        padded_dem = np.pad(dem_filled, 1, mode='edge')
        
        for i in range(ny):
            for j in range(nx):
                row, col = i + 1, j + 1 
                current_elv = padded_dem[row, col]
                min_neighbor = min(padded_dem[row - 1, col], padded_dem[row + 1, col],
                                padded_dem[row, col - 1], padded_dem[row, col + 1])
                # Sink detection
                if current_elv < min_neighbor: 
                    new_elv = min_neighbor + slope
                    padded_dem[row, col] = new_elv
                    dem_filled[i, j] = new_elv 
                    changed = True

    if iter_count == max_iter:
        print("Sink filling reached the maximum iteration threshold.")

    return dem_filled

def load_dem_from_file(file_path, preprocess=True, smooth_sigma=0.5, fill_sinks_flag=False, no_data_value=None):
    """
    Load a Digital Elevation Model from a file.
    Supports various formats including numpy arrays, GeoTIFFs, and ASCII grids.
    Includes options for preprocessing like smoothing and sink filling.
    Identifies and returns a mask of valid data areas.

    Parameters:
    -----------
    file_path : str : Path to the DEM file
    preprocess : bool : Whether to preprocess the DEM (smooth, remove artifacts)
    smooth_sigma : float : Sigma parameter for Gaussian smoothing if preprocessing is enabled
    fill_sinks_flag : bool : Whether to fill sinks in the DEM
    no_data_value : float or None : Explicitly specify the NoData value if not read from metadata (e.g., for .npy/.txt). If None, attempts to read from GeoTIFF or defaults to large negative numbers/NaN.

    Returns:
    --------
    dem : numpy.ndarray : The loaded and optionally preprocessed DEM.
    valid_mask : numpy.ndarray (bool) : Mask indicating valid data cells (True) vs. NoData/NaN cells (False).
    metadata : dict or None : Metadata extracted from GeoTIFF (transform, crs) if using rasterio, otherwise None.
    """

    # Helper function 
    def standardize_dem(dem, nodata_value):
        """
        Checks for NoData values in the DEM
        """
        if nodata_value is not None:
            valid_mask = (dem != nodata_value) & (~np.isnan(dem))
            dem[~valid_mask] = np.nan  # Standardize invalid values to NaN
        else:
            valid_mask = ~np.isnan(dem)
        return dem, valid_mask


    # Initialize return variables
    metadata = None
    valid_mask = None 
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Load based on file type

    # Numpy binary format
    if ext == '.npy':
        dem = np.load(file_path).astype(np.float64)
        dem, valid_mask = standardize_dem(dem, no_data_value)

    # GeoTIFF format
    elif ext in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                dem = src.read(1).astype(np.float64)
                metadata = {'transform': src.transform, 'crs': src.crs}
                # Get NoData value from metadata if available
                meta_nodata = src.nodata
                no_data_value = meta_nodata if meta_nodata is not None else no_data_value
                dem, valid_mask = standardize_dem(dem, no_data_value)
        except ImportError:
            metadata = None
            try:
                from PIL import Image
                img = Image.open(file_path)
                dem = np.array(img).astype(np.float64)
                dem, valid_mask = standardize_dem(dem, no_data_value)
            except Exception as e: 
                raise ImportError(f"Could not read GeoTIFF: {e}.")

    # ASCII grid format or text file
    elif ext in ['.asc', '.txt']:
        meta_nodata = None
        try:
            if ext == '.asc':
                with open(file_path, 'r') as f:
                    for _ in range(6): 
                        line = f.readline().lower()
                        if 'nodata_value' in line:
                                meta_nodata = float(line.split()[-1])
                                break
            # Use header value if found, otherwise user value, otherwise assume NaN
            no_data_value = meta_nodata if meta_nodata is not None else no_data_value
            dem = np.loadtxt(file_path, skiprows=6 if ext == '.asc' else 0).astype(np.float64)
            dem, valid_mask = standardize_dem(dem, no_data_value)
        except Exception as e: 
            raise ValueError(f"Could not load ASCII grid {file_path}: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Ensure float64
    dem = dem.astype(np.float64)
    # Ensure valid_mask exists even if all data was valid initially
    if valid_mask is None:
        valid_mask = np.ones(dem.shape, dtype=bool)

    # Prepossessing by smoothing and filling sinks
    if preprocess:
        # Smooth the DEM to reduce noise and artifacts
        if smooth_sigma > 0 and np.any(valid_mask):
            # Temporarily replace NaN with interpolation or mean for smoothing
            dem_filled_for_smooth = dem.copy()
            if np.any(~valid_mask): 
                coords = np.array(np.nonzero(valid_mask)).T
                values = dem[valid_mask]
                invalid_coords = np.array(np.nonzero(~valid_mask)).T
                if coords.size > 0 and invalid_coords.size > 0:
                    from scipy.interpolate import griddata
                    try:
                        interpolated_values = griddata(coords, values, invalid_coords, method='nearest')
                        dem_filled_for_smooth[~valid_mask] = interpolated_values
                    except Exception as e: 
                        valid_mean = np.nanmean(dem) if np.any(valid_mask) else 0
                        dem_filled_for_smooth[~valid_mask] = valid_mean if not np.isnan(valid_mean) else 0
                # Handle cases where only valid or only invalid exist
                elif np.any(valid_mask): 
                    valid_mean = np.nanmean(dem)
                    dem_filled_for_smooth[~valid_mask] = valid_mean if not np.isnan(valid_mean) else 0

            smoothed_dem = gaussian_filter(dem_filled_for_smooth, sigma=smooth_sigma)
            dem[valid_mask] = smoothed_dem[valid_mask] 

    # Sink Filling 
    if fill_sinks_flag:
        print("Applying sink filling (respecting valid mask)...")
        dem_filled_temp = dem.copy()
        fill_value_temp = np.nanmean(dem) if np.any(valid_mask) else 0
        if np.isnan(fill_value_temp): 
            fill_value_temp = 0
        dem_filled_temp[~valid_mask] = fill_value_temp # Fill NaN temporarily

        dem_filled_result = fill_sinks(dem_filled_temp) # Apply sink fill globally
    # Copy back only valid areas
        dem[valid_mask] = dem_filled_result[valid_mask] 
        print("Sink filling complete.")

    # Final check: ensure invalid areas are NaN
    dem[~valid_mask] = np.nan

    return dem, valid_mask, metadata

def generate_boundary_mask(dem, valid_mask, outlet_threshold_percentile=5.0):
    """
    Generates a boundary mask for potentially irregular shapes defined by valid_mask.
    Codes: -1: Outside/NoData, 0: Internal, 1: Wall, 2: Open Outlet.

    Parameters:
    -----------
    dem : numpy.ndarray : The input Digital Elevation Model (potentially with NaNs).
    valid_mask : numpy.ndarray (bool) : Mask == True for valid data cells, False for NoData/NaN cells.
    outlet_threshold_percentile : float : Points along the *true boundary* below this elevation percentile are considered potential outlets.

    Returns:
    --------
    boundary_mask : numpy.ndarray (int) : Integer mask (-1, 0, 1, 2).
    """
    ny, nx = dem.shape
    # Initialize mask: -1 outside, 0 inside valid area
    boundary_mask = np.full((ny, nx), -1, dtype=np.int32)
    boundary_mask[valid_mask] = 0 # Mark valid cells as Internal=0 initially

    true_boundary_indices = []
    true_boundary_elevations = []

    # Iterate through valid cells to find those adjacent to invalid cells
    rows, cols = np.where(valid_mask) # Get indices of valid cells
    for r, c in zip(rows, cols):
        is_boundary = False
        # Check 4 neighbors (N, S, E, W)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_r, neighbor_c = r + dr, c + dc
            # Check if neighbor is within array bounds
            if 0 <= neighbor_r < ny and 0 <= neighbor_c < nx:
                # If neighbor is invalid, then current valid cell (r, c) is a boundary
                if not valid_mask[neighbor_r, neighbor_c]:
                    is_boundary = True
                    break 
            else:
                 # Cells on the edge of the *array* are also boundaries if they are valid
                is_boundary = True
                break

        if is_boundary:
            boundary_mask[r, c] = 1 # Mark as Default Wall=1
            true_boundary_indices.append((r, c))
            true_boundary_elevations.append(dem[r, c])

    if not true_boundary_indices:
        print("Warning: No true boundary cells identified. Check DEM and valid_mask.")
        return boundary_mask # Return mask with only 0 and -1

    # Identify potential outlets AMONG the true boundary cells
    # Allow disabling outlet detection by setting threshold to 100% or higher
    if outlet_threshold_percentile < 100: 
        # Filter out NaN elevations just in case
        valid_boundary_elevations = [e for e in true_boundary_elevations if not np.isnan(e)]
        if not valid_boundary_elevations:
            print("Warning: All true boundary cells have NaN elevation. Cannot identify outlets.")
            return boundary_mask

        outlet_elevation_threshold = np.percentile(valid_boundary_elevations, outlet_threshold_percentile)

        # Mark low points *on the true boundary* as potential outlets (Open=2)
        for idx, elev in zip(true_boundary_indices, true_boundary_elevations):
            if not np.isnan(elev) and elev <= outlet_elevation_threshold:
                boundary_mask[idx] = 2 

    num_outlets = np.sum(boundary_mask == 2)
    num_walls = np.sum(boundary_mask == 1)
    print(f"Generated boundary mask: {num_walls} wall cells, {num_outlets} potential outlet cells identified.")
    return boundary_mask
