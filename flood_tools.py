import os
import heapq
import numpy as np
from scipy.ndimage import gaussian_filter

def fill_sinks(dem, slope=0.001):
    """
    Fills sinks in a Digital Elevation Model using the Priority-Flood algorithm.
    Handles NaN values as NoData/boundaries.
    Based on the algorithm described by Wang & Liu (2006) and Barnes et al. (2014).

    Parameters:
    -----------
    dem : numpy.ndarray : The input DEM (2D array). NaN values are treated as NoData.
    slope : float : A small value to add to the minimum neighbor elevation to ensure a slope.

    Returns:
    --------
    dem_filled : numpy.ndarray : The DEM with sinks filled. NaN values remain NaN.
    """
    
    print("Starting Priority-Flood sink filling...")
    ny, nx = dem.shape
    # Initialize output DEM with infinity where valid, NaN where invalid
    dem_filled = np.full_like(dem, np.inf, dtype=dem.dtype)
    is_nan = np.isnan(dem)
    dem_filled[is_nan] = np.nan

    # Keep track of cells whose final elevation is determined
    processed = np.zeros_like(dem, dtype=bool)
    # Don't process NaN cells
    processed[is_nan] = True 

    # Priority queue (min-heap). Stores tuples of (elevation, row, col)
    pq = []

    # Initialize queue with boundary cells:
    # 1. Cells on the actual edges of the array IF they are not NaN
    # 2. Valid cells that are adjacent to a NaN cell
    print("Initializing priority queue with boundary cells...")
    init_count = 0
    for r in range(ny):
        for c in range(nx):
            # Skip NaN cells
            if processed[r, c]: 
                continue

            is_boundary = False
            # Check array edges
            if r == 0 or r == ny - 1 or c == 0 or c == nx - 1:
                is_boundary = True
            else:
                # Check neighbors for NaN
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    # Bounds check isn't strictly needed here as we are checking neighbors of interior cells
                    if is_nan[r + dr, c + dc]:
                        # Found a NaN neighbor
                        is_boundary = True
                        break 

            if is_boundary:
                elev_orig = dem[r, c]
                # Set initial filled elevation
                dem_filled[r, c] = elev_orig + slope
                heapq.heappush(pq, (elev_orig, r, c))
                init_count += 1

    print(f"Initialized queue with {init_count} boundary cells.")
    print("Processing cells...")
    processed_count = 0
    total_valid_cells = np.sum(~is_nan)

    # Main processing loop
    while pq:
        elev_c, r, c = heapq.heappop(pq)

        # Check if already processed (due to potential duplicates in PQ with different priorities)
        if processed[r, c]:
            continue

        # Mark as processed - its elevation is now final
        processed[r, c] = True
        processed_count += 1
        # Print progress every 5% of valid cells processed
        if processed_count % (total_valid_cells // 20 + 1) == 0: 
            print(f"  Processed {processed_count}/{total_valid_cells} ({processed_count * 100.0 / total_valid_cells:.1f}%)")

        # Process neighbors (N, S, E, W)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_r, neighbor_c = r + dr, c + dc

            # Check bounds and if neighbor is already processed (or NaN)
            if 0 <= neighbor_r < ny and 0 <= neighbor_c < nx and not processed[neighbor_r, neighbor_c]:
                elev_n_orig = dem[neighbor_r, neighbor_c] # Original elevation of neighbor

                # Determine the elevation to spill into the neighbor
                # It must be at least the current cell's *filled* elevation, and also at least the neighbor's *original* elevation.
                elev_n_new = max(elev_c, elev_n_orig)

                # Update neighbor's elevation in the output grid
                # The check 'elev_n_new < dem_filled[neighbor_r, neighbor_c]' ensures we only update if we 
                # found a lower "spill elevation" path, and handles the initial np.inf state.
                if elev_n_new < dem_filled[neighbor_r, neighbor_c]:
                    dem_filled[neighbor_r, neighbor_c] = elev_n_new + slope
                    # Push the neighbor onto the queue with its new potential filled elevation as the priority.
                    heapq.heappush(pq, (elev_n_new, neighbor_r, neighbor_c))

    unprocessed_count = total_valid_cells - processed_count
    if unprocessed_count > 0:
         print(f"Warning: {unprocessed_count} valid cells were not reached. This might indicate disconnected regions in the DEM.")

    print("Priority-Flood sink filling complete.")

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
                valid_mean = np.nanmean(dem[valid_mask]) if np.any(valid_mask) else 0
                fill_value = valid_mean if not np.isnan(valid_mean) else 0
                dem_filled_for_smooth[~valid_mask] = fill_value

                # Scrapped for being too slow (Keep for reference)
                # coords = np.array(np.nonzero(valid_mask)).T
                # values = dem[valid_mask]
                # invalid_coords = np.array(np.nonzero(~valid_mask)).T
                # if coords.size > 0 and invalid_coords.size > 0:
                #     from scipy.interpolate import griddata
                #     try:
                #         interpolated_values = griddata(coords, values, invalid_coords, method='nearest')
                #         dem_filled_for_smooth[~valid_mask] = interpolated_values
                #     except Exception as e: 
                #         valid_mean = np.nanmean(dem) if np.any(valid_mask) else 0
                #         dem_filled_for_smooth[~valid_mask] = valid_mean if not np.isnan(valid_mean) else 0
                # # Handle cases where only valid or only invalid exist
                # elif np.any(valid_mask): 
                # valid_mean = np.nanmean(dem)
                # dem_filled_for_smooth[~valid_mask] = valid_mean if not np.isnan(valid_mean) else 0


            print("Applying Gaussian smoothing...")
            smoothed_dem = gaussian_filter(dem_filled_for_smooth, sigma=smooth_sigma)
            dem[valid_mask] = smoothed_dem[valid_mask] 
            print("Smoothing complete.")

    # Sink Filling 
    if fill_sinks_flag:
        print("Applying Priority-Flood sink filling...")
        dem = fill_sinks(dem, slope=0.001) 
        print("Sink filling complete.")

    # Ensure invalid areas are NaN
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

if __name__ == "__main__":
    # Example usage of the functions
    dem_file = './Cagayan_Valley_ESPG4326.tif'  # Replace with your DEM file path
    dem, valid_mask, metadata = load_dem_from_file(dem_file, preprocess=True , fill_sinks_flag=False)
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 10))
    plt.imshow(dem, cmap='terrain')      
    plt.colorbar(label='Elevation (m)')
    plt.title('Filled DEM')
    plt.show()


    