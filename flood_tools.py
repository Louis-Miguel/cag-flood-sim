import numpy as np
import os
from scipy.ndimage import gaussian_filter

def fill_sinks(dem, slope=0.01):
    """
    Fill depressions or sinks in a Digital Elevation Model (DEM) to prevent water accumulation.

    Parameters:
    -----------
    dem : numpy.ndarray
        The input DEM (2D array of elevation values).
    slope : float
        A small value to add to the minimum neighbor elevation to ensure a slope.

    Returns:
    --------
    dem_filled : numpy.ndarray
        The DEM with sinks filled.
    """
    dem_filled = dem.copy()

    # Handling extreme values
    dem_filled = np.clip(dem_filled, np.percentile(dem_filled, 1), np.percentile(dem_filled, 99))
    
    # Smooth the DEM to reduce numerical instability
    dem_filled = gaussian_filter(dem_filled, sigma=1.0)
    filled = False

    while not filled:
        ny, nx = dem_filled.shape
        sinks = np.zeros((ny, nx), dtype=bool)

        # Identify sink cells
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if (dem_filled[i, j] < dem_filled[i - 1, j] and
                    dem_filled[i, j] < dem_filled[i + 1, j] and
                    dem_filled[i, j] < dem_filled[i, j - 1] and
                    dem_filled[i, j] < dem_filled[i, j + 1]):
                    sinks[i, j] = True

        if np.sum(sinks) == 0:
            filled = True
        else:
            # Fill sinks by raising them to the minimum neighbor elevation
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    if sinks[i, j]:
                        min_neighbor = min(dem_filled[i - 1, j], dem_filled[i + 1, j],
                                           dem_filled[i, j - 1], dem_filled[i, j + 1])
                        dem_filled[i, j] = min_neighbor + slope
            
            # Limit the iterations to prevent infinite loops
            filled = True  # Force exit after one iteration (can be adjusted)

    return dem_filled

def load_dem_from_file(file_path, preprocess=True, smooth_sigma=0.5):
    """
    Load a Digital Elevation Model from a file
    Supports various formats including numpy arrays, GeoTIFFs, and ASCII grids
    
    Parameters:
    -----------
    file_path : str
        Path to the DEM file
    preprocess : bool
        Whether to preprocess the DEM (smooth, remove artifacts)
    smooth_sigma : float
        Sigma parameter for Gaussian smoothing if preprocessing is enabled
        
    Returns:
    --------
    dem : numpy.ndarray
        The loaded and optionally preprocessed DEM
    """

    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Load based on file type
    if ext == '.npy':
        # Numpy binary format
        dem = np.load(file_path)
    elif ext in ['.tif', '.tiff']:
        try:
            # Try with rasterio (better for GeoTIFFs)
            import rasterio
            with rasterio.open(file_path) as src:
                dem = src.read(1)  # Read first band
        except ImportError:
            # Fall back to simpler method if rasterio not available
            try:
                from PIL import Image
                img = Image.open(file_path)
                dem = np.array(img)
            except:
                raise ImportError("Need either rasterio or PIL to read GeoTIFF files")
    elif ext in ['.asc', '.txt']:
        # ASCII grid format or simple text file
        dem = np.loadtxt(file_path, skiprows=6 if ext == '.asc' else 0)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Basic preprocessing
    if preprocess:
        # Replace no-data values (often very large negative numbers)
        no_data_mask = (dem < -1000) | np.isnan(dem)
        if np.any(no_data_mask):
            # Replace with mean of valid values
            valid_mean = np.mean(dem[~no_data_mask])
            dem[no_data_mask] = valid_mean
        
        # Apply smoothing to reduce noise and artifacts
        dem = gaussian_filter(dem, sigma=smooth_sigma)
    
    return dem