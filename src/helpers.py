import xarray as xr
import numpy as np
from scipy import interpolate

def add_ghost_points(ds,lon_name='longitude',lon_bounds=[0,360]):
    
    """
    Add extra columns to xarray dataset at the edges of the longitude domain that are the average of the two edges of the dataset to ensure it wraps fully and NaNs don't appear at the edges when interpolating to a finer grid

    Args:
        ds: regular gridded xarray dataset with a longitude dimension that is intended to be periodic
        lon_name: string, key for the longitude variable
        lon_bounds: list of 2 floats, indicates whether range is [-180,180] or [0, 360]

    Returns:
        ds: xarray dataset with extra ghost point added at each edge of the dataset
    """
    
    if ds[lon_name][-1] != lon_bounds[-1]:
        ds = xr.concat([ds, ds.isel({lon_name: 0})], dim=lon_name)

    if ds[lon_name][0] != lon_bounds[0]:
        ds = xr.concat([ds.isel({lon_name: -1}), ds], dim=lon_name)

    for var in ds.data_vars:
        ds[var][{ds[lon_name].name: 0}] = ds[var][{ds[lon_name].name: -1}] = (
        ds[var][{ds[lon_name].name: 0}] + ds[var][{ds[lon_name].name: -1}]
        ) / 2.0
    lon = np.array(ds[lon_name])
    lon[0] = lon_bounds[0]
    lon[-1] = lon_bounds[-1]

    ds[lon_name] = lon
    return ds

