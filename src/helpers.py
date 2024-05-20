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

def add_ice_mask(ds_neurost,ds_sst):
    
    """
    Take sea ice mask from MUR SST and apply it to NeurOST SSH

    Args:
        ds_neurost: standard NeurOST xarray dataset
        ds_sst: standard MUR xarray dataset

    Returns:
        masked_ds: xarray dataset with NeurOST SSH with NaN mask applied to sea ice
    """
    
    
    ds_sst = ds_sst[['sea_ice_fraction']]
    ds_sst = ds_sst.isel(lon=slice(None,None,10), lat=slice(None,None,10))
    # ds_sst = ds_sst.coarsen({'lon':10,'lat':10},boundary='trim').mean()
    ds_sst['lon'] = ds_sst['lon']%360
    ds_sst = ds_sst.sortby('lon')
    ice_mask = xr.where(ds_sst.sea_ice_fraction > 0.01, 1, 0)
    ds_sst['ice_mask'] = ice_mask
    ds_sst = ds_sst.rename({'lon':'longitude','lat':'latitude'})
    ds_sst = ds_sst.isel(time=0).interp_like(ds_neurost.isel(time=0),method = 'nearest')
    subset = ds_sst['ice_mask'].isel(longitude=-1)
    subset[:] = ds_sst['ice_mask'].isel(longitude=-2).copy()
    masked_ds = ds_neurost.where(ds_sst['ice_mask'] != 1, np.nan)
    return masked_ds
