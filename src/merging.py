import numpy as np
import pyinterp
from tqdm import tqdm
import xarray as xr
import pandas as pd
import os
import datetime
from scipy.spatial import cKDTree
from scipy.signal import convolve


# check if file exists and delete if it does
def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File "+file_path+" already existed, deleting")

# shift longitude coordinates to Greenwich meridian to avoid dateline issues when interpolating
def shift_to_greenwich(lon,lon0):
    lon_shifted = lon - lon0
    lon_shifted[lon_shifted<-180]+=360
    lon_shifted[lon_shifted>180]-=360
    return lon_shifted

# routine used for creating smaller local sub-grid for inserting patch into global grid
def find_closest_element(arr, target, above=True):
    if above:
        idx = np.searchsorted(arr, target, side='right')
    else:
        idx = np.searchsorted(arr, target, side='left') - 1
    
    # edge cases where index is out of bounds
    if idx == len(arr):
        return arr[-1]
    elif idx == -1:
        return arr[0]
    
    return arr[idx]

def create_kernel(L,n=128,dx=7.5e3):
    """
    Create Gaussian weighting kernel of size (n,n) centered on the middle point with decay scale L.

    Args:
        L: float, decay scale in m.
        n: int, number of pixels on grid.
        dx: float, grid spacing in m

    Returns:
        kernel: numpy ndarray (n,n) with the weights
    """
    x = np.linspace(-dx * (n-1) / 2, dx * (n-1) / 2, n)
    y = np.linspace(-dx * (n-1) / 2, dx * (n-1) / 2, n)
    xx, yy = np.meshgrid(x, y)

    r = np.sqrt(xx**2 + yy**2)

    gaussian_kernel = np.exp(-r**2 / (L**2))
    gaussian_kernel[gaussian_kernel<1e-2] = 0

    return gaussian_kernel

def numerical_derivative(data,axis,order=1,N=9,method='classic',h=7.5e3):
    """
    Calculate first order partial derivatives using wide range of finite difference stencil schemes.

    Args:
        data: numpy array of shape (samples, rows [y decreasing], columns [x increasing])
        axis: 'x' = Eastward or 'y' = Northward
        order: order of derivative, currently only 1st order supported so defaults to 1
        N: odd integer specifying size of stencil, defaults to 9
        method: specifies stencil method classic, SNR=smooth-noise robust (n=2), and SNR4=smooth-noise robust (n=4)
        h: float, grid spacing in m assumed to be uniform in x and y, defaults to 7.5km

    Returns:
        kernel: numpy ndarray (n,n) with the weights
    """
    if order==1:
        if axis == 'x':
            ax_idx = 2
            shift_sign = 1
        elif axis == 'y':
            ax_idx = 1
            shift_sign = -1
        else:
            raise Exception("axis must be 'x' or 'y'")
    else:
        raise Exception("only 1st derivatives implemented to date")
    if N%2==0:
        raise Exception("N must be odd")
        
    if N==11:
        if order==1:
            if method=='SNR4':
                coeff = np.array([-11,-32,39,256,322,0,-322,-256,-39,32,11])
                denom = 1536*h
    if N==9:
        if order==1:
            if method=='SNR4':
                coeff = np.array([-2,-1,16,27,0,-27,-16,1,2])
                denom = 96*h
            elif method=='SNR2':
                coeff = np.array([1,6,14,14,0,-14,-14,-6,-1])
                denom = 128*h
            else:
                coeff = np.array([-3,32,-168,672,0,-672,168,-32,3])
                denom = 840*h
    if N==7:
        if order==1:
            if method=='SNR4':
                coeff = np.array([-5,12,39,0,-39,-12,5])
                denom = 96*h
            elif method=='SNR2':
                coeff = np.array([1,4,5,0,-5,-4,-1])
                denom = 32*h
            else:
                coeff = np.array([1,-9,45,0,-45,9,-1])
                denom = 60*h
    if N==5:
        if order==1:
            if method=='classic':
                coeff = np.array([-1,8,0,-8,1])
                denom = 12*h
    if N==3:
        if order==1:
            if method=='classic':
                coeff=np.array([1,0,-1])
                denom = 2*h
        
    aux = np.zeros((data.shape[0],N,data.shape[1],data.shape[2]))
    if order==1:
        for i in range(N):
            shift_val = int(i - (N-1)/2)
            aux[:,i,:,:] = coeff[i]*np.roll(data,shift = shift_sign*shift_val, axis = ax_idx)/denom
        return np.sum(aux,axis=1)
    
    
def numerical_derivative_conv(data, axis, order=1, N=9, method='classic', h=7.5e3):
    if order != 1:
        raise Exception("Only 1st derivatives implemented to date")
    if axis not in ['x', 'y']:
        raise Exception("axis must be 'x' or 'y'")
    if N % 2 == 0:
        raise Exception("N must be odd")

    ax_idx = 1 if axis == 'y' else 0
    shift_sign = 1 if axis == 'x' else -1

    if N == 9:
        if method == 'SNR4':
            coeff = np.array([-2, -1, 16, 27, 0, -27, -16, 1, 2])
            denom = 96 * h

        kernel = coeff.reshape(1, -1) if axis == 'x' else -coeff.reshape(-1, 1)
        kernel = np.expand_dims(kernel,axis=0)/denom
        
        result = convolve(data, kernel, mode='same')

        return result


def bilinear_interpolation(lat_data, lon_data, values, lat_regular, lon_regular):
    latlon_data = np.column_stack((lat_data, lon_data))
    latlon_regular = np.dstack(np.meshgrid(lat_regular, lon_regular)).reshape(-1, 2)
    
    # KD-Tree for efficient nearest-neighbour search
    tree = cKDTree(latlon_data)
    distances, indices = tree.query(latlon_regular, k=4)  # Find 4 nearest neighbors
    
    # weights for bilinear interpolation
    weights = 1.0 / distances**2
    weights/=np.sum(weights, axis=1, keepdims=True)
    
    # bilinear interpolation
    interpolated_values = np.sum(values[indices]*weights, axis=1)
    
    return interpolated_values.reshape(lon_regular.shape[0],lat_regular.shape[0])

def merge_maps(data, kernel, lon_min = -180,lon_max = 180, lat_min = -70, lat_max = 80, res = 1/10, progress = True):
    """
    Use kernel weighted averaging to piece together all the nxn local patch reconstructions onto a regular global lat/lon grid using inverse-distance weighted interpolation.

    Args:
        data: numpy ndarray of shape (patches, n, n, 3), where the final axis corresponds to (lon, lat, SSH prediction).
        kernel: numpy ndarray of shape (n,n) defining the weighting kernel applied to each prediction.
        res: resolution in degrees of the target global lat/lon grid
        progress: Boolean shows progress bar if True

    Returns:
        ssh: numpy ndarray (m,n) giving the merged global SSH map.
        lon: numpy ndarray (m,n) giving the longitude coordinates [lon_min,lon_max-res].
        lat: numpy ndarray (m,n) giving the latitude coordinates [lat_min,lat_max].
    """
    n_regions = data.shape[0]
    n_vars = data.shape[-1]-2
    
    # progress bar using tqdm
    if progress:
        progress_bar = tqdm(total=n_regions, ncols=80)
    
    x0, x1 = lon_min, lon_max
    y0, y1 = lat_min, lat_max
    x_lin = np.arange(x0, x1, res)
    y_lin = np.arange(y0, y1, res)
    mx, my = np.meshgrid(np.arange(x0, x1, res),
                            np.arange(y0, y1, res),
                            indexing='ij')
    pred_sum = np.zeros((mx.shape[0],mx.shape[1],n_vars))
    kernel_sum = np.zeros((mx.shape[0],mx.shape[1]))
    for r in range(n_regions):
        if progress:
            progress_bar.update(1)
        # shift the coordinates to the Greenwich meridian to do interpolation to avoid wrapping around dateline
        lon = data[r,:,:,0]
        lat = data[r,:,:,1]
        lon0 = find_closest_element(x_lin,0.25*(lon[lon.shape[0]//2-1,lon.shape[1]//2-1]+lon[lon.shape[0]//2-1,lon.shape[1]//2]+lon[lon.shape[0]//2,lon.shape[1]//2-1]+lon[lon.shape[0]//2,lon.shape[1]//2]),above=False)

        lon = shift_to_greenwich(lon,lon0)
        lon_grid_min = find_closest_element(x_lin,np.min(lon),above=False)
        lon_grid_max = find_closest_element(x_lin,np.max(lon),above=True)
        lat_grid_min = find_closest_element(y_lin,np.min(lat),above=False)
        lat_grid_max = find_closest_element(y_lin,np.max(lat),above=True)
        if lat_grid_max>lat_max:
            lat_grid_max = lat_max
        if lat_grid_min<lat_min:
            lat_grid_min = lat_min
        lon_grid,lat_grid = np.meshgrid(np.arange(lon_grid_min, lon_grid_max, res),
                            np.arange(lat_grid_min, lat_grid_max, res),
                            indexing='ij')
        
        
        kernel_interp = bilinear_interpolation(lat.ravel(), lon.ravel(), kernel.ravel(), np.arange(lat_grid_min, lat_grid_max, res), np.arange(lon_grid_min, lon_grid_max, res))
        
        pred_interp = []
        
        for var in range(n_vars):
            var_interp = bilinear_interpolation(lat.ravel(), lon.ravel(), data[r,:,:,var + 2].ravel(), np.arange(lat_grid_min, lat_grid_max, res), np.arange(lon_grid_min, lon_grid_max, res))
            var_interp[np.isnan(var_interp)] = 0
            pred_interp.append(var_interp.ravel())
    
        
        
        # shift the coordinates back:
        lon_grid += lon0
        lon_idx = (np.round((lon_grid-lon_min)/res)%mx.shape[0]).astype('int').ravel()
        lat_idx = (np.round((lat_grid-lat_min)/res)%mx.shape[1]).astype('int').ravel()
        indices = np.stack((lon_idx,lat_idx),axis=-1)
        
        kernel_interp = kernel_interp.ravel()
        kernel_interp[np.isnan(kernel_interp)] = 0
        
        kernel_sum[indices[:, 0], indices[:, 1]] += kernel_interp
        
        for var in range(n_vars):
            for idx in range(indices.shape[0]):
                pred_sum[indices[idx, 0], indices[idx, 1], var] += pred_interp[var][idx]*kernel_interp[idx]
        
    if progress:
        progress_bar.close()
    return pred_sum/np.expand_dims(kernel_sum,axis=-1), mx, my




def map_to_xarray(sla, lon, lat, date, ds_mask, ds_dist, ds_mdt, with_grads = False, dsla_dy = None, dsla_dx = None, d2sla_dx2 = None, d2sla_dy2 = None, d2sla_dxy = None, mask_coast_dist=10, network_name = 'SimVP_SSH'):
    """
    Takes global SLA map numpy array and create nicely formatted xarray dataset.

    Args:
        sla: numpy ndarray of shape (m,n), SLA prediction.
        lon: numpy ndarray of shape (m,n), lon coordinates in range [-180,180].
        lat: numpy ndarray of shape (m,n), lat coordinates in range [lat_min,lat_max].
        date: datetime.datetime.date object, day of prediction.
        with_grads: Boolean, up to 2nd order spatial SLA derivatives also included as variables if True.
        dsla_dx, etc.: numpy ndarray (m,n), derivatives of SLA predictions.
        mdt_ds: xarray dataset, standard CMEMS MDT product
        ds_mask: xarray dataset, land mask from 2023a_SSH_mapping_OSE data challenge with coords renamed and shifted to longitude in [0,360]
        ds_dist: xarray dataset, distance to nearest coast from 2023a_SSH_mapping_OSE data challenge with coords renamed and shifted to longitude in [0,360]
        network_name: string, name of NN method used to produce the predictions
        

    Returns:
        ds: xarray dataset with appropiate variables on lon grid [0,360] with coords named 'longitude' and 'latitude'.
    """

    lon_da = lon[:,0]
    lat_da = lat[0,:]
    time = pd.date_range(str(date),periods=1)
    
    if with_grads==False:
        da = xr.DataArray(data=np.expand_dims(np.swapaxes(sla,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da.attrs['long_name'] = 'Sea Level Anomaly'
        da.attrs['units'] = 'm'
        da.attrs['description'] = 'SLA mapped using '+ network_name
        ds_mdt = ds_mdt.interp_like(da, method = 'linear')
        da_mdt = ds_mdt['mdt']
        da_adt = da_mdt + da
        da_adt.attrs['long_name'] = 'Absolute Dynamic Topography'
        da_adt.attrs['units'] = 'm'
        da_adt.attrs['description'] = 'ADT from CNES/CLS MDT + SLA mapped using '+ network_name
        ds = xr.Dataset({'sla':da, 'adt':da_adt}) 
        ds['longitude'] = np.mod(ds['longitude'],360) # convert to 0-360 for easier comparison to other maps
        ds = ds.sortby('longitude')
        ds['latitude'] = ds['latitude'].assign_attrs({'units':'degrees_north','_CoordinateAxisType':'Lat'})
        ds['longitude'] = ds['longitude'].assign_attrs({'units':'degrees_east','_CoordinateAxisType':'Lon'})
    
    else:
        
        da = xr.DataArray(data=np.expand_dims(np.swapaxes(sla,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da.attrs['long_name'] = 'Sea Level Anomaly'
        da.attrs['units'] = 'm'
        da.attrs['description'] = 'SLA mapped using '+ network_name
        
        da_dx = xr.DataArray(data=np.expand_dims(np.swapaxes(dsla_dx,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da_dx.attrs['long_name'] = 'dSLA/dx'
        da_dx.attrs['units'] = ''
        da_dx.attrs['description'] = 'dSLA/dx mapped using '+ network_name
        
        
        da_dy = xr.DataArray(data=np.expand_dims(np.swapaxes(dsla_dy,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da_dy.attrs['long_name'] = 'dSLA/dy'
        da_dy.attrs['units'] = ''
        da_dy.attrs['description'] = 'dSLA/dy mapped using '+ network_name
        
        da_dx2 = xr.DataArray(data=np.expand_dims(np.swapaxes(d2sla_dx2,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da_dx2.attrs['long_name'] = 'd2SLA/dx2'
        da_dx2.attrs['units'] = '-1/m'
        da_dx2.attrs['description'] = 'd2SLA/dx2 mapped using '+ network_name
        
        da_dy2 = xr.DataArray(data=np.expand_dims(np.swapaxes(d2sla_dy2,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da_dy2.attrs['long_name'] = 'd2SLA/dy2'
        da_dy2.attrs['units'] = '-1/m'
        da_dy2.attrs['description'] = 'd2SLA/dy2 mapped using '+ network_name
        
        da_dxy = xr.DataArray(data=np.expand_dims(np.swapaxes(d2sla_dxy,0,1),-1),
                           dims=["latitude", "longitude", "time"],
                           coords=dict(longitude=("longitude", lon_da), latitude=("latitude", lat_da), time=("time", time)))
        da_dxy.attrs['long_name'] = 'd2SLA/dxy'
        da_dxy.attrs['units'] = '-1/m'
        da_dxy.attrs['description'] = 'd2SLA/dxy mapped using '+ network_name
            
        ds_mdt = ds_mdt.interp_like(da, method = 'linear')
        da_mdt = ds_mdt['mdt']
        da_adt = da_mdt + da
        da_adt.attrs['long_name'] = 'Absolute Dynamic Topography'
        da_adt.attrs['units'] = 'm'
        da_adt.attrs['description'] = 'ADT from CNES/CLS MDT + SLA mapped using '+ network_name
        ds = xr.Dataset({'sla':da, 'adt':da_adt, 'dSLA_dx':da_dx, 'dSLA_dy':da_dy, 'd2SLA_dx2':da_dx2, 'd2SLA_dy2':da_dy2, 'd2SLA_dxy':da_dxy}) 
        ds['longitude'] = np.mod(ds['longitude'],360) # convert to 0-360 for easier comparison to other maps
        ds = ds.sortby('longitude')
        ds['latitude'] = ds['latitude'].assign_attrs({'units':'degrees_north','_CoordinateAxisType':'Lat'})
        ds['longitude'] = ds['longitude'].assign_attrs({'units':'degrees_east','_CoordinateAxisType':'Lon'})

    ds_dist = ds_dist.interp(latitude=ds.latitude, longitude=ds.longitude, method='nearest')
    ds_mask = ds_mask.interp(latitude=ds.latitude, longitude=ds.longitude, method='nearest')
    
    ds = ds.where(ds_mask['mask'] == 0, np.nan) # mask land points
    ds = ds.where(ds_dist['distance'] > mask_coast_dist, np.nan) # mask points close to coastlines
    
    return ds

def merge_maps_and_save(pred_dir, pred_file_pattern, pred_date, output_nc_dir, mask_filename, dist_filename, mdt_filename, network_name, available_regions, L=200e3, crop_pixels=4, dx=7.5e3, with_grads=False, mask_coast_dist=10, lon_min=-180 ,lon_max=180, lat_min=-70, lat_max=80, res=1/10, progress=True):
    
    # add doc string
    
    print(f'Mapping {pred_date}')
    # load and re-format the land_mask netcdf
    ds_mask = xr.open_dataset(mask_filename)
    # ds_mask = ds_mask.rename({'lon': 'longitude', 'lat': 'latitude'})
    # ds_mask = ds_mask.assign_coords(longitude=(ds_mask.longitude % 360)).roll(longitude=(ds_mask.dims['longitude'] // 2), roll_coords=True)
    
    # load and re-format the coast distance netcdf
    ds_dist = xr.open_dataset(dist_filename)
    # ds_dist = ds_dist.rename({'lon': 'longitude', 'lat': 'latitude'})
    # ds_dist = ds_dist.assign_coords(longitude=(ds_dist.longitude % 360)).roll(longitude=(ds_dist.dims['longitude'] // 2), roll_coords=True)
    
    ds_mdt = xr.open_dataset(mdt_filename).isel(time=0)
    
    if with_grads:
        data = np.load(pred_dir+pred_file_pattern+str(pred_date)+'.npy')
        deta_dx = numerical_derivative_conv(data,axis='x',order=1,N=9,method='SNR4',h=7.5e3)
        deta_dy = numerical_derivative_conv(data,axis='y',order=1,N=9,method='SNR4',h=7.5e3)
        d2eta_dx2 = numerical_derivative_conv(deta_dx,axis='x',order=1,N=9,method='SNR4',h=7.5e3)
        d2eta_dy2 = numerical_derivative_conv(deta_dy,axis='y',order=1,N=9,method='SNR4',h=7.5e3)
        d2eta_dxy = numerical_derivative_conv(deta_dx,axis='y',order=1,N=9,method='SNR4',h=7.5e3)
        
        coords = np.load(pred_dir+'coord_grids.npy')
        coords_new = np.zeros((np.size(available_regions),128,128,2))
        for i, r in enumerate(available_regions):
            coords_new[i,] = coords[r,]
        coords = coords_new.copy()
        data = np.stack((data,deta_dx,deta_dy,d2eta_dx2,d2eta_dy2,d2eta_dxy),axis=-1)
        data = np.concatenate((coords,data),axis=-1)
    else:
        data = np.expand_dims(np.load(pred_dir+pred_file_pattern+str(pred_date)+'.npy'),axis=-1)
        coords = np.load(pred_dir+'coord_grids.npy')
        coords_new = np.zeros((np.size(available_regions),128,128,2))
        for i, r in enumerate(available_regions):
            coords_new[i,] = coords[r,]
        coords = coords_new.copy()
        data = np.concatenate((coords,data),axis=-1)
        
        
    if crop_pixels!=0:
        data = data[:,crop_pixels:-crop_pixels,crop_pixels:-crop_pixels,:]
        
    n = data.shape[1]
    kernel = create_kernel(L,n,dx)
    
    if (with_grads==False):
        sla, lon, lat = merge_maps(data, kernel, lon_min, lon_max, lat_min, lat_max, res, progress)
        sla = np.reshape(sla,(sla.shape[0],sla.shape[1]))
        ds = map_to_xarray(sla, lon, lat, pred_date, ds_mask, ds_dist, ds_mdt, with_grads=with_grads, mask_coast_dist=mask_coast_dist, network_name = network_name)
        date_today = datetime.date.today()
        save_path = output_nc_dir + network_name + f'_L{int(L/1e3)}km' + '_mappedSLA_' + str(pred_date).replace('-','') + '_' + str(date_today).replace('-','') + '.nc'
        remove_file(save_path)
        ds.to_netcdf(save_path)
    else:
        interps, lon, lat = merge_maps(data, kernel, lon_min, lon_max, lat_min, lat_max, res, progress)
        ds = map_to_xarray(interps[:,:,0], lon, lat, pred_date, ds_mask, ds_dist, ds_mdt, with_grads=with_grads, dsla_dx = interps[:,:,1], dsla_dy = interps[:,:,2], d2sla_dx2 = interps[:,:,3], d2sla_dy2 = interps[:,:,4], d2sla_dxy = interps[:,:,5], mask_coast_dist=mask_coast_dist, network_name = network_name)
        date_today = datetime.date.today()
        save_path = output_nc_dir + network_name + f'_L{int(L/1e3)}km' + '_mappedSLA_' + str(pred_date).replace('-','') + '_' + str(date_today).replace('-','') + '.nc'
        remove_file(save_path)
        ds.to_netcdf(save_path)
        