import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
#import matplotlib.path as mpltPath
import xarray as xr 
#import time
from datetime import date, timedelta
import os
#import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
#from random import shuffle
#import copy

# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )

# convert ECEF coords to lat,lon:
def xyz2ll(x,y,z, lat_org, lon_org, alt_org, transformer1, transformer2):

    # transform origin of local tangent plane to ECEF coordinates (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
    ecef_org=np.array([[x_org,y_org,z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)

    # transform ENU coords to ECEF by rotating
    ecefDelta = rotMatrix.T.dot(np.stack([x,y,np.zeros_like(x)],axis=-1).T)
    # add offset of all corrds on tangent plane to get all points in ECEF
    ecef = ecefDelta+ecef_org
    # transform to geodetic coordinates
    lon, lat, alt = transformer2.transform( ecef[0,:],ecef[1,:],ecef[2,:],radians=False)
    # only return lat, lon since we're interested in points on Earth. 
    # N.B. this amounts to doing an inverse stereographic projection from ENU to lat, lon so shouldn't be used to directly back calculate lat, lon from tangent plane coords
    # this is instead achieved by binning the data's lat/long variables onto the grid in the same way as is done for the variable of interest
    return lat, lon


# convert lat, lon to ECEF coords
def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer):

    # transform geodetic coords to ECEF (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x, y, z = transformer.transform( lon,lat, np.zeros_like(lon),radians=False)
    x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
    # define position of all points relative to origin of local tangent plane
    vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)    

    # rotate ECEF coordinates to ENU
    enu = rotMatrix.dot(vec)
    X = enu.T[0,:,0]
    Y = enu.T[0,:,1]
    Z = enu.T[0,:,2]
    return X, Y, Z

# generate coords for points on a square with prescribed side length and center
def box(x_bounds, y_bounds, refinement=100):
    xs = np.zeros(int(4*refinement))
    ys = np.zeros(int(4*refinement))
    
    xs[:refinement] = np.linspace(x_bounds[0], x_bounds[-1], num=refinement)
    ys[:refinement] = np.linspace(y_bounds[0], y_bounds[0], num=refinement)
                
    xs[refinement:2*refinement] = np.linspace(x_bounds[-1], x_bounds[-1], num=refinement)
    ys[refinement:2*refinement] = np.linspace(y_bounds[0], y_bounds[-1], num=refinement)
                
    xs[2*refinement:3*refinement] = np.linspace(x_bounds[-1], x_bounds[0], num=refinement)
    ys[2*refinement:3*refinement] = np.linspace(y_bounds[-1], y_bounds[-1], num=refinement)
    
    xs[3*refinement:] = np.linspace(x_bounds[0], x_bounds[0], num=refinement)
    ys[3*refinement:] = np.linspace(y_bounds[-1], y_bounds[0], num=refinement)
    
    return xs, ys

# # bin average high res MUR L4 SST (MW+IR observations)
# def grid_sst_hr(data_sst_hr, n_t, n, L_x, L_y, lon0, lat0, coord_grid):
#     ds = data_sst_hr
    
#     lon_grid = coord_grid[:,:,0].flatten()
#     lat_grid = coord_grid[:,:,1].flatten()
#     lat_max = np.max(lat_grid)+0.1
#     lat_min = np.min(lat_grid)-0.1

#     if ((np.size(lon_grid[lon_grid>175])>0) and (np.size(lon_grid[lon_grid<-175])>0)):
#         long_max_unshifted = np.max(lon_grid[lon_grid<0])+0.1
#         long_min_unshifted = np.min(lon_grid[lon_grid>0])-0.1
#     else:
#         long_max_unshifted = np.max(lon_grid)+0.1
#         long_min_unshifted = np.min(lon_grid)-0.1

#     if long_max_unshifted>long_min_unshifted:
#         ds = ds.isel(lon = (ds.lon < long_max_unshifted) & (ds.lon > long_min_unshifted),drop = True)
#     else:
#         ds = ds.isel(lon = (ds.lon < long_max_unshifted) | (ds.lon > long_min_unshifted),drop = True)
#     ds = ds.sel(lat=slice(lat_min,lat_max), drop = True)
#     ds = ds.load()
#     ds['lon'] = (ds['lon']-lon0+180)%360-180

#     lon = np.array(ds['lon'])
#     lat = np.array(ds['lat'])
#     lon, lat = np.meshgrid(lon, lat)

#     lon = lon.flatten()
#     lat = lat.flatten()
#     sst_list = []
#     for t in range(n_t):
#         sst = np.array(ds['analysed_sst'].isel(time=t)).ravel()
#         sst[np.isnan(sst)] = 0
#         sst_list.append(sst)
    
#     # calculate ENU coords of data on tangent plane
#     x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)
#     sst_grids, _,_,_ = stats.binned_statistic_2d(x, y, sst_list, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
#     for i,sst_grid in enumerate(sst_grids):
#         sst_grid = np.rot90(sst_grid)
#         sst_grid[sst_grid<270] = 0
#         sst_grids[i] = sst_grid
    
#     return sst_grids

def grid_sst_hr(ds, n_t, n, L_x, L_y, lon0, lat0, coord_grid):
    
    lon_grid = coord_grid[:,:,0].flatten()
    lat_grid = coord_grid[:,:,1].flatten()
    lat_max = np.max(lat_grid)+0.1
    lat_min = np.min(lat_grid)-0.1

    
    ds = ds.sel(lat=slice(lat_min,lat_max), drop = True)
    
    if ((np.size(lon_grid[lon_grid>175])>0) and (np.size(lon_grid[lon_grid<-175])>0)):
        long_max_unshifted = np.max(lon_grid[lon_grid<0]) + 0.1
        long_min_unshifted = np.min(lon_grid[lon_grid>0]) - 0.1
        
    else:
        long_max_unshifted = np.max(lon_grid) + 0.1
        long_min_unshifted = np.min(lon_grid) - 0.1
 
    if long_max_unshifted>long_min_unshifted:
        ds = ds.isel(lon = (ds.lon < long_max_unshifted) & (ds.lon > long_min_unshifted),drop = True)
    else:
        ds1 = ds.isel(lon = (ds.lon < long_max_unshifted),drop=True)
        ds2 = ds.isel(lon = (ds.lon > long_min_unshifted),drop=True)
        ds = xr.concat([ds1,ds2],'lon')
        # ds = ds.isel(lon = (ds.lon < long_max_unshifted) | (ds.lon > long_min_unshifted),drop = True)
    
    ds = ds.load()

    ds['lon'] = (ds['lon']-lon0+180)%360-180

    lon = np.array(ds['lon'])
    lat = np.array(ds['lat'])
    lon, lat = np.meshgrid(lon, lat)

    lon = lon.flatten()
    lat = lat.flatten()
    
    sst_list = []
    for t in range(n_t):
        sst = np.array(ds['analysed_sst'].isel(time=t)).ravel()
        sst[np.isnan(sst)] = 0
        sst_list.append(sst)
    
    
    sst = np.array(ds['analysed_sst']).ravel()
    sst[np.isnan(sst)] = 0
        # sst_list.append(sst)
    # calculate ENU coords of data on tangent plane
    x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)
    sst_grids, _,_,_ = stats.binned_statistic_2d(x, y, sst_list, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
    for i,sst_grid in enumerate(sst_grids):
        sst_grid = np.rot90(sst_grid)
        sst_grid[sst_grid<270] = 0
        sst_grids[i] = sst_grid
    
    return sst_grids

def load_multisat_ssh(ssh_files):
    datasets = []
    ds = xr.open_dataset(ssh_files[0])
    sat = str(os.path.dirname(ssh_files[0]).split('cmems_sla/')[1])
    ds['time_variable'] = ds['time']
    ds['time'] = np.arange(ds['time'].shape[0])
    ds['sat'] = xr.DataArray([sat] * ds['time'].shape[0], dims=['time'])    
    ds = ds.rename({'time':'n_obs'})
    ds = ds.rename({'time_variable':'time'})
    datasets.append(ds)
    length = ds['n_obs'].shape[0]
    for f in ssh_files[1:]:
        ds = xr.open_dataset(f)
        sat = str(os.path.dirname(f).split('cmems_sla/')[1])
        ds['time_variable'] = ds['time']
        ds['time'] = np.arange(length,length+ds['time'].shape[0])
        ds['sat'] = xr.DataArray([sat] * ds['time'].shape[0], dims=['time'])
        ds = ds.rename({'time':'n_obs'})
        ds = ds.rename({'time_variable':'time'})
        datasets.append(ds)
        length+=ds['n_obs'].shape[0]
        # ds = xr.concat([ds,ds2],dim='n_obs')
    ds = xr.concat(datasets,dim='n_obs')
    ds = ds.sortby('time')
    ds['n_obs'] = ds['time']
    ds = ds.drop_vars(['time'])
    ds = ds.rename({'n_obs':'time'})
    
    return ds

# find coords of along track observations on local grid
def extract_tracked(ds, date_min, date_max, L_x, L_y, lon0, lat0, transformer_ll2xyz, withhold_sats = None):
    ds = ds.sel(time = slice(date_min, date_max))

    ds['longitude'] = ds['longitude']%360
    ds['longitude'] = (ds['longitude']-lon0+180)%360-180
    
    longitude = np.array(ds['longitude']).flatten()
    
    latitude = np.array(ds['latitude']).flatten()
    sla_f = np.array(ds['sla_filtered']).flatten()

    sla_uf = np.array(ds['sla_unfiltered']).flatten()
    sat = np.array(ds['sat']).flatten()
    t = np.array(ds['time']).flatten()
    # calculate ENU coords of along-track obs
    x,y,z = ll2xyz(latitude, longitude, 0, lat0, 0, 0, transformer_ll2xyz)
    sla_f = sla_f[z>-1000e3]
    sla_uf = sla_uf[z>-1000e3]
    y = y[z>-1000e3]
    x = x[z>-1000e3]
    sat = sat[z>-1000e3]
    t = t[z>-1000e3]
    
    sla_f = sla_f[x<L_x/2]
    sla_uf = sla_uf[x<L_x/2]
    y = y[x<L_x/2]
    t = t[x<L_x/2]
    sat = sat[x<L_x/2]
    x = x[x<L_x/2]

    sla_f = sla_f[x>-L_x/2]
    sla_uf = sla_uf[x>-L_x/2]
    y = y[x>-L_x/2]
    t = t[x>-L_x/2]
    sat = sat[x>-L_x/2]
    x = x[x>-L_x/2]
    
    sla_f = sla_f[y<L_y/2]
    sla_uf = sla_uf[y<L_y/2]
    x = x[y<L_y/2]
    t = t[y<L_y/2]
    sat = sat[y<L_y/2]
    y = y[y<L_y/2]
    
    sla_f = sla_f[y>-L_y/2]
    sla_uf = sla_uf[y>-L_y/2]
    x = x[y>-L_y/2]
    t = t[y>-L_y/2]
    sat = sat[y>-L_y/2]
    y = y[y>-L_y/2]
    
    if withhold_sats is not None:
        mask = np.array([s not in withhold_sats for s in sat])
        x_in = x[mask]
        y_in = y[mask]
        t_in = t[mask]
        sla_f_in = sla_f[mask]
        sla_uf_in = sla_uf[mask]
        x_out = x[~mask]
        y_out = y[~mask]
        t_out = t[~mask]
        sla_f_out = sla_f[~mask]
        sla_uf_out = sla_uf[~mask]
        tracks_in = {'x':x_in, 'y':y_in, 'time':t_in, 'sla_filtered':sla_f_in, 'sla_unfiltered':sla_uf_in}
        tracks_out = {'x':x_out, 'y':y_out, 'time':t_out, 'sla_filtered':sla_f_out, 'sla_unfiltered':sla_uf_out}
        return tracks_in, tracks_out
    else:
        tracks = {'x':x, 'y':y, 'time':t, 'sla_filtered':sla_f, 'sla_unfiltered':sla_uf}
        return tracks

def grid_ssh(tracks,n,N_t,L_x,L_y,start_date,filtered=False):
    x = tracks['x']
    y = tracks['y']
    t = tracks['time']
    ssh = tracks['sla_filtered'] if filtered else tracks['sla_unfiltered']
    days_since_start = [dt - np.datetime64(start_date) for dt in tracks['time']]
    days_since_start = (np.array(days_since_start).astype('timedelta64[D]') / np.timedelta64(1, 'D')).astype('int').tolist()
    unique_values = np.unique(days_since_start)
    missing_days = [t for t in range(N_t) if t not in unique_values]
    # print(unique_values)
    first_indices = [np.where(days_since_start == value)[0][0] for value in unique_values]
    # print(len(first_indices))
    data_final = np.zeros((N_t,n,n))
    missed_days = 0
    for day in range(N_t):
        if day not in missing_days:
            if day == N_t-1:
                ssh_loop = ssh[first_indices[day-missed_days]:]
                x_loop = x[first_indices[day-missed_days]:]
                y_loop = y[first_indices[day-missed_days]:]
            else:
                ssh_loop = ssh[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
                x_loop = x[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
                y_loop = y[first_indices[day-missed_days]:first_indices[day-missed_days+1]]

            x_loop = x_loop[~np.isnan(ssh_loop)]
            y_loop = y_loop[~np.isnan(ssh_loop)]
            ssh_loop = ssh_loop[~np.isnan(ssh_loop)]
        else:
            missed_days+=1
            x_loop = np.zeros(1)
            y_loop = np.zeros(1)
            ssh_loop = np.zeros(1)
        input_grid, _,_,_ = stats.binned_statistic_2d(x_loop, y_loop, ssh_loop, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0
        data_final[day,:,:] = input_grid

    return data_final

def normalise_ssh(ssh, mean_ssh, std_ssh):    
    return (ssh-mean_ssh)/std_ssh

def rescale_x(x, L_x, n):
    return (x + 0.5*L_x)*(n - 1)/L_x

def rescale_y(y, L_y, n): 
    return (-y + 0.5*L_y)*(n - 1)/L_y

def reformat_output_tracks(tracks, max_outvar_length, N_t, n, L_x, L_y, start_date, mean_ssh, std_ssh, filtered=False):
    x = tracks['x']
    y = tracks['y']
    t = tracks['time']
    ssh = tracks['sla_filtered'] if filtered else tracks['sla_unfiltered']
    days_since_start = [dt - np.datetime64(start_date) for dt in tracks['time']]
    days_since_start = (np.array(days_since_start).astype('timedelta64[D]') / np.timedelta64(1, 'D')).astype('int').tolist()
    unique_values = np.unique(days_since_start)
    missing_days = [t for t in range(N_t) if t not in unique_values]
    # print(unique_values)
    first_indices = [np.where(days_since_start == value)[0][0] for value in unique_values]
    # print(len(first_indices))
    data_final = np.zeros((N_t,max_outvar_length,3))
    missed_days = 0
    for day in range(N_t):
        if day not in missing_days:
            if day == N_t-1:
                ssh_loop = ssh[first_indices[day-missed_days]:]
                x_loop = x[first_indices[day-missed_days]:]
                y_loop = y[first_indices[day-missed_days]:]
            else:
                ssh_loop = ssh[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
                x_loop = x[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
                y_loop = y[first_indices[day-missed_days]:first_indices[day-missed_days+1]]

            x_loop = x_loop[~np.isnan(ssh_loop)]
            y_loop = y_loop[~np.isnan(ssh_loop)]
            ssh_loop = ssh_loop[~np.isnan(ssh_loop)]
            
            x_loop = rescale_x(x_loop, L_x, n)
            y_loop = rescale_y(y_loop, L_y, n)
            ssh_loop = normalise_ssh(ssh_loop, mean_ssh, std_ssh)
            
            n_obs = ssh_loop.shape[0]
            if n_obs<max_outvar_length:
                data_final[day-missed_days,:n_obs,0] = x_loop
                data_final[day-missed_days,:n_obs,1] = y_loop
                data_final[day-missed_days,:n_obs,2] = ssh_loop
            else:
                data_final[day-missed_days,:,0] = x_loop[:max_outvar_length]
                data_final[day-missed_days,:,1] = y_loop[:max_outvar_length]
                data_final[day-missed_days,:,2] = ssh_loop[:max_outvar_length]
            
                
        else:
            missed_days+=1
            
    return data_final