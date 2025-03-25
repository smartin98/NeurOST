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

# bin average high res MUR L4 SST (MW+IR observations)
def grid_sst_hr(data_sst_hr, n_t, n, L_x, L_y, lon0, lat0, coord_grid):
    ds = data_sst_hr
    
    lon_grid = coord_grid[:,:,0].ravel()
    lat_grid = coord_grid[:,:,1].ravel()
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
    
    # calculate ENU coords of data on tangent plane
    x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)
    sst_grids, _,_,_ = stats.binned_statistic_2d(x, y, sst_list, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
    for i,sst_grid in enumerate(sst_grids):
        sst_grid = np.rot90(sst_grid)
        sst_grid[sst_grid<273] = 0
        sst_grids[i] = sst_grid
    
    return sst_grids

def load_multisat_ssh(ssh_files):
    ds = xr.open_dataset(ssh_files[0])
    ds['time_variable'] = ds['time']
    ds['time'] = np.arange(ds['time'].shape[0])
    ds = ds.rename({'time':'n_obs'})
    ds = ds.rename({'time_variable':'time'})
    for f in ssh_files[1:]:
        ds2 = xr.open_dataset(f)
        ds2['time_variable'] = ds2['time']
        ds2['time'] = np.arange(ds['n_obs'].shape[0],ds['n_obs'].shape[0]+ds2['time'].shape[0])
        ds2 = ds2.rename({'time':'n_obs'})
        ds2 = ds2.rename({'time_variable':'time'})
        ds = xr.concat([ds,ds2],dim='n_obs')
    ds = ds.sortby('time')
    ds['n_obs'] = ds['time']
    ds = ds.drop_vars(['time'])
    ds = ds.rename({'n_obs':'time'})
    
    return ds

# find coords of along track SSH observations on local grid
def extract_tracked(ds, L_x, L_y, lon0, lat0, transformer_ll2xyz):
    # ds = ds.copy()
    # if nrt==False:
    
    # else:
        # ds['longitude'] = ds['longitude']%360
        # ds['longitude'] = (ds['longitude']-lon0+180)%360-180
    
    longitude = np.array(ds['longitude']).flatten()
    longitude = (longitude-lon0+180)%360-180
    
    latitude = np.array(ds['latitude']).flatten()
    sla_f = np.array(ds['sla_filtered']).flatten()

    sla_uf = np.array(ds['sla_unfiltered']).flatten()

    # calculate ENU coords of along-track obs
    x,y,z = ll2xyz(latitude, longitude, 0, lat0, 0, 0, transformer_ll2xyz)
    
    mask = (z > -1000e3) & (x < L_x / 2) & (x > -L_x / 2) & (y < L_y / 2) & (y > -L_y / 2)
    x, y, sla_f, sla_uf = x[mask], y[mask], sla_f[mask], sla_uf[mask]
    
    tracks = np.stack([x, y, sla_f, sla_uf], axis = -1)
    return tracks