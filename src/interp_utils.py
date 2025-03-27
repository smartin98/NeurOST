import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import xarray as xr 
from datetime import date, timedelta
import os
from tqdm import tqdm
import pandas as pd
import shutil
import h5py

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

def load_multisat_ssh_single_day(ssh_files, satellite_names):
    datasets = []
    for f, sat_name in zip(ssh_files, satellite_names):
        ds = xr.open_dataset(f)
        sat_var = xr.DataArray(np.full(ds.sizes['time'], sat_name, dtype=object), dims=['time'])
        ds = ds.assign(satellite=sat_var)
        datasets.append(ds)
    
    ds_merged = xr.concat(datasets, dim='time')
    return ds_merged

def get_next_dir_level(dir_name, file_paths):
    return [path.split(dir_name + os.sep, 1)[-1].split(os.sep, 1)[0] for path in file_paths if dir_name + os.sep in path]

def load_ssh_by_date_range(start_date, end_date, dir_name = 'input_data/cmems_sla'):
    ssh_files_total = GetListOfFiles(dir_name)
    n_days = (end_date - start_date).days
    dates = [start_date + timedelta(days = t) for t in range(n_days)]
    ssh_datasets = []
    print('Number of days to load: '+str(n_days))
    for t in tqdm(range(n_days), desc="SSH loading progress"):
        
        ssh_files = [f for f in ssh_files_total if date(int(f[-20:-16]), int(f[-16:-14]), int(f[-14:-12])) == dates[t]]
        sat_names = get_next_dir_level(dir_name, ssh_files)

        if len(ssh_files)>0:
            ds = load_multisat_ssh_single_day(ssh_files, sat_names)
            ssh_datasets.append(ds)

    return dates, ssh_datasets

# empty a directory of any files
def empty_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)


def convert_np_cache_to_hdf5(np_dir = 'input_data/sla_cache_2024-03-01_2024-04-01', hdf5_path = 'input_data/sla_cache_2024-03-01_2024-04-01.h5',delete_np=False):
    with h5py.File(hdf5_path, 'w') as h5f:
        for fname in tqdm(os.listdir(np_dir), desc = 'Converting .npy cache to single hdf5 file:'):
            if fname.endswith('.npy') and not fname.endswith('_sats.npy'):
                base, _ = os.path.splitext(fname)
                group = h5f.require_group(base)
                data = np.load(os.path.join(np_dir, fname), allow_pickle=False)
                group.create_dataset('data', data=data, compression="gzip")

                sats_fname = base + '_sats.npy'
                sats_path = os.path.join(np_dir, sats_fname)
                if os.path.exists(sats_path):
                    sats_data = np.load(sats_path, allow_pickle=True)
                    group.create_dataset('sats', data=sats_data, compression="gzip")
    if delete_np:
        print('deleting .npy cache at: '+np_dir)
        shutil.rmtree(np_dir)

def create_sla_chunks(start_date, end_date, chunk_dir = 'input_data/sla_cache', time_bin_size = 10, lon_bin_size = 10, lat_bin_size = 10, n_t = 30, cmems_dir = 'input_data/cmems_sla',force_recache=False):
    
    output_dir = chunk_dir + '_' + str(start_date) + '_' + str(end_date)
    if (os.path.exists(output_dir+'.h5') and force_recache) or (not os.path.exists(output_dir+'.h5')):
    
        dates, ssh_datasets = load_ssh_by_date_range(start_date - timedelta(days = n_t//2), end_date + timedelta(days = n_t//2), dir_name = cmems_dir)
        ds = xr.concat(ssh_datasets, dim = 'time')

        del ssh_datasets

        ds['day_idx'] = (ds['time'] - np.datetime64(start_date - timedelta(days = n_t//2), 'ns'))// np.timedelta64(1, 'D')
        time_vals = ds['day_idx'].values
        time_bins = np.arange(0,365,time_bin_size)
        lat_bins = np.arange(-90, 90, lat_bin_size)
        lon_bins = np.arange(-180, 180, lon_bin_size)

        # Compute bin indices
        time_idx = np.digitize(time_vals, time_bins) - 1
        lat_vals = ds['latitude'].values.ravel()
        lon_vals = ds['longitude'].values.ravel()
        lat_idx = np.digitize(lat_vals, lat_bins) - 1
        lon_idx = np.digitize(lon_vals, lon_bins) - 1

        df = pd.DataFrame({
            'time_bin': time_idx,
            'lat_bin': lat_idx,
            'lon_bin': lon_idx
        })

        # group data indices by bin
        grouped = df.groupby(['time_bin', 'lat_bin', 'lon_bin']).indices
    
        os.makedirs(output_dir, exist_ok=True)
        #empty any existing files left over from previous cache under same name
        empty_directory(output_dir)

        for (t_bin, lat_bin, lon_bin), idx in tqdm(grouped.items(), desc="Saving chunked SSH into cache at "+output_dir):
            # Extract data efficiently using precomputed indices
            lat_subset = lat_vals[idx]
            lon_subset = lon_vals[idx]
            time_subset = time_vals[idx]
            sla_unfiltered_subset = ds['sla_unfiltered'].values[idx]
            sla_filtered_subset = ds['sla_filtered'].values[idx]
            sat_subset = ds['satellite'].values[idx]

            # Save data
            filename = os.path.join(output_dir, f'bin_t{t_bin}_lat{lat_bin}_lon{lon_bin}.npy')
            np.save(filename, np.column_stack((lat_subset, lon_subset, time_subset, sla_unfiltered_subset, sla_filtered_subset)))
            np.save(filename[:-4] + '_sats.npy', sat_subset)
        
        
        convert_np_cache_to_hdf5(np_dir = output_dir, hdf5_path = output_dir+'.h5',delete_np=True)
    else:
        print('skipping cache saving since already exists and force_recache=False')
        

        
def get_query_bins(query_time_start, query_time_end, query_lat_min, query_lat_max, query_lon_min, query_lon_max, time_bins, lat_bins, lon_bins):
    # Compute full bin edges (intervals)
    time_bin_edges = np.append(time_bins, np.inf)  # Ensure upper edge coverage
    lat_bin_edges = np.append(lat_bins, np.inf)
    lon_bin_edges = np.append(lon_bins, np.inf)

    # Find bins that overlap the query bounds
    q_time_idx = np.where((time_bin_edges[:-1] <= query_time_end) & (time_bin_edges[1:] > query_time_start))[0]
    q_lat_idx = np.where((lat_bin_edges[:-1] <= query_lat_max) & (lat_bin_edges[1:] > query_lat_min))[0]
    q_lon_idx = np.where((lon_bin_edges[:-1] <= query_lon_max) & (lon_bin_edges[1:] > query_lon_min))[0]

    return q_time_idx, q_lat_idx, q_lon_idx


def load_query_data_h5(query_time_start, query_time_end, query_lat_min, query_lat_max, 
                    query_lon_min, query_lon_max, time_bins, lat_bins, lon_bins, h5f):
    # Compute bin edges and indices just as before
    time_bin_edges = np.append(time_bins, np.inf)
    lat_bin_edges  = np.append(lat_bins, np.inf)
    lon_bin_edges  = np.append(lon_bins, np.inf)
    
    q_time_idx = np.where((time_bin_edges[:-1] <= query_time_end) & (time_bin_edges[1:] > query_time_start))[0]
    q_lat_idx  = np.where((lat_bin_edges[:-1] <= query_lat_max) & (lat_bin_edges[1:] > query_lat_min))[0]
    q_lon_idx  = np.where((lon_bin_edges[:-1] <= query_lon_max) & (lon_bin_edges[1:] > query_lon_min))[0]

    data_list = []
    sat_list = []
    

    # Loop over the bins that intersect the query bounds.
    for t in q_time_idx:
        for lat_i in q_lat_idx:
            for lon_i in q_lon_idx:
                group_name = f'bin_t{t}_lat{lat_i}_lon{lon_i}'
                if group_name in h5f:
                    group = h5f[group_name]
                    # Load main data and satellite data from the group.
                    bin_data = group['data'][()]
                    sat_data = group['sats'][()]
                    sat_data = np.char.mod('%s', sat_data)
                    # sat_data = sat_data.astype(str)
                    data_list.append(bin_data)
                    sat_list.append(sat_data)
    
    data = np.concatenate(data_list) if data_list else np.empty((0,))
    sats = np.concatenate(sat_list) if sat_list else np.empty((0,))
    return data, sats

def extract_tracks_h5(t_mid, lon0, lat0, coord_grid, transformer_ll2xyz, time_bins, lon_bins, lat_bins, h5f, n_t = 30, L_x=960e3, L_y=960e3, filtered=False):
    lon_grid = coord_grid[:,:,0]
    lat_grid = coord_grid[:,:,1]
    lat_max = np.max(lat_grid)+0.1
    lat_min = np.min(lat_grid)-0.1
    lon_max = np.max(lon_grid[:,-1]) + 0.1 #handles wrap around dateline
    lon_min = np.min(lon_grid[:,0]) - 0.1 #handles wrap around dateline

    
    
    data, sat = load_query_data_h5(query_time_start = t_mid - n_t//2,
                                 query_time_end = t_mid + n_t//2,
                                 query_lat_min = lat_min,
                                 query_lat_max = lat_max,
                                 query_lon_min = lon_min,
                                 query_lon_max = lon_max,
                                 time_bins = time_bins,
                                 lat_bins = lat_bins,
                                 lon_bins = lon_bins,
                                 h5f = h5f
                                )
    
    if np.size(data)>0:
        latitude = data[:,0]
        longitude = data[:,1]
        day = data[:,2] - (t_mid - n_t//2)

        if filtered:
            sla = data[:,4]
        else:
            sla = data[:,3]

        mask = (day >= 0) & (day<n_t)

        longitude, latitude, sla, day, sat = longitude[mask], latitude[mask], sla[mask], day[mask], sat[mask]

        # Normalize longitude
        longitude = (longitude - lon0 + 180) % 360 - 180

        # Calculate ENU coordinates
        x, y, z = ll2xyz(latitude, longitude, 0, lat0, 0, 0, transformer_ll2xyz)

        mask = (z > -1e6) & (-L_x / 2 < x) & (x < L_x / 2) & (-L_y / 2 < y) & (y < L_y / 2)

        return np.column_stack((x[mask], y[mask], sla[mask], day[mask])), sat[mask]
    else:
        return np.zeros((1,4)), None


def bin_ssh(data, L_x = 960e3, L_y = 960e3, n = 128, n_t = 30, filtered = False):
    ssh_grid = np.zeros((n_t,n,n))
    data[np.isnan(data)] = 0
    
    for t in range(n_t):
        mask = (data[:,3] == t)
        x, y, ssh = data[mask, 0], data[mask, 1], data[mask, 2]
        
        input_grid, _,_,_ = stats.binned_statistic_2d(x,y,ssh, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0

        ssh_grid[t,:,:] = input_grid
        
    return ssh_grid


def get_ssh_h5(r,t,coord_grid,transformer_ll2xyz, time_bins, lon_bins, lat_bins, h5f, n_t = 30, n = 128, L_x = 960e3, L_y = 960e3, leave_out_altimeters=True, withhold_sat='random', filtered=False):

    mid = n//2
    neighbors = [(r, mid-1, mid-1), (r, mid-1, mid), (r, mid, mid-1), (r, mid, mid)]
    lon0 = np.mean([coord_grid[neighbor[0], neighbor[1], neighbor[2], 0] for neighbor in neighbors])
    lat0 = np.mean([coord_grid[neighbor[0], neighbor[1], neighbor[2], 1] for neighbor in neighbors])
    
    d, s = extract_tracks_h5(t_mid = t,
                          lon0 = lon0,
                          lat0 = lat0,
                          coord_grid = coord_grid[r,],
                          transformer_ll2xyz = transformer_ll2xyz, 
                          time_bins=time_bins, 
                          lon_bins=lon_bins, 
                          lat_bins=lat_bins,
                          n_t = n_t,
                          L_x = L_x,
                          L_y = L_y,
                          filtered = filtered,
                          h5f = h5f
                         )
    
    if d.shape[0]>1:
    
        if leave_out_altimeters:
            sats_all = np.unique(s)
            if withhold_sat == 'random':
                # randomly select 1 satellite to withhold (e.g. for ground truth during training)
                withhold = np.random.choice(sats_all)
            else:
                # specify either 1 sat or list of sats to withhold (e.g. for inference while maintaining independent sat for validation)
                withhold = withhold_sat
            mask = (s != withhold) if isinstance(withhold, str) else ~np.isin(s, withhold)

            d_out = d[~mask]
            d, s = d[mask], s[mask]

            out_tracks = []
            for t in range(n_t):
                mask = (d_out[:,3] == t)
                out_tracks.append(d_out[mask,:])

            len_max = max([d.shape[0] for d in out_tracks])
            out_data = np.zeros((n_t,len_max,3))
            for t in range(n_t):
                out_data[t,:out_tracks[t].shape[0],] = out_tracks[t][:,:3]

        else:
            out_data = None

        ssh_grid = bin_ssh(d, L_x = L_x, L_y = L_y, n = n, n_t = n_t, filtered = filtered)
    else:
        ssh_grid = np.zeros((n_t,n,n))
        out_data = None
    
    return ssh_grid, out_data