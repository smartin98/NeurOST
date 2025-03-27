import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from src.interp_utils import *
from datetime import date, timedelta
import os
import numpy as np
import pyproj


class NeurOST_dataset(Dataset):
    def __init__(self, sst_zarr, start_date, end_date, N_t, mean_ssh, std_ssh, mean_sst, std_sst, coord_grids, n = 128, L_x = 960e3, L_y = 960e3, force_recache = False, leave_out_altimeters=True, withhold_sat='random', filtered=False, use_sst=True, time_bin_size = 10, lon_bin_size = 10, lat_bin_size = 10):
        self.sst_zarr = sst_zarr
        self.start_date = start_date
        self.end_date = end_date
        self.N_t = N_t
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst
        self.coord_grids = coord_grids
        self.n = n
        self.L_x = L_x
        self.L_y = L_y
        files = sorted(os.listdir(self.sst_zarr))
        self.zarr_paths = [self.sst_zarr + f for f in files]
        self.force_recache = force_recache
        self.leave_out_altimeters = leave_out_altimeters
        self.withhold_sat = withhold_sat
        self.filtered = filtered
        self.use_sst = use_sst
        self.time_bin_size = time_bin_size
        self.lon_bin_size = lon_bin_size
        self.lat_bin_size = lat_bin_size
        
                
        self.ds_sst = xr.open_mfdataset(self.zarr_paths, engine="zarr", combine="by_coords", parallel=True)
        if np.min(self.ds_sst['time']) > np.datetime64(str(self.start_date - timedelta(days = self.N_t//2)),'ns'):
            raise ValueError("MUR SST zarr file missing dates at beginning of desired time range")
        if np.max(self.ds_sst['time']) < np.datetime64(str(self.end_date + timedelta(days = self.N_t//2)),'ns'):
            raise ValueError("MUR SST zarr file missing dates at end of desired time range")
        
        self.ds_sst = self.ds_sst.sel(time=slice(str(self.start_date - timedelta(days = self.N_t//2)), str(self.end_date + timedelta(days = self.N_t//2))))
        t_idxs = np.arange(self.N_t//2, self.ds_sst['time'].shape[0] - self.N_t//2, 1)
        r_idxs = np.arange(self.coord_grids.shape[0])
        
        create_sla_chunks(self.start_date, self.end_date, chunk_dir = 'input_data/sla_cache', time_bin_size = self.time_bin_size, lon_bin_size = self.lon_bin_size, lat_bin_size = self.lat_bin_size, n_t = self.N_t, cmems_dir = 'input_data/cmems_sla', force_recache = self.force_recache)
        self.chunk_dir = 'input_data/sla_cache' + '_' + str(self.start_date) + '_' + str(self.end_date)
        
        self.time_bins = np.arange(0,self.ds_sst['time'].shape[0], self.time_bin_size)
        self.lon_bins = np.arange(-180,180, self.lon_bin_size)
        self.lat_bins = np.arange(-90,90, self.lat_bin_size)
        
        t_idxs, r_idxs = np.meshgrid(t_idxs, r_idxs)
        
        self.t_idxs, self.r_idxs = t_idxs.ravel(), r_idxs.ravel()
        
        self.transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
        
        
    def __len__(self):
        return np.size(self.t_idxs)
    
    def __getitem__(self, idx):
        
        r = self.r_idxs[idx]
        t = self.t_idxs[idx]
        
        lon0 = 0.25*(self.coord_grids[r,int(self.n/2)-1,int(self.n/2)-1,0]+self.coord_grids[r,int(self.n/2)-1,int(self.n/2),0]+self.coord_grids[r,int(self.n/2),int(self.n/2)-1,0]+self.coord_grids[r,int(self.n/2),int(self.n/2),0])
        lat0 = 0.25*(self.coord_grids[r,int(self.n/2)-1,int(self.n/2)-1,1]+self.coord_grids[r,int(self.n/2)-1,int(self.n/2),1]+self.coord_grids[r,int(self.n/2),int(self.n/2)-1,1]+self.coord_grids[r,int(self.n/2),int(self.n/2),1])
        
        if self.use_sst:
        
            sst = grid_sst_hr(self.ds_sst.isel(time=slice(t-self.N_t//2, t+self.N_t//2)),self.N_t,self.n, self.L_x, self.L_y, lon0, lat0, self.coord_grids[r,])

            sst[sst!=0] = (sst[sst!=0]-self.mean_sst)/self.std_sst
        
        ssh_in, ssh_out = get_ssh(r, 
                                  t, 
                                  self.coord_grids, 
                                  self.transformer_ll2xyz, 
                                  self.time_bins, 
                                  self.lon_bins, 
                                  self.lat_bins, 
                                  n_t=self.N_t, 
                                  n=self.n, 
                                  L_x=self.L_x, 
                                  L_y=self.L_y, 
                                  leave_out_altimeters=self.leave_out_altimeters, 
                                  withhold_sat=self.withhold_sat, 
                                  filtered=self.filtered, 
                                  chunk_dir=self.chunk_dir
                                 )
        
        ssh_in[ssh_in!=0] = (ssh_in[ssh_in!=0]-self.mean_ssh)/self.std_ssh
        
        if ssh_out is not None:
            x = ssh_out[:,:,0]
            y = ssh_out[:,:,1]
            ssh = ssh_out[:,:,2]
            x[x!=0] = (x[x!=0] + 0.5 * self.L_x) * (self.n-1) / self.L_x # convert to pixel coords for loss function
            y[y!=0] = (0.5 * self.L_y - y[y!=0]) * (self.n-1) / self.L_y # convert to pixel coords for loss function
            ssh[ssh!=0] = (ssh[ssh!=0]-self.mean_ssh)/self.std_ssh
            ssh_out = np.stack((x,y,ssh), axis = -1)
        
        if self.use_sst:
            if ssh_out is not None:
                return torch.from_numpy(np.stack((sst, ssh_in), axis = 1).astype(np.float32)), torch.from_numpy(ssh_out.astype(np.float32))
            else:
                return torch.from_numpy(np.stack((sst, ssh_in), axis = 1).astype(np.float32))
        else:
            if ssh_out is not None:
                return torch.from_numpy(np.expand_dims(ssh_in, axis = 1).astype(np.float32)), torch.from_numpy(ssh_out.astype(np.float32))
            else:
                return torch.from_numpy(np.expand_dims(ssh_in, axis = 1).astype(np.float32))