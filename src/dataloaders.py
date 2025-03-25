import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from src.interp_utils import *
from datetime import date, timedelta
import os
import numpy as np


class NeurOST_dataset(Dataset):
    def __init__(self, zarr_dir, N_t, mean_ssh, std_ssh, mean_sst, std_sst, coord_grids, n = 128, L_x = 960e3, L_y = 960e3):
        self.zarr_dir = zarr_dir
        self.N_t = N_t
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst
        self.coord_grids = coord_grids
        self.n = n
        self.L_x = L_x
        self.L_y = L_y
        files = sorted(os.listdir(self.zarr_dir))
        self.zarr_paths = [self.zarr_dir + f for f in files]
        
        self.ds_sst = xr.open_mfdataset(self.zarr_paths, engine="zarr", combine="by_coords", parallel=True)
        self.t_range = self.ds_sst['time'].shape[0]
        print(self.t_range)
        
    def __len__(self):
        return int(10000)
        # return self.coord_grids.shape[0]
    
    def __getitem__(self, idx):
        # print('fetching idx ' + str(idx))
        
        r = np.random.randint(self.coord_grids.shape[0])
        t = np.random.randint(self.t_range - self.N_t)
        
        lon0 = 0.25*(self.coord_grids[r,int(self.n/2)-1,int(self.n/2)-1,0]+self.coord_grids[r,int(self.n/2)-1,int(self.n/2),0]+self.coord_grids[r,int(self.n/2),int(self.n/2)-1,0]+self.coord_grids[r,int(self.n/2),int(self.n/2),0])
        lat0 = 0.25*(self.coord_grids[r,int(self.n/2)-1,int(self.n/2)-1,1]+self.coord_grids[r,int(self.n/2)-1,int(self.n/2),1]+self.coord_grids[r,int(self.n/2),int(self.n/2)-1,1]+self.coord_grids[r,int(self.n/2),int(self.n/2),1])
        
        sst = grid_sst_hr(self.ds_sst.isel(time=slice(t, t+self.N_t)),self.N_t,self.n, self.L_x, self.L_y, lon0, lat0, self.coord_grids[r,])

        sst[sst!=0] = (sst[sst!=0]-self.mean_sst)/self.std_sst

#         tracks_in, tracks_out = extract_tracked(self.worker_ds_ssh, self.start_date, self.start_date+timedelta(days=self.N_t+2), self.L_x, self.L_y, lon0, lat0, transformer_ll2xyz, withhold_sats = ['s3a','s3b'])
#         ssh_in = grid_ssh(tracks_in, self.n, self.N_t, self.L_x, self.L_y, self.start_date,self.filtered_sla)
#         ssh_out = reformat_output_tracks(tracks_out, self.max_outvar_length, self.N_t, self.n, self.L_x, self.L_y, self.start_date, self.mean_ssh, self.std_ssh, self.filtered_sla)

#         ssh_in[ssh_in!=0] = (ssh_in[ssh_in!=0]-self.mean_ssh)/self.std_ssh
        # invar = torch.from_numpy(np.stack((sst, ssh_in), axis = 1).astype(np.float32))
        # outvar = torch.from_numpy(ssh_out)
        
        return sst