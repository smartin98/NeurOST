import numpy as np
import torch
import os
import sys
sys.path.append('src')
from src.simvp_model import *
from src.dataloaders import *
import datetime
import zarr
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--start', type = str, help = 'start date for SSH mapping (YYYYMMDD)')
parser.add_argument('--end', type = str, help = 'end date for SSH mapping (YYYYMMDD)')
parser.add_argument('--sst_zarr_path', type = str, default = 'input_data/mur_coarse_zarrs', help = "Path to SST zarr store")
parser.add_argument('--experiment_name', type = str, default = 'NeurOST_SSH-SST_', help = 'name of SSH mapping experiment (will be used in filenames of output)')
parser.add_argument("--withheld_sats", nargs = "+", type = str, help = "List all satellites to withhold from input using the standard short names from CMEMS (e.g. s3a, al, swon, etc.)")
parser.add_argument('--no_sst', action = "store_true", help = "Use only SSH in input")
parser.add_argument('--coord_grid_path', type = str, default = "input_data/coord_grids.npy", help = "Path to npy file containing coordinates of local patches for reconstruction")
parser.add_argument('--model_weights_path', type = str, help = "Path to saved torch model weights (not necessary if just doing inference with the pre-trained model)")
parser.add_argument('--mean_ssh', type = float, default = 0.074, help = "Global mean SSH for normalisation")
parser.add_argument('--std_ssh', type = float, default = 0.0986, help = "Global std SSH for normalisation")
parser.add_argument('--mean_sst', type = float, default = 293.307, help = "Global mean SST for normalisation")
parser.add_argument('--std_sst', type = float, default = 8.726, help = "Global std SST for normalisation")



args = parser.parse_args()

if args.start is None:
    raise ValueError('Need to specify start date for mapping range using --start YYYYMMDD')
if args.end is None:
    raise ValueError('Need to specify end date for mapping range using --end YYYYMMDD')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

start_date = datetime.date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
end_date = datetime.date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:]))


dataset = NeurOST_dataset()