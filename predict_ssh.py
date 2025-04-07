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
import signal
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--start', type = str, help = 'start date for SSH mapping (YYYYMMDD)')
parser.add_argument('--end', type = str, help = 'end date for SSH mapping (YYYYMMDD)')
parser.add_argument('--sst_zarr_path', type = str, default = 'input_data/mur_coarse_zarrs', help = "Path to SST zarr store")
parser.add_argument('--experiment_name', type = str, default = 'NeurOST_SSH-SST', help = 'name of SSH mapping experiment (will be used in filenames of output)')
parser.add_argument("--withheld_sats", nargs = "+", type = str, help = "List all satellites to withhold from input using the standard short names from CMEMS (e.g. s3a, al, swon, etc.)")
parser.add_argument('--no_sst', action = "store_true", help = "Use only SSH in input")
parser.add_argument('--coord_grid_path', type = str, default = "input_data/coord_grids.npy", help = "Path to npy file containing coordinates of local patches for reconstruction")
parser.add_argument('--model_weights_path', type = str, help = "Path to saved torch model weights (not necessary if just doing inference with the pre-trained model)")
parser.add_argument('--mean_ssh', type = float, default = 0.074, help = "Global mean SSH for normalisation")
parser.add_argument('--std_ssh', type = float, default = 0.0986, help = "Global std SSH for normalisation")
parser.add_argument('--mean_sst', type = float, default = 293.307, help = "Global mean SST for normalisation")
parser.add_argument('--std_sst', type = float, default = 8.726, help = "Global std SST for normalisation")
parser.add_argument('--n_t', type = int, default = 30, help = "Length (in days) of the temporal mapping window for the network")
parser.add_argument('--n', type = int, default = 128, help = "Size of the spatial mapping window for the network (integer number of pixels)")
parser.add_argument('--L_x', type = float, default = 960e3, help = "Size of the spatial mapping window for the network in the zonal direction (float in km)")
parser.add_argument('--L_y', type = float, default = 960e3, help = "Size of the spatial mapping window for the network in the meridional direction (float in km)")
parser.add_argument('--force_recache', action = "store_true", help = "Re-create SSH HDF5 cache for desired mapping range even if covered by existing cache")
parser.add_argument('--filtered_sla', action = "store_true", help = "Use sla_filtered variable from CMEMS product rather than sla_unfiltered (default).")
parser.add_argument('--time_bin_size', type = float, default = 10, help = "Time bin size (in days) for SSH cache")
parser.add_argument('--lon_bin_size', type = float, default = 10, help = "Longitude bin size (in deg) for SSH cache")
parser.add_argument('--lat_bin_size', type = float, default = 10, help = "Latitude bin size (in deg) for SSH cache")
parser.add_argument('--output_zarr_dir', type = str, default = 'predictions/unmerged_zarrs/', help = "Path to directory within which unmerged predictions will be stored in zarr store.")
parser.add_argument('--n_cpu_workers', type = int, default = 1, help = "Number of CPU workers used by dataloader to parallelize data loading (strongly recommend setting as high as your resources allow to ensure GPU saturation)")
parser.add_argument('--batch_size', type = int, default = 32, help = "Number of examples per batch, adjust based on your GPU memory.")
parser.add_argument('--prefetch_factor', type = int, default = 2, help = "Dataloader prefetch factor.")

args = parser.parse_args()

if args.start is None:
    raise ValueError('Need to specify start date for mapping range using --start YYYYMMDD')
if args.end is None:
    raise ValueError('Need to specify end date for mapping range using --end YYYYMMDD')
    
if args.withheld_sats is None:
    leave_out_altimeters = False
else:
    leave_out_altimeters = True

if args.n_cpu_workers <=1:
    multiprocessing = False
else:
    multiprocessing = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ')
print(device)

start_date = datetime.date(int(args.start.replace('-','')[:4]), int(args.start.replace('-','')[4:6]), int(args.start.replace('-','')[6:]))
end_date = datetime.date(int(args.end.replace('-','')[:4]), int(args.end.replace('-','')[4:6]), int(args.end.replace('-','')[6:]))
coord_grids = np.load(args.coord_grid_path)


dataset = NeurOST_dataset(sst_zarr = args.sst_zarr_path,
                          start_date = start_date,
                          end_date = end_date,
                          N_t = args.n_t,
                          mean_ssh = args.mean_ssh,
                          std_ssh = args.std_ssh,
                          mean_sst = args.mean_sst,
                          std_sst = args.std_sst,
                          coord_grids = coord_grids,
                          n = args.n,
                          L_x = args.L_x,
                          L_y = args.L_y,
                          force_recache = args.force_recache,
                          leave_out_altimeters = leave_out_altimeters,
                          withhold_sat = args.withheld_sats,
                          filtered = args.filtered_sla,
                          use_sst = not args.no_sst,
                          time_bin_size = args.time_bin_size,
                          lon_bin_size = args.lon_bin_size,
                          lat_bin_size = args.lat_bin_size,
                          ssh_out_n_max = 1000,
                         )


if args.no_sst:
    model = SimVP_Model_no_skip(in_shape=(args.n_t,1,args.n,args.n),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(device)
else:
    model = SimVP_Model_no_skip_sst(in_shape=(args.n_t,2,args.n,args.n),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(device)

if args.model_weights_path is None:
    if args.no_sst:
        model_weights_path = "input_data/simvp_ssh_ns1000000_global_weights_epoch45"
    else:
        model_weights_path = "input_data/model_weights/simvp_ssh_sst_ns1000000global_weights_epoch46"
else:
    model_weights_path = args.model_weights_path
    
state_dict = torch.load(model_weights_path)['model_state_dict']

if "module." in list(state_dict.keys())[0]:
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
try:
    model.load_state_dict(state_dict)
except:
    raise ValueError("Error loading model weights, check if --model_weights_path argument is consistent with the architecture (e.g. SSH-only weights with SSH-SST network?)")

model.eval()

pred_zarr_path = os.path.join(args.output_zarr_dir, args.experiment_name + '_unmerged_preds_' + str(start_date).replace('-','') + '_' + str(end_date).replace('-','') + '.zarr')

if multiprocessing:
    dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            shuffle = False, 
                            num_workers = args.n_cpu_workers, 
                            worker_init_fn = worker_init_fn, # defined in src.dataloaders...
                            persistent_workers = True,
                            prefetch_factor = args.prefetch_factor
                           )
else:
    dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            shuffle = False,
                           )
    


#########
# handle case where existing zarr file would be over-written. user is asked to confirm if they want to over-write. If no (or if no response in 60 s), we ensure doesn't over-write by adding integer to end of file path.
def timeout_handler(signum, frame):
    raise TimeoutError
    
def get_new_filepath(filepath):
    base, ext = os.path.splitext(filepath)
    i = 0
    while os.path.exists(f"{base}{i}{ext}"):  # Ensure unique filename
        i += 1
    return f"{base}{i}{ext}"

if os.path.exists(pred_zarr_path):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # Set timeout to 60 seconds
    try:
        confirm = input("Zarr pred path exists already, do you want to over-write? (yes/no): ").strip().lower()
    except TimeoutError:
        confirm = "no"
        print("\nNo response. Defaulting to 'no'.")

    signal.alarm(0)  # Disable alarm

    if confirm != "yes":
        new_path = get_new_filepath(pred_zarr_path)
        print(f"Renaming file to avoid overwrite: {new_path}")
        pred_zarr_path = new_path
##########

# initialise zarr store to store unmerged patch predictions
pred_store = zarr.open(pred_zarr_path, mode = 'w', shape=((end_date-start_date).days + 1, coord_grids.shape[0], args.n, args.n), chunks=(10, 10, args.n, args.n), dtype="float32")

data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu_workers)

current_idx = 0

r_idxs = np.arange(coord_grids.shape[0])
t_idxs = np.arange((end_date - start_date).days + 1)

t_idxs, r_idxs = np.meshgrid(t_idxs, r_idxs)

t_idxs, r_idxs = t_idxs.ravel(), r_idxs.ravel()

with torch.no_grad():
    for input_data, _ in tqdm(data_loader, desc = "Running batch predictions"):
        
        input_data = input_data.to(device)
        
        t_indexer = t_idxs[current_idx:current_idx + len(input_data)]
        r_indexer = r_idxs[current_idx:current_idx + len(input_data)]
        
        preds = model(input_data)
        preds = preds.cpu().numpy()[:, int(args.n_t / 2), 0, :, :]
        preds = preds * args.std_ssh + args.mean_ssh

        if pred_store.shape[0] > args.batch_size:
            # save slice if batch all within one region (faster writing)
            if r_indexer[0] == r_indexer[-1]:
                pred_store[t_indexer[0]:t_indexer[-1]+1,r_indexer[0]] = preds
            else:
                # resort to saving each batch item individually if not all within one region
                for i in range(t_indexer.shape[0]):
                    pred_store[int(t_indexer[i]),int(r_indexer[i])] = preds[i,]
        else:
            # resort to saving each batch item individually if not all within one region
            for i in range(t_indexer.shape[0]):
                pred_store[int(t_indexer[i]),int(r_indexer[i])] = preds[i,]
        current_idx += len(input_data)
        
print('Inference complete! Un-merged patch predictions saved at: ' + pred_zarr_path + ', run map_merging script to create final gridded product.')

