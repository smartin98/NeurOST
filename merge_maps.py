import numpy as np
import datetime
import multiprocessing
from src.merging import *
import gc
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--start', type = str, help = 'start date for SSH map merging (YYYYMMDD)')
    parser.add_argument('--end', type = str, help = 'end date for SSH map merging (YYYYMMDD)')
    parser.add_argument('--zarr_start_date', type = str, help = 'start date of the zarr pred file in case different from --start (YYYYMMDD)')
    parser.add_argument('--zarr_end_date', type = str, help = 'end date of the zarr pred file in case different from --end (YYYYMMDD)')
    parser.add_argument('--experiment_name', type = str, default = 'NeurOST_SSH-SST', help = 'name of SSH mapping experiment (will be used in filenames of output)')
    parser.add_argument('--n_cpu_workers', type = int, default = 1, help = "Number of CPU workers used to parallelize map merging (note each worker requires relatively high memory so careful not to set too high)")
    parser.add_argument('--mask_filename', type = str, default = 'input_data/auxilliary/land_water_mask_10grid.nc', help = 'path to land-water mask file')
    parser.add_argument('--dist_filename', type = str, default = 'input_data/auxilliary/distance_to_nearest_coastlines_10grid.nc', help = 'path to coast distance file')
    parser.add_argument('--mdt_filename', type = str, default = 'input_data/auxilliary/mdt_hybrid_cnes_cls18_cmems2020_global.nc', help = 'path to MDT file')
    parser.add_argument('--network_name', type = str, default = 'SimVP_SSH_SST_1M_global', help = 'name of neural network architecture (saved in meta data of output nc files)')
    parser.add_argument('--coord_grid_path', type = str, default = 'input_data/coord_grids.npy', help = 'path to coord grid .npy file')
    parser.add_argument('--L', type = float, default = 250e3, help = 'width of Gaussian kernel used in merging averaging [in km]')
    
    args = parser.parse_args()
    
    if args.zarr_start_date is None:
        args.zarr_start_date = args.start
    if args.zarr_end_date is None:
        args.zarr_end_date = args.end
    
    return args



def worker(lock, batches):
    while True:
        with lock:
            if not batches:
                break 

            batch = batches.pop(0)

        merge_maps_and_save_zarr(pred_path, zarr_start_date, batch, output_nc_dir, args.mask_filename, args.dist_filename, args.mdt_filename, args.network_name, coord_grid_path = args.coord_grid_path,L=args.L, crop_pixels=9, dx=7.5e3, with_grads=True, mask_coast_dist=0, lon_min=-180 ,lon_max=180, lat_min=-70, lat_max=80, res=1/10, progress=False, mask_ice=True, sst_zarr_path = 'input_data/mur_coarse_zarrs', experiment_name = experiment_name)

        gc.collect()

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':

    args = parse_args()

    # experiment_name, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    start = args.start.replace('-','')
    end = args.end.replace('-','')
    zarr_start = args.zarr_start_date.replace('-','')
    zarr_end = args.zarr_end_date.replace('-','')
    experiment_name = args.experiment_name
    
    zarr_start_date = datetime.date(int(zarr_start[:4]),int(zarr_start[4:6]),int(zarr_start[6:]))
    zarr_end_date = datetime.date(int(zarr_end[:4]),int(zarr_end[4:6]),int(zarr_end[6:]))
    start_date = datetime.date(int(start[:4]),int(start[4:6]),int(start[6:]))
    end_date = datetime.date(int(end[:4]),int(end[4:6]),int(end[6:]))
    pred_dates = [start_date+datetime.timedelta(days=t) for t in range((end_date-start_date).days+1)]
    pred_path = 'predictions/unmerged_zarrs/' + experiment_name + '_unmerged_preds_' + str(zarr_start_date).replace('-','') + '_' + str(zarr_end_date).replace('-','') + '.zarr'
    output_nc_dir = 'predictions/merged_products/' + experiment_name
    os.makedirs(output_nc_dir, exist_ok = True)
    
    N_workers = args.n_cpu_workers
    
    centers = pred_dates
    
    lock = multiprocessing.Lock()
    num_workers = N_workers
    batches_split = create_sublists(centers, num_workers)
   
    processes = []
    centers_per_worker = len(centers) // num_workers
    worker_center_list = []
    for w in range(num_workers-1):
        worker_center_list.append(centers[int(w*centers_per_worker):int((w+1)*centers_per_worker)])
    worker_center_list.append(centers[int((num_workers-1)*centers_per_worker):])
    
    for i in range(num_workers):
        worker_batches = batches_split[i]
        centers = centers[centers_per_worker:]

        process = multiprocessing.Process(target=worker, args=(lock, worker_batches))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
