import numpy as np
import datetime
import multiprocessing
from src.merging import *
import gc
import sys

def worker(lock, batches):
    while True:
        with lock:
            if not batches:
                break 

            batch = batches.pop(0)

        merge_maps_and_save_zarr(pred_path, zarr_start_date, batch, output_nc_dir, mask_filename, dist_filename, mdt_filename, network_name, coord_grid_path = 'input_data/coord_grids.npy',L=250e3, crop_pixels=9, dx=7.5e3, with_grads=True, mask_coast_dist=0, lon_min=-180 ,lon_max=180, lat_min=-70, lat_max=80, res=1/10, progress=False, mask_ice=True, sst_zarr_path = 'input_data/mur_coarse_zarrs')

        gc.collect()

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':

    experiment_name, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    start = start.replace('-','')
    end = end.replace('-','')
    
    zarr_start_date = datetime.date(int(start[:4]),int(start[4:6]),int(start[6:]))
    start_date = datetime.date(int(start[:4]),int(start[4:6]),int(start[6:]))
    end_date = datetime.date(int(end[:4]),int(end[4:6]),int(end[6:]))
    pred_dates = [start_date+datetime.timedelta(days=t) for t in range((end_date-start_date).days+1)]
    pred_path = 'predictions/unmerged_zarrs/' + experiment_name + '_unmerged_preds_' + str(start_date).replace('-','') + '_' + str(end_date).replace('-','') + '.zarr'
    output_nc_dir = 'predictions/merged_products/' + experiment_name
    os.makedirs(output_nc_dir, exist_ok = True)
    mask_filename = 'input_data/auxilliary/land_water_mask_10grid.nc'
    dist_filename = 'input_data/auxilliary/distance_to_nearest_coastlines_10grid.nc'
    mdt_filename = 'input_data/auxilliary/mdt_hybrid_cnes_cls18_cmems2020_global.nc'
    network_name = 'SimVP_SSH_SST_1M_global'
    N_workers = 3
    
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
