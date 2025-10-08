#!/usr/bin/env python3
import os
import sys
import subprocess
from datetime import datetime, date, timedelta
from glob import glob
import xarray as xr
import copernicusmarine as cm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# ---------- Config ----------
remove_mur_nc = True
cmems_dt_to_nrt = date(2024, 6, 14)  # transition date used for 'mixed' mode
# ---------- Satellites ----------
sats_nrt = ['al','c2n','h2b','j3n','s3a','s3b','s6a-hr','swon']
sats_delayed = ['c2','c2n','en','enn','e1','e1g','e2','g2','h2a','h2ag','h2b',
                'j1','j1g','j1n','j2','j2n','j2g','j3','j3n','al','alg',
                's3a','s3b','s6a-lr','swon','swonc','tp','tpn']

# ---------- Helpers ----------
def format_date_to_string(date_object):
    """YYYY-MM-DDTHH:MM:SSZ"""
    return datetime(date_object.year, date_object.month, date_object.day, 0, 0, 0).isoformat() + 'Z'

def generate_date_list(start_date, end_date):
    """Return list of CMEMS filters like '*/YYYY/MM/*.nc' (unique)."""
    lst = [f"*/{(start_date + timedelta(days=i)).strftime('%Y/%m')}/*.nc" for i in range((end_date - start_date).days + 1)]
    return sorted(set(lst))

# ---------- MUR: per-worker process ----------
def download_and_coarsen_one_day(args):
    """
    args -> (date_obj, worker_id)
    Downloads MUR for that single day into worker-specific tmp dir and coarsens files found there.
    """
    d, worker_id = args
    date_str = format_date_to_string(d)
    day_label = d.strftime("%Y-%m-%d")
    tmp_dir = f'input_data/mur_sst_tmp/worker_{worker_id}'
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[MUR worker {worker_id}] Starting {day_label}")

    # build command list for subprocess
    cmd = [
        "podaac-data-downloader",
        "-c", "MUR-JPL-L4-GLOB-v4.1",
        "-d", tmp_dir,
        "--start-date", date_str,
        "--end-date", date_str,
        "-b=-180,-90,180,90"
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[MUR worker {worker_id}] Downloader failed for {day_label}: {e}")
        return

    files = sorted(glob(os.path.join(tmp_dir, '*.nc')))
    if not files:
        print(f"[MUR worker {worker_id}] No files found for {day_label} in {tmp_dir}")
    for f in files:
        fname = os.path.basename(f)
        try:
            print(f"[MUR worker {worker_id}] Coarsening {fname}")
            ds_sst = xr.open_dataset(f)
            zarr_path = os.path.join('input_data', 'mur_coarse_zarrs', str(ds_sst['time'].values[0])[:10].replace('-', '') + '.zarr')
            if os.path.isdir(zarr_path):
                print(f"[MUR worker {worker_id}] {zarr_path} already exists; skipping")
            else:
                sst = (ds_sst[['analysed_sst', 'sea_ice_fraction']]
                       .load()
                       .astype('float32')
                       .coarsen({'lon': 5, 'lat': 5}, boundary='trim')
                       .mean())
                sst = sst.chunk({'time': 1, 'lon': 1000, 'lat': 1000})
                sst.to_zarr(zarr_path)
            ds_sst.close()
        except Exception as e:
            print(f"[MUR worker {worker_id}] Error coarsening {fname}: {e}")
        finally:
            if remove_mur_nc:
                try:
                    os.remove(f)
                except OSError:
                    pass
    # keep worker tmp dir persistent (not removed) to avoid race with other workers

# ---------- CMEMS: per-(sat,search) threaded -->
def cmems_download_task(args):
    """
    args -> (sat, search_pattern, mode) where mode in {'nrt', 'delayed'}
    """
    sat, search, mode = args
    outdir = os.path.join('input_data', 'cmems_sla', sat)
    os.makedirs(outdir, exist_ok=True)
    ds_id = ("cmems_obs-sl_glo_phy-ssh_nrt_" + sat + "-l3-duacs_PT1S") if mode == 'nrt' else ("cmems_obs-sl_glo_phy-ssh_my_" + sat + "-l3-duacs_PT1S")
    try:
        print(f"[CMEMS {mode}] {sat} -> {search}")
        cm.get(
            dataset_id = ds_id,
            filter = search,
            output_directory = outdir,
            no_directories = True,
            overwrite = True
        )
    except Exception as e:
        print(f"[CMEMS {mode}] Error downloading {sat} {search}: {e}")

# ---------- Main ----------
def main():
    if len(sys.argv) < 4:
        print("Usage: python downloader.py <start_date> <end_date> <n_workers>")
        sys.exit(1)

    date_start_input, date_end_input, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    date_start = datetime.strptime(date_start_input, "%Y-%m-%d").date()
    date_end = datetime.strptime(date_end_input, "%Y-%m-%d").date()
    if date_end < date_start:
        raise SystemExit("end_date must be >= start_date")

    # ensure copernicus credentials
    cred_path = os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials")
    if not os.path.exists(cred_path):
        print("Logging in to Copernicus (cm)...")
        cm.login()

    # make base dirs
    os.makedirs('input_data/mur_sst_tmp', exist_ok=True)
    os.makedirs('input_data/mur_coarse_zarrs', exist_ok=True)
    os.makedirs('input_data/cmems_sla', exist_ok=True)

    # ------------- MUR (process pool) -------------
    dates = [date_start + timedelta(days=i) for i in range((date_end - date_start).days + 1)]
    # Round-robin assign dates to worker IDs so worker-owned tmp dirs are reused
    worker_assignments = [(d, i % n_workers) for i, d in enumerate(dates)]

    print(f"Starting MUR downloads/coarsening for {len(dates)} days with {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(download_and_coarsen_one_day, worker_assignments))

    # ------------- CMEMS (thread pool) -------------
    # Decide mode (nrt/delayed/mixed)
    if date_start < cmems_dt_to_nrt:
        mode = 'delayed' if date_end < cmems_dt_to_nrt else 'mixed'
    else:
        mode = 'nrt'
    print(f"CMEMS mode = {mode}")

    tasks = []
    if mode == 'nrt':
        search_strings = generate_date_list(date_start, date_end)
        for sat in sats_nrt:
            for s in search_strings:
                tasks.append((sat, s, 'nrt'))
    elif mode == 'delayed':
        search_strings = generate_date_list(date_start, date_end)
        for sat in sats_delayed:
            for s in search_strings:
                tasks.append((sat, s, 'delayed'))
    elif mode == 'mixed':
        # delayed portion (date_start .. cmems_dt_to_nrt)
        delayed_searches = generate_date_list(date_start, cmems_dt_to_nrt)
        for sat in sats_delayed:
            for s in delayed_searches:
                tasks.append((sat, s, 'delayed'))
        # nrt portion (cmems_dt_to_nrt .. date_end)
        nrt_searches = generate_date_list(cmems_dt_to_nrt, date_end)
        for sat in sats_nrt:
            for s in nrt_searches:
                tasks.append((sat, s, 'nrt'))

    if tasks:
        print(f"Starting CMEMS downloads with {n_workers} threads ({len(tasks)} tasks)...")
        with ThreadPoolExecutor(max_workers=n_workers) as tpool:
            list(tpool.map(cmems_download_task, tasks))
    else:
        print("No CMEMS tasks to run for given dates/mode.")

    print("All downloads complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
