#!/usr/bin/env python3
"""Build quarterly SST and SLA cache files from raw MUR zarrs and CMEMS SLA files.

Run once (or re-run with --force to rebuild). Idempotent: skips quarters whose
output files already exist.

Usage:
    python create_cache.py \
        --start 1993-01-01 --end 2024-01-01 \
        --sst_nc_dir input_data/mur_coarse_nc \
        --cmems_sla_dir input_data/cmems_sla \
        --sst_out_dir input_data/sst_cache \
        --sla_out_dir input_data/sla_cache \
        --n_t 30
"""

import argparse
import datetime
import math
import os
import sys
import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.interp_utils import EPOCH, load_ssh_by_date_range

# ---------- Chunk generation ----------

def generate_quarterly_chunks(start_date, end_date):
    """Return list of (chunk_start, chunk_end) date pairs covering [start_date, end_date)."""
    chunks = []
    year = start_date.year
    while True:
        for month in [1, 4, 7, 10]:
            cs = datetime.date(year, month, 1)
            ce = datetime.date(year + 1, 1, 1) if month == 10 else datetime.date(year, month + 3, 1)
            if ce <= start_date:
                continue
            if cs >= end_date:
                return chunks
            chunks.append((cs, ce))
        year += 1

# ---------- SST cache ----------

def build_sst_chunk(chunk_start, chunk_end, sst_nc_dir, sst_out_dir, n_t, force):
    """Build one SST cache file: (N_days_buf, N_lat, N_lon) float16 numpy array.

    Index 0 corresponds to day (chunk_start - n_t//2).
    Land/ice/missing values stored as 0.0.
    """
    out_path = os.path.join(sst_out_dir, f'sst_{chunk_start}_{chunk_end}.npy')
    lat_path = os.path.join(sst_out_dir, 'sst_lat.npy')
    lon_path = os.path.join(sst_out_dir, 'sst_lon.npy')

    if os.path.exists(out_path) and not force:
        print(f'  [SST] Skipping {out_path} (exists)')
        return

    buf_start = chunk_start - datetime.timedelta(days=n_t // 2)
    buf_end   = chunk_end   + datetime.timedelta(days=n_t // 2)
    n_days_buf = (buf_end - buf_start).days

    sst_lat = None
    sst_lon = None
    sst_shape = None
    sst_arrays = []

    for i in tqdm(range(n_days_buf), desc=f'SST {chunk_start}–{chunk_end}'):
        day = buf_start + datetime.timedelta(days=i)
        fpath = os.path.join(sst_nc_dir, day.strftime('%Y%m%d') + '.nc')

        if not os.path.exists(fpath):
            sst_arrays.append(None)
            continue

        ds = xr.open_dataset(fpath)
        sst = ds['analysed_sst'].values.squeeze().astype(np.float32)
        sst[np.isnan(sst)] = 0.0
        sst_arrays.append(sst.astype(np.float16))

        if sst_lat is None:
            sst_lat = ds['lat'].values.astype(np.float32)
            sst_lon = ds['lon'].values.astype(np.float32)
            sst_shape = sst.shape
        ds.close()

    if sst_lat is None:
        print(f'  [SST] WARNING: no zarr files found for {chunk_start}–{chunk_end}, skipping')
        return

    # Fill missing days with zeros
    zero_fill = np.zeros(sst_shape, dtype=np.float16)
    sst_arrays = [a if a is not None else zero_fill for a in sst_arrays]

    stacked = np.stack(sst_arrays, axis=0)  # (N_days_buf, N_lat, N_lon)
    print(f'  [SST] Saving {out_path}  shape={stacked.shape}  dtype=float16')
    np.save(out_path, stacked)

    if not os.path.exists(lat_path):
        np.save(lat_path, sst_lat)
        print(f'  [SST] Saved {lat_path}')
    if not os.path.exists(lon_path):
        np.save(lon_path, sst_lon)
        print(f'  [SST] Saved {lon_path}')

# ---------- SLA cache ----------

def build_sla_chunk(chunk_start, chunk_end, cmems_dir, sla_out_dir, n_t,
                    t_bin_size, lat_bin_size, lon_bin_size, force):
    """Build SSH quarterly cache: flat sorted observation array + 3D bin index.

    Output files:
        sla_{cs}_{ce}_data.npy    (N_obs, 5) float32: lat, lon, day_abs, sla_unfilt, sla_filt
        sla_{cs}_{ce}_sats.npy   (N_obs,)   int8:    satellite ID per observation
        sla_{cs}_{ce}_idx.npy    (n_t_bins, n_lat_bins, n_lon_bins, 2) int32: [row_start, row_end]
        sla_{cs}_{ce}_satnames.npy (N_sats,) str:     maps int8 ID → satellite name string

    t_offset (absolute day of array row 0) = (chunk_start - n_t//2 - EPOCH).days
    """
    prefix = os.path.join(sla_out_dir, f'sla_{chunk_start}_{chunk_end}')
    if all(os.path.exists(f'{prefix}_{s}.npy') for s in ['data', 'sats', 'idx', 'satnames']) and not force:
        print(f'  [SLA] Skipping {prefix} (exists)')
        return

    buf_start = chunk_start - datetime.timedelta(days=n_t // 2)
    buf_end   = chunk_end   + datetime.timedelta(days=n_t // 2)
    n_days_buf = (buf_end - buf_start).days
    t_offset = (buf_start - EPOCH).days

    print(f'  [SLA] Loading SSH {buf_start} – {buf_end}  (t_offset={t_offset})')
    dates, ssh_datasets = load_ssh_by_date_range(buf_start, buf_end, dir_name=cmems_dir)

    if not ssh_datasets:
        print(f'  [SLA] WARNING: no data found, skipping')
        return

    ds = xr.concat(ssh_datasets, dim='time')
    del ssh_datasets

    lat_vals = ds['latitude'].values.ravel().astype(np.float32)
    lon_vals = ds['longitude'].values.ravel().astype(np.float32)
    lon_vals = ((lon_vals + 180) % 360) - 180

    epoch_np = np.datetime64(EPOCH, 'D')
    day_abs = ((ds['time'].values.astype('datetime64[D]') - epoch_np)
               / np.timedelta64(1, 'D')).astype(np.float32).ravel()

    sla_unfiltered = ds['sla_unfiltered'].values.ravel().astype(np.float32)
    sla_filtered   = ds['sla_filtered'].values.ravel().astype(np.float32)
    sat_names_raw  = ds['satellite'].values.ravel()
    del ds

    # Satellite name → int8 ID mapping
    unique_sats = sorted(set(str(s) for s in sat_names_raw))
    sat2id  = {name: np.int8(i) for i, name in enumerate(unique_sats)}
    satnames = np.array(unique_sats)
    sat_ids  = np.array([sat2id[str(s)] for s in sat_names_raw], dtype=np.int8)

    # Bin counts
    n_t_bins   = math.ceil(n_days_buf / t_bin_size)
    n_lat_bins = math.ceil(180 / lat_bin_size)
    n_lon_bins = math.ceil(360 / lon_bin_size)

    t_rel = day_abs - t_offset
    t_b   = np.clip((t_rel // t_bin_size).astype(np.int64), 0, n_t_bins   - 1)
    lat_b = np.clip(((lat_vals + 90)  // lat_bin_size).astype(np.int64), 0, n_lat_bins - 1)
    lon_b = np.clip(((lon_vals + 180) // lon_bin_size).astype(np.int64), 0, n_lon_bins - 1)

    sort_key = t_b * (n_lat_bins * n_lon_bins) + lat_b * n_lon_bins + lon_b
    order = np.argsort(sort_key, kind='stable')

    data_sorted = np.column_stack(
        (lat_vals, lon_vals, day_abs, sla_unfiltered, sla_filtered)
    )[order].astype(np.float32)
    sats_sorted = sat_ids[order]
    keys_sorted = sort_key[order]

    # Build 3D bin index from sorted keys
    ssh_idx = np.zeros((n_t_bins, n_lat_bins, n_lon_bins, 2), dtype=np.int32)
    unique_keys, first_occ = np.unique(keys_sorted, return_index=True)

    for k_pos in range(len(unique_keys)):
        key = int(unique_keys[k_pos])
        tb  = key // (n_lat_bins * n_lon_bins)
        rem = key  % (n_lat_bins * n_lon_bins)
        lb  = rem  // n_lon_bins
        lob = rem   % n_lon_bins
        r0  = int(first_occ[k_pos])
        r1  = int(first_occ[k_pos + 1]) if k_pos + 1 < len(first_occ) else len(order)
        ssh_idx[tb, lb, lob] = [r0, r1]

    # Metadata: [t_bin_size, lat_bin_size, lon_bin_size, t_offset]
    meta = np.array([t_bin_size, lat_bin_size, lon_bin_size, t_offset], dtype=np.int32)

    print(f'  [SLA] Saving {prefix}_*.npy  N_obs={len(data_sorted)}  '
          f'n_t_bins={n_t_bins}  sats={list(unique_sats)}')
    np.save(f'{prefix}_data.npy',     data_sorted)
    np.save(f'{prefix}_sats.npy',     sats_sorted)
    np.save(f'{prefix}_idx.npy',      ssh_idx)
    np.save(f'{prefix}_satnames.npy', satnames)
    np.save(f'{prefix}_meta.npy',     meta)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description='Build quarterly NeurOST training/inference cache')
    parser.add_argument('--start',         required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end',           required=True, help='End date YYYY-MM-DD (exclusive)')
    parser.add_argument('--sst_nc_dir',    default='input_data/mur_coarse_nc')
    parser.add_argument('--cmems_sla_dir', default='input_data/cmems_sla')
    parser.add_argument('--sst_out_dir',   default='input_data/sst_cache')
    parser.add_argument('--sla_out_dir',   default='input_data/sla_cache')
    parser.add_argument('--n_t',           type=int, default=30)
    parser.add_argument('--t_bin_size',    type=int, default=10)
    parser.add_argument('--lat_bin_size',  type=int, default=10)
    parser.add_argument('--lon_bin_size',  type=int, default=10)
    parser.add_argument('--no_sst', action='store_true', help='Skip SST cache creation')
    parser.add_argument('--no_sla', action='store_true', help='Skip SLA cache creation')
    parser.add_argument('--force',  action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)

    os.makedirs(args.sst_out_dir, exist_ok=True)
    os.makedirs(args.sla_out_dir, exist_ok=True)

    chunks = generate_quarterly_chunks(start_date, end_date)
    print(f'Processing {len(chunks)} quarterly chunks: {start_date} → {end_date}')

    for chunk_start, chunk_end in chunks:
        print(f'\n=== Chunk {chunk_start} – {chunk_end} ===')
        if not args.no_sst:
            build_sst_chunk(chunk_start, chunk_end,
                            args.sst_nc_dir, args.sst_out_dir,
                            args.n_t, args.force)
        if not args.no_sla:
            build_sla_chunk(chunk_start, chunk_end,
                            args.cmems_sla_dir, args.sla_out_dir,
                            args.n_t,
                            args.t_bin_size, args.lat_bin_size, args.lon_bin_size,
                            args.force)

    print('\nCache build complete.')

if __name__ == '__main__':
    main()
