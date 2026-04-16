import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform
import xarray as xr
import datetime
import os
from tqdm import tqdm

# Reference epoch for absolute day indexing (standard CMEMS/AVISO epoch)
EPOCH = datetime.date(1993, 1, 1)

# ---------- File utilities ----------

def GetListOfFiles(dirName, ext='.nc'):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)
    return allFiles

def empty_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

# ---------- Coordinate transforms ----------

transformer_ll2xyz = pyproj.Transformer.from_crs(
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
)
transformer_xyz2ll = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
)

def xyz2ll(x, y, z, lat_org, lon_org, alt_org, transformer1, transformer2):
    x_org, y_org, z_org = transformer1.transform(lon_org, lat_org, alt_org, radians=False)
    ecef_org = np.array([[x_org, y_org, z_org]]).T
    rot1 = scipy.spatial.transform.Rotation.from_euler('x', -(90 - lat_org), degrees=True).as_matrix()
    rot3 = scipy.spatial.transform.Rotation.from_euler('z', -(90 + lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)
    ecefDelta = rotMatrix.T.dot(np.stack([x, y, np.zeros_like(x)], axis=-1).T)
    ecef = ecefDelta + ecef_org
    lon, lat, alt = transformer2.transform(ecef[0, :], ecef[1, :], ecef[2, :], radians=False)
    return lat, lon

def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer, rot_matrix=None, ecef_origin=None):
    """Convert lat/lon to ENU coordinates. Accepts pre-computed rot_matrix and ecef_origin
    to skip Euler reconstruction when called in a tight loop."""
    x, y, z = transformer.transform(lon, lat, np.zeros_like(lon), radians=False)

    if ecef_origin is not None:
        x_org, y_org, z_org = ecef_origin[0], ecef_origin[1], ecef_origin[2]
    else:
        x_org, y_org, z_org = transformer.transform(lon_org, lat_org, alt_org, radians=False)

    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T

    if rot_matrix is not None:
        rotMatrix = rot_matrix
    else:
        rot1 = scipy.spatial.transform.Rotation.from_euler('x', -(90 - lat_org), degrees=True).as_matrix()
        rot3 = scipy.spatial.transform.Rotation.from_euler('z', -(90 + lon_org), degrees=True).as_matrix()
        rotMatrix = rot1.dot(rot3)

    enu = rotMatrix.dot(vec)
    X = enu.T[0, :, 0]
    Y = enu.T[0, :, 1]
    Z = enu.T[0, :, 2]
    return X, Y, Z

def box(x_bounds, y_bounds, refinement=100):
    xs = np.zeros(int(4 * refinement))
    ys = np.zeros(int(4 * refinement))
    xs[:refinement] = np.linspace(x_bounds[0], x_bounds[-1], num=refinement)
    ys[:refinement] = np.linspace(y_bounds[0], y_bounds[0], num=refinement)
    xs[refinement:2 * refinement] = np.linspace(x_bounds[-1], x_bounds[-1], num=refinement)
    ys[refinement:2 * refinement] = np.linspace(y_bounds[0], y_bounds[-1], num=refinement)
    xs[2 * refinement:3 * refinement] = np.linspace(x_bounds[-1], x_bounds[0], num=refinement)
    ys[2 * refinement:3 * refinement] = np.linspace(y_bounds[-1], y_bounds[-1], num=refinement)
    xs[3 * refinement:] = np.linspace(x_bounds[0], x_bounds[0], num=refinement)
    ys[3 * refinement:] = np.linspace(y_bounds[-1], y_bounds[0], num=refinement)
    return xs, ys

# ---------- SSH loading from raw CMEMS files (used by create_cache.py) ----------

def load_multisat_ssh_single_day(ssh_files, satellite_names):
    datasets = []
    for f, sat_name in zip(ssh_files, satellite_names):
        ds = xr.open_dataset(f)
        sat_var = xr.DataArray(np.full(ds.sizes['time'], sat_name, dtype=object), dims=['time'])
        ds = ds.assign(satellite=sat_var)
        datasets.append(ds)
    return xr.concat(datasets, dim='time')

def get_next_dir_level(dir_name, file_paths):
    return [path.split(dir_name + os.sep, 1)[-1].split(os.sep, 1)[0]
            for path in file_paths if dir_name + os.sep in path]

def load_ssh_by_date_range(start_date, end_date, dir_name='input_data/cmems_sla'):
    ssh_files_total = GetListOfFiles(dir_name)
    n_days = (end_date - start_date).days
    dates = [start_date + datetime.timedelta(days=t) for t in range(n_days)]
    ssh_datasets = []
    print('Number of days to load: ' + str(n_days))
    for t in tqdm(range(n_days), desc="SSH loading progress"):
        ssh_files = [f for f in ssh_files_total
                     if datetime.date(int(f[-20:-16]), int(f[-16:-14]), int(f[-14:-12])) == dates[t]]
        sat_names = get_next_dir_level(dir_name, ssh_files)
        if len(ssh_files) > 0:
            ds = load_multisat_ssh_single_day(ssh_files, sat_names)
            ssh_datasets.append(ds)
    return dates, ssh_datasets

# ---------- SSH gridding ----------

def fast_bin_2d(x, y, values, n, x_range, y_range):
    """Mean-binned statistic on a regular 2D grid.

    Drop-in replacement for scipy.stats.binned_statistic_2d using digitize + bincount.
    Returns (n, n) with x as first axis, y as second (same convention as scipy).
    """
    if x.size == 0:
        return np.full((n, n), np.nan)

    x_min, x_max = x_range
    y_min, y_max = y_range

    x_edges = np.linspace(x_min, x_max, n + 1)
    y_edges = np.linspace(y_min, y_max, n + 1)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1

    valid = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    xi, yi = xi[valid], yi[valid]
    vals = values[valid].astype(np.float64)

    linear_idx = xi * n + yi
    sum_grid = np.bincount(linear_idx, weights=vals, minlength=n * n).reshape(n, n)
    cnt_grid = np.bincount(linear_idx, minlength=n * n).reshape(n, n)

    with np.errstate(invalid='ignore'):
        mean_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)

    return mean_grid

def bin_ssh(data, L_x=960e3, L_y=960e3, n=128, n_t=30, filtered=False):
    ssh_grid = np.zeros((n_t, n, n))
    data[np.isnan(data)] = 0

    for t in range(n_t):
        mask = (data[:, 3] == t)
        x, y, ssh = data[mask, 0], data[mask, 1], data[mask, 2]
        input_grid = fast_bin_2d(x, y, ssh, n=n,
                                 x_range=[-L_x / 2, L_x / 2],
                                 y_range=[-L_y / 2, L_y / 2])
        input_grid = np.rot90(input_grid)
        input_grid[np.isnan(input_grid)] = 0
        ssh_grid[t, :, :] = input_grid

    return ssh_grid

# ---------- Pre-computation (one-time at dataset init) ----------

def precompute_region_rotation_matrices(coord_grids, n=128):
    """Pre-compute ENU rotation matrices and ECEF origins for all regions.

    Returns:
        rot_matrices:  (N_regions, 3, 3) float64
        ecef_origins:  (N_regions, 3)    float64
    """
    N_regions = coord_grids.shape[0]
    rot_matrices = np.zeros((N_regions, 3, 3), dtype=np.float64)
    ecef_origins = np.zeros((N_regions, 3), dtype=np.float64)

    mid = n // 2
    for r in range(N_regions):
        lon0 = 0.25 * (coord_grids[r, mid - 1, mid - 1, 0] + coord_grids[r, mid - 1, mid, 0] +
                       coord_grids[r, mid, mid - 1, 0] + coord_grids[r, mid, mid, 0])
        lat0 = 0.25 * (coord_grids[r, mid - 1, mid - 1, 1] + coord_grids[r, mid - 1, mid, 1] +
                       coord_grids[r, mid, mid - 1, 1] + coord_grids[r, mid, mid, 1])

        x_org, y_org, z_org = transformer_ll2xyz.transform(lon0, lat0, 0.0, radians=False)
        ecef_origins[r] = [x_org, y_org, z_org]

        rot1 = scipy.spatial.transform.Rotation.from_euler('x', -(90 - lat0), degrees=True).as_matrix()
        rot3 = scipy.spatial.transform.Rotation.from_euler('z', -(90 + lon0), degrees=True).as_matrix()
        rot_matrices[r] = rot1.dot(rot3)

    return rot_matrices, ecef_origins

def precompute_sst_bilinear_weights(coord_grids, sst_lat, sst_lon, n=128):
    """Pre-compute bilinear interpolation weights from SST grid to all ENU patches.

    Args:
        coord_grids: (N_regions, n, n, 2) — lon/lat of each output pixel
        sst_lat:     (N_lat,) 1D latitude coordinate of the SST grid (ascending)
        sst_lon:     (N_lon,) 1D longitude coordinate of the SST grid (ascending)

    Returns:
        bilinear_idx: (N_regions, n*n, 4) int32, flat indices into flattened (N_lat, N_lon) SST
        bilinear_w:   (N_regions, n*n, 4) float32, weights summing to 1
    """
    N_regions = coord_grids.shape[0]
    N_pix = n * n
    N_lat = len(sst_lat)
    N_lon = len(sst_lon)

    bilinear_idx = np.zeros((N_regions, N_pix, 4), dtype=np.int32)
    bilinear_w = np.zeros((N_regions, N_pix, 4), dtype=np.float32)

    dlat = float(sst_lat[1] - sst_lat[0])
    dlon = float(sst_lon[1] - sst_lon[0])
    lat0_sst = float(sst_lat[0])
    lon0_sst = float(sst_lon[0])
    lon_range = N_lon * dlon

    for r in tqdm(range(N_regions), desc="Pre-computing SST bilinear weights"):
        lons = coord_grids[r, :, :, 0].ravel().astype(np.float64)
        lats = coord_grids[r, :, :, 1].ravel().astype(np.float64)

        i_f = (lats - lat0_sst) / dlat
        j_f = ((lons - lon0_sst) % lon_range) / dlon

        i0 = np.floor(i_f).astype(np.int32)
        j0 = np.floor(j_f).astype(np.int32)
        i1 = i0 + 1
        j1 = j0 + 1

        di = (i_f - i0).astype(np.float32)
        dj = (j_f - j0).astype(np.float32)

        i0 = np.clip(i0, 0, N_lat - 1)
        i1 = np.clip(i1, 0, N_lat - 1)
        j0 = j0 % N_lon
        j1 = j1 % N_lon

        bilinear_idx[r, :, 0] = i0 * N_lon + j0
        bilinear_idx[r, :, 1] = i0 * N_lon + j1
        bilinear_idx[r, :, 2] = i1 * N_lon + j0
        bilinear_idx[r, :, 3] = i1 * N_lon + j1

        bilinear_w[r, :, 0] = (1 - di) * (1 - dj)
        bilinear_w[r, :, 1] = (1 - di) * dj
        bilinear_w[r, :, 2] = di * (1 - dj)
        bilinear_w[r, :, 3] = di * dj

    return bilinear_idx, bilinear_w

def apply_sst_bilinear(sst_slice, bilinear_idx_r, bilinear_w_r, n=128):
    """Apply pre-computed bilinear weights to extract an SST patch from a pre-loaded array.

    Args:
        sst_slice:       (N_t, N_lat, N_lon) float16/float32 — pre-loaded SST, 0 = no-data
        bilinear_idx_r:  (n*n, 4) int32 — flat indices into (N_lat, N_lon) for one region
        bilinear_w_r:    (n*n, 4) float32 — bilinear weights for that region

    Returns:
        sst_patch: (N_t, n, n) float32, 0 where no data
    """
    N_t = sst_slice.shape[0]
    sst_flat = sst_slice.reshape(N_t, -1).astype(np.float32)  # (N_t, N_lat*N_lon)

    # sst_flat[:, bilinear_idx_r] → (N_t, N_pix, 4), then weighted sum → (N_t, N_pix)
    sst_patch = (sst_flat[:, bilinear_idx_r] * bilinear_w_r[np.newaxis]).sum(axis=-1)

    # Match original masking: pixels below freezing are fill (land/ice) → zero
    sst_patch[sst_patch < 273.0] = 0.0

    return sst_patch.reshape(N_t, n, n)

# ---------- In-RAM SSH query ----------

def load_query_data_np(query_t_start, query_t_end,
                       query_lat_min, query_lat_max,
                       query_lon_min, query_lon_max,
                       ssh_data, ssh_sats, ssh_idx,
                       t_bin_size=10, lat_bin_size=10, lon_bin_size=10,
                       t_offset=0):
    """Query flat in-RAM SSH array for observations within a spatiotemporal bounding box.

    Args:
        query_t_start/end:   absolute day indices (days since EPOCH)
        query_lat/lon_*:     geographic bounding box (degrees)
        ssh_data:            (N_obs, 5) float32 sorted by (t_bin, lat_bin, lon_bin)
                             columns: lat, lon, day_abs, sla_unfiltered, sla_filtered
        ssh_sats:            (N_obs,) int8 satellite IDs
        ssh_idx:             (n_t_bins, n_lat_bins, n_lon_bins, 2) int32 row start/end per bin
        t_offset:            absolute day of the first element of the t-bin axis in ssh_idx
    """
    n_t_bins, n_lat_bins, n_lon_bins = ssh_idx.shape[:3]

    t0_bin = max(0, int((query_t_start - t_offset) // t_bin_size))
    t1_bin = min(n_t_bins - 1, int((query_t_end - t_offset) // t_bin_size))

    lat0_bin = max(0, int((query_lat_min + 90) // lat_bin_size))
    lat1_bin = min(n_lat_bins - 1, int((query_lat_max + 90) // lat_bin_size))

    if t0_bin > t1_bin or lat0_bin > lat1_bin:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int8)

    # Longitude — handle dateline crossing
    if query_lon_min <= query_lon_max:
        lon0_bin = max(0, int((query_lon_min + 180) // lon_bin_size))
        lon1_bin = min(n_lon_bins - 1, int((query_lon_max + 180) // lon_bin_size))
        lon_ranges = [(lon0_bin, lon1_bin)]
    else:
        # wraps across ±180
        lon_a0 = max(0, int((query_lon_min + 180) // lon_bin_size))
        lon_a1 = n_lon_bins - 1
        lon_b0 = 0
        lon_b1 = min(n_lon_bins - 1, int((query_lon_max + 180) // lon_bin_size))
        lon_ranges = [(lon_a0, lon_a1), (lon_b0, lon_b1)]

    data_list = []
    sat_list = []

    for t_b in range(t0_bin, t1_bin + 1):
        for lat_b in range(lat0_bin, lat1_bin + 1):
            for lon_lo, lon_hi in lon_ranges:
                for lon_b in range(lon_lo, lon_hi + 1):
                    r0, r1 = ssh_idx[t_b, lat_b, lon_b]
                    if r1 > r0:
                        data_list.append(ssh_data[r0:r1])
                        sat_list.append(ssh_sats[r0:r1])

    if data_list:
        return np.concatenate(data_list, axis=0), np.concatenate(sat_list, axis=0)
    return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int8)

def extract_tracks_np(t_abs_mid, lon0, lat0, coord_grid, rot_matrix, ecef_origin,
                      ssh_data, ssh_sats, ssh_satnames, ssh_idx,
                      t_offset=0,
                      t_bin_size=10, lat_bin_size=10, lon_bin_size=10,
                      n_t=30, L_x=960e3, L_y=960e3, filtered=False):
    """Extract SSH tracks for a given region and time window using in-RAM arrays.

    Returns:
        data:     (N, 4) float32  columns: x_enu, y_enu, sla, day_relative
        sats:     (N,)   str      satellite names
    Returns (zeros(1,4), None) if no data found.
    """
    lon_grid = coord_grid[:, :, 0]
    lat_grid = coord_grid[:, :, 1]
    lat_max = np.max(lat_grid) + 0.1
    lat_min = np.min(lat_grid) - 0.1
    lon_max = np.max(lon_grid[:, -1]) + 0.1
    lon_min = np.min(lon_grid[:, 0]) - 0.1

    t_abs_start = t_abs_mid - n_t // 2
    t_abs_end = t_abs_mid + n_t // 2

    data, sats_int = load_query_data_np(
        query_t_start=t_abs_start,
        query_t_end=t_abs_end,
        query_lat_min=lat_min,
        query_lat_max=lat_max,
        query_lon_min=lon_min,
        query_lon_max=lon_max,
        ssh_data=ssh_data,
        ssh_sats=ssh_sats,
        ssh_idx=ssh_idx,
        t_bin_size=t_bin_size,
        lat_bin_size=lat_bin_size,
        lon_bin_size=lon_bin_size,
        t_offset=t_offset,
    )

    if data.shape[0] == 0:
        return np.zeros((1, 4), dtype=np.float32), None

    lat = data[:, 0]
    lon = data[:, 1]
    day_abs = data[:, 2]
    sla = data[:, 4] if filtered else data[:, 3]

    day = (day_abs - t_abs_start).astype(np.float32)
    valid = (day >= 0) & (day < n_t)
    lon, lat, sla, day, sats_int = lon[valid], lat[valid], sla[valid], day[valid], sats_int[valid]

    if lon.size == 0:
        return np.zeros((1, 4), dtype=np.float32), None

    lon = (lon - lon0 + 180) % 360 - 180

    x, y, z = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz,
                     rot_matrix=rot_matrix, ecef_origin=ecef_origin)

    in_patch = (z > -1e6) & (-L_x / 2 < x) & (x < L_x / 2) & (-L_y / 2 < y) & (y < L_y / 2)
    x, y, sla, day, sats_int = x[in_patch], y[in_patch], sla[in_patch], day[in_patch], sats_int[in_patch]

    if x.size == 0:
        return np.zeros((1, 4), dtype=np.float32), None

    sats = np.array([ssh_satnames[s] for s in sats_int])
    return np.column_stack((x, y, sla, day)).astype(np.float32), sats

def get_ssh_np(r, t_abs, coord_grids, rot_matrices, ecef_origins,
               ssh_data, ssh_sats, ssh_satnames, ssh_idx,
               t_offset=0,
               t_bin_size=10, lat_bin_size=10, lon_bin_size=10,
               n_t=30, n=128, L_x=960e3, L_y=960e3,
               leave_out_altimeters=True, withhold_sat='random', filtered=False):
    """Get SSH input grid and withheld ground-truth tracks for one region and time step.

    Args:
        r:       region index
        t_abs:   absolute day index (days since EPOCH = 1993-01-01)

    Returns:
        ssh_grid:  (n_t, n, n) float32
        out_data:  (n_t, max_pts, 3) float32 or None
    """
    mid = n // 2
    lon0 = 0.25 * (coord_grids[r, mid - 1, mid - 1, 0] + coord_grids[r, mid - 1, mid, 0] +
                   coord_grids[r, mid, mid - 1, 0] + coord_grids[r, mid, mid, 0])
    lat0 = 0.25 * (coord_grids[r, mid - 1, mid - 1, 1] + coord_grids[r, mid - 1, mid, 1] +
                   coord_grids[r, mid, mid - 1, 1] + coord_grids[r, mid, mid, 1])

    d, s = extract_tracks_np(
        t_abs_mid=t_abs,
        lon0=lon0,
        lat0=lat0,
        coord_grid=coord_grids[r],
        rot_matrix=rot_matrices[r],
        ecef_origin=ecef_origins[r],
        ssh_data=ssh_data,
        ssh_sats=ssh_sats,
        ssh_satnames=ssh_satnames,
        ssh_idx=ssh_idx,
        t_offset=t_offset,
        t_bin_size=t_bin_size,
        lat_bin_size=lat_bin_size,
        lon_bin_size=lon_bin_size,
        n_t=n_t,
        L_x=L_x,
        L_y=L_y,
        filtered=filtered,
    )

    if d.shape[0] > 1:
        if leave_out_altimeters:
            sats_all = np.unique(s)
            if withhold_sat == 'random':
                withhold = np.random.choice(sats_all)
            else:
                withhold = withhold_sat

            mask = (s != withhold) if isinstance(withhold, str) else ~np.isin(s, withhold)
            d_out = d[~mask]
            d, s = d[mask], s[mask]

            out_tracks = []
            for t in range(n_t):
                t_mask = (d_out[:, 3] == t)
                out_tracks.append(d_out[t_mask, :])

            len_max = max(tr.shape[0] for tr in out_tracks) if any(tr.shape[0] > 0 for tr in out_tracks) else 1
            out_data = np.zeros((n_t, len_max, 3), dtype=np.float32)
            for t in range(n_t):
                if out_tracks[t].shape[0] > 0:
                    out_data[t, :out_tracks[t].shape[0]] = out_tracks[t][:, :3]
        else:
            out_data = None

        ssh_grid = bin_ssh(d, L_x=L_x, L_y=L_y, n=n, n_t=n_t)
    else:
        ssh_grid = np.zeros((n_t, n, n), dtype=np.float32)
        out_data = None

    return ssh_grid, out_data
