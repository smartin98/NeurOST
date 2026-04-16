import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import numpy as np
import datetime
import os
from src.interp_utils import (
    EPOCH,
    precompute_sst_bilinear_weights,
    precompute_region_rotation_matrices,
    apply_sst_bilinear,
    get_ssh_np,
)

# ---------- Internal helpers ----------

def _quarterly_chunks_for_range(start_date, end_date):
    """Return list of (chunk_start, chunk_end) quarterly dates overlapping [start_date, end_date]."""
    chunks = []
    year = start_date.year
    while True:
        for month in [1, 4, 7, 10]:
            cs = datetime.date(year, month, 1)
            ce = (datetime.date(year + 1, 1, 1) if month == 10
                  else datetime.date(year, month + 3, 1))
            if ce <= start_date:
                continue
            if cs > end_date:
                return chunks
            chunks.append((cs, ce))
        year += 1

def _load_sla_chunk(sla_cache_dir, chunk_start, chunk_end):
    """Load one quarterly SLA chunk from disk into a dict of numpy arrays."""
    prefix = os.path.join(sla_cache_dir, f'sla_{chunk_start}_{chunk_end}')
    meta = np.load(f'{prefix}_meta.npy')
    return {
        'data':           np.load(f'{prefix}_data.npy'),
        'sats':           np.load(f'{prefix}_sats.npy'),
        'idx':            np.load(f'{prefix}_idx.npy'),
        'satnames':       np.load(f'{prefix}_satnames.npy', allow_pickle=True),
        't_bin_size':     int(meta[0]),
        'lat_bin_size':   int(meta[1]),
        'lon_bin_size':   int(meta[2]),
        't_offset':       int(meta[3]),
        'chunk_start_abs': (chunk_start - EPOCH).days,
        'chunk_end_abs':   (chunk_end   - EPOCH).days,
    }

def _get_sample(r, t_abs,
                sst_data, sst_buf_start_abs,
                ssh_chunk,
                coord_grids, rot_matrices, ecef_origins,
                bilinear_idx, bilinear_w,
                N_t, n, L_x, L_y,
                mean_ssh, std_ssh, mean_sst, std_sst,
                leave_out_altimeters, withhold_sat, filtered,
                use_sst, ssh_out_n_max):
    """Build one (input, output) training/inference sample from pre-loaded in-RAM data."""
    c = ssh_chunk
    ssh_in, ssh_out = get_ssh_np(
        r=r, t_abs=t_abs,
        coord_grids=coord_grids,
        rot_matrices=rot_matrices,
        ecef_origins=ecef_origins,
        ssh_data=c['data'], ssh_sats=c['sats'],
        ssh_satnames=c['satnames'], ssh_idx=c['idx'],
        t_offset=c['t_offset'],
        t_bin_size=c['t_bin_size'],
        lat_bin_size=c['lat_bin_size'],
        lon_bin_size=c['lon_bin_size'],
        n_t=N_t, n=n, L_x=L_x, L_y=L_y,
        leave_out_altimeters=leave_out_altimeters,
        withhold_sat=withhold_sat,
        filtered=filtered,
    )

    ssh_in[ssh_in != 0] = (ssh_in[ssh_in != 0] - mean_ssh) / std_ssh

    if use_sst:
        t_sst = (t_abs - N_t // 2) - sst_buf_start_abs
        sst = apply_sst_bilinear(
            sst_data[t_sst:t_sst + N_t],
            bilinear_idx[r], bilinear_w[r], n=n
        )
        sst[sst != 0] = (sst[sst != 0] - mean_sst) / std_sst

    if ssh_out is not None:
        x   = ssh_out[:, :, 0].copy()
        y   = ssh_out[:, :, 1].copy()
        ssh = ssh_out[:, :, 2].copy()
        x[x != 0]     = (x[x != 0] + 0.5 * L_x) * (n - 1) / L_x
        y[y != 0]     = (0.5 * L_y - y[y != 0]) * (n - 1) / L_y
        ssh[ssh != 0] = (ssh[ssh != 0] - mean_ssh) / std_ssh
        ssh_out = np.stack((x, y, ssh), axis=-1)
        if ssh_out.shape[1] < ssh_out_n_max:
            out = np.zeros((N_t, ssh_out_n_max, 3), dtype=np.float32)
            out[:, :ssh_out.shape[1]] = ssh_out
        else:
            out = ssh_out[:, :ssh_out_n_max, :]
    else:
        out = np.zeros((N_t, ssh_out_n_max, 3), dtype=np.float32)

    if use_sst:
        inp = np.stack((ssh_in, sst), axis=1).astype(np.float32)
    else:
        inp = np.expand_dims(ssh_in, axis=1).astype(np.float32)

    return torch.from_numpy(inp), torch.from_numpy(out.astype(np.float32))


# ---------- Inference / map-style dataset ----------

class NeurOST_dataset(Dataset):
    """Map-style dataset for inference. Loads quarterly cache files into RAM on first access."""

    def __init__(self,
                 sst_cache_dir,
                 sla_cache_dir,
                 start_date,
                 end_date,
                 N_t,
                 mean_ssh,
                 std_ssh,
                 mean_sst,
                 std_sst,
                 coord_grids,
                 n=128,
                 L_x=960e3,
                 L_y=960e3,
                 leave_out_altimeters=True,
                 withhold_sat='random',
                 filtered=False,
                 use_sst=True,
                 ssh_out_n_max=1000):

        self.sst_cache_dir  = sst_cache_dir
        self.sla_cache_dir  = sla_cache_dir
        self.start_date     = start_date
        self.end_date       = end_date
        self.N_t            = N_t
        self.mean_ssh       = mean_ssh
        self.std_ssh        = std_ssh
        self.mean_sst       = mean_sst
        self.std_sst        = std_sst
        self.coord_grids    = coord_grids
        self.n              = n
        self.L_x            = L_x
        self.L_y            = L_y
        self.leave_out_altimeters = leave_out_altimeters
        self.withhold_sat   = withhold_sat
        self.filtered       = filtered
        self.use_sst        = use_sst
        self.ssh_out_n_max  = ssh_out_n_max

        # Find quarterly chunks that contain at least one sample center day in [start_date, end_date].
        # Each chunk's buffered SST/SSH files already cover ±N_t//2 around their sample range.
        self.covering_chunks = _quarterly_chunks_for_range(start_date, end_date)
        if not self.covering_chunks:
            raise RuntimeError(
                f'No quarterly cache chunks found for date range {start_date}–{end_date}. '
                f'Run create_cache.py first.'
            )

        # Pre-compute bilinear weights and rotation matrices once at init
        sst_lat = np.load(os.path.join(sst_cache_dir, 'sst_lat.npy'))
        sst_lon = np.load(os.path.join(sst_cache_dir, 'sst_lon.npy'))
        self.bilinear_idx, self.bilinear_w = precompute_sst_bilinear_weights(
            coord_grids, sst_lat, sst_lon, n=n
        )
        self.rot_matrices, self.ecef_origins = precompute_region_rotation_matrices(
            coord_grids, n=n
        )

        # Build flat index arrays (absolute day indices × region indices)
        t_abs_start = (start_date - EPOCH).days
        t_abs_end   = (end_date   - EPOCH).days
        t_idxs = np.arange(t_abs_start, t_abs_end + 1)
        r_idxs = np.arange(coord_grids.shape[0])
        t_idxs, r_idxs = np.meshgrid(t_idxs, r_idxs)
        self.t_idxs = t_idxs.ravel()
        self.r_idxs = r_idxs.ravel()

        # Lazily loaded in-RAM data (populated by _load_data or worker_init_fn)
        self.sst_data         = None
        self.sst_buf_start_abs = None
        self.ssh_chunks       = None  # list of chunk dicts, ordered by chunk_start

    def __len__(self):
        return np.size(self.t_idxs)

    def _load_data(self):
        """Load all covering quarterly chunks into RAM. Called once per worker process."""
        # Build a combined SST array covering [start - N_t//2, end + N_t//2)
        buf_start_abs = (self.start_date - datetime.timedelta(days=self.N_t // 2) - EPOCH).days
        buf_end_abs   = (self.end_date   + datetime.timedelta(days=self.N_t // 2) - EPOCH).days + 1
        total_days    = buf_end_abs - buf_start_abs

        # Probe shape from first valid SST file
        sample_arr = None
        for cs, ce in self.covering_chunks:
            p = os.path.join(self.sst_cache_dir, f'sst_{cs}_{ce}.npy')
            if os.path.exists(p):
                sample_arr = np.load(p, mmap_mode='r')
                break
        if sample_arr is None:
            raise RuntimeError('No SST cache files found for the inference range.')

        N_lat, N_lon = sample_arr.shape[1], sample_arr.shape[2]
        sst_combined = np.zeros((total_days, N_lat, N_lon), dtype=np.float16)

        for cs, ce in self.covering_chunks:
            p = os.path.join(self.sst_cache_dir, f'sst_{cs}_{ce}.npy')
            if not os.path.exists(p):
                continue
            arr = np.load(p)   # (n_days_buf, N_lat, N_lon)
            chunk_buf_abs = ((cs - datetime.timedelta(days=self.N_t // 2)) - EPOCH).days
            offset = chunk_buf_abs - buf_start_abs
            dst0 = max(0, offset)
            dst1 = min(total_days, offset + arr.shape[0])
            src0 = dst0 - offset
            src1 = dst1 - offset
            sst_combined[dst0:dst1] = arr[src0:src1]

        self.sst_data          = sst_combined
        self.sst_buf_start_abs = buf_start_abs

        # Load SSH quarterly chunks
        self.ssh_chunks = []
        for cs, ce in self.covering_chunks:
            prefix = os.path.join(self.sla_cache_dir, f'sla_{cs}_{ce}')
            if not os.path.exists(f'{prefix}_data.npy'):
                continue
            self.ssh_chunks.append(_load_sla_chunk(self.sla_cache_dir, cs, ce))

        if not self.ssh_chunks:
            raise RuntimeError('No SLA cache files found for the inference range.')

    def _find_ssh_chunk(self, t_abs):
        """Return the SSH chunk whose unbuffered range contains t_abs."""
        for c in self.ssh_chunks:
            if c['chunk_start_abs'] <= t_abs < c['chunk_end_abs']:
                return c
        # Fall back to nearest chunk (handles boundary edge cases)
        return min(self.ssh_chunks, key=lambda c: min(
            abs(t_abs - c['chunk_start_abs']), abs(t_abs - c['chunk_end_abs'])
        ))

    def __getitem__(self, idx):
        if self.sst_data is None:
            self._load_data()

        r     = self.r_idxs[idx]
        t_abs = self.t_idxs[idx]

        return _get_sample(
            r=r, t_abs=t_abs,
            sst_data=self.sst_data,
            sst_buf_start_abs=self.sst_buf_start_abs,
            ssh_chunk=self._find_ssh_chunk(t_abs),
            coord_grids=self.coord_grids,
            rot_matrices=self.rot_matrices,
            ecef_origins=self.ecef_origins,
            bilinear_idx=self.bilinear_idx,
            bilinear_w=self.bilinear_w,
            N_t=self.N_t, n=self.n, L_x=self.L_x, L_y=self.L_y,
            mean_ssh=self.mean_ssh, std_ssh=self.std_ssh,
            mean_sst=self.mean_sst, std_sst=self.std_sst,
            leave_out_altimeters=self.leave_out_altimeters,
            withhold_sat=self.withhold_sat,
            filtered=self.filtered,
            use_sst=self.use_sst,
            ssh_out_n_max=self.ssh_out_n_max,
        )


# ---------- Training / iterable dataset ----------

class NeurOSTChunkedIterableDataset(IterableDataset):
    """Iterable dataset for training. Each worker loads one quarterly chunk into RAM and
    draws samples randomly, refreshing to a new random chunk every refresh_interval samples.

    chunk_list: list of (chunk_start, chunk_end) date pairs — pass all available quarters.
    """

    def __init__(self,
                 sst_cache_dir,
                 sla_cache_dir,
                 chunk_list,
                 coord_grids,
                 N_t=30,
                 n=128,
                 L_x=960e3,
                 L_y=960e3,
                 mean_ssh=0.074,
                 std_ssh=0.0986,
                 mean_sst=293.307,
                 std_sst=8.726,
                 refresh_interval=50000,
                 use_sst=True,
                 leave_out_altimeters=True,
                 withhold_sat='random',
                 filtered=False,
                 ssh_out_n_max=1000):

        self.sst_cache_dir    = sst_cache_dir
        self.sla_cache_dir    = sla_cache_dir
        self.chunk_list       = chunk_list
        self.coord_grids      = coord_grids
        self.N_t              = N_t
        self.n                = n
        self.L_x              = L_x
        self.L_y              = L_y
        self.mean_ssh         = mean_ssh
        self.std_ssh          = std_ssh
        self.mean_sst         = mean_sst
        self.std_sst          = std_sst
        self.refresh_interval = refresh_interval
        self.use_sst          = use_sst
        self.leave_out_altimeters = leave_out_altimeters
        self.withhold_sat     = withhold_sat
        self.filtered         = filtered
        self.ssh_out_n_max    = ssh_out_n_max

        # Pre-compute bilinear weights + rotation matrices once; shared across workers via fork
        sst_lat = np.load(os.path.join(sst_cache_dir, 'sst_lat.npy'))
        sst_lon = np.load(os.path.join(sst_cache_dir, 'sst_lon.npy'))
        self.bilinear_idx, self.bilinear_w = precompute_sst_bilinear_weights(
            coord_grids, sst_lat, sst_lon, n=n
        )
        self.rot_matrices, self.ecef_origins = precompute_region_rotation_matrices(
            coord_grids, n=n
        )

        # Per-worker state (populated in __iter__)
        self.sst_data          = None
        self.sst_buf_start_abs = None
        self.ssh_chunk         = None
        self.sample_queue      = None
        self.queue_pos         = 0

    def _load_chunk(self, chunk_idx):
        """Load one quarterly chunk into this worker's RAM. ~0.6 s sequential Lustre read."""
        cs, ce = self.chunk_list[chunk_idx]

        self.sst_data = np.load(
            os.path.join(self.sst_cache_dir, f'sst_{cs}_{ce}.npy')
        )
        self.sst_buf_start_abs = ((cs - datetime.timedelta(days=self.N_t // 2)) - EPOCH).days

        self.ssh_chunk = _load_sla_chunk(self.sla_cache_dir, cs, ce)

        # Build shuffled (r, t_abs) sample list for this chunk
        cs_abs = (cs - EPOCH).days
        ce_abs = (ce - EPOCH).days
        t_range = np.arange(cs_abs, ce_abs, dtype=np.int32)
        r_range = np.arange(self.coord_grids.shape[0], dtype=np.int32)
        tt, rr  = np.meshgrid(t_range, r_range)
        pairs   = np.column_stack([rr.ravel(), tt.ravel()])
        np.random.shuffle(pairs)
        self.sample_queue = pairs
        self.queue_pos    = 0

    def __iter__(self):
        info      = get_worker_info()
        worker_id = info.id if info else 0
        n_chunks  = len(self.chunk_list)

        chunk_idx = worker_id % n_chunks
        self._load_chunk(chunk_idx)

        samples_yielded = 0
        while True:
            if samples_yielded > 0 and samples_yielded % self.refresh_interval == 0:
                self._load_chunk(int(np.random.randint(n_chunks)))

            if self.queue_pos >= len(self.sample_queue):
                np.random.shuffle(self.sample_queue)
                self.queue_pos = 0

            r, t_abs = int(self.sample_queue[self.queue_pos, 0]), int(self.sample_queue[self.queue_pos, 1])
            self.queue_pos  += 1
            samples_yielded += 1

            yield _get_sample(
                r=r, t_abs=t_abs,
                sst_data=self.sst_data,
                sst_buf_start_abs=self.sst_buf_start_abs,
                ssh_chunk=self.ssh_chunk,
                coord_grids=self.coord_grids,
                rot_matrices=self.rot_matrices,
                ecef_origins=self.ecef_origins,
                bilinear_idx=self.bilinear_idx,
                bilinear_w=self.bilinear_w,
                N_t=self.N_t, n=self.n, L_x=self.L_x, L_y=self.L_y,
                mean_ssh=self.mean_ssh, std_ssh=self.std_ssh,
                mean_sst=self.mean_sst, std_sst=self.std_sst,
                leave_out_altimeters=self.leave_out_altimeters,
                withhold_sat=self.withhold_sat,
                filtered=self.filtered,
                use_sst=self.use_sst,
                ssh_out_n_max=self.ssh_out_n_max,
            )


# ---------- Worker initialisation for NeurOST_dataset ----------

def worker_init_fn(worker_id):
    """Eagerly load quarterly cache into RAM at DataLoader worker startup."""
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset._load_data()
