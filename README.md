# NeurOST
Neural Ocean Surface Topography (NeurOST). High-resolution global sea surface height and surface current mapping through deep learning synthesis of multimodal satellite observations.

:warning: WORK IN PROGRESS :warning:

Instructions:

1. **Installation.** Install all dependencies in conda environment `conda env create -f environment.yml`.
2. **Setup.** Run `./setup.sh` to build directories and download auxilliary datasets from their respective repositories.
3. **Signup.** Make sure you have log-in credentials for [Copernicus Marine Service](https://data.marine.copernicus.eu/register) and for [NASA Earth-data](https://urs.earthdata.nasa.gov/) which we will use to access satellite datasets.
4. **Store credentials.** Run `python gen_netrc.py` to store login credential files for both Copernicus and NASA.
5. **Download data.** Run `python src/downloaders.py startdate enddate` using date format (YYYYMMDD) to download and pre-process observations. For long date ranges, this can take a while. The script downloads the [NASA MUR L4 SST product](https://doi.org/10.5067/GHGMR-4FJ04) and coarsens it to ML-ready zarr stores, then downloads L3 nadir SSH observations from the [CMEMS near real-time product](https://doi.org/10.48670/moi-00147). Note, to use the pre-trained model you need to make sure you have 15 days either side of your intended mapping date range.
6. **Neural network inference.** Run the trained neural network to generate predictions for local patches. This has a few options, to run inference with all available altimeters (as is done to generate the [NeurOST PO.DAAC product](https://doi.org/10.5067/NEURO-STV24)) use `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50`. This script should ideally be run with access to a GPU, update batch_size according to your GPU memory and n_cpu_workers depending on your resources. Explore the script for all the options, but here are some other example commands: `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50 --no_sst` uses the SSH-only model (which results in lower resolution maps), and `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50 --withheld_sats swon al` allows you to specify a list of satellite altimeters to withhold from the input (e.g. for independent mapping validation). Note on resources: running on a node on NASA Pleiades with A100 GPU and 50 CPU workers, inference and zarr store saving takes ~3-4 mins per global mapping day.
7. **Merge patches.** The neural network inference step results in a zarr store with gridded SSH predictions on over-lapping, locally flat patches (see our [GRL paper]( https://doi.org/10.1029/2024GL110059) for details). Run `python merge_maps.py --start YYYYMMDD --end YYYYMMDD --n_cpu_workers N` to merge the overlapping patches into a single gridded SSH product and output as NetCDF. Each CPU worker processes each date separately in parallel, taking ~10 mins per global mapping day in our tests. You will likely have to cap n_cpu_workers based on your system's available memory.
8. **Enjoy your AI-generated global SSH maps** :sunglasses:

Advanced Usage:

* **Re-train SimVP on your prefered data.** Change params like the length of the temporal window, or use different training date range but use the same NN architecture. FUNCTIONALITY COMING SOON :construction:
* **Plug in your own NN architecture.** Provided you have a torch model that takes the same input dimensions, you can achieve this with minor modifications to predict_ssh.py.
* **Include different observation inputs.** For gridded fields, save in zarr stores with similar format to what we use for MUR SST, then adapt the `get_sst()` in `src.interp_utils.py` for your needs.