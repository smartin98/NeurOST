# NeurOST
Neural Ocean Surface Topography (NeurOST). High-resolution global sea surface height and surface current mapping through deep learning synthesis of multimodal satellite observations.

## Purpose of the code
This repo reproduces the workflow used to create the NeurOST global gridded sea surface height (SSH) and surface geostrophic currents [product available through NASA PO.DAAC](https://doi.org/10.5067/NEURO-STV24). The algorithm provides state-of-the-art global SSH and surface current maps which better resolve mesoscale eddy dynamics by replacing linear objective analysis methods with a data-driven neural network and by using high-resolution sea surface temperature (SST) as an additional predictor in the SSH mapping. The NeurOST methodology and validation against independent observations are all outlined in our papers (see Citations below).

By providing a streamlined open-source pipeline, we hope to allow users to experiment with the methodology, build on it, and further evaluate the approach. This repo should provide the tools needed for the following envisaged use cases:
* Fully reproduce the production of the NeurOST (SSH-SST) global product distributed on NASA PO.DAAC
* Create NeurOST maps with custom altimeter configurations (helpful for evaluation purposes)
* Produce maps using the SSH-only variant of NeurOST for fairer comparison to SSH-only methods
* Change the layout of the local patch grids used for predictions (e.g. could decreasing the patch spacing improve the maps?) :construction: COMING SOON :construction:
* Re-train NeurOST on different date range :construction: COMING SOON :construction:
* Re-train NeurOST with different time input window length :construction: COMING SOON :construction:
* Train a different neural network architecture within the same pipeline :construction: COMING SOON :construction:
* Re-train NeurOST with different inputs (e.g. different L4 SST product, L3 SST rather than L4, salinity?, spatiotemporal coords?) :construction: COMING SOON :construction:
* Fine-tune NeurOST on region of interest :construction: COMING SOON :construction:

## Citations

If you find this code helpful in your work, please include the following citations:
* Martin, S. A., Manucharyan, G. E., & Klein, P., 2024. Deep Learning Improves Global Satellite Observations of Ocean Eddy Dynamics, _Geophysical Research Letters_, 51, e2024GL110059. [DOI](https://doi.org/10.1029/2024GL110059)
* Martin, S. A., Manucharyan, G. E., & Klein, P., 2023. Synthesizing sea surface temperature and satellite altimetry observations using deep learning improves the accuracy and resolution of gridded sea surface height anomalies., _Journal of Advances in Modeling Earth Systems_, 15, e2022MS003589. [DOI](https://doi.org/10.1029/2022MS003589)
* Martin, S. A., (2025). NeurOST (code), _GitHub_, https://github.com/smartin98/NeurOST.

## Funding Acknowledgement

This work was funded by the National Aeronautics and Space Administration under Grant 80NSSC21K1187 issued through the Science Mission Directorate, Ocean Surface Topography Science Team program and additional financial support from the Theodore H. and Marie M. Sarchin Endowed Fellowship in Oceanography.

## Contributing

If you come across bugs, raise issues in the GitHub. If you would build additional functionality that you think may be of use to others, open a pull request. If you have questions feel free to reach out by email: smart1n@uw.edu.

## Basic Instructions

1. **Installation.** Install all dependencies in conda environment `conda env create -f environment.yml`.
2. **Setup.** Run `./setup.sh` to build directories and download auxilliary datasets from their respective repositories.
3. **Signup.** Make sure you have log-in credentials for [Copernicus Marine Service](https://data.marine.copernicus.eu/register) and for [NASA Earth-data](https://urs.earthdata.nasa.gov/) which we will use to access satellite datasets.
4. **Store credentials.** Run `python gen_netrc.py` to store login credential files for both Copernicus and NASA.
5. **Download data.** Run `python src/downloaders.py startdate enddate n_workers` using date format (YYYY-MM-DD) to download and pre-process observations. For long date ranges, this can take a while but it can be parallelized by setting n_workers which you can tune based on your system's memory. The script downloads the [NASA MUR L4 SST product](https://doi.org/10.5067/GHGMR-4FJ04) and coarsens it to ML-ready zarr stores, then downloads L3 nadir SSH observations from the [CMEMS near real-time product](https://doi.org/10.48670/moi-00147). Note, to use the pre-trained model you need to make sure you have 15 days either side of your intended mapping date range.
6. **Neural network inference.** Run the trained neural network to generate predictions for local patches. This has a few options, to run inference with all available altimeters (as is done to generate the [NeurOST PO.DAAC product](https://doi.org/10.5067/NEURO-STV24)) use `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50`. This script should ideally be run with access to a GPU, update batch_size according to your GPU memory and n_cpu_workers depending on your resources. Explore the script for all the options, but here are some other example commands: `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50 --no_sst` uses the SSH-only model (which results in lower resolution maps), and `python predict_ssh.py --start YYYYMMDD --end YYYYMMDD --batch_size 50 --n_cpu_workers 50 --withheld_sats swon al` allows you to specify a list of satellite altimeters to withhold from the input (e.g. for independent mapping validation). Note on resources: running on a node on NASA Pleiades with A100 GPU and 50 CPU workers, inference and zarr store saving takes ~3-4 mins per global mapping day.
7. **Merge patches.** The neural network inference step results in a zarr store with gridded SSH predictions on over-lapping, locally flat patches (see our [GRL paper](https://doi.org/10.1029/2024GL110059) for details). Run `python merge_maps.py --start YYYYMMDD --end YYYYMMDD --n_cpu_workers N` to merge the overlapping patches into a single gridded SSH product and output as NetCDF. Each CPU worker processes each date separately in parallel, taking ~10 mins per global mapping day in our tests. You will likely have to cap n_cpu_workers based on your system's available memory.
8. **Enjoy your AI-generated global SSH maps** :sunglasses:

## Advanced Usage

* **Re-train SimVP on your prefered data.** Change params like the length of the temporal window, or use different training date range but use the same NN architecture. FUNCTIONALITY COMING SOON :construction:
* **Plug in your own NN architecture.** Provided you have a torch model that takes the same input dimensions, you can achieve this with minor modifications to predict_ssh.py.
* **Include different observation inputs.** For gridded fields, save in zarr stores with similar format to what we use for MUR SST, then adapt the `get_sst()` in `src.interp_utils.py` for your needs.