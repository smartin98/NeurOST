mkdir predictions
mkdir predictions/unmerged_zarrs
mkdir predictions/merged_products
mkdir input_data
mkdir input_data/mur_sst_tmp
mkdir input_data/mur_coarse_zarrs
mkdir input_data/cmems_sla
mkdir input_data/sla_cache
mkdir input_data/auxilliary
cd input_data/auxilliary
wget https://zenodo.org/records/15116366/files/distance_to_nearest_coastlines_10grid.nc
wget https://zenodo.org/records/15116366/files/land_water_mask_10grid.nc
wget https://zenodo.org/records/15116366/files/mdt_currents_0.1degree_grid.nc
wget https://zenodo.org/records/15116366/files/mdt_hybrid_cnes_cls18_cmems2020_global.nc
cd ..
wget https://dataverse.harvard.edu/api/access/datafile/8111925 -O coord_grids.npy
mkdir model_weights
cd model_weights
wget https://dataverse.harvard.edu/api/access/datafile/8131431 -O 
simvp_ssh_sst_ns1000000global_weights_epoch46
wget https://dataverse.harvard.edu/api/access/datafile/8131432 -O simvp_ssh_ns1000000_global_weights_epoch45
cd ..
cd ..