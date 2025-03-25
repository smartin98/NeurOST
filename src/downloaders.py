import os
from datetime import datetime, date, timedelta
import sys
from glob import glob
import copernicusmarine as cm
import xarray as xr
cm.login()

# #############################################
# # MUR SST
# #############################################


def format_date_to_string(date_object):
    """Formats a datetime.date object to YYYY-MM-DD'T'HH:MM:00Z string.

    Args:
      date_object: A datetime.date object.

    Returns:
      A string in the format YYYY-MM-DD'T'HH:MM:00Z.
    """
    # Combine the date object with time (set hours, minutes, seconds to 0, and timezone to 'Z')
    formatted_string = datetime(date_object.year, date_object.month, date_object.day, 0, 0, 0).isoformat() + 'Z'
    return formatted_string


date_start_input = sys.argv[1]
date_end_input = sys.argv[2]

# def download_mur_local(date_start_input,date_end_input):
date_start = datetime.strptime(date_start_input, "%Y-%m-%d").date()
date_end = datetime.strptime(date_end_input, "%Y-%m-%d").date()


start_date_string = format_date_to_string(date_start)
end_date_string = format_date_to_string(date_end)

n_days = (date_end-date_start).days
for t in range(n_days):

    command = 'podaac-data-downloader -c MUR-JPL-L4-GLOB-v4.1 -d ./input_data/mur_sst_tmp --start-date '+format_date_to_string(date_start+timedelta(days=t))+' --end-date '+format_date_to_string(date_start+timedelta(days=t))+' -b="-180,-90,180,90"'
    
    os.system(command)
    
    # ### coarsen MUR 5x and save to zarr to speed up pre-processing functions
    sst_dir = 'input_data/mur_sst_tmp/'
    files = sorted(glob(sst_dir+'*.nc'))
    
    for f in files:
        print('coarsening: '+ f)
        ds_sst = xr.open_dataset(f)
        if os.path.isdir('input_data/mur_coarse_zarrs/' + str(ds_sst['time'].values[0])[:10].replace('-','') +'.zarr'):
            print('already coarsened, skipping')
            if remove_mur_nc:
                os.remove(f)
        else:
            sst = ds_sst['analysed_sst'].load().astype('float32').coarsen({'lon':5,'lat':5},boundary = 'trim').mean()
            sst = sst.chunk({'time':1,'lon':1000,'lat':1000})
            sst.to_zarr('input_data/mur_coarse_zarrs/' + str(sst['time'].values[0])[:10].replace('-','') +'.zarr')
            if remove_mur_nc:
                os.remove(f)

#############################################
# CMEMS SLA
#############################################


sats = ['al','c2n','h2b','j3n','s3a','s3b','s6a-hr','swon']


def generate_date_list(start_date, end_date):
    date_list = ['*/'+(start_date + timedelta(days=i)).strftime('%Y/%m') +'/*.nc' for i in range((end_date - start_date).days + 1)]
    return list(set(date_list))

date_start_input = sys.argv[1]
date_end_input = sys.argv[2]

# def download_mur_local(date_start_input,date_end_input):
date_start = datetime.strptime(date_start_input, "%Y-%m-%d").date()
date_end = datetime.strptime(date_end_input, "%Y-%m-%d").date()


search_strings = generate_date_list(date_start, date_end)

for sat in sats:
    print('Downloading data for '+sat)
    for search in search_strings:
        print(search)
        cm.get(
            dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_"+sat+"-l3-duacs_PT1S",
            filter = search,
            output_directory = 'input_data/cmems_sla/'+sat+'/',
            no_directories = True,
            force_download = True
        )