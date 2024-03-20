import os
from datetime import datetime, date
import sys
from glob import glob

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

command = 'podaac-data-downloader -c MUR-JPL-L4-GLOB-v4.1 -d ./input_data/mur_sst --start-date '+start_date_string+' --end-date '+end_date_string+' -b="-180,-90,180,90"'

os.system(command)

# # drop unnecessary variables to save disk space
# raw_files = glob('input_data/mur_sst/*.nc')
# raw_files = [f for f in raw_files if 'compressed' not in f]

# for f in raw_files:
#     ds = xr.open_dataset(f)
#     ds = ds.drop(['analysis_error','mask','dt_1km_data'])
#     ds.to_netcdf(f[:-3]+'_compressed.nc')
#     os.system('rm '+f)