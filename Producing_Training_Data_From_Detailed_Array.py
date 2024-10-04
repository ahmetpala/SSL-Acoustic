import numpy as np
import time
import xarray as xr
from tqdm import tqdm
from batch.data_transforms.db_with_limits import db_with_limits
from concurrent.futures import ThreadPoolExecutor
import pickle

print('YENISI BU')
# The code for saving the list corresponding to the undersampled patch data array for each training year. We save data and other details.
def process_patch(patch_index):
    sv_data = sv_array

    ping_slice = slice(center_coordinates['center_x'][patch_index] - patch_size // 2,
                       center_coordinates['center_x'][patch_index] + patch_size // 2)
    range_slice = slice(center_coordinates['center_y'][patch_index] - patch_size // 2,
                        center_coordinates['center_y'][patch_index] + patch_size // 2)

    dat1 = sv_data.isel(ping_time=ping_slice, range=range_slice).sel(frequency=frequencies)
    dat1_all = [db_with_limits(dat1.values, 1, 2, frequencies)[0]]

    return dat1_all

start_time = time.time()

years = [2007,2008,2009,2010,2011,2013,2014,2015,2016,2017,2018]
year_codes = [2007205,2008205,2009107,2010205,2011206,2013842,2014807,2015837,2016837,2017843,2018823]
patch_size = 8

detailed_data_array = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based4/detailed_8w_IntensityBased4_16Batch_200Ep_100Iter_320000.npy'
                              ,allow_pickle=True)

all_data_lists = []

for y in range(len(years)-2):
    sv_array = xr.open_zarr(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_sv.zarr').sv
    filtered_detailed_data_array = detailed_data_array[detailed_data_array['total_elements']==years[y]]
    center_coordinates = filtered_detailed_data_array[['center_x', 'center_y']]

    frequencies = [18, 38, 120, 200]
    patch_size = 8

    # Process patches in parallel
    with ThreadPoolExecutor() as executor:
        data_list = list(tqdm(executor.map(process_patch, range(len(center_coordinates))),
                              total=len(center_coordinates),
                              desc=f"Processing Patches for year {years[y]}"))
        #print(np.concatenate(data_list).shape)

    all_data_lists.append(np.concatenate(data_list)) # Append the data list for this year to the list of all data lists

data_array = np.concatenate(all_data_lists)

# Save numpy arrays
np.save('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based4/data_8w_IntensityBased4_16Batch_200Ep_100Iter_320000.npy', data_array)

# Measure execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)