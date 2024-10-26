import numpy as np
import time
import xarray as xr
from tqdm import tqdm
from batch.data_transforms.db_with_limits import db_with_limits
from concurrent.futures import ThreadPoolExecutor
import pickle

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

# Timing the script
start_time = time.time()

years = [2007,2008,2009,2010,2011,2013,2014,2015,2016,2017]
year_codes = [2007205,2008205,2009107,2010205,2011206,2013842,2014807,2015837,2016837,2017843]
patch_size = 8


for y in range(9,len(years)):
    sv_array = xr.open_zarr(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_sv.zarr').sv
    undersampled_patch_data_array_all = np.load(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/'
                                                f'ACOUSTIC/GRIDDED/{year_codes[y]}_undersampled_patch_data_array_all_{patch_size}w.npy'
                                                , allow_pickle=True)
    center_coordinates = undersampled_patch_data_array_all[['center_x', 'center_y']]

    frequencies = [18, 38, 120, 200]
    patch_size = 8

    with ThreadPoolExecutor() as executor:
        data_list = list(tqdm(executor.map(process_patch, range(len(center_coordinates))),
                              total=len(center_coordinates),
                              desc=f"Processing Patches for year {years[y]}"))

    data_array = np.concatenate(data_list)

    final_list = {'data': data_array,
                     'labels': undersampled_patch_data_array_all['label'],
                     'center_coordinates': [undersampled_patch_data_array_all['center_y'],undersampled_patch_data_array_all['center_x']],
                     'patch_class': undersampled_patch_data_array_all['label'],
                     'year': np.full_like(undersampled_patch_data_array_all['count_1'], years[y]),
                     'n_other': undersampled_patch_data_array_all['count_1'],
                     'n_sandeel': undersampled_patch_data_array_all['count_2'],
                     'n_bottom': undersampled_patch_data_array_all['count_3'],
                     'Sv_200_mean': undersampled_patch_data_array_all['Sv_all_mean'],
                     'n_total': undersampled_patch_data_array_all['total_elements']}

    with open(f'/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/list_undersampled_{years[y]}_8w_.pkl', "wb") as f:
        pickle.dump(final_list, f)

# Measure execution time
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)