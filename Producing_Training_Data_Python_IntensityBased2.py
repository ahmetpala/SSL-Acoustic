import numpy as np
import time
import xarray as xr
from tqdm import tqdm
from batch.data_transforms.db_with_limits import db_with_limits

years = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/years_8w_IntensityBased2_16Batch_200Ep_100Iter_320000.npy')
center_coordinates = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/center_coordinates_8w_IntensityBased2_16Batch_200Ep_100Iter_320000.npy')

sv_array = [xr.open_zarr('/scratch/disk5/ahmet/data/2007/2007205/ACOUSTIC/GRIDDED/2007205_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2008/2008205/ACOUSTIC/GRIDDED/2008205_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2009/2009107/ACOUSTIC/GRIDDED/2009107_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2010/2010205/ACOUSTIC/GRIDDED/2010205_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2011/2011206/ACOUSTIC/GRIDDED/2011206_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2013/2013842/ACOUSTIC/GRIDDED/2013842_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2014/2014807/ACOUSTIC/GRIDDED/2014807_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2015/2015837/ACOUSTIC/GRIDDED/2015837_sv.zarr').sv,
            xr.open_zarr('/scratch/disk5/ahmet/data/2016/2016837/ACOUSTIC/GRIDDED/2016837_sv.zarr').sv]
train_years = [2007,2008,2009,2010,2011,2013,2014,2015,2016]


start_time = time.time()


frequencies = [18, 38, 120, 200]
patch_size = 8
data_list = []

for i in tqdm(range(len(years)), desc="Processing Patches"):
    year_index = np.where([train_years == years[i]])[1][0]
    sv_data = sv_array[year_index]

    ping_slice = slice(center_coordinates[i][1] - patch_size // 2, center_coordinates[i][1] + patch_size // 2)
    range_slice = slice(center_coordinates[i][0] - patch_size // 2, center_coordinates[i][0] + patch_size // 2)

    dat1 = sv_data.isel(ping_time=ping_slice, range=range_slice).sel(frequency=frequencies)
    dat1_all = [db_with_limits(dat1.values, 1, 2, frequencies)[0]]

    # Storing data in lists
    data_list.append(dat1_all)


# Convert lists to numpy arrays
data_array = np.concatenate(data_list)

np.save('/scratch/disk5/ahmet/data/data_8w_IntensityBased3_16Batch_200Ep_100Iter_320000_FIXED.npy', data_array)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")
