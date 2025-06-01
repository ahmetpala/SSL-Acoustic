import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

years = [2007,2008,2009,2010,2011,2013,2014,2015,2016] #,2017,2018
year_codes = [2007205,2008205,2009107,2010205,2011206,2013842,2014807,2015837,2016837] #,2017843,2018823
patch_size = 8

for y in tqdm(range(len(years)), desc="Loading Data"):
    var_name = f'patch_data_array_all_{years[y]}'
    globals()[var_name] = np.load(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/'
                                  f'ACOUSTIC/GRIDDED/{year_codes[y]}_patch_data_array_all_{patch_size}w.npy',
                                  allow_pickle=True)
    indices_to_remove = np.where(globals()[var_name]['label'] == 'ignore')[0]
    globals()[var_name] = np.delete(globals()[var_name], indices_to_remove, axis=0)


data_arrays = []

for y in range(len(years)):
    var_name = f'patch_data_array_all_{years[y]}'
    data_arrays.append(globals()[var_name])

concatenated_data = np.concatenate(data_arrays, axis=0)

# Visualization: equal bins (200)
#plt.rcParams["axes.grid"] = False
plt.figure(figsize=(10,8))
# Plot histogram of the average values
plt.hist(concatenated_data['Sv_all_mean'], bins=199, color='blue', alpha=0.7)
plt.xlabel('Mean $S_v$')
plt.ylabel('Frequency')
plt.xlim(-76, 1)
plt.savefig('Sv_mean_all_data_equal_bins', dpi=600)
plt.show()

# Visualization: Custom Bins
plt.figure(figsize=(10,8))
plt.hist(concatenated_data['Sv_all_mean'], bins=[-75, -70, -65, -60, -55, 0], color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Mean $S_v$')
plt.ylabel('Frequency')
plt.xlim(-76, 1)
plt.savefig('Sv_mean_all_data_custom_bins', dpi=600)
plt.show()


data = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/data_8w_IntensityBased2FIXED_16Batch_200Ep_100Iter_320000_FIXED.npy')
patch_class = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/patch_class_8w_IntensityBased2_16Batch_200Ep_100Iter_320000.npy')
years = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/years_8w_IntensityBased2_16Batch_200Ep_100Iter_320000.npy')
center_coordinates = np.load('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Intensity_Based2/center_coordinates_8w_IntensityBased2_16Batch_200Ep_100Iter_320000.npy')


average_values = np.mean(data, axis=(1, 2, 3))
plt.rcParams["axes.grid"] = False

plt.figure(figsize=(10,8))
# Plot histogram of the average values
plt.hist(average_values, bins=200, color='blue', alpha=0.7)
plt.xlabel('Mean $S_v$')
plt.ylabel('Frequency')
plt.xlim(-76, 1)
plt.savefig('Sv_mean_sampled_data_equal_bins', dpi=600)
plt.show()


plt.figure(figsize=(10,8))
plt.hist(average_values, bins=[-75, -70, -65, -60, -55, 0], color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Mean $S_v$')
plt.ylabel('Frequency')
plt.xlim(-76, 1)
plt.savefig('Sv_mean_smpled_data_custom_bins', dpi=600)
plt.show()