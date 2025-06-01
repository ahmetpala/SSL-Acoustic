import numpy as np
from data.data_reader import DataReaderZarr, get_data_readers
import pandas as pd


years = [2007,2008,2009,2010,2011,2013,2014,2015,2016,2017,2018]
year_codes = [2007205,2008205,2009107,2010205,2011206,2013842,2014807,2015837,2016837,2017843,2018823]
y = 10

survey_path = f'/scratch/disk5/ahmet/data/{years[y]}' #'/Users/apa055/Desktop/data'

surveys = [f'/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_sv.zarr']
readers = [DataReaderZarr(survey_path + zarr_file) for zarr_file in surveys]

print(years[y])
survey = readers[0]

patch_size = 8

loaded_patch_data_array = np.load(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_patch_data_array_all_{patch_size}w.npy')
np.array(np.unique(loaded_patch_data_array['label'], return_counts=True))



np.random.seed(1)
# Undersampling the patch_data_array_all

df = pd.DataFrame(loaded_patch_data_array)

df_filtered = df[df['label'] != 'ignore']
df_filtered = df_filtered[df_filtered['label'] != 'ignored_fish']

# Group by 'label' and sample 6000 instances for each group
min_number = 38444 # number of sandeel patches
undersampled_df = df_filtered.groupby('label', group_keys=False).apply(lambda x: x.sample(min(51771, len(x)), replace=False))

undersampled_patch_data_array_all = undersampled_df.to_records(index=False)

np.save(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_undersampled_patch_data_array_all_{patch_size}w.npy', np.array(undersampled_patch_data_array_all))
