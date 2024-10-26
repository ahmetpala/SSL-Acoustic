import numpy as np
import pandas as pd
from data.dataloaders import define_data_loaders
from data.data_reader import DataReaderZarr, get_data_readers, get_zarr_readers
from data.partition import DataZarr
from batch.data_transforms.db_with_limits import db_with_limits
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time

plt.rcParams["axes.grid"] = False

start_time = time.time()

sampling_strategy_acoustic = 'Complete_Random'
print(sampling_strategy_acoustic)
window_size = [8, 8]
num_workers = 0
batch_size_per_gpu = 16
train_iters = 100
epochs = 200
train_years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

data_object = DataZarr(frequencies=[18, 38, 120, 200], patch_size=window_size, partition_train='selected surveys',
                       train_surveys=train_years,
                       validation_surveys=[2017], partition_predict=[2017],
                       evaluation_surveys=[2017], save_prediction_surveys=False, eval_mode='all',
                       sampling_strategy=sampling_strategy_acoustic, patch_overlap=20)
data_loader = define_data_loaders(data_object, batch_size=batch_size_per_gpu, iterations=train_iters
                                  , test_iter=2, patch_size=window_size, meta_channels=[],
                                  num_workers=num_workers)[0]  # train dataloader

# Initialize lists to store extracted data
data_list = []
center_coordinates_list = []
year_list = []
patch_class_list = []

for ep in range(1, epochs + 1):
    tqdm.write(f"Iteration {ep}/{epochs}")
    for images_hepsi in tqdm(data_loader, desc='Processing batches', leave=False):

        images_a = images_hepsi['data']

        center_coordinates = images_hepsi['center_coordinates']
        year = images_hepsi['year']
        patch_class = images_hepsi['patch_class']

        data_list.append(images_hepsi['data'])
        center_coordinates_list.append(center_coordinates)
        year_list.append(year)
        patch_class_list.append(patch_class)

data_array = np.concatenate(data_list)
center_coordinates_array = np.concatenate(center_coordinates_list)
year_array = np.concatenate(year_list)
patch_class_array = np.concatenate(patch_class_list)

np.save('/scratch/disk5/ahmet/data/data_8w_IntensityBased_16Batch_200Ep_100Iter_320000.npy', data_array)
np.save('/scratch/disk5/ahmet/data/center_coordinates_8w_IntensityBased_16Batch_200Ep_100Iter_320000.npy', center_coordinates_array)
np.save('/scratch/disk5/ahmet/data/years_8w_IntensityBased_16Batch_200Ep_100Iter_320000.npy', year_array)
np.save('/scratch/disk5/ahmet/data/patch_class_8w_IntensityBased_16Batch_200Ep_100Iter_320000.npy', patch_class_array)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")
