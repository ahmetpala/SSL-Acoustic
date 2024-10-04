import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.dataloaders import define_data_loaders
from data.partition import DataZarr

plt.rcParams["axes.grid"] = False

sampling_strategy_acoustic = 'Echogram_Painting_Overlapping'
window_size = [8, 8]
num_workers = 0
batch_size_per_gpu = 25
train_iters = 1
test_iter = 205
epochs = 1
train_years = [2007]  #

data_object = DataZarr(frequencies=[18, 38, 120, 200], patch_size=window_size, partition_train='selected surveys',
                       train_surveys=train_years,
                       validation_surveys=[2017], partition_predict=[2017],
                       evaluation_surveys=[], save_prediction_surveys=False, eval_mode='all',
                       sampling_strategy=sampling_strategy_acoustic, patch_overlap=20)
data_loader = define_data_loaders(data_object, batch_size=batch_size_per_gpu, iterations=train_iters
                                  , test_iter=test_iter, patch_size=window_size, meta_channels=[],
                                  num_workers=num_workers)[1]  # train dataloader

loader_output = {'data': [],
                 'labels': [],
                 'center_coordinates': [],
                 'patch_class': [],
                 'year': [],
                 'n_other': [],
                 'n_sandeel': [],
                 'n_bottom': [],
                 'Sv_200_mean': [],
                 'n_total': []}

# Inner loop with progress bar for the data loader batches
for out in tqdm(data_loader, desc='Processing batches', leave=False):
    for key in loader_output.keys():
        if len(loader_output[key]) == 0:
            loader_output[key] = out[key]  # Initialize with the first non-empty array
        else:
            loader_output[key] = np.concatenate((loader_output[key], out[key]), axis=0)

with open("list_overlapping_echogram_painting_2017_8w_ping_1286200.pkl", "wb") as f:
    pickle.dump(loader_output, f)


