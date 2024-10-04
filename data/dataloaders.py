"""This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

import time

from torch.utils.data import DataLoader

from batch.dataset import Dataset, DatasetTest
from batch.transforms import (define_data_augmentation, define_data_transform, define_data_transform_test,
                              define_label_transform_train, define_label_transform_test, is_use_metadata)
from paths import *
from utils_unet.general import (seed_worker)


def define_data_loaders(data_obj,
                        batch_size,
                        iterations,
                        test_iter,
                        patch_size,
                        meta_channels,
                        num_workers,
                        **kwargs):
    # Are we training with metadata?
    use_metadata = is_use_metadata(meta_channels)
    frequencies = data_obj.frequencies

    # Divide data into training and test
    print("Preparing data samplers")
    start = time.time()
    readers_train, readers_test = data_obj.partition_data_train()
    samplers_train, samplers_test, sampler_probs = data_obj.get_samplers_train(readers_train, readers_test)
    print(f"Executed time for preparing samples (s): {np.round((time.time() - start), 2)}\n")

    print("Preparing data loaders")
    # Define data augmentation, and data and label transforms for training
    data_augmentation = define_data_augmentation(use_metadata)
    label_transform_train = define_label_transform_train(frequencies)
    data_transform = define_data_transform(use_metadata)

    # Prepare dataset and dataloader for training
    dataset_train = Dataset(
        samplers_train,
        patch_size,
        frequencies,
        meta_channels,
        n_samples=batch_size * iterations,
        sampler_probs=sampler_probs, #,augmentation_function=data_augmentation
        label_transform_function=None,
        data_transform_function=data_transform,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    # Define label and data transform for testing
    label_transform_test = define_label_transform_test(frequencies, label_masks="all", patch_overlap=0)

    # TODO consider same transform as in test -> currently included to match testing scheme from earlier code
    # Values outside data boundary set to 0
    data_transform_test = define_data_transform_test(use_metadata)

    # Create test dataloader
    dataset_test = DatasetTest(
        samplers_test,
        patch_size,
        frequencies,
        meta_channels,
        n_samples=batch_size * test_iter,
        sampler_probs=sampler_probs,
        augmentation_function=None,
        label_transform_function=None,
        data_transform_function=data_transform_test,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    print(f"Executed time for preparing dataloaders (s): {np.round((time.time() - start), 2)}\n")
    return dataloader_train, dataloader_test