""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
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

import numpy as np

class GriddedPreDefinedAll():
    # Class variable to store the current index
    current_index = 0

    def __init__(self, zarr_files, window_size):
        self.zarr_files = zarr_files
        self.window_size = window_size


    def get_sample(self):
        zarr_rand = np.random.choice(self.zarr_files)
        undersampled_patch_data = zarr_rand.undersampled_patch_data_array_all
        x, y, label, n_other, n_sandeel, n_bottom, Sv_200_mean, n_total = undersampled_patch_data[self.current_index]
        GriddedPreDefinedAll.current_index = (GriddedPreDefinedAll.current_index + 1) % len(undersampled_patch_data)

        return [y, x], zarr_rand, label, n_other, n_sandeel, n_bottom, Sv_200_mean, n_total

