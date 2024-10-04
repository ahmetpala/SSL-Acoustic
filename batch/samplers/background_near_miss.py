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
from utils_unet.np import getGrid, nearest_interpolation


class Background():
    def __init__(self, echograms, window_size):
        """
        :param echograms: A list of all echograms in set
        :param window_size: (tuple), [height, width]
        """
        self.echograms = echograms
        self.window_size = window_size

    def get_sample(self):
        """
        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        # Random echogram
        ech_index = np.random.randint(len(self.echograms))
        ech = self.echograms[ech_index]

        # Random x, y-loc above seabed
        # If window width is greater than echogram width, set x = echogram senter
        if ech.shape[1] <= self.window_size[1]:
            x = ech.shape[1] // 2
        else:
            half_patch_width = self.window_size[1] // 2 - 20
            x = np.random.randint(half_patch_width, ech.shape[1] - half_patch_width)

        # Select random location above seabed, or in the middle of the echogram if window height > echogram height
        seabed = int(ech.get_seabed(x))
        if seabed <= self.window_size[0]:
            y = ech.shape[0] // 2
        else:
            y = np.random.randint(self.window_size[0] // 2, seabed - self.window_size[0] // 2)

        # Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y, x], 1), 1)
        labels = nearest_interpolation(ech.label_memmap(), grid, boundary_val=0, out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample()  # Draw new sample

        return [y, x], ech


class NMBackgroundZarr():
    def __init__(self, zarr_files, n_sandeel=5, window_size=(256, 256)):
        """
        Sample from zarr-files
        :param zarr_files: (list)
        :param window_size: (tuple), height, width
        """
        self.zarr_files = zarr_files
        self.window_size = window_size
        self.n_sandeel = n_sandeel
        self.thresholds = {2011: 33.22945957485141, 2013: 33.167769220736616, 2014: 46.48532133305009,
                           2015: 45.866204078489865, 2016: 34.2591259264971, 2017: 36.18716532013442}

    def get_sample(self):
        # Select random zarr file in list
        zarr_rand = np.random.choice(self.zarr_files)

        # Select random sample within Near Miss Samples (Equal number of sandeel fish schools)
        rownumber = np.random.randint(0, ((zarr_rand.get_objects_file().category == 27).sum())*self.n_sandeel)

        # Select random sample within Near Miss Samples (Based on threshold calculated from sandeel schools)
        # rownumber = np.random.randint(0, (zarr_rand.distances['fast_avr_dist'] < self.thresholds[zarr_rand.year]).sum())

        # Get ping time index
        x = int(zarr_rand.distances.iloc[rownumber].ping_time)

        # Get ping time index
        y = int(zarr_rand.distances.iloc[rownumber].range)
        if y < self.window_size[0] // 2: y = self.window_size[0] // 2  # Modifying exceeding patches

        return [y, x], zarr_rand
