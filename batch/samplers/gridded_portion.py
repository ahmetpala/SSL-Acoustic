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

class GriddedPortion():
    # Class variable to store the current index
    current_index = 0

    def __init__(self, zarr_files, window_size, start_ping, end_ping, start_range, end_range):
        self.zarr_files = zarr_files
        self.window_size = window_size
        self.start_ping = start_ping
        self.end_ping = end_ping
        self.start_range = start_range
        self.end_range = end_range

        # Check if coordinates have been initialized
        if not hasattr(self, 'coordinates'):
            start_x = max(0, (self.start_ping//self.window_size[0])*self.window_size[0] - self.window_size[0]//2) #self.start_ping + self.window_size[0]//2
            end_x = start_x + ((self.end_ping - start_x)//self.window_size[0])*self.window_size[0] + 1
            step_size_x = self.window_size[1]

            start_y = self.start_range + self.window_size[0]//2
            end_y = self.end_range
            step_size_y = self.window_size[0]

            x_values = list(range(start_x, end_x, step_size_x))
            y_values = list(range(start_y, end_y, step_size_y))

            self.coordinates = [[y, x] for x in x_values for y in y_values]

    def get_sample(self):
        zarr_rand = np.random.choice(self.zarr_files)
        y, x = self.coordinates[self.current_index]
        GriddedPortion.current_index = (GriddedPortion.current_index + 1) % len(self.coordinates)

        return [y, x], zarr_rand, 'echogram_paint', 31, 31, 31, 31, 31

