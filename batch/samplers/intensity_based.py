
import numpy as np


class IntensityBased():
    def __init__(self, zarr_files, window_size=(256, 256), thresh_l=-75, thresh_up=0):
        """
        Sample from zarr-files
        :param zarr_files: (list)
        :param window_size: (tuple), height, width
        """
        self.zarr_files = zarr_files
        self.window_size = window_size
        self.thresh_l = thresh_l
        self.thresh_up = thresh_up

    def get_sample(self):
        # Select random zarr file in list
        zarr_rand = np.random.choice(self.zarr_files)

        # Filtering the patch_data_array_all based on thresholds
        filtered_rows = zarr_rand.patch_data_array_all[
            (zarr_rand.patch_data_array_all['Sv_all_mean'].astype(float) >= self.thresh_l) &
            (zarr_rand.patch_data_array_all['Sv_all_mean'].astype(float) < self.thresh_up)]

        # Take a random row from the filtered data
        random_row = np.random.choice(filtered_rows)

        return [random_row['center_y'], random_row['center_x']], zarr_rand