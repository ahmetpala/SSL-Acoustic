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

import os
import numpy as np
import matplotlib.colors as mcolors
from scipy.signal import convolve2d as conv2d
import xarray as xr
import pandas as pd
from glob import glob
from pathlib import Path

import paths
from batch.data_transforms.db_with_limits import db
from data.missing_korona_depth_measurements import depth_excluded_echograms

import matplotlib.pyplot as plt

class DataReaderZarr():
    """
    Data reader for zarr files. Expectation is that the zarr file contains data from one year only
    :param path: path to survey directory (i.e. /cruise_data/2017/S2017843)
    """

    def __init__(self, path):
        # Get all paths
        self.sv_path = os.path.abspath(path)
        self.name = os.path.split(self.sv_path)[-1].replace('_sv.zarr', '')
        self.path = os.path.split(self.sv_path)[0]
        self.annotation_path = os.path.join(*[self.path, f'{self.name}_labels.zarr'])
        self.seabed_path = os.path.join(*[self.path, f'{self.name}_bottom.zarr'])
        self.work_path = os.path.join(*[self.path, f'{self.name}_labels.parquet'])
        self.objects_df_path = os.path.join(*[self.path, f'{self.name}_labels.parquet.csv'])

        #self.distances_df_path = os.path.join(*[self.path, f'{self.name}_distances.pkl'])  # AHMET
        #self.distances = pd.read_pickle(self.distances_df_path)  # AHMET
        #self.n_sandeel = None  # AHMET
        #try:
        #    self.patch_data_array_all = np.load(os.path.join(*[self.path, f'{self.name}_patch_data_array_all_8w.npy']),allow_pickle=True)
        #    # Find the indices of rows with 'ignore' in the 'label' column
        #    indices_to_remove = np.where(self.patch_data_array_all['label'] == 'ignore')[0]

         #   # Remove rows with 'ignore' in the 'label' column
         #   self.patch_data_array_all = np.delete(self.patch_data_array_all, indices_to_remove, axis=0)
        #except FileNotFoundError:
        #    print("The file 'patch_data_array_all' does not exist. Ignoring.")

        #try:
        #    self.undersampled_patch_data_array_all = np.load(os.path.join(*[self.path, f'{self.name}_undersampled_patch_data_array_all_8w.npy']),allow_pickle=True)
        #except FileNotFoundError:
        #    print("The file 'undersampled_patch_data_array_all' does not exist. Ignoring.")

        self.data_format = 'zarr'
        assert os.path.isdir(self.sv_path), f"No Sv data found at {self.sv_path}"

        # Load data
        self.ds = xr.open_zarr(self.sv_path)  # , chunks={'frequency': 'auto'})

        # Read coordinates
        self.frequencies = self.ds.frequency.astype(int)
        #self.heave = self.ds.heave
        #self.channel_id = self.ds.get('channelID')
        #self.latitude = self.ds.get('latitude')
        #self.longitude = self.ds.get('longitude')
        self.range_vector = self.ds.range
        self.time_vector = self.ds.ping_time
        self.year = int(self.ds.ping_time[0].dt.year)
        #self.date_range = (self.ds.ping_time[0], self.ds.ping_time[-1])
        self.shape = (self.ds.sizes['ping_time'], self.ds.sizes['range'])
        #self.raw_file = self.ds.raw_file  # List of raw files, length = nr of pings

        # TODO: slow
        #self.raw_file_included = np.unique(self.ds.raw_file.values)  # list of unique raw files contained in zarr file
        #self.raw_file_excluded = []
        #self.raw_file_start = None

        # Used for seabed estimation
        # transducer_offset = self.ds.transducer_draft.mean()
        # self.transducer_offset_pixels = int(transducer_offset/(self.range_vector.diff(dim='range').mean()).values)

        # Load annotations files
        self.annotation = None
        if os.path.isdir(self.annotation_path):
            self.annotation = xr.open_zarr(self.annotation_path)
            self.labels = self.annotation.annotation
           # self.objects = self.annotation.object

            # Fish categories used in survey
            self.fish_categories = [cat for cat in self.annotation.category.values if cat != -1]
        else:
            print(f'No annotation file found at {self.annotation_path}')

        # Load seabed file
        self.seabed = None
        if os.path.isdir(self.seabed_path):
            self.seabed = xr.open_zarr(self.seabed_path)

        # Get valid pings
        self.valid_pings = None

        # Objects dataframe
        self.objects_df = None
        self.data_format = 'zarr'

    def get_valid_pings(self):
        if self.valid_pings is not None:
            return self.valid_pings
        else:
            csv_dir = Path(self.path).parents[1]
            csv_path = os.path.join(*[csv_dir, 'STOX', self.name.replace('S', '') + '_transects.csv'])
            if not os.path.isfile(csv_path):
                self.valid_pings = np.array([[0, self.shape[0]]]).astype(np.int32)
                return self.valid_pings

           # assert os.path.isfile(csv_path), print("No stox file found at", csv_path)

            valid_df = pd.read_csv(csv_path)

            start_pings = []
            end_pings = []
            for _, row in valid_df.iterrows():
                start_pings.append(self.get_ping_index(np.datetime64(row.StartDateTime)))
                end_pings.append(self.get_ping_index(np.datetime64(row.StopDateTime)))

            self.valid_pings = np.array([start_pings, end_pings]).astype(np.int32).T
            return self.valid_pings  # np.sort(self.valid_pings, axis=0)

    def get_ping_index(self, ping_time):
        """
        Due to rounding errors, the ping_time variable for labels and data are not exactly equal.
        This function returns the closest index to the input ping time
        :param ping_time: (np.datetime64)
        :return: (int) index of closest index in data time_vector
        """
        return int(np.abs((self.time_vector - ping_time)).argmin().values)

    def get_range_index(self, range):
        """
        Get closest index in range_vector
        """
        return int(np.abs((self.range_vector - range)).argmin().values)

    def get_coord_index(self, coord):
        """
        Get closest index based on coordinate (latitude, longitude)
        """
        return np.nanargmin(np.sqrt(np.power(self.ds.latitude.values - coord[0], 2)
                                    + np.power(self.ds.longitude.values - coord[1], 2)))

    def get_fish_schools(self, category='all'):
        """
        Get all bounding boxes for the input categories
        :param category: Categories to include ('all', or list)
        :return: dataframe with bounding boxes
        """
        df = self.get_objects_file()
        if category == 'all':
            category = self.fish_categories

        if not isinstance(category, (list, np.ndarray)):
            category = [category]

        return df.loc[(df.category.isin(category)) & (df.valid_object)]

    def get_objects_file(self):
        """
        Get or compute dataframe with bounding box indexes for all fish schools
        :return: Pandas dataframe with object info and bounding boxes
        """
        if self.objects_df is not None:
            return self.objects_df

        parsed_objects_file_path = os.path.join(os.path.split(self.objects_df_path)[0],
                                                self.name + '_objects_parsed.csv')

        if os.path.isfile(parsed_objects_file_path):
            return pd.read_csv(parsed_objects_file_path, index_col=0)
        elif os.path.isfile(self.objects_df_path):# and os.path.isfile(self.work_path):
            print('Generating objects file with seabed distances ... ')

            # Create parsed objects file from object file and work file
            df = pd.read_csv(self.objects_df_path, header=0)
            df = df.rename(columns={"upperdept": "upperdepth",
                                    "lowerdept": "lowerdepth",
                                    "upperdeptindex": "upperdepthindex",
                                    "lowerdeptindex": "lowerdepthindex"})

            categories = df.category.values
            upperdeptindices = df.upperdepthindex.values
            lowerdeptindices = df.lowerdepthindex.values
            startpingindices = df.startpingindex.values
            endpingindices = df.endpingindex.values

            distance_to_seabed = np.zeros_like(upperdeptindices, dtype=np.float32)
            distance_to_seabed[:] = np.nan
            valid_object = np.zeros_like(upperdeptindices, dtype='bool')

            assert len(df['object']) == len(df), print('Object IDs not unique!')
            #
            for idx, (category, upperdeptindex, lowerdeptindex, startpingindex, endpingindex) in \
                    enumerate(zip(categories, upperdeptindices, lowerdeptindices, startpingindices, endpingindices)):

                # Skip objects with ping errors og of category -1
                # TODO better solution for this? Fix ping errors?
                if startpingindex > endpingindex or category == -1:
                    valid_object[idx] = False
                    continue

                # Add distance to seabed
                if os.path.isdir(self.seabed_path):
                    center_ping_idx = startpingindex + int((endpingindex - startpingindex) / 2)
                    distance_to_seabed[idx] = self.get_seabed(center_ping_idx) - lowerdeptindex

                valid_object[idx] = True

            # # Save parsed objecs file
            df['distance_to_seabed'] = distance_to_seabed
            df['valid_object'] = valid_object
            df.to_csv(parsed_objects_file_path)
            self.objects_df = df
            self.n_sandeel = (self.objects_df.category == 27).sum()  # AHMET
            return df
        else:
            # Cannot return object file
            raise FileNotFoundError(
                f'Cannot compute objects dataframe from {self.objects_df_path}')# and {self.work_path}')

    def get_data_slice(self, idx_ping, n_pings: (int, None) = None, idx_range: (int, None) = None,
                       n_range: (int, None) = None,
                       frequencies: (int, list, None) = None, drop_na=False, return_numpy=True):
        '''
        Get slice of xarray.Dataset based on indices in terms of (frequency, ping_time, range).
        Arguments for 'ping_time' and 'range' indices are given as the start index and the number of subsequent indices.
        'range' and 'frequency' arguments are optional.

        :param idx_ping: (int) First ping_time index of the slice
        :param n_pings: (int) Number of subsequent ping_time indices of the slice
        :param idx_range: (int | None) First range index of the slice (None slices from first range index)
        :param n_range: (int | None) Number of subsequent range indices of the slice (None slices to last range index)
        :param frequencies: (int | list[int] | None) Frequencies in slice (None returns all frequencies)
        :return: Sliced xarray.Dataset

        Example:
        ds_slice = ds.get_slice(idx_ping=20000, n_pings=256) # xarray.Dataset sliced in 'ping_time' dimension [20000:20256]
        sv_data = ds_slice.sv # xarray.DataArray of underlying sv data
        sv_data_numpy = sv_data.values # numpy.ndarray of underlying sv data
        '''

        assert isinstance(idx_ping, (int, np.integer, type(None)))
        assert isinstance(n_pings, (int, np.integer, type(None)))
        assert isinstance(idx_range, (int, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))
        assert isinstance(frequencies, (int, np.integer, list, np.ndarray, type(None)))
        if isinstance(frequencies, list):
            assert all([isinstance(f, (int, np.integer)) for f in frequencies])

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            slice_range = slice(None, n_range)  # Valid for n_range int, None
        elif n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range, idx_range + n_range)

        if frequencies is None:
            frequencies = self.frequencies

        # Make sure frequencies is array-like to preserve dims when slicing
        if isinstance(frequencies, (int, np.integer)):
            frequencies = [frequencies]

        data = self.ds.sv.sel(frequency=frequencies).isel(ping_time=slice_ping_time, range=slice_range)

        if drop_na:
            data = data.dropna(dim='range')

        if return_numpy:
            return data.values
        else:
            return data

    def get_label_slice(self, idx_ping: int, n_pings: int, idx_range: (int, None) = None, n_range: (int, None) = None,
                        drop_na=False, categories=None, return_numpy=True, correct_transducer_offset=False,
                        mask=True):
        """
        Get slice of labels
        :param idx_ping: (int) Index of start ping
        :param n_pings: (int) Width of slice
        :param idx_range: (int) Index of start range
        :param n_range: (int) Height of slice
        :param drop_na: (bool) Drop nans at the bottom of data (data is padded with nans since echograms have different heights)
        :return: np.array with labels
        """
        assert isinstance(idx_ping, (int, np.integer))
        assert isinstance(n_pings, (int, np.integer))
        assert isinstance(idx_range, (int, np.integer, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            slice_range = slice(None, n_range)  # Valid for n_range int, None
        elif n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range, idx_range + n_range)

        # Convert labels from set of binary masks to 2D segmentation mask
        if categories is None:
            categories = np.array(self.fish_categories)

        # Initialize label mask and fill
        label_slice = self.labels.isel(ping_time=slice_ping_time, range=slice_range)

        if mask:
            # TODO figure out whether -1 cat should be ignored
            labels = label_slice.sel(category=-1) * 0

            # labels = self.labels.sel(category=categories[0]).isel(ping_time=slice_ping_time, range=slice_range) * categories[0]
            for cat in categories:
                labels = labels.where(label_slice.sel(category=cat) <= 0,
                                      cat)  # Where condition is False, fill with cat

            # Drop nans in range dimension
            if drop_na:
                labels = labels.dropna(dim='range')
        else:
            # TODO: mask away -1?
            labels = label_slice.sel(category=categories)

        # Convert to np array
        if return_numpy:
            return labels.values
        else:
            return labels

    def get_seabed_mask(self, idx_ping: int, n_pings: int, idx_range: (int, None) = None, n_range: (int, None) = None,
                        return_numpy=False, seabed_pad=0):
        """
        Get seabed mask from slice
        :param idx_ping: Start ping index (int)
        :param n_pings: End ping index (int)
        :param idx_range: Number of pings (int)
        :param n_range: Number of vertical samples to include (int)
        :param return_numpy: Return mask as numpy array
        :return: Mask where everything below seafloor is marked with 1, everything above is marked with 0
        """

        assert isinstance(idx_ping, (int, np.integer))
        assert isinstance(n_pings, (int, np.integer))
        assert isinstance(idx_range, (int, np.integer, type(None)))
        assert isinstance(n_range, (int, np.integer, type(None)))

        slice_ping_time = slice(idx_ping, idx_ping + n_pings)

        if idx_range is None:
            idx_range = 0

        if n_range is None:
            slice_range = slice(idx_range, None)
        else:
            slice_range = slice(idx_range,
                                idx_range + n_range)

        # Everything below seafloor has value 1, everything above has value 0
        seabed_slice = self.seabed.bottom_range.isel(ping_time=slice_ping_time, range=slice_range).fillna(0)

        # TODO write for negative and positive padding?
        if seabed_pad != 0:
            seabed_slice_pad = xr.zeros_like(seabed_slice).values.copy()
            seabed_slice_pad[:, seabed_pad:, ] = seabed_slice[:, :-seabed_pad]

            return seabed_slice_pad

        if return_numpy:
            return seabed_slice.values
        else:
            return seabed_slice

    def get_seabed(self, idx_ping: int, n_pings: (int) = 1, idx_range: (int, None) = None, n_range: (int, None) = None,
                   return_numpy=True):
        """
        Get vector of range indices for the seabed
        WARNING slow for large stretches of data

        :param idx_ping: index of start ping (int)
        :param n_pings: number of pings to include (int)
        :return: vector with seabed range indices (np.array)
        """

        # Get seabed mask for the specified pings
        seabed_mask = self.get_seabed_mask(idx_ping, n_pings, idx_range, n_range, return_numpy=False)
        seabed = seabed_mask.argmax(dim="range")

        if return_numpy:
            return seabed.values.astype(int)
        else:
            return seabed

    def get_rawfile_index(self, rawfile):
        relevant_pings = np.argwhere(self.raw_file.values == rawfile).ravel()
        start_ping = relevant_pings[0]
        n_pings = len(relevant_pings)
        return start_ping, n_pings

    # These two functions are (currently) necessary to predict on zarr-data
    def get_data_rawfile(self, rawfile, frequencies, drop_na):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_data_slice(idx_ping=start_ping, n_pings=n_pings, frequencies=frequencies, drop_na=drop_na,
                                   return_numpy=True)

    def get_labels_rawfile(self, rawfile):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_label_slice(idx_ping=start_ping, n_pings=n_pings, return_numpy=True)

    def get_seabed_rawfile(self, rawfile):
        start_ping, n_pings = self.get_rawfile_index(rawfile)

        return self.get_seabed(idx_ping=start_ping, n_pings=n_pings)

    def visualize(self,
                  ping_idx=None,
                  n_pings=2000,
                  range_idx=None,
                  n_range=None,
                  raw_file=None,
                  frequencies=None,
                  draw_seabed=True,
                  show_labels=True,
                  predictions=None,
                  data_transform=db):
        """
        Visualize data from xarray
        :param ping_idx: Index of start ping (int)
        :param n_pings: Nr of pings to visualize (int)
        :param range_idx: Index of start range (int)
        :param n_range: Nr of range samples to visualize (int)
        :param raw_file: Visualize data from a single raw file (overrides ping index arguments!) (str)
        :param frequencies: Frequencies to visualize (list)
        :param draw_seabed: Draw seabed on plots (bool)
        :param show_labels: Show annotation (bool)
        :param predictions: Predictions data variables should follow annotation format or be presented as a numpy array (xarray.Dataset, numpy.ndarray)
        :param data_transform: Data transform before visualization (db transform recommended) (function)
        """

        # Visualize data from a single raw file
        if raw_file is not None:
            idxs = np.argwhere(self.raw_file.values == raw_file).ravel()
            ping_idx = idxs[0]
            n_pings = len(idxs)

        # retrieve data
        if ping_idx is None:
            ping_idx = np.random.randint(0, len(self.time_vector) - n_pings)
        if frequencies is None:
            frequencies = list(self.frequencies.values)
        if range_idx is None:
            range_idx = 0
        if n_range is None:
            n_range = self.shape[1]

        data = self.get_data_slice(ping_idx, n_pings, range_idx, n_range, frequencies, drop_na=True)

        # Optionally transform data
        if data_transform != None:
            data = data_transform(data)

        # Initialize plot
        nrows = len(frequencies) + int(show_labels)
        if predictions is not None:
            nrows += 1
        fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(16, 16), sharex=True)
        axs = axs.ravel()
        plt.tight_layout()

        # Get tick labels
        tick_idx_y = np.arange(start=0, stop=data.shape[-1], step=int(data.shape[-1] / 4))
        tick_labels_y = self.range_vector[range_idx:range_idx + n_range].values
        tick_labels_y = np.round(tick_labels_y[tick_idx_y], decimals=0).astype(np.int32)

        tick_idx_x = np.arange(start=0, stop=n_pings, step=int(n_pings / 6))
        tick_labels_x = self.time_vector[ping_idx:ping_idx + n_pings]
        tick_labels_x = tick_labels_x[tick_idx_x].values
        tick_labels_x = [np.datetime_as_string(t, unit='s').replace('T', '\n') for t in tick_labels_x]
        #
        plt.setp(axs, xticks=tick_idx_x, xticklabels=tick_labels_x,
                 yticks=tick_idx_y, yticklabels=tick_labels_y)

        # Format settings
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'green', 'red'])  # green = other, red = sandeel
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        # Get seabed
        if draw_seabed:
            seabed = self.get_seabed(idx_ping=ping_idx, n_pings=n_pings, idx_range=range_idx, n_range=n_range).astype(
                np.float)
            seabed[seabed >= data.shape[-1]] = None

        # Plot data
        for i in range(data.shape[0]):
            axs[i].imshow(data[i, :, :].T, cmap='jet', aspect='auto')
            axs[i].set_title(f"{str(frequencies[i])} Hz", fontsize=8)
            axs[i].set_ylabel('Range (m)')

        # Optionally plot labels
        if show_labels:
            labels = self.get_label_slice(ping_idx, n_pings, range_idx, n_range, drop_na=True)

            # crop labels
            labels = labels[:, :data.shape[-1]]
            axs[i + 1].imshow(labels.T, cmap=cmap_labels, norm=norm_labels, aspect='auto')
            axs[i + 1].set_ylabel('Range (m)')
            axs[i + 1].set_title('Annotations')

        # Optionally draw seabed
        if draw_seabed:
            for ax in axs:
                ax.plot(np.arange(data.shape[1]), seabed, c='white', lw=1)

        if predictions is not None:
            if type(predictions) != np.ndarray:
                predictions = predictions.annotation.sel(category=27)[range_idx:range_idx + n_range,
                              ping_idx:ping_idx + n_pings].values.astype(np.float32)

            # crop predictions (since we cut nans from the data)
            predictions = predictions[:data.shape[-1], :]

            assert predictions.shape == data[0, :, :].T.shape, print(
                f"Prediction shape {predictions.shape} does not match data shape {data.T.shape}")
            axs[i + 2].imshow(predictions, cmap='twilight_shifted', vmin=0, vmax=1, aspect='auto')
            axs[i + 2].set_title('Prediction (sandeel)')

        plt.xlabel('Ping time')
        plt.show()

    def estimate_seabed(self, raw_file=None, save_to_file=True):
        """ Return, load or calculate seabed for entire reader"""
        if self.seabed_dataset is not None:
            if raw_file is None:
                return self.seabed_dataset.seabed
            else:
                return self.seabed_dataset.seabed.where(self.seabed_dataset.raw_file == raw_file, drop=True).astype(
                    int).values
        elif os.path.isdir(self.seabed_path):
            self.seabed_dataset = xr.open_zarr(self.seabed_path)
            return self.get_seabed(raw_file)
        else:
            print("Estimate seabed")

            def seabed_gradient(data):
                gradient_filter_1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradient_filter_2 = np.array([[1, 5, 1], [-2, -10, -2], [1, 5, 1]])
                grad_1 = conv2d(data, gradient_filter_1, mode='same')
                grad_2 = conv2d(data, gradient_filter_2, mode='same')
                return np.multiply(np.heaviside(grad_1, 0), grad_2)

            data = self.ds.sv.fillna(0)  # fill nans with 0

            # Number of pixel rows at top of image (noise) not included when computing the maximal gradient
            n = 150  # 10*int(0.05*500)

            # Vertical shift of seabed approximation line (to give a conservative line)
            a = int(0.004 * 500)

            seabed = xr.DataArray(data=np.zeros((data.shape[:2])),
                                  dims=['frequency', 'ping_time'],
                                  coords={'frequency': data.frequency,
                                          'ping_time': data.ping_time,
                                          'raw_file': ("ping_time", data.raw_file)})

            for i in range(data.shape[0]):
                seabed_grad = xr.apply_ufunc(seabed_gradient, data[i, :, :], dask='allowed')
                seabed[i, :] = -a + n + seabed_grad[:, n:].argmax(axis=1)

            # Repair large jumps in seabed altitude
            # Set start/stop for repair interval [i_edge:-i_edge] to avoid repair at edge of echogram
            i_edge = 2

            # Use rolling mean and rolling std with window of 500 to find jumps in the seabed altitude
            repair_threshold = 0.75
            window_size = 500
            sb_max = seabed - seabed.rolling(ping_time=window_size, min_periods=1, center=True).mean()
            sb_max *= 1 / seabed.rolling(ping_time=window_size, min_periods=1, center=True).std()

            for f in range(sb_max.shape[0]):
                i = i_edge

                # Get indices of
                to_fix = np.argwhere(abs(sb_max[f, i:]).values > repair_threshold).ravel() + i
                k = 0
                while k < len(to_fix):
                    idx_0 = to_fix[k]

                    # Check if there is multiple subsequent indexes that needs repair
                    c = 0
                    while to_fix[k + c] == idx_0 + c:
                        c += 1
                        if k + c == len(to_fix):
                            break
                    idx_1 = idx_0 + c - 1

                    if idx_0 <= i_edge:
                        seabed[f, idx_0:idx_1 + 1] = seabed[f, idx_1 + 1]
                    elif idx_1 >= sb_max.shape[1] - i_edge:
                        seabed[f, idx_0:idx_1 + 1] = seabed[f, idx_0 - 1]
                    else:
                        seabed[f, idx_0:idx_1 + 1] = (seabed[f, [idx_0 - 1, idx_1 + 1]]).mean()

                    k += c

            s = xr.ufuncs.rint(seabed.median(dim='frequency'))
            self.seabed_dataset = xr.Dataset(data_vars={'seabed': s.astype(int)}, coords={'ping_time': s.ping_time})

            # save to zarr file
            if save_to_file:
                self.seabed_dataset.to_zarr(self.seabed_path)
            return self.get_seabed(raw_file=raw_file)

    # TODO Save to file, not in zarr?
    def _create_label_mask(self, heave=True):
        parquet_path = os.path.join(self.path.split('.')[0] + '_work.parquet')
        transducer_offset = self.ds.transducer_draft.mean(dim='frequency')

        if os.path.isfile(parquet_path):
            # read parquet data
            parquet_data = pd.read_parquet(os.path.join(parquet_path), engine='pyarrow')
            labels = np.zeros(shape=(self.ds.dims['ping_time'], self.ds.dims['range']))

            # add labels as variable to zarr
            self.ds["labels"] = (['ping_time', 'range'], labels)

            for _, row in parquet_data.iterrows():
                x0 = row['mask_depth_upper'] - transducer_offset.loc[row['pingTime']]
                x1 = row['mask_depth_lower'] - transducer_offset.loc[row['pingTime']]
                fish_id = int(row['ID'].split('-')[-1])

                if heave:
                    h = self.heave.loc[row['pingTime']]
                    if h == 0:
                        self.ds["labels_heave"].loc[row['pingTime'], x0:x1] = fish_id
                    else:
                        self.ds["labels_heave"].loc[row['pingTime'], x0 - h:x1 - h] = fish_id
                else:
                    # add fish observation to label mask
                    self.ds["labels"].loc[row['pingTime'], x0:x1] = fish_id


def get_zarr_readers(years='all', frequencies=np.array([18, 38, 120, 200]), minimum_shape=256,
                     path_to_zarr_files=None):
    if path_to_zarr_files is None:
        path_to_zarr_files = paths.path_to_zarr_files()

    if years == 'all':
        zarr_files = sorted([z_file for z_file in glob(path_to_zarr_files + '/**/*sv.zarr', recursive=True)])
    else:
        assert type(years) is list, f"Uknown years variable format: {type(years)}"
        zarr_files = []
        for year in years:
            zarr_files += glob(path_to_zarr_files + f'{year}/*/ACOUSTIC/GRIDDED/*sv.zarr', recursive=True)


    assert len(zarr_files) > 0, f"No survey data found at {path_to_zarr_files}"
    zarr_readers = [DataReaderZarr(zarr_file) for zarr_file in zarr_files]

    # Filter on frequencies
    zarr_readers = [z for z in zarr_readers if all([f in z.frequencies for f in frequencies])]

    # Ensure both sandeel and other categories are in zarr files
    zarr_readers = [z for z in zarr_readers if all([cat in z.fish_categories for cat in [27, 1]])]

    return zarr_readers


def get_echograms(years='all', path_to_echograms=None, frequencies=[18, 38, 120, 200], minimum_shape=256):
    """ Returns all the echograms for a given year that contain the given frequencies"""

    if path_to_echograms is None:
        path_to_echograms = paths.path_to_echograms()

    eg_names = os.listdir(path_to_echograms)
    eg_names = sorted(eg_names)  # To visualize echogram predictions in the same order with two different models
    eg_names = [name for name in eg_names if
                '.' not in name]  # Include folders only: exclude all root files (e.g. '.tar')

    echograms = [Echogram(os.path.join(path_to_echograms, e)) for e in eg_names]

    # Filter on frequencies
    echograms = [e for e in echograms if all([f in e.frequencies for f in frequencies])]

    # Filter on shape: minimum size
    echograms = [e for e in echograms if (e.shape[0] > minimum_shape) & (e.shape[1] > minimum_shape)]

    # Filter on shape of time_vector vs. image data: discard echograms with shape deviation
    echograms = [e for e in echograms if e.shape[1] == e.time_vector.shape[0]]

    # Filter on Korona depth measurements: discard echograms with missing depth files or deviating shape
    echograms = [e for e in echograms if e.name not in depth_excluded_echograms]

    # Filter on shape of heave vs. image data: discard echograms with shape deviation
    echograms = [e for e in echograms if e.shape[1] == e.heave.shape[0]]

    if years == 'all':
        return echograms
    else:
        # Make sure years is an itterable
        if type(years) not in [list, tuple, np.array]:
            years = [years]

        # Filter on years
        echograms = [e for e in echograms if e.year in years]

        return echograms


def get_data_readers(years='all', frequencies=[18, 38, 120, 200], minimum_shape=50, mode='zarr'):
    if mode == 'memm':
        return get_echograms(years=years, frequencies=frequencies, minimum_shape=minimum_shape)
    elif mode == 'zarr':
        return get_zarr_readers(years, frequencies, minimum_shape)


if __name__ == '__main__':
    pass

