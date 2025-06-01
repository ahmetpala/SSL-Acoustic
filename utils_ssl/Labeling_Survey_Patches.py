
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from data.data_reader import DataReaderZarr, get_data_readers
from batch.data_transforms.db_with_limits import db_with_limits
from batch.label_transforms.refine_label_boundary import refine_label_boundary

years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017]
year_codes = [2007205, 2008205, 2009107, 2010205, 2011206, 2013842, 2014807, 2015837, 2016837, 2017843]

frequencies = [18, 38, 120, 200]
patch_size = 8
threshold_percentage = 2

patch_data = []
Sv_patch_data = []

batch_size = 12800

for y in range(len(years)):

    survey_path = f'/scratch/disk5/ahmet/data/{years[y]}'  # '/Users/apa055/Desktop/data'
    surveys = [f'/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_sv.zarr']
    readers = [DataReaderZarr(survey_path + zarr_file) for zarr_file in surveys]
    survey = readers[0]

    son_hesap = patch_size * ((survey.shape[0] - patch_size) // patch_size)

    for ping_start in tqdm(range(0, son_hesap, batch_size), desc="Processing batches"):
        ping_end = min(ping_start + batch_size, son_hesap)
        ping_slice = slice(ping_start, ping_end)

        range_start = 4
        range_end = 708
        range_slice = slice(range_start, range_end)

        dat1 = survey.ds.sv.isel(ping_time=ping_slice, range=range_slice).sel(frequency=frequencies)
        dat1_200 = db_with_limits(dat1[3].values, 1, 2, frequencies)[0]
        dat1_all = db_with_limits(dat1.values, 1, 2, frequencies)[0]

        seabed = survey.get_seabed_mask(idx_ping=ping_start, n_pings=ping_end - ping_start, idx_range=range_start,
                                        n_range=range_end - range_start, return_numpy=False)
        labels = survey.get_label_slice(idx_ping=ping_start, n_pings=ping_end - ping_start, idx_range=range_start,
                                        n_range=range_end - range_start, drop_na=False, return_numpy=False,
                                        categories=survey.fish_categories)
        labels_values = labels.values

        labels_values[labels_values == 27.0] = 2.0
        labels_values[labels_values == 5027.0] = 4.0
        labels_values[labels_values == 6009.0] = 5.0

        # Annotation Modification
        modified = refine_label_boundary(ignore_zero_inside_bbox=False).__call__(dat1.values, labels_values, [1], [1])[
            1]
        modified[seabed == 1.0] = 3.0

        for i in range(0, modified.shape[0], patch_size):
            for j in range(0, modified.shape[1], patch_size):
                # Sv_200_mean = dat1_200[i:i+patch_size, j:j+patch_size].mean()
                Sv_all_mean = dat1_all[:, i:i + patch_size, j:j + patch_size].mean()
                patch = modified[i:i + patch_size, j:j + patch_size]

                sv_patch = dat1_all[:, i:i + patch_size, j:j + patch_size]

                count_0 = np.sum(patch == 0)
                count_1 = np.sum(patch == 1)
                count_2 = np.sum(patch == 2)
                count_3 = np.sum(patch == 3)

                count_4 = np.sum(patch == 4)
                count_5 = np.sum(patch == 5)

                total_elements = patch.size

                percentage_0 = count_0 / total_elements * 100
                percentage_1 = count_1 / total_elements * 100
                percentage_2 = count_2 / total_elements * 100
                percentage_3 = count_3 / total_elements * 100

                if percentage_2 > threshold_percentage:
                    patch_label = 'sandeel'
                if percentage_1 > threshold_percentage:
                    patch_label = 'other'
                if percentage_1 <= threshold_percentage and percentage_2 <= threshold_percentage:
                    patch_label = 'background'
                if percentage_1 > threshold_percentage and percentage_2 > threshold_percentage:
                    patch_label = 'sandeel_other'
                if count_3 >= 50:
                    patch_label = 'ignore'

                if count_4 > 4 or count_5 > 4:
                    patch_label = 'ignored_fish'
                if 'sandeel' in patch_label and count_3 > 2:
                    patch_label = 'sandeel_seabed'
                elif 'other' in patch_label and count_3 > 2:
                    patch_label = 'other_seabed'
                elif 'background' in patch_label and 0 < count_3 < total_elements - 8:
                    patch_label = 'seabed'
                elif 'sandeel_other' in patch_label and count_3 > 0:
                    patch_label = 'sandeel_other_seabed'

                center_x = i + patch_size // 2 + ping_start
                center_y = j + patch_size // 2
                if not np.isnan(sv_patch).any() and patch_label != 'ignore':
                    patch_data.append(
                        (center_x, center_y, patch_label, count_1, count_2, count_3, Sv_all_mean, total_elements))
                    Sv_patch_data.append(sv_patch)
                else:
                    pass

    dtype = [('center_x', int), ('center_y', int), ('label', 'U20'), ('count_1', int), ('count_2', int),
             ('count_3', int),
             ('Sv_all_mean', float), ('total_elements', int)]

    patch_data_array_all = np.array(patch_data, dtype=dtype)

    np.save(f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_patch_data_array_all_{patch_size}w.npy', patch_data_array_all)
