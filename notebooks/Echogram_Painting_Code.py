import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, OPTICS, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KNeighborsClassifier

from batch.data_transforms.db_with_limits import db_with_limits
from batch.label_transforms.refine_label_boundary import refine_label_boundary
from data.data_reader import DataReaderZarr
import pickle



# Survey Definition

years = [2007,2008,2009,2010,2011,2013,2014,2015,2016,2017,2018]
year_codes = [2007205,2008205,2009107,2010205,2011206,2013842,2014807,2015837,2016837,2017843,2018823]
y = 9
#  Path to surveys
survey_path = f'/scratch/disk5/ahmet/data/{years[y]}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_sv.zarr' #'/Users/apa055/Desktop/data'
survey = DataReaderZarr(survey_path)
loaded_patch_data_array = np.load(f'/scratch/disk5/ahmet/data/{survey.year}/{year_codes[y]}/ACOUSTIC/GRIDDED/{year_codes[y]}_patch_data_array_all_8w.npy')
print(survey.year)


features_echogram = pd.read_csv('/scratch/disk5/ahmet/dino_output/extracted_features/Model33_EchogramPainting.csv', header=None, sep=r',|\t', engine='python') #1289700

# Preparation of Original Sv Values
with open('/scratch/disk5/ahmet/data/8w_Data_EchogramPaint_2017/list_echogram_painting_2017_8w_model33.pkl', "rb") as f: # list_echogram_painting_2017_8w_.pkl for model 25
                                                                                                                                        # list_echogram_painting_2017_8w_model33_ping_1286200.pkl - model 33, sadece sandeel
    echogram_Sv_list = pickle.load(f)

Sv_flattened = echogram_Sv_list['data'].reshape(echogram_Sv_list['data'].shape[0], echogram_Sv_list['data'].shape[1]*echogram_Sv_list['data'].shape[2]*echogram_Sv_list['data'].shape[3])
# Reshape one-dimensional arrays to have shape (n, 1)
echogram_Sv_list_patch_class = echogram_Sv_list['patch_class'].reshape(-1, 1)
echogram_Sv_list_year = echogram_Sv_list['year'].reshape(-1, 1)
echogram_Sv_list_center_coordinates = echogram_Sv_list['center_coordinates']
echogram_Sv_list_n_other = echogram_Sv_list['n_other'].reshape(-1, 1)
echogram_Sv_list_n_sandeel = echogram_Sv_list['n_sandeel'].reshape(-1, 1)
echogram_Sv_list_n_bottom = echogram_Sv_list['n_bottom'].reshape(-1, 1)
echogram_Sv_list_Sv_200_mean = echogram_Sv_list['Sv_200_mean'].reshape(-1, 1)
echogram_Sv_list_n_total = echogram_Sv_list['n_total'].reshape(-1, 1)
Sv_flattened_reshaped = Sv_flattened


# Concatenate arrays
Sv_ayarlanmis_data = np.concatenate([np.array(features_echogram[0]).reshape(-1, 1),echogram_Sv_list_patch_class,echogram_Sv_list_year,echogram_Sv_list_center_coordinates,
    echogram_Sv_list_patch_class,echogram_Sv_list_n_other,echogram_Sv_list_n_sandeel,echogram_Sv_list_n_bottom,echogram_Sv_list_Sv_200_mean,echogram_Sv_list_n_total,
    Sv_flattened_reshaped], axis=1)

def visualize_data(ping_start_value, ping_end_value, range_start, range_end, num_clusters,
                   Sv_ayarlanmis_data, features_echogram, cmap_for_clusters, plot_last=True,
                   plot_Sv_clusters=True, savefig=False):
    # Applying k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #kmeans = OPTICS(xi=0.0005, min_cluster_size=200, min_samples=5)
    #kmeans = DBSCAN(eps=0.5, min_samples=200, metric='minkowski', leaf_size=5, p=1)
    kmeans_classes = kmeans.fit_predict(features_echogram.iloc[:, 11:139])
    kmeans_Sv_original_classes = kmeans.fit_predict(np.array(Sv_ayarlanmis_data[:, 11:], dtype=np.float64))

    # Predicting Corresponding Patches
    knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean').fit(features_echogram.iloc[:100, 11:139], features_echogram.iloc[:100, 1])
    echogram_classes = knn.predict(features_echogram.iloc[:, 11:139])
    final_data = pd.concat \
        ([pd.DataFrame(kmeans_classes), pd.DataFrame(echogram_classes), pd.DataFrame(kmeans_Sv_original_classes), features_echogram.iloc[: ,3:5]], axis=1)
    final_data.columns = ['kmeans_cluster', 'class', 'k_means_Sv', 'center_y', 'center_x']

    additi = -10  # -150000
    patch_size = 8
    frequencies = [18, 38, 120, 200]

    ping_start = ((ping_start_value + additi + 8) // patch_size) * patch_size
    ping_end = ((ping_end_value + additi) // patch_size) * patch_size

    ping_slice = slice(ping_start, ping_end)
    range_slice = slice(range_start, range_end)

    yeni_array = final_data[
        (final_data['center_x'] > ping_start) &
        (final_data['center_x'] < ping_end) &
        (final_data['center_y'] < range_end)
        ]

    gercek_array = loaded_patch_data_array[
        (loaded_patch_data_array['center_x'] > ping_start) &
        (loaded_patch_data_array['center_x'] < ping_end) &
        (loaded_patch_data_array['center_y'] < range_end)
        ]

    yeni_array['center_x'] = yeni_array['center_x'] - yeni_array['center_x'].min() + patch_size // 2
    gercek_array['center_x'] = gercek_array['center_x'] - gercek_array['center_x'].min() + patch_size // 2

    dat1 = survey.ds.sv.isel(ping_time=ping_slice, range=range_slice).sel(frequency=frequencies)
    seabed = survey.get_seabed_mask(idx_ping=ping_start, n_pings=ping_end - ping_start, idx_range=range_start, n_range=range_end - range_start, return_numpy=False)
    labels = survey.get_label_slice(idx_ping=ping_start, n_pings=ping_end - ping_start, idx_range=range_start, n_range=range_end - range_start, drop_na=False, return_numpy=False, categories=survey.fish_categories)
    labels_values = labels.values

    labels_values[labels_values == 27.0] = 2.0
    labels_values[labels_values == 5027.0] = 4.0
    labels_values[labels_values == 6009.0] = 4.0

    modified = refine_label_boundary(ignore_zero_inside_bbox=False).__call__(dat1.values, labels_values, [1], [1])[1]
    modified[seabed == 1.0] = 0.0

    labels_from_patch_data = np.empty_like(modified, dtype=object)
    clusters_from_patch_data = np.empty_like(modified, dtype=object)
    clusters_Sv_from_patch_data = np.empty_like(modified, dtype=object)
    labels_from_gercek_patch_data = np.empty_like(modified, dtype=object)

    for cluster, label, cluster_Sv, center_y, center_x in np.array(yeni_array):
        start_x = max(0, center_x - patch_size // 2)
        end_x = min(modified.shape[0], center_x + patch_size // 2)
        start_y = max(0, center_y - patch_size // 2)
        end_y = min(modified.shape[1], center_y + patch_size // 2)

        labels_from_patch_data[start_x:end_x, start_y:end_y] = label
        clusters_from_patch_data[start_x:end_x, start_y:end_y] = cluster
        clusters_Sv_from_patch_data[start_x:end_x, start_y:end_y] = cluster_Sv

    for center_x_gercek, center_y_gercek, label_gercek, _, _, _, _, _ in gercek_array:
        start_x = max(0, center_x_gercek - patch_size // 2)
        end_x = min(modified.shape[0], center_x_gercek + patch_size // 2)
        start_y = max(0, center_y_gercek - patch_size // 2)
        end_y = min(modified.shape[1], center_y_gercek + patch_size // 2)

        labels_from_gercek_patch_data[start_x:end_x, start_y:end_y] = label_gercek

    transposed_modified = modified.T
    transposed_labels = labels_from_patch_data.T
    transposed_gercek_labels = labels_from_gercek_patch_data.T
    transposed_clusters_from_patch_data = clusters_from_patch_data.T
    transposed_clusters_Sv_from_patch_data = clusters_Sv_from_patch_data.T

    transposed_labels[transposed_labels == None] = 'ignore'
    transposed_clusters_from_patch_data[transposed_clusters_from_patch_data == None] = num_clusters + 1
    transposed_clusters_Sv_from_patch_data[transposed_clusters_Sv_from_patch_data == None] = num_clusters + 1


    fig, ax = plt.subplots(3 +plot_last +plot_Sv_clusters, 1, figsize=(12, 14), sharex=True)

    im0 = ax[0].imshow(db_with_limits(dat1[3].T.values, 1, 2, frequencies)[0], cmap='viridis', aspect="auto")
    ax[0].set_title('Acoustic Data in 200 kHz')
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="1%", pad=0.05)
    cbar0 = plt.colorbar(im0, cax=cax0, label='$S_v$')

    im1 = ax[1].imshow(transposed_modified, aspect="auto")
    ax[1].set_title('Modified Annotations')
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="1%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, ticks=[0, 1, 2, 4], label='Pixel class')
    cbar1.set_ticklabels(['background', 'other', 'sandeel', 'ignore'])

    if plot_Sv_clusters:
        im2 = ax[2].imshow(transposed_clusters_Sv_from_patch_data.astype(float), aspect="auto",
                           interpolation='nearest', cmap=cmap_for_clusters, alpha=0.7)
        ax[2].set_title('KMeans Clusters - Original $S_v$')
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes("right", size="1%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2, label='Cluster number')

    im3 = ax[ 2 +plot_Sv_clusters].imshow(transposed_clusters_from_patch_data.astype(float), aspect="auto",
                                          interpolation='nearest', cmap=cmap_for_clusters, alpha=0.7)
    ax[ 2 +plot_Sv_clusters].set_title('KMeans Clusters - SSL Features')
    divider3 = make_axes_locatable(ax[ 2 +plot_Sv_clusters])
    cax3 = divider3.append_axes("right", size="1%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3, label='Cluster number')

    if plot_last:
        im4 = ax[2 +plot_last +plot_Sv_clusters].imshow(numeric_labels_gercek, vmin=0, vmax=len(label_mapping) - 1, aspect="auto")
        ax[2 +plot_last +plot_Sv_clusters].set_title('Corresponding Real Patch Classes')
        divider4 = make_axes_locatable(ax[2 +plot_last +plot_Sv_clusters])
        cax4 = divider4.append_axes("right", size="1%", pad=0.05)
        cbar4 = plt.colorbar(im4, cax=cax4, ticks=[0, 1, 4, 5], label='Patch class')
        cbar4.set_ticklabels(['background', 'other', 'sandeel', 'ignore'])

    # Add x label
    fig.text(0.5, 0.07, 'ping time', ha='center')
    # Add y label and position it at the middle
    fig.text(0.07, 0.5, 'range', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0.3)
    if savefig: plt.savefig('echogram_painting_model_25_ping_1289700_plain', dpi=600)
    plt.show()

# Example usage:
ping_start_value = 1289700 #1289700
ping_end_value = 1290700 #1290700
range_start = 0
range_end = 320
num_clusters = 6
Sv_ayarlanmis_data = Sv_ayarlanmis_data
features_echogram = features_echogram

# Plot without the last subplot
visualize_data(ping_start_value, ping_end_value,
               range_start, range_end, num_clusters, Sv_ayarlanmis_data,
               features_echogram, 'inferno',
               plot_last=False, plot_Sv_clusters=True, savefig=False)
