# Modified Background Sampler for y axis - Patches are selected within the water column only, no exceed
# Near field (first 20 rows) are also excluded. the interval = [20, (seabed - 128)]
import numpy as np


class BackgroundZarr_ahmet():
    def __init__(self, zarr_files, window_size=(256, 256)):
        """
        Sample from zarr-files
        :param zarr_files: (list)
        :param window_size: (tuple)
        """
        self.zarr_files = zarr_files
        self.window_size = window_size

    def get_sample(self):
        # Select random zarr file in list
        zarr_rand = np.random.choice(self.zarr_files)

        # select random ping in zarr file
        x = np.random.randint(self.window_size[1] // 2, zarr_rand.shape[0] - self.window_size[1] // 2)

        # Get y-loc above seabed
        seabed = int(zarr_rand.get_seabed(x))


        if seabed - (self.window_size[0]+20) <= 0:
            return self.get_sample()
        y = np.random.randint(self.window_size[0]//2 + 20, seabed-self.window_size[0]//2)


        # Check if any fish_labels in the crop
        labels = zarr_rand.get_label_slice(idx_ping=x-self.window_size[1]//2,
                                           n_pings=self.window_size[1],
                                           idx_range=max(0, y-self.window_size[0]//2),
                                           n_range=self.window_size[0],
                                           drop_na=False,
                                           return_numpy=False)

        # Check if any fish-labels in crop
        if (labels > 0).any() or (labels == -1).all(): # Possible bottleneck?
            return self.get_sample()

        return [y, x], zarr_rand