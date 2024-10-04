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

from batch.samplers.background_near_miss import NMBackgroundZarr
from data.data_reader import get_data_readers
from batch.samplers.background import BackgroundZarr
from batch.samplers.seabed import SeabedZarr
from batch.samplers.school import SchoolZarr
from batch.samplers.school_seabed import SchoolSeabedZarr
from batch.samplers.gridded import Gridded
from batch.samplers.ahmet_background import BackgroundZarr_ahmet
from batch.samplers.gridded_portion import GriddedPortion
from batch.samplers.gridded_predefined_all_locations import GriddedPreDefinedAll
from batch.samplers.intensity_based import IntensityBased


class DataZarr:
    """  Partition zarr data into training, test and validation datasets """

    def __init__(self, frequencies, patch_size, partition_train, train_surveys, validation_surveys,
                 partition_predict, evaluation_surveys, save_prediction_surveys, eval_mode, sampling_strategy,
                 patch_overlap=20,
                 **kwargs):

        self.frequencies = sorted([freq for freq in frequencies])  # multiply by 1000 if frequency in Hz
        self.window_size = patch_size  # height, width

        # Get list of all memmap data readers (Echograms)
        # self.readers = get_data_readers(
        #     frequencies=self.frequencies,
        #     minimum_shape=self.window_size[0],
        #     mode="zarr",
        # )

        self.partition_train = partition_train  # Random, selected or all
        self.train_surveys = train_surveys  # List of surveys used for training
        self.validation_surveys = validation_surveys  # List of surveys used for testing

        # Evaluation / inference
        self.partition_predict = partition_predict
        self.evaluation_surveys = evaluation_surveys
        self.save_prediction_surveys = save_prediction_surveys  # List of surveys for which to save predictions
        self.eval_mode = eval_mode
        self.patch_overlap = patch_overlap

        self.sampling_strategy = sampling_strategy  # AHMET

        # print(f"{len(self.readers)} found:", [z.name for z in self.readers])

    # Partition data into train, test, val
    def partition_data_train(self):
        """
        Choose partitioning of data
        Currently only the partition 'single survey' can be used, i.e. we train and validate on the same surveys
        This should be changed in the future when the training procedure changes according to the zarr pre-processed format
        """

        assert self.partition_train in [
            "random",
            "selected surveys",
            "all surveys",
        ], "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"

        # Random partition of all surveys
        if self.partition_train == "random":
            readers = get_data_readers(
                years='all',
                frequencies=self.frequencies,
                minimum_shape=self.window_size[0],
                mode="zarr",
            )

            portion_train = 0.85

            # Set random seed to get the same partition every time
            np.random.seed(seed=10)
            np.random.shuffle(readers)

            train = readers[: int(portion_train * len(readers))]
            test = readers[int(portion_train * len(readers)):]

            # Reset random seed to generate random crops during training
            np.random.seed(seed=None)

        elif self.partition_train == "selected surveys":
            train = get_data_readers(self.train_surveys, frequencies=self.frequencies,
                                     minimum_shape=self.window_size[0],
                                     mode="zarr")
            test = get_data_readers(self.validation_surveys, frequencies=self.frequencies,
                                    minimum_shape=self.window_size[0],
                                    mode="zarr")

        elif self.partition_train == "all surveys":
            train_surveys = list(range(2007, 2019))
            train = get_data_readers(train_surveys, frequencies=self.frequencies,
                                     minimum_shape=self.window_size[0],
                                     mode="zarr")
            test = [survey for survey in train if survey.year == 2017]  # use 2017 survey as test after training on all

        else:
            raise ValueError(
                "Parameter 'partition' must equal 'random' or 'selected surveys' or 'single survey' or 'all surveys'"
            )

        len_train = 0
        n_pings_train = 0
        for ii in range(len(train)):
            #len_train += len(train[ii].raw_file_included)
            n_pings_train += train[ii].shape[0]

        len_test = 0
        n_pings_test = 0
        for ii in range(len(test)):
            #len_test += len(test[ii].raw_file_included)
            n_pings_test += test[ii].shape[0]

        print("Train: {} surveys, {} raw files, {} pings\nValidation: {} surveys, {} raw files, {} pings"
              .format(len(train), len_train, n_pings_train, len(test), len_test, n_pings_test))

        return train, test

    def get_samplers_train(self, readers_train=None, readers_test=None):
        """
        Provides a list of the samplers used to draw samples for training and validation
        :return list of the samplers used to draw samples for training,
        list of the samplers used to draw samples for validation and
        list of the sampling probabilities awarded to each of the samplers
        """
        if readers_train is None or readers_test is None:
            readers_train, readers_test = self.partition_data_train()

        if self.sampling_strategy == 'Near_Miss':
            samplers_train = [
                NMBackgroundZarr(readers_train, self.window_size),
                SchoolZarr(readers_train, self.window_size, 27),
                SchoolZarr(readers_train, self.window_size, 1)]

            # Also same random samplers for testing during training
            samplers_test = [
                NMBackgroundZarr(readers_test, self.window_size),
                SchoolZarr(readers_test, self.window_size, 27),
                SchoolZarr(readers_test, self.window_size, 1)]

            sampler_probs = [1, 1, 1]

        elif self.sampling_strategy == 'Random_Bg':
            samplers_train = [
                BackgroundZarr(readers_train, self.window_size, check_seabed=False, check_fish=True),
                SchoolZarr(readers_train, self.window_size, 27, check_seabed=False),
                SchoolZarr(readers_train, self.window_size, 1, check_seabed=False)]

            # Also same random samplers for testing during training
            samplers_test = [
                BackgroundZarr(readers_test, self.window_size, check_seabed=False, check_fish=True),
                SchoolZarr(readers_test, self.window_size, 27, check_seabed=False, pure_fish=True),
                SchoolZarr(readers_test, self.window_size, 1, check_seabed=False, pure_fish=True)]

            sampler_probs = [1, 1, 1]

        elif self.sampling_strategy == 'Complete_Random':
            samplers_train = [BackgroundZarr(readers_train, self.window_size, check_seabed=False, check_fish=False)]

            # Also same random samplers for testing during training
            samplers_test = [BackgroundZarr(readers_test, self.window_size, check_seabed=False, check_fish=False)]
            #samplers_test = [SchoolZarr(readers_test, self.window_size, 1, check_seabed=False)]

            sampler_probs = [1]

        elif self.sampling_strategy == 'NM_and_Random_Bg':
            samplers_train = [
                NMBackgroundZarr(readers_train, self.window_size),
                BackgroundZarr(readers_train, self.window_size),
                SchoolZarr(readers_train, self.window_size, 27),
                SchoolZarr(readers_train, self.window_size, 1)]

            # Also same random samplers for testing during training
            samplers_test = [
                NMBackgroundZarr(readers_test, self.window_size),
                BackgroundZarr(readers_test, self.window_size),
                SchoolZarr(readers_test, self.window_size, 27),
                SchoolZarr(readers_test, self.window_size, 1)]

            sampler_probs = [1, 1, 2, 2]

        elif self.sampling_strategy == '6_Classes':
            samplers_train = [
                BackgroundZarr(readers_train, self.window_size, check_seabed=True, check_fish=True),
                SeabedZarr(readers_train, self.window_size),
                SchoolZarr(readers_train, self.window_size, 27, check_seabed=True),
                SchoolZarr(readers_train, self.window_size, 1, check_seabed=True),
                SchoolSeabedZarr(
                    readers_train,
                    self.window_size,
                    max_dist_to_seabed=self.window_size[0] // 2,
                    fish_type=27,
                ),
                SchoolSeabedZarr(
                    readers_train,
                    self.window_size,
                    max_dist_to_seabed=self.window_size[0] // 2,
                    fish_type=1,
                ),
            ]

            # Also same random samplers for testing during training
            samplers_test = [
                BackgroundZarr(readers_test, self.window_size, check_seabed=True, check_fish=True),
                SeabedZarr(readers_test, self.window_size),
                SchoolZarr(readers_test, self.window_size, 27, check_seabed=True),
                SchoolZarr(readers_test, self.window_size, 1, check_seabed=True),
                SchoolSeabedZarr(
                    readers_test,
                    self.window_size,
                    max_dist_to_seabed=self.window_size[0] // 2,
                    fish_type=27,
                ),
                SchoolSeabedZarr(
                    readers_test,
                    self.window_size,
                    max_dist_to_seabed=self.window_size[0] // 2,
                    fish_type=1,
                ),
            ]

            sampler_probs = [1, 1, 1, 1, 1, 1]

        elif self.sampling_strategy == 'Echogram_Painting':
            samplers_train = [GriddedPortion(readers_train, self.window_size, start_ping=1286200, end_ping=1287200, start_range=0, end_range=328)]

            # Also same random samplers for testing during training
            samplers_test = [GriddedPortion(readers_test, self.window_size, start_ping=1286200, end_ping=1287200, start_range=0, end_range=328)]

            sampler_probs = [1]

        elif self.sampling_strategy == 'Gridded_PreDefined_All':
            samplers_train = [GriddedPreDefinedAll(readers_test, self.window_size)]

            # Also same random samplers for testing during training
            samplers_test = [GriddedPreDefinedAll(readers_test, self.window_size)]

            sampler_probs = [1]

        elif self.sampling_strategy == 'Intensity_Based':
            samplers_train = [IntensityBased(readers_train, self.window_size, thresh_l=-75, thresh_up=-60),
                              IntensityBased(readers_train, self.window_size, thresh_l=-60, thresh_up=-45),
                              IntensityBased(readers_train, self.window_size, thresh_l=-45, thresh_up=-30),
                              IntensityBased(readers_train, self.window_size, thresh_l=-30, thresh_up=-15),
                              IntensityBased(readers_train, self.window_size, thresh_l=-15, thresh_up=0.1)]

            # Also same random samplers for testing during training
            samplers_test = [IntensityBased(readers_test, self.window_size, thresh_l=-75, thresh_up=-60),
                              IntensityBased(readers_test, self.window_size, thresh_l=-60, thresh_up=-45),
                              IntensityBased(readers_test, self.window_size, thresh_l=-45, thresh_up=-30),
                              IntensityBased(readers_test, self.window_size, thresh_l=-30, thresh_up=-15),
                              IntensityBased(readers_test, self.window_size, thresh_l=-15, thresh_up=0.1)]

            sampler_probs = [1, 1, 1, 1, 1]

        elif self.sampling_strategy == 'Intensity_Based_2':
            samplers_train = [IntensityBased(readers_train, self.window_size, thresh_l=-74.9, thresh_up=-70),
                              IntensityBased(readers_train, self.window_size, thresh_l=-70, thresh_up=-65),
                              IntensityBased(readers_train, self.window_size, thresh_l=-65, thresh_up=-60),
                              IntensityBased(readers_train, self.window_size, thresh_l=-60, thresh_up=-55),
                              IntensityBased(readers_train, self.window_size, thresh_l=-55, thresh_up=0.1)]

            # Also same random samplers for testing during training
            samplers_test = [IntensityBased(readers_test, self.window_size, thresh_l=-74.9, thresh_up=-70),
                              IntensityBased(readers_test, self.window_size, thresh_l=-70, thresh_up=-65),
                              IntensityBased(readers_test, self.window_size, thresh_l=-65, thresh_up=-60),
                              IntensityBased(readers_test, self.window_size, thresh_l=-60, thresh_up=-55),
                              IntensityBased(readers_test, self.window_size, thresh_l=-55, thresh_up=0.1)]

            sampler_probs = [1, 1, 1, 1, 1]


        elif self.sampling_strategy == 'Intensity_Based_3':
            samplers_train = [IntensityBased(readers_train, self.window_size, thresh_l=-74.9, thresh_up=-71.87),
                              IntensityBased(readers_train, self.window_size, thresh_l=-71.87, thresh_up=-68.83),
                              IntensityBased(readers_train, self.window_size, thresh_l=-68.83, thresh_up=-65.79),
                              IntensityBased(readers_train, self.window_size, thresh_l=-65.79, thresh_up=-62.76),
                              IntensityBased(readers_train, self.window_size, thresh_l=-62.76, thresh_up=-59.73)]

            # Also same random samplers for testing during training
            samplers_test = [IntensityBased(readers_train, self.window_size, thresh_l=-74.9, thresh_up=-71.87),
                              IntensityBased(readers_train, self.window_size, thresh_l=-71.87, thresh_up=-68.83),
                              IntensityBased(readers_train, self.window_size, thresh_l=-68.83, thresh_up=-65.79),
                              IntensityBased(readers_train, self.window_size, thresh_l=-65.79, thresh_up=-62.76),
                              IntensityBased(readers_train, self.window_size, thresh_l=-62.76, thresh_up=-59.73)]

            sampler_probs = [1, 1, 1, 1, 1]


        elif self.sampling_strategy == 'Intensity_Based_4':
            samplers_train = [IntensityBased(readers_train, self.window_size, thresh_l=-74.9, thresh_up=-71.11),
                              IntensityBased(readers_train, self.window_size, thresh_l=-71.11, thresh_up=-67.31),
                              IntensityBased(readers_train, self.window_size, thresh_l=-67.31, thresh_up=-63.52),
                              IntensityBased(readers_train, self.window_size, thresh_l=-63.52, thresh_up=-59.73),
                              IntensityBased(readers_train, self.window_size, thresh_l=-59.73, thresh_up=0.1)]

            # Also same random samplers for testing during training
            samplers_test = [IntensityBased(readers_train, self.window_size, thresh_l=-74.9, thresh_up=-71.11),
                              IntensityBased(readers_train, self.window_size, thresh_l=-71.11, thresh_up=-67.31),
                              IntensityBased(readers_train, self.window_size, thresh_l=-67.31, thresh_up=-63.52),
                              IntensityBased(readers_train, self.window_size, thresh_l=-63.52, thresh_up=-59.73),
                              IntensityBased(readers_train, self.window_size, thresh_l=-59.73, thresh_up=0.1)]

            sampler_probs = [1, 1, 1, 1, 1]


        else:
            raise ValueError(f"Undefined sampling strategy: {self.sampling_strategy}")

        assert len(sampler_probs) == len(samplers_train)
        assert len(sampler_probs) == len(samplers_test)
        print(f"Sampling strategy for training and validation: {self.sampling_strategy}, {sampler_probs}")

        return samplers_train, samplers_test, sampler_probs

    def get_evaluation_surveys(self):
        """Get list of surveys to get predictions for / calculate evaluation metrics for"""
        if self.partition_predict == "all surveys":
            evaluation_survey_years = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
        elif self.partition_predict == "selected surveys":
            evaluation_survey_years = self.evaluation_surveys
        else:
            raise ValueError(f"Partition options: Options: selected surveys or all surveys - "
                             f"default: 'all surveys', not {self.partition_predict}")
        return evaluation_survey_years

    def get_gridded_survey_sampler(self, year):
        """ Create a gridded sampler which covers all data in one survey """
        surveys = get_data_readers([year], frequencies=self.frequencies,
                                   minimum_shape=self.window_size[0],
                                   mode="zarr")

        samplers = [Gridded(surveys,
                            window_size=self.window_size,
                            patch_overlap=self.patch_overlap,
                            mode=self.eval_mode)]

        return samplers

    def get_survey_readers(self, survey):
        return get_data_readers([survey], frequencies=self.frequencies,
                                minimum_shape=self.window_size[0],
                                mode="zarr")
