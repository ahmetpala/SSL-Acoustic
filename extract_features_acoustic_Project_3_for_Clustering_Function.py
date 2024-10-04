# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Write out a table of features - code ripped from eval_knn

import os
import random
import sys
import argparse

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

import utils
import vision_transformer as vits
from data.dataloaders import define_data_loaders
from data.partition import DataZarr

from modified_resnet import resnet18 as modified_resnet
from modified_resnet_64output import resnet18 as modified_resnet_resnet18_64
from modified_resnet_128output import resnet18 as modified_resnet_resnet18_128


# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# Set the seed for reproducibility
set_seed(42)  # Use any integer value as seed


def Extraction_for_Clustering(test_list):
    # Set the seed for reproducibility
    set_seed(42)  # Use any integer value as seed
    print('Hoho')
    if dist.is_initialized():
        dist.destroy_process_group()

    torch.cuda.empty_cache()

    class CustomDatasetTest:
        def __init__(self, data_array):
            self.data_array = data_array

        def __len__(self):
            return len(self.data_array)

        def __getitem__(self, idx):
            return self.data_array[idx]

    def extract_feature_pipeline(args):

        custom_dataset = CustomDatasetTest(test_list)
        # Create a dataloader
        data_loader = DataLoader(custom_dataset, args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers)

        # print(f"{args.batch_size_per_gpu * args.test_iter} patches from acoustic data will be predicted.")

        # ============ building network ... ============
        if "vit" in args.arch:
            model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            # print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
        elif "xcit" in args.arch:
            model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch](num_classes=0)
            model.fc = nn.Identity()
        elif args.arch == 'modified_resnet18':
            model = modified_resnet(num_classes=0)
            model.fc = nn.Identity()
        elif args.arch == 'modified_resnet18_64':
            model = modified_resnet_resnet18_64(num_classes=0)
            model.fc = nn.Identity()
        elif args.arch == 'modified_resnet18_128':
            model = modified_resnet_resnet18_128(num_classes=0)
            model.fc = nn.Identity()
        else:
            print(f"Architecture {args.arch} non supported")
            sys.exit(1)

        if args.use_cuda: model.cuda()
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        model.eval()

        # ============ extract features ... ============
        print("Extracting features...")
        features = extract_features(model, data_loader, args.use_cuda)

        if utils.get_rank() == 0:  # what does this do?
            features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def extract_features(model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None  # Initialize features

        for i, loader_output in enumerate(metric_logger.log_every(data_loader, 10000)):

            index = torch.add(torch.arange(0, len(loader_output)),
                              (i * args.batch_size_per_gpu))  # Defining index numbers
            samples = loader_output.to(torch.float32)  # Acoustic data definition

            if use_cuda:
                samples = samples.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # init storage feature matrix
            if dist.get_rank() == 0 and features is None:
                features = torch.zeros((args.batch_size_per_gpu * args.test_iter), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                # print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        return features

    class ReturnIndexDataset(datasets.ImageFolder):
        def __getitem__(self, idx):
            img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
            return img, idx

    parser = argparse.ArgumentParser('extract_features', description="Extract features from a data set")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size')  # 64 for Undersampled_Test, 25 for Echogram Painting
    parser.add_argument('--pretrained_weights',
                        default='/scratch/disk5/ahmet/dino_output/100ep_acoustics_8w_Intensity_Based2FIXED_9_TrainSurveys_NEAREST_ResNet18Modified_128NONNormalized_8192Out_FixAug_2048Batch_FixedData(NoBlur)/checkpoint.pth',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,  # Cuda usage gives error for now
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='modified_resnet18_128', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--test_iter', default=-(-test_list.shape[0] // 64), type=int,
                        # 205 for Echogram Painting, 801 for Gridded_Predefined (5000 for Model25_Balanced_320000_8w, and 64 batch size)
                        help='Number of iterations for test dataloader.')  # len(undersampled_patch)/batch_size for Undersampled_Test_8w_Specific_Year!!!!!
    parser.add_argument('--test_year', default=2017, type=int,
                        # Test years for features. Saved data array will be used.
                        help='Number of iterations for test dataloader.')
    parser.add_argument('--output',
                        default='/scratch/disk5/ahmet/dino_plain//Project_3/Overlapping_features_echogram_1289700_2017.csv',
                        help='Filename for saving computed features.')
    window_size = [8, 8]
    parser.add_argument("--sampling_strategy_acoustic", default='Echogram_Painting_Project_3',
                        type=str, help='Sampling strategy for acoustic data.')

    # Check if running in a Jupyter Notebook
    if 'ipykernel' in sys.modules:
        class Args:
            batch_size_per_gpu = 64
            pretrained_weights = '/scratch/disk5/ahmet/dino_output/100ep_acoustics_8w_Intensity_Based2FIXED_9_TrainSurveys_NEAREST_ResNet18Modified_128NONNormalized_8192Out_FixAug_2048Batch_FixedData(NoBlur)/checkpoint.pth'
            use_cuda = True
            arch = 'modified_resnet18_128'
            patch_size = 16
            checkpoint_key = "teacher"
            num_workers = 0
            dist_url = "env://"
            data_path = None
            test_iter = -(-test_list.shape[0] // 64)
            test_year = 2017
            output = '/scratch/disk5/ahmet/dino_plain//Project_3/Overlapping_features_echogram_1289700_2017.csv'
            sampling_strategy_acoustic = 'Echogram_Painting_Project_3'

        args = Args()
    else:
        args = parser.parse_args()

    if not dist.is_initialized():
        utils.init_distributed_mode(args)
    else:
        pass

    cudnn.benchmark = False

    # Extract the features
    features = extract_feature_pipeline(args)

    feats = features.cpu()
    feats = feats[:test_list.shape[0], :]  # Taking the features only affiliated with the data.

    return feats
