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
import pickle
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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CustomDatasetTest(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list[next(iter(self.data_list))])

    def __getitem__(self, idx):
        # Initialize an empty dictionary to hold samples for each key
        samples = {}

        # Iterate over each key and extract data for the corresponding index
        for key in self.data_list.keys():
            samples[key] = self.data_list[key][idx]

        return samples

def extract_feature_pipeline(args):
    # ========== preparing data ... ==========
    #transform = pth_transforms.Compose([
    #    pth_transforms.Resize(256, interpolation=3),
    #    pth_transforms.CenterCrop(224),
    #    pth_transforms.ToTensor(),
    #    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #])

    # dataset = ReturnIndexDataset(args.data_path, transform=transform)

    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    # data_loader = torch.utils.data.DataLoader(
    #    dataset,
    #    sampler=sampler,
    #    batch_size=args.batch_size_per_gpu,
    #    num_workers=args.num_workers,
    #    pin_memory=True,
    #    drop_last=False,
    # )

    # Create a custom dataset
    if args.sampling_strategy_acoustic == 'Undersampled_Test_ESKI_KAYMIS':
        with open('/scratch/disk5/ahmet/data/8w_Test_2017/list_undersampled_2017_8w_.pkl', "rb") as f:
            test_list = pickle.load(f)
    elif args.sampling_strategy_acoustic == 'Echogram_Painting':
        with open('/scratch/disk5/ahmet/data/8w_Data_EchogramPaint_2017/list_echogram_painting_2017_8w_model33.pkl', "rb") as f:
            test_list = pickle.load(f)
    elif args.sampling_strategy_acoustic == 'Balanced_320000_8w': # List Data for balanced sampling (saved manually)
        with open('/scratch/disk5/ahmet/data/8w_Sampled_Data_9Years/Balanced/List/list_balanced_data_9Years_8w_.pkl', "rb") as f:
            test_list = pickle.load(f)
    elif args.sampling_strategy_acoustic == 'Undersampled_Test_8w_Specific_Year':
        with open(f'/scratch/disk5/ahmet/data/8w_Test_9Years/list_undersampled_{args.test_year}_8w_.pkl', "rb") as f:
            test_list = pickle.load(f)
            test_list['center_coordinates'] = np.array(test_list['center_coordinates']).T # Fixing inconsistency

    custom_dataset = CustomDatasetTest(test_list)
    # Create a dataloader
    data_loader = DataLoader(custom_dataset, args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers)


    print(f"{args.batch_size_per_gpu * args.test_iter} patches from acoustic data will be predicted.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
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
    (features, labels, center_coordinates, years,
     n_other, n_sandeel, n_bottom, Sv_200_mean, n_total) = extract_features(model, data_loader, args.use_cuda)

    if utils.get_rank() == 0:  # what does this do?
        features = nn.functional.normalize(features, dim=1, p=2)

    labels = np.array(labels) # Converting list to np array
    years = np.array(years) # Converting list to np array

    return features, labels, center_coordinates, years, n_other, n_sandeel, n_bottom, Sv_200_mean, n_total


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None # Initialize features
    labels = [] # Initialize labels
    n_other = [] # Initialize others
    n_sandeel = [] # Initialize sandeel
    n_bottom = [] # Initialize bottom
    Sv_200_mean = [] # Initialize Sv_200_mean
    n_total = [] # Initialize total
    years = [] # Initialize labels
    center_coordinates = torch.empty(0, 2, dtype=torch.int16) # Init center coordinates
    composed_transform = pth_transforms.Compose([
        pth_transforms.Resize(224, interpolation=InterpolationMode.NEAREST, antialias=True),  # Resize
        #pth_transforms.CenterCrop(224)  # Center crop
    ])

    for i, loader_output in enumerate(metric_logger.log_every(data_loader, 5)):

        index = torch.add(torch.arange(0, len(loader_output['data'])), (i*args.batch_size_per_gpu))  # Defining index numbers
        #samples = composed_transform(loader_output['data']).to(torch.float32)  # Bunu kaldirdik, resize a gerek yok
        samples = loader_output['data'].to(torch.float32)  # Acoustic data definition

        #import matplotlib.pyplot as plt
        #print(loader_output['patch_class'])
        #plt.imshow(loader_output['data'][0, 3, :, :])
        #plt.title('Original Image')
        #plt.colorbar()
        #plt.show()

        #plt.imshow(samples[0, 3, :, :])
        #plt.title('Final Image')
        #plt.colorbar()
        #plt.show()

        if use_cuda:
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros((args.batch_size_per_gpu*args.test_iter), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # Append labels
        labels.extend(loader_output['patch_class'])
        years.extend(loader_output['year'])
        n_other.extend(loader_output['n_other'])
        n_sandeel.extend(loader_output['n_sandeel'])
        n_bottom.extend(loader_output['n_bottom'])
        Sv_200_mean.extend(loader_output['Sv_200_mean'])
        n_total.extend(loader_output['n_total'])

        # Append center coordinates
        center_coordinates = torch.cat((center_coordinates,loader_output['center_coordinates']))

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
    return features, labels, center_coordinates, years, n_other, n_sandeel, n_bottom, Sv_200_mean, n_total


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('extract_features', description="Extract features from a data set")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size') # 64 for Undersampled_Test, 25 for Echogram Painting
    parser.add_argument('--pretrained_weights',
                        default='/scratch/disk5/ahmet/dino_output/100ep_acoustics_8w_YENI_Intensity_Based_9_TrainSurveys_NEAREST_ResNet18Modified_128NONNormalized_8192Out_FixAug_2048Batch_FixedData(NoBlur)/checkpoint.pth',
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
    parser.add_argument('--test_iter', default=1987, type=int, # 205 for Echogram Painting, 801 for Gridded_Predefined (5000 for Model25_Balanced_320000_8w, and 64 batch size)
                        help='Number of iterations for test dataloader.') # len(undersampled_patch)/batch_size for Undersampled_Test_8w_Specific_Year!!!!!
    parser.add_argument('--test_year', default=2017, type=int,
                        # Test years for features. Saved data array will be used.
                        help='Number of iterations for test dataloader.')
    parser.add_argument('--output', default='/scratch/disk5/ahmet/dino_output/extracted_features/100ep_acoustics_8w_YENI_Intensity_Based_9_TrainSurveys_NEAREST_ResNet18Modified_128NONNormalized_8192Out_FixAug_2048Batch_FixedData(NoBlur)_2017_GriddedAll_NoCenterCrop_IgnoredFish_FixedTestData.csv',
                        help='Filename for saving computed features.')
    parser.add_argument("--window_size", default=[8, 8], type=int, help='Window size for acoustic data.')
    parser.add_argument("--sampling_strategy_acoustic", default='Undersampled_Test_8w_Specific_Year',
                        type=str, help='Sampling strategy for acoustic data.')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Extract the features
    (features, labels, coordinates, years,
     n_other, n_sandeel, n_bottom, Sv_200_mean, n_total) = extract_feature_pipeline(args)

    # save features and labels as CSV
    print('Saving the extracted features...')
    feats = features.cpu()
    if args.output and dist.get_rank() == 0:
        with open(args.output, "w") as f:
            for i, (lab, year, nother, nsandeel, nbottom, Sv200_mean, ntotal) in enumerate(zip(labels, years, n_other, n_sandeel, n_bottom, Sv_200_mean, n_total)):
                f.write(f'{i}\t{str(lab)}\t{str(year)}\t{coordinates[i,0]}\t{coordinates[i,1]}\t{labels[i]}\t{nother}\t{nsandeel}\t{nbottom}\t{Sv200_mean}\t{ntotal}\t')
                for j in feats[i]:
                    f.write(str(float(j)))
                    f.write(',')
                f.write('del\n')

    # this does...?
    dist.barrier()
