# Self-Supervised Feature Learning for Acoustic Data Analysis

This repository contains the implementation of self-supervised learning methods for acoustic data analysis, focusing on fisheries echosounder data. The primary goal of this study was to develop a deep learning model inspired by the DINO architecture to extract acoustic features without requiring manual annotations. The model was trained using multiple data sampling strategies to address class imbalance and improve the discriminative power of features in downstream tasks such as classification and regression.

## Main Model Overview

This repository implements a Self-Supervised Learning (SSL) model designed specifically for acoustic data analysis. The training process involves the following steps:

1. **Data Preparation**: Extract acoustic data patches from the training dataset using a predefined sampling scheme to form the training set.
2. **View Generation**: For each patch `x`, generate two global views and eight local views, forming sets `V_G` and `V_L`. The views are chosen in pairs from the combined set `V` for training.
3. **Network Assignment**:
   - The **teacher network** receives only global views (`V_G`).
   - The **student network** can receive both global and local views (`V`).
4. **Training Mechanism**: The teacher and student networks align their outputs by minimizing the dissimilarity between the chosen views.
5. **Parameter Updates**: 
   - **Student network** parameters (`θ_s`) are updated using the AdamW optimizer.
   - **Teacher network** parameters (`θ_t`) are updated using an exponential moving average (EMA) technique.


![SSL Model Overview](utils_unet/SSL_Framework_Figure_NEW.jpg)

*Figure: Overview of the self-supervised learning model applied in the study.*

## Repository Structure


## Prerequisites

- numpy~=1.26.2
- matplotlib~=3.8.1
- xarray~=0.20.0
- pandas~=1.2.3
- scipy~=1.11.3
- torch~=2.0.1
- PyYAML~=6.0.1
- Pillow~=10.0.1
- torchvision~=0.16.1
- tqdm~=4.66.1
- joblib~=1.3.2
- scikit-learn~=1.3.2
- zarr~=2.6.1
- numcodecs~=0.12.1
- opencv-python~=4.8.1.78
- requests~=2.28.1
- scikit-image~=0.22.0

## Usage


## Data Availability

The datasets used in this project are securely stored on servers managed by the Institute of Marine Research (IMR). Due to the large size of the dataset, access can be requested by contacting the corresponding author for an S3 access token. Please refer to the manuscript for detailed information on the dataset.

## Contact

For questions or collaboration inquiries, please reach out to the corresponding author, Ahmet Pala.
