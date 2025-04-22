import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

# Only "illness" dataset is available
DSETS = ['ecg']

def get_dls(params):
    assert params.dset in DSETS, f"Unrecognized dataset (`{params.dset}`). Only 'ecg' is available."

    if not hasattr(params, 'use_time_features'):
        params.use_time_features = True

    # Set correct dataset path
    root_path = 'C:/Users/mehja/Desktop/zahra/zehra_thesis/codes'
    data_path = 'cleaned_achen_data_v2.csv'  # Only available dataset

    size = [params.context_points, 0, params.target_points]

    dls = DataLoaders(
        datasetCls=Dataset_Custom,
        dataset_kwargs={
            'root_path': root_path,
            'data_path': data_path,
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },
        batch_size=params.batch_size,
        workers=params.num_workers,
    )

    # Ensure the dataset is loaded correctly before accessing its attributes
    sample_data = dls.train.dataset[0]
    if sample_data:
        dls.vars, dls.len = sample_data[0].shape[1], params.context_points
        dls.c = sample_data[1].shape[0]
    else:
        raise ValueError("Dataset is empty or incorrectly loaded.")

    return dls


if __name__ == "__main__":
    class Params:
        def __init__(self):
            self.dset = 'ecg'  # Only ecg dataset is available
            self.context_points = 384
            self.target_points = 96
            self.batch_size = 64
            self.num_workers = 8
            self.with_ray = False
            self.features = 'M'

    params = Params()  # Correctly instantiate Params
    dls = get_dls(params)

    # Debugging: Print dataset details
    print(f"Dataset loaded successfully: {params.dset}")
    print(f"Training data sample shape: {dls.train.dataset[0][0].shape}, {dls.train.dataset[0][1].shape}")
