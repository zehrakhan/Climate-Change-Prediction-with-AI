import sys
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import numpy as np
import pandas as pd
import os
from torch import nn
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls

import time
import random
import argparse
import datetime
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig
from xlstm import xlstm

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
assert torch.__version__ >= '1.8.0', "DDP-based MoE requires Pytorch >= 1.8.0"

from dataclasses import dataclass
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n1', type=int, default=256, help='First Embedded representation')
parser.add_argument('--n2', type=int, default=256, help='Second Embedded representation')
parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
parser.add_argument('--d_state', type=int, default=1024, help='d_state parameter of Mamba')
parser.add_argument('--dconv', type=int, default=4, help='d_conv parameter of Mamba')
parser.add_argument('--e_fact', type=int, default=2, help='expand factor parameter of Mamba')
parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')

parser.add_argument('--model_name2', type=str, default='xLSTMTime', help='model_name2')
parser.add_argument('--dset', type=str, default='ecg', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')

parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=1, help='whether to use time features or not')
parser.add_argument('--patch_len', type=int, default=24, help='patch length')
parser.add_argument('--stride', type=int, default=24, help='stride between patch')
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
parser.add_argument('--n_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--d_model', type=int, default=1024, help='Transformer d_model')
parser.add_argument('--dropout', type=float, default=0.3, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
parser.add_argument('--is_train', type=int, default=0, help='training the model')

parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

args = parser.parse_args()
args.save_model_name = str(args.model_name2) + '_cw' + str(args.context_points) + '_tw' + str(args.target_points) + '_patch' + str(args.patch_len) + '_stride' + str(args.stride) + '_epochs' + str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

configs = args

def get_model(c_in, args):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)
    model = xlstm(configs, enc_in=c_in)
    return model

def combined_loss(input, target, alpha=0.5):
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)

def find_lr():
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = torch.nn.L1Loss(reduction='mean')
    cbs = [RevInCB(dls.vars)] if args.revin else []
    learn = Learner(dls, model, loss_func, cbs=cbs)
    return learn.lr_finder()

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mae, mse, r2

import plotly.graph_objects as go
import numpy as np

def plot_loss(train_losses, valid_losses):
    fig = go.Figure()

    # Plot the training loss
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(train_losses)),
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue'),
            hoverinfo='x+y',
        )
    )

    # Plot the validation loss
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(valid_losses)),
            y=valid_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red'),
            hoverinfo='x+y',
        )
    )

    # Update the layout of the plot
    fig.update_layout(
        title="Training and Validation Loss Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
    )

    # Show the plot
    fig.show()

def train_func(lr=args.lr):
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = torch.nn.L1Loss(reduction='mean')
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [SaveModelCB(monitor='valid_loss', fname=args.save_model_name, path=args.save_path)]
    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs, metrics=[mse, mae])
    # Train the model and capture training and validation loss at each epoch
    loss_history = learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)

    # Check if loss_history is returned and contains both train and validation losses
    if loss_history is not None:
        # Extract the losses from the history dictionary
        train_losses = [epoch['train_loss'] for epoch in loss_history]
        valid_losses = [epoch['valid_loss'] for epoch in loss_history]

        # Plot the losses
        plot_loss(train_losses, valid_losses)
    else:
        print("Training did not return loss history.")

def test_func():
    weight_path = args.save_path + '/' + args.save_model_name + '.pth'
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    cbs = [RevInCB(dls.vars)] if args.revin else []
    learn = Learner(dls, model, cbs=cbs)

    # test the model
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])  # out: a list of [pred, targ, score_values]

    # Check if out[1] is a PyTorch tensor, then move to CPU if necessary
    actual = out[1]
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()  # Actual values (3D array: num_samples, sequence_length, num_features)
    else:
        actual = np.array(actual)  # If it's already a numpy array, convert to numpy (this step might not be needed)

    # Check if out[0] is a PyTorch tensor, then move to CPU if necessary
    predicted = out[0]
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()  # Predicted values (3D array: num_samples, sequence_length, num_features)
    else:
        predicted = np.array(predicted)  # If it's already a numpy array, convert to numpy (this step might not be needed)

    # Flatten the arrays for R² calculation
    actual_flat = actual.reshape(-1)  # Flatten to 1D array
    predicted_flat = predicted.reshape(-1)  # Flatten to 1D array

    # Calculate metrics
    mae_value = mean_absolute_error(actual_flat, predicted_flat)
    mse_value = mean_squared_error(actual_flat, predicted_flat)
    r2 = r2_score(actual_flat, predicted_flat)

    print(f"MAE: {mae_value}")
    print(f"MSE: {mse_value}")
    print(f"R²: {r2}")

    return out

import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go

def plot_feature_actual_vs_predicted(actual, predicted, feature_idx, dataset, seq_idx):
    """
    Plot the actual vs predicted values for a specific feature and sequence.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    - dataset (Dataset_Custom): The dataset object to access the inverse_transform method.
    - seq_idx (int): Index of the sequence to plot.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Select the specific sequence for the given feature index
    actual_feature = actual[seq_idx, :, feature_idx]  # Shape: (seq_len,)
    predicted_feature = predicted[seq_idx, :, feature_idx]  # Shape: (seq_len,)

    # Reshape the actual and predicted values into 2D arrays for inverse transformation
    actual_feature_2d = actual_feature.reshape(-1, 1)  # Shape: (seq_len, 1)
    predicted_feature_2d = predicted_feature.reshape(-1, 1)  # Shape: (seq_len, 1)

    # Create dummy arrays with the same number of features as the dataset
    dummy_actual = np.zeros((actual_feature_2d.shape[0], dataset.data_x.shape[1]))
    dummy_predicted = np.zeros((predicted_feature_2d.shape[0], dataset.data_x.shape[1]))

    # Fill only the target feature column
    dummy_actual[:, feature_idx] = actual_feature_2d.flatten()
    dummy_predicted[:, feature_idx] = predicted_feature_2d.flatten()

    # Apply inverse transformation to get the original scale
    original_actual = dataset.inverse_transform(dummy_actual)[:, feature_idx]
    original_predicted = dataset.inverse_transform(dummy_predicted)[:, feature_idx]
    print('original_predicted',original_predicted)
    # Create a Plotly figure
    fig = go.Figure()

    # Add actual values trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_actual)),
            y=original_actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue'),
            hoverinfo='x+y',
        )
    )

    # Add predicted values trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_predicted)),
            y=original_predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash'),
            hoverinfo='x+y',
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Actual vs Predicted for Feature {feature_idx}, Sequence {seq_idx}",
        xaxis_title="Time Steps",
        yaxis_title="Average Temperature",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
    )

    # Show the plot
    fig.show()


def plot_forecasted_interactive(actual, predicted, feature_idx, dataset, seq_idx):
    """
    Plot historical and forecasted values for a specific feature and sequence.
    The data is plotted in its original form by applying inverse transformation.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    - dataset (Dataset_Custom): The dataset object to access the inverse_transform method.
    - seq_idx (int): Index of the sequence to plot.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Select the specific sequence for the given feature index
    actual_feature = actual[seq_idx, :, feature_idx]  # Shape: (seq_len,)
    predicted_feature = predicted[seq_idx, :, feature_idx]  # Shape: (pred_len,)

    # Combine historical values and predicted values for the current sequence
    combined_values = np.concatenate([actual_feature, predicted_feature])  # Shape: (seq_len + pred_len,)

    # Reshape the combined values for inverse transformation
    # Create a dummy array with the same number of features as the dataset
    dummy_data = np.zeros((len(combined_values), dataset.data_x.shape[1]))
    dummy_data[:, feature_idx] = combined_values  # Fill only the target feature column

    # Apply inverse transformation to get the original scale
    original_values = dataset.inverse_transform(dummy_data)[:, feature_idx]

    # Split the original values back into historical and forecasted parts
    original_historical = original_values[:actual_feature.shape[0]]  # Shape: (seq_len,)
    original_forecasted = original_values[actual_feature.shape[0]:]  # Shape: (pred_len,)

    # Create a Plotly figure for the current sequence
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_historical)),
            y=original_historical,
            mode='lines',
            name='Historical',
            line=dict(color='blue'),
            hoverinfo='x+y',
        )
    )

    # Add forecasted data trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_historical), len(original_values)),
            y=original_forecasted,
            mode='lines',
            name='Forecasted',
            line=dict(color='red', dash='dash'),
            hoverinfo='x+y',
        )
    )

    # Add a vertical line to indicate the start of the forecast
    fig.add_vline(
        x=len(original_historical),
        line=dict(color='gray', dash='dash'),
        annotation_text='Forecast Start',
        annotation_position='top right',
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Historical and Forecasted Values for Feature {feature_idx} (Sequence {seq_idx})",
        xaxis_title="Time Steps",
        yaxis_title="Average Temperature",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
    )

    # Show the plot
    fig.show()


import plotly.graph_objects as go
import numpy as np
import torch

def plot_combined(actual, predicted, feature_idx, dataset, seq_idx=0):
    """
    Plot actual vs predicted values and forecasted values for a specific feature and sequence in one graph.
    The data is plotted in its original form by applying inverse transformation.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    - dataset (Dataset_Custom): The dataset object to access the inverse_transform method.
    - seq_idx (int): Index of the sequence to plot.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Select the specific sequence for the given feature index
    actual_feature = actual[seq_idx, :, feature_idx]  # Shape: (seq_len,)
    predicted_feature = predicted[seq_idx, :, feature_idx]  # Shape: (pred_len,)

    # Combine historical values and predicted values for the current sequence
    combined_values = np.concatenate([actual_feature, predicted_feature])  # Shape: (seq_len + pred_len,)

    # Reshape the combined values for inverse transformation
    combined_values_2d = combined_values.reshape(-1, 1)  # Shape: (n, 1)

    # Create a dummy array with the same number of features as the dataset
    dummy_data = np.zeros((combined_values_2d.shape[0], dataset.data_x.shape[1]))
    dummy_data[:, feature_idx] = combined_values_2d.flatten()  # Fill only the target feature column

    # Apply inverse transformation to get the original scale
    original_values = dataset.inverse_transform(dummy_data)[:, feature_idx]

    # Split the original values into historical, predicted, and forecasted parts
    original_historical = original_values[:len(actual_feature)]  # Shape: (seq_len,)
    original_predicted = original_values[len(actual_feature):len(actual_feature) + len(actual_feature)]  # Shape: (seq_len,)
    original_forecasted = original_values[len(actual_feature):]  # Shape: (pred_len,)

    # Create a Plotly figure
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_historical)),
            y=original_historical,
            mode='lines',
            name='Historical',
            line=dict(color='blue'),
            hoverinfo='x+y',
        )
    )

    # Add predicted data trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_historical)),
            y=original_predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash'),
            hoverinfo='x+y',
        )
    )

    # Add forecasted data trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(original_historical), len(original_values)),
            y=original_forecasted,
            mode='lines',
            name='Forecasted',
            line=dict(color='red', dash='dash'),
            hoverinfo='x+y',
        )
    )

    # Add a vertical line to indicate the start of the forecast
    fig.add_vline(
        x=len(original_historical),
        line=dict(color='gray', dash='dash'),
        annotation_text='Forecast Start',
        annotation_position='top right',
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Actual, Predicted, and Forecasted Values for Feature {feature_idx}, Sequence {seq_idx}",
        xaxis_title="Time Steps",
        yaxis_title="Average Temperature",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
    )

    # Show the plot
    fig.show()

if __name__ == '__main__':
    if args.is_train:
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)

        dls = get_dls(args)
        dataset = dls.train.dataset
        feature_idx = 0
        seq_idx = 549
        plot_forecasted_interactive(out[1], out[0], feature_idx, dataset, seq_idx)
        plot_feature_actual_vs_predicted(out[1], out[0], feature_idx, dataset, seq_idx)
        plot_combined(out[1], out[0], feature_idx, dataset, seq_idx)

    print('----------- Complete! -----------')