#good
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
#import torchcd 
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
parser.add_argument('--n1',type=int,default=128,help='First Embedded representation')#256
parser.add_argument('--n2',type=int,default=256,help='Second Embedded representation')
parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
parser.add_argument('--d_state', type=int, default=128, help='d_state parameter of Mamba')#256
parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
parser.add_argument('--e_fact', type=int, default=2, help='expand factor parameter of Mamba')
parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')



parser.add_argument('--model_name2', type=str, default='xLSTMTime', help='model_name2')
# IntegratedModel   model1 model2 dlinear
parser.add_argument('--dset', type=str, default='ecg', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64    , help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')

parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=1, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
# parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
#parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=3, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model')



#parser = argparse.ArgumentParser(description='Swin Transformer training and evaluation script', add_help=False)

#parser.add_argument('Swin Transformer training and evaluation script', add_help=False)
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)




args = parser.parse_args()
#print('args:', args)
#args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_IMAGE_SIZE'+str(args.IMAGE_SIZE)+'_NUM_CLASSES'+str(args.NUM_CLASSES) +'_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
#args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
#if not os.path.exists(args.save_path): os.makedirs(args.save_path)
args.save_model_name = str(args.model_name2)+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset #/My model/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


configs =args


def get_model(c_in,args):
    """
    c_in: number of input variables
    """

    #get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)


    ## get model
    model =  xlstm ( configs,enc_in=c_in,
           )
    return model

def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)

def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls,model,  loss_func , cbs=cbs  )  #cbs=cbs                      
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    #print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(dls.vars, args)
    #model = get_model(dls.vars, args, model_type)

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss

    #delta = 0.25
    #loss_func = HuberLoss(delta)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
    #cbs = [
         #PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model , loss_func,
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse,mae]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path =args.save_path+'/' + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)#cbs=cbs
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
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

'''
if __name__ == '__main__':
   
    if args.is_train:

        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)

    else:   # testing mode

        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
        
        for feature_idx in range(1):  # Assuming there are 7 features
           # plot_feature_actual_vs_predicted(out[1], out[0], feature_idx)
            plot_forecasted_interactive (out[1], out[0], feature_idx)
'''
if __name__ == '__main__':
    if args.is_train:
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:  # Testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)

        # Load the dataset
        dls = get_dls(args)  # Get the data loaders
        dataset = dls.train.dataset  # Access the dataset object from the data loader

        # Plot for a specific feature (e.g., feature_idx = 0) and sequence (e.g., seq_idx = 0)
        feature_idx = 0  # Change this to the desired feature index
        seq_idx = 549  # Change this to the desired sequence index
        plot_forecasted_interactive(out[1], out[0], feature_idx, dataset, seq_idx)
        plot_feature_actual_vs_predicted(out[1], out[0], feature_idx, dataset, seq_idx)
        plot_combined(out[1], out[0], feature_idx, dataset, seq_idx)
   
    print('----------- Complete! -----------')