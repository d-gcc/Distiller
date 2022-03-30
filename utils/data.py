from pathlib import Path
from dataclasses import dataclass
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import cast, Any, Dict, List, Tuple, Optional

import torch.quantization.quantize_fx as quantize_fx
import copy
import time

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation
from sklearn.metrics import mean_squared_error
from math import sqrt

@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float):
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=None)
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))



def load_ucr_data(config, use_encoder=True) -> Tuple[InputData, InputData]:

    train = np.loadtxt(config.data_folder / config.experiment /f'{config.experiment}_TRAIN.tsv', delimiter='\t')
    test = np.loadtxt(config.data_folder / config.experiment /f'{config.experiment}_TEST.tsv', delimiter='\t')

    if use_encoder:
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
        y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))
    else:
        y_train = np.expand_dims(train[:, 0], axis=-1)
        y_test = np.expand_dims(test[:, 0], axis=-1)

    if y_train.shape[1] == 2:
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
    
    original_data = train[:, 1:]
    test_data = test[:, 1:]
    
    if config.use_sax > 0:
        transform_data = np.array([])

        for time_series in original_data:
            scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
            dataset = scaler.fit_transform(time_series.reshape(1, -1))

            n_sax_symbols = config.sax_symbols
            if config.paa_segments > int(dataset.shape[1]/2):
                n_paa_symbols = int(dataset.shape[1]/2)
            else:
                n_paa_symbols = config.paa_segments
            
            sax = SymbolicAggregateApproximation(n_segments=n_paa_symbols,alphabet_size_avg=n_sax_symbols)
            sax_dataset = sax.fit_transform(dataset)
            
            sax_dataset_inv = sax.inverse_transform(sax_dataset)
            total_training_sax += sqrt(mean_squared_error(dataset[0].ravel(), sax_dataset_inv[0].ravel()))
            
            scaler2 = TimeSeriesScalerMeanVariance(mu=0., std=1.)
            transform_data = np.append(transform_data,scaler2.fit_transform(sax_dataset).squeeze())

        transformed_torch = torch.from_numpy(transform_data.reshape(-1,sax_dataset.squeeze().shape[0])).unsqueeze(1).float()
        train_input = InputData(x=transformed_torch,
                                y=torch.from_numpy(y_train))
        
        transform_test_data = np.array([])

        for time_series in test_data:
            scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
            dataset_test = scaler.fit_transform(time_series.reshape(1, -1))
            
            if config.use_sax == 2:
                n_sax_symbols = config.sax_symbols
                if config.paa_segments > int(dataset.shape[1]/2):
                    n_paa_symbols = int(dataset.shape[1]/2)
                else:
                    n_paa_symbols = config.paa_segments
                sax = SymbolicAggregateApproximation(n_segments=n_paa_symbols,alphabet_size_avg=n_sax_symbols)
                sax_dataset_test = sax.fit_transform(dataset_test)

                test_dataset_inv = sax.inverse_transform(sax_dataset_test)
                total_testing_sax += sqrt(mean_squared_error(dataset_test[0].ravel(), test_dataset_inv[0].ravel()))
                
                scaler2 = TimeSeriesScalerMeanVariance(mu=0., std=1.)
                transform_test_data = np.append(transform_test_data,scaler2.fit_transform(sax_dataset_test).squeeze())
                
                test_torch = torch.from_numpy(transform_test_data.reshape(-1,sax_dataset_test.squeeze().shape[0])).unsqueeze(1).float()
            else:
                scaler2 = TimeSeriesScalerMeanVariance(mu=0., std=1.)
                transform_test_data = np.append(transform_test_data,scaler2.fit_transform(dataset_test).squeeze())
            
                test_torch = torch.from_numpy(transform_test_data.reshape(-1,dataset_test.squeeze().shape[0])).unsqueeze(1).float()

        test_input = InputData(x=test_torch,
                           y=torch.from_numpy(y_test))
        
        
    else:
        train_input = InputData(x=torch.from_numpy(train[:, 1:]).unsqueeze(1).float(),
                                y=torch.from_numpy(y_train))
        test_input = InputData(x=torch.from_numpy(test[:, 1:]).unsqueeze(1).float(),
                           y=torch.from_numpy(y_test))

    return train_input, test_input

def get_loaders(config):

    train_data, test_data = load_ucr_data(config)
    train_data, val_data = train_data.split(config.val_size)

    train_loader = DataLoader(TensorDataset(train_data.x, train_data.y),batch_size=config.batch_size,shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data.x, val_data.y),batch_size=config.batch_size,shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data.x, test_data.y),batch_size=config.batch_size,shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_raw_data(config):

    train_data, test_data = load_ucr_data(config, use_encoder=False)
    train_data, val_data = train_data.split(config.val_size)
    
    return train_data, val_data, test_data

def get_kfold_loaders(config):
    train_loaders = []
    test_loaders = []
    kfold = KFold(n_splits=config.cross_validation)
    train_data, _ = load_ucr_data(config)
    for fold, (train_index, test_index) in enumerate(kfold.split(train_data.x, train_data.y)):

        x_train_fold = train_data.x[train_index]
        x_test_fold = train_data.x[test_index]
        y_train_fold = train_data.y[train_index]
        y_test_fold = train_data.y[test_index]

        train_loaders.append(DataLoader(TensorDataset(x_train_fold, y_train_fold),batch_size=config.batch_size,shuffle=False))
        test_loaders.append(DataLoader(TensorDataset(x_test_fold, y_test_fold),batch_size=config.batch_size,shuffle=False))
        
    return train_loaders, test_loaders