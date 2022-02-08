#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import torch
import numpy as np
import pandas as pd
import copy
import os
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from pathlib import Path
from utils.data import get_loaders
from utils.util import str2bool, get_free_device 
from utils.inception import InceptionModel
from utils.distiller import DistillKL, KDEnsemble
from utils.trainer import train_single, train_distilled, evaluate


# In[2]:


def RunTeacher(model, config):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader, val_loader, test_loader = get_loaders(config)

    for epoch in range(1, config.epochs + 1):
        train_single(epoch, train_loader, val_loader, model, optimizer, config)
    
    if (config.distiller == 'teacher'):
        if not os.path.exists('./teachers/'):
            os.makedirs('./teachers/')
        model_name = f'Inception_{config.experiment}_{config.init_seed}_teacher.pkl'
        savepath = "./teachers/" + model_name
        torch.save(model.state_dict(), savepath)

    evaluate(test_loader, model, config)


# In[3]:


def RunStudent(model, config):
    data = torch.randn(7, 1, 400).to(config.device) 

    model_s = model
    model_s.eval()
    model_s = model_s.to(config.device)
    feat_s, _ = model_s(data)

    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_list = nn.ModuleList([])
    criterion_list.append(nn.CrossEntropyLoss())
    
    if config.distiller == 'kd':
        criterion_list.append(DistillKL(config.kd_temperature))
    elif config.distiller == 'kd_baseline':
        criterion_list.append(KDEnsemble(config.kd_temperature))

    optimizer = torch.optim.Adam(model_s.parameters(), lr=config.lr)
    
    # Teachers
    for teacher in range(0,config.teachers):
        savepath = Path('./teachers/Inception_'+config.experiment+ '_' + str(teacher) + '_teacher.pkl')
        teacher_config = copy.deepcopy(config)
        teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = config.bits
        model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=teacher_config)
        
        model_t.load_state_dict(torch.load(savepath,map_location=config.device))
        model_t.eval()
        model_t = model_t.to(config.device)
        feat_t, _ = model_t(data)
        module_list.append(model_t)

    module_list.to(config.device)
    criterion_list.to(config.device)
    train_loader, val_loader, test_loader = get_loaders(config)
    for epoch in range(1, config.epochs + 1):
        train_distilled(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, config)
    
    return evaluate(test_loader, model_s, config)


# In[4]:


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="A dummy argument for Jupyter", default="1")
    parser.add_argument('--experiment', type=str, default='SyntheticControl') 
    #ECG5000, ItalyPowerDemand, Plane, SyntheticControl

    # Quantization
    parser.add_argument('--bits', type=int, default=32)
    parser.add_argument('--bit1', type=int, default=13)
    parser.add_argument('--bit2', type=int, default=12)
    parser.add_argument('--bit3', type=int, default=8)
    parser.add_argument('--std_dev', type=float, default=0)
    parser.add_argument('--power_two', type=str2bool, default=False)
    parser.add_argument('--additive', type=str2bool, default=False)
    parser.add_argument('--grad_scale', type=int, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1500)
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--pid', type=int, default=0)

    # Distillation
    parser.add_argument('--distiller', type=str, default='kd', choices=['teacher', 'kd','kd_baseline'])
    parser.add_argument('--kd_temperature', type=float, default=4)
    parser.add_argument('--teachers', type=int, default=1)

    parser.add_argument('--w_ce', type=float, default=1, help='weight for cross entropy')
    parser.add_argument('--w_kl', type=float, default=0.9, help='weight for KL')
    parser.add_argument('--w_other', type=float, default=0.1, help='weight for other losses')
    
    # SAX - PAA
    parser.add_argument('--use_sax', type=int, default=0)
    parser.add_argument('--sax_symbols', type=int, default=8)
    parser.add_argument('--paa_segments', type=int, default=10)
    
    config = parser.parse_args()
    
    if config.device == -1:
        config.device = torch.device(get_free_device())
    else:
        config.device = torch.device("cuda:" + str(config.device))
    
    if config.init_seed > -1:
        np.random.seed(config.init_seed)
        torch.manual_seed(config.init_seed)
        torch.cuda.manual_seed(config.init_seed)
        torch.backends.cudnn.deterministic = True
    
    df = pd.read_csv('TimeSeries.csv',header=None)
    num_classes = int(df[(df == config.experiment).any(axis=1)][1])
    if num_classes == 2:
        num_classes = 1
    config.num_classes = num_classes

    if config.device == torch.device('cpu'):
        config.data_folder = Path('./dataset/TimeSeriesClassification')
    elif os.uname()[1] == 'cs-gpu04':
        config.data_folder = Path('/data/dgcc/TimeSeriesClassification')
    else:
        config.data_folder = Path('/data/cs.aau.dk/dgcc/TimeSeriesClassification')
    

    if config.distiller == 'teacher':
        teacher_config = config
        teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = config.bits
        model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=teacher_config)
        model_t = model_t.to(config.device)
        
        for teacher in range(0,config.teachers):
            config.init_seed = teacher
            np.random.seed(teacher)
            torch.manual_seed(teacher)
            torch.cuda.manual_seed(teacher)
            torch.backends.cudnn.deterministic = True
            RunTeacher(model_t, config)
    else:
        model_s = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=config)
        model_s = model_s.to(config.device)
        RunStudent(model_s, config)

