from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
import torch
import torch.nn.functional as F
import os

from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from .util import _to_1d_binary, insert_SQL 
from utils.inception import InceptionModel
import copy


def train_single(epoch, train_loader, model, optimizer, config):
    model.train()

    for idx, data in enumerate(train_loader):
        input, target = data
        input = input.float()
        input = input.to(config.device)
        target = target.to(config.device)
        feat_s, logit_s = model(input)
              
        if len(target.shape) == 1:
            loss = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config):

    for module in module_list:
        module.eval()

    module_list[0].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_weights = module_list[-1]

    total_kl = 0
    total_ce_loss = 0
        
    total_teacher_losses = np.empty(config.teachers)
    
    for idx, data in enumerate(train_loader):
        batch_loss = 0
        teachers_loss = torch.zeros(config.teachers, dtype=torch.float32, device = config.device)
        
        input, target = data
        index = len(input)

        input = input.float()

        input = input.to(config.device)
        target = target.to(config.device)

        feat_s, logit_s = model_s(input)
        loss_cls = criterion_cls(logit_s, target.argmax(dim=-1))
        
        if config.distiller == 'kd':
            for teacher in range(0,config.teachers):
                model_t = module_list[teacher+1]

                with torch.no_grad():
                    feat_t, logit_t = model_t(input)

                loss_div = criterion_div(logit_s, logit_t)
                teachers_loss[teacher] += loss_div
                batch_loss += loss_div
                
            with torch.no_grad():
                teacher_losses, ensemble_weights = model_weights(teachers_loss)
                ensemble_loss = torch.sum(teacher_losses)
            
        elif config.distiller == 'kd_baseline':
            logit_list = []
            for teacher in range(0,config.teachers):
                model_t = module_list[teacher+1]

                with torch.no_grad():
                    feat_t, logit_t = model_t(input)
                    logit_list.append(logit_t)

            loss_div = criterion_div(logit_s, logit_list)
            batch_loss += loss_div
            ensemble_loss = batch_loss

        loss_kd = 0
              
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')       

        loss = config.w_ce * loss_cls + config.w_kl * batch_loss + config.w_other * loss_kd

        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
def validation(epoch, val_loader, module_list, criterion_list, optimizer, config):

    for module in module_list:
        module.eval()

    module_list[-1].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_weights = module_list[-1]

    total_kl = 0
    total_ce_loss = 0
        
    total_teacher_losses = np.empty(config.teachers)
    
    for idx, data in enumerate(val_loader):
        batch_loss = 0
        teachers_loss = torch.zeros(config.teachers, dtype=torch.float32, device = config.device)
        
        input, target = data
        index = len(input)

        input = input.float()

        input = input.to(config.device)
        target = target.to(config.device)

        with torch.no_grad():
            feat_s, logit_s = model_s(input)

        loss_cls = criterion_cls(logit_s, target.argmax(dim=-1))
        
        for teacher in range(0,config.teachers):
            model_t = module_list[teacher+1]

            with torch.no_grad():
                feat_t, logit_t = model_t(input)

            loss_div = criterion_div(logit_s, logit_t)
            teachers_loss[teacher] += loss_div
            batch_loss += loss_div
            
        loss_kd = 0
              
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')       

        teacher_losses, ensemble_weights = model_weights(teachers_loss)
        ensemble_loss = torch.sum(teacher_losses)

        loss = config.w_ce * loss_cls + config.w_kl * ensemble_loss + config.w_other * loss_kd

        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    config.teacher_weights = ensemble_weights.tolist()

            
def evaluate(test_loader, model, config):
    model.eval()

    with torch.no_grad():
        true_list, preds_list = [], []
        for x, y in test_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():
                true_list.append(y.cpu().detach().numpy())
                _, preds = model(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.cpu().detach().numpy())

        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)
        accuracy = accuracy_score(*_to_1d_binary(true_np, preds_np), normalize=True)

        try:
            roc_auc = roc_auc_score(true_np, preds_np)
            pr_auc = average_precision_score(true_np, preds_np)
        except Exception as e:
            print("PR and ROC one class undefinition")


        if config.distiller == 'teacher':
            type_q = "Full precision: " + str(config.bits)
            insert_SQL("Inception", config.pid, config.experiment, "Parameter", 0, type_q, config.bits, config.distiller,
                       accuracy, "Seed", config.init_seed, "Metric 2", 0, "Metric 3", 0, "Metric 4", 0) 
        elif config.leaving_out:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            insert_SQL("Inception", config.pid, config.experiment, "Parameter", 0, type_q, config.bits, config.distiller,
                       accuracy, "Temperature", config.kd_temperature, "w_kl", 
                       config.w_kl, " ".join(str(e) for e in config.teacher_setting), config.teachers, "Metric 4", 0) 
         
        elif config.learned_kl_w:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            teacher_w = "-".join(str(round(e, 3)) for e in config.teacher_weights)
            insert_SQL("Inception", config.pid, config.experiment, "Teacher weights", teacher_w, type_q, config.bits, 
                       config.distiller, accuracy, "Temperature", config.kd_temperature, "init_seed", config.init_seed, 
                       "Teachers", config.teachers, "Epochs", config.epochs) 
        else:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            insert_SQL("Inception", config.pid, config.experiment, "Parameter", 0, type_q, config.bits, config.distiller,
                       accuracy, "Temperature", config.kd_temperature, "w_kl", config.w_kl, "Teachers", config.teachers, 
                       "Metric 4", 0) 
        
        return accuracy

def evaluate_ensemble(test_loader, config):
    
    teachers = [i for i in range(0,config.teachers)]
    ensemble_result = []
    for teacher in teachers:
        savepath = Path('./teachers/Inception_' + config.experiment + '_' + str(teacher) + '_teacher.pkl')
        teacher_config = copy.deepcopy(config)
        teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = config.bits
        model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=teacher_config)

        model_t.load_state_dict(torch.load(savepath, map_location=config.device))
        model_t.eval()
        model_t = model_t.to(config.device)

        with torch.no_grad():
            true_list, preds_list = [], []
            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                with torch.no_grad():
                    true_list.append(y.cpu().detach().numpy())
                    _, preds = model_t(x)
                    if len(y.shape) == 1:
                        preds = torch.sigmoid(preds)
                    else:
                        preds = torch.softmax(preds, dim=-1)
                    preds_list.append(preds)

            true_np, preds_tensor = np.concatenate(true_list), torch.cat(preds_list)
        ensemble_result.append(preds_tensor)

    sum_probabilities = torch.stack(ensemble_result).sum(dim=0)
    sum_np = sum_probabilities.cpu().detach().numpy()

    accuracy = accuracy_score(*_to_1d_binary(true_np, sum_np), normalize=True)

    type_q = "Full precision: " + str(config.bits)
    insert_SQL("Inception", config.pid, config.experiment, "Teacher Ensemble", 0, type_q, config.bits, config.distiller,
               accuracy, "Metric 1", 0, "Metric 2", 0, "Metric 3", 0, "Metric 4", 0) 

    
    return accuracy