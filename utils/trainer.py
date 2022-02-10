from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
import torch
import torch.nn.functional as F
import os

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from .util import _to_1d_binary, insert_SQL 
import copy


def train_single(epoch, train_loader, val_loader, model, optimizer, config):
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


def train_distilled(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, config, teacher_weights=None):

    for module in module_list:
        module.eval()

    module_list[0].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    total_kl = 0
    total_ce_loss = 0
    
    total_teacher_losses = np.empty(config.teachers)
    
    for idx, data in enumerate(train_loader):
        batch_loss = 0
        teachers_loss = torch.zeros(1, dtype=torch.float32)
        
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
                if config.learned_kl_w:
                    teachers_loss = teachers_loss.add(torch.multiply(teacher_weights[teacher], loss_div))
                batch_loss += loss_div
            
        elif config.distiller == 'kd_baseline':
            logit_list = []
            for teacher in range(0,config.teachers):
                model_t = module_list[teacher+1]

                with torch.no_grad():
                    feat_t, logit_t = model_t(input)
                    logit_list.append(logit_t)

            loss_div = criterion_div(logit_s, logit_list)
            batch_loss += loss_div
            
        loss_kd = 0
              
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')        

        if config.learned_kl_w:
            loss = config.w_ce * loss_cls + teachers_loss + config.w_other * loss_kd 
        else:
            #loss = config.w_ce * loss_cls + config.w_kl * batch_loss + config.w_other * loss_kd 
            loss = config.w_ce * loss_cls + 1/config.teachers * batch_loss + config.w_other * loss_kd 

        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if config.learned_kl_w:
        with torch.no_grad():
            teacher_weights -= config.lr * 0.1 * teacher_weights.grad
            teacher_weights.grad = None
        return teacher_weights

            
def evaluate(val_loader, model, config):
    model.eval()

    model_eval = model
    trainingTime = time.time() 
    with torch.no_grad():
        true_list, preds_list = [], []
        for x, y in val_loader:
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
            insert_SQL("Inception", config.pid, config.experiment, 0, "Parameter", type_q, config.bits, config.distiller,
                       accuracy, "Seed", config.init_seed, "Metric 2", 0, "Metric 3", 0, "Metric 4", 0) 
        elif config.leaving_out:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            insert_SQL("Inception", config.pid, config.experiment, 0, "Parameter", type_q, config.bits, config.distiller,
                       accuracy, "Temperature", config.kd_temperature, "w_kl", config.w_kl, " ".join(str(e) for e in config.teacher_setting), config.teachers, "Metric 4", 0) 
         
        elif config.learned_kl_w:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            teacher_w = " ".join(str(round(e, 3)) for e in config.teacher_weights)
            insert_SQL("Inception", config.pid, config.experiment, teacher_w, "Teacher weights", type_q, config.bits, config.distiller,
                       accuracy, "Temperature", config.kd_temperature, "init_seed", config.init_seed, "Teachers", config.teachers, "Metric 4", 0) 
        else:
            type_q = "Mixed: " + str(config.bit1) + "-" + str(config.bit2) + "-" + str(config.bit3)
            insert_SQL("Inception", config.pid, config.experiment, 0, "Parameter", type_q, config.bits, config.distiller,
                       accuracy, "Temperature", config.kd_temperature, "w_kl", config.w_kl, "Teachers", config.teachers, "Metric 4", 0) 

        
        return accuracy
