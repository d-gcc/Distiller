from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
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


def train_distilled(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""

    for module in module_list:
        module.eval()

    module_list[0].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    
    total_kl = 0
    total_ce_loss = 0
    total_teacher_losses = np.empty(opt.teachers)
    
    for idx, data in enumerate(train_loader):
        batch_loss = 0
        teacher_losses = np.empty(opt.teachers)
        
        input, target = data
        index = len(input)

        input = input.float()

        input = input.to(opt.device)
        target = target.to(opt.device)


        # ===================forward=====================
        feat_s, logit_s = model_s(input)
        #logit_s = model_s(input)
        loss_cls = criterion_cls(logit_s, target.argmax(dim=-1))
        
        
        if opt.distiller == 'kd':
            count_t = 0
            for teacher in range(0,opt.teachers):
                count_t += 1
                model_t = module_list[teacher+1]

                with torch.no_grad():
                    feat_t, logit_t = model_t(input)

                loss_div = criterion_div(logit_s, logit_t)
                batch_loss += loss_div
            batch_loss = batch_loss * 100
            
        elif opt.distiller == 'kd_unified':
            logit_list = []
            for teacher in range(0,opt.paa_segments):
                if teacher in [0,2,4,9]:
                    model_t = module_list[teacher+1]

                    with torch.no_grad():
                        feat_t, logit_t = model_t(input)
                        logit_list.append(logit_t)

            loss_div = criterion_div(logit_s, logit_list)
            batch_loss += loss_div

        elif opt.distiller == 'kd_single':
            model_t = module_list[opt.teacher_number]

            with torch.no_grad():
                feat_t, logit_t = model_t(input)
            batch_loss = criterion_div(logit_s, logit_t)
           

        if opt.distiller == 'kd' or opt.distiller == 'kd_unified' or opt.distiller == 'kd_single':
            loss_kd = 0
              
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')        

        #loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd  
        loss = opt.gamma * loss_cls + opt.alpha * batch_loss + opt.beta * loss_kd  
        
        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
            
def evaluate(val_loader, model, config):

    # switch to evaluate mode
    model.eval()

    #model = model_quantized

    model_eval = model
    trainingTime = time.time() 
    with torch.no_grad():
        true_list, preds_list = [], []
        for x, y in val_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():
                true_list.append(y.cpu().detach().numpy())
                _, preds = model(x)
                #preds = model_eval(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.cpu().detach().numpy())

        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)
        accuracy = accuracy_score(*_to_1d_binary(true_np, preds_np), normalize=True)
        #print('Accuracy: ' + str(top1))
        inferenceTime = time.time() 
        #print('Inference time in seconds: ' + str(inferenceTime-trainingTime))
        roc = 0 #inferenceTime-trainingTime #roc_auc_score(true_np, preds_np)
        pr = 0 #average_precision_score(true_np, preds_np)


        if config.bits == 32:
            type_q = "Full precision"
        else:
            if config.power_two:
                if config.additive:
                    type_q = "Additive power-of-two"
                else:
                    type_q = "Power-of-two"

            else:
                type_q = "Uniform"

#         insert_SQL(config.model_s, config.pid, config.experiment, config.num_classes, type_q, 
#                    config.bits, config.distill, 1, 1, top1, 
#                    roc, timeBefore)        
        #insert_SQL(config.model_s, config.pid, config.experiment, config.sax_symbols, type_q, config.bits, 
        #           config.distill, config.std_dev, config.paa_segments, top1, config.epochs, config.teacher_number)       
        insert_SQL("Inception", config.pid, config.experiment, 12, "Parameter", type_q, config.bits, config.distiller,
                   accuracy, 13, "Metric 1", 14, "Metric 2", 15, "Metric 3", 16, "Metric 4") 
        
        return accuracy
