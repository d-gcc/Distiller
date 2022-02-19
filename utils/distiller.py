from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherWeights(nn.Module):
    def __init__(self, config, weights):
        super(TeacherWeights,self).__init__()
        self.W = torch.nn.Parameter(weights)  #torch.nn.Parameter(torch.rand(config.teachers)) 
        
    def forward(self, x):
        teacher_weights = F.softmax(self.W, dim=0) 
        teacher_loss = torch.multiply(teacher_weights, x)
        return teacher_loss, teacher_weights

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class KDEnsemble(nn.Module):
    def __init__(self, T, device):
        super(KDEnsemble, self).__init__()
        self.T = T
        self.device = device

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        if y_t:
            p_t = torch.zeros(y_t[0].size()).to(self.device)
            for y_ti in y_t:
                p_tf = F.softmax(y_ti/self.T, dim=1)
                p_t = torch.add(p_t,p_tf)
                
        loss = F.kl_div(p_s, p_t/len(y_t), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss