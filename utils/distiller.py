from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class KDEnsemble(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KDEnsemble, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        if y_t:
            p_t = torch.zeros(y_t[0].size()).to("cuda:0")
            for y_ti in y_t:
                p_tf = F.softmax(y_ti/self.T, dim=1)
                p_t = torch.add(p_t,p_tf)
                
        loss = F.kl_div(p_s, p_t/len(y_t), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss