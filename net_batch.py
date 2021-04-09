# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:22:51 2020

@author: Allen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import v_wrap, set_init

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.l2_const = 1e-4
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # for Actor
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2*self.s_dim*5, a_dim)
        
        self.distribution = torch.distributions.Categorical
        
    def forward(self, x):
        # Common layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Actor
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 2*self.s_dim*5)
        logits = F.softmax(self.act_fc1(x_act), dim = 1)
        
        return logits
    
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.l2_const = 1e-4
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # for Critic
        self.cri_conv1 = nn.Conv2d(128, 1, kernel_size=1)
        self.cri_fc1 = nn.Linear(1*self.s_dim*5, 64)
        self.cri_fc2 = nn.Linear(64, a_dim)
        
    def forward(self, x):
        # Common layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Critic
        x_cri = F.relu(self.cri_conv1(x))
        x_cri = x_cri.view(-1, 1*self.s_dim*5)
        x_cri = F.relu(self.cri_fc1(x_cri))
        values = self.cri_fc2(x_cri)

        return values
    
