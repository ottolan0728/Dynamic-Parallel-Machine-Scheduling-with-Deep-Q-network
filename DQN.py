# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:30:41 2020

@author: Allen
"""

import torch
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
from net_batch import Critic
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DQN(object):
    def __init__(self, gamma, tau, s_dim, a_dim, checkpoint_dir=None):
        self.gamma = torch.tensor(gamma)
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.distribution = torch.distributions.Categorical
        # early stopping
        self.capacity = 900
        self.memory = []
        self.position = 0
        self.gap_stop = 0.05
        self.epoch = 0
        self.check_epoch = 200000
        
        # Record losses
        self.polciy_losses = []
        self.value_losses = []
        self.total_losses = []
        
        self.policy_record, self.value_record, self.best_policy, self.best_value  = 0,0,9999,9999
        self.policy_stop, self.value_stop = False, False
        
        # action & reward buffer
        self.rewards = []
        self.batch_memory = []

        # Define the critic
        self.critic = Critic(self.s_dim, self.a_dim).to(device)
        self.critic_target = Critic(self.s_dim, self.a_dim).to(device)
        
        # Define the optimizers for Critic networks
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=1e-3,
                                                 weight_decay=1e-2
                                                 )  # optimizer for the critic network
        
        # Make sure both targets are with the same weight
        hard_update(self.critic_target, self.critic)
        
    def choose_action(self, s, rate, test = False, explore = True):
        self.critic.eval()  # Sets the actor in evaluation mode
        actions_value = self.critic(s)
        self.critic.train()  # Sets the actor in training mode
        
        # explore rate = 0.9 rate decay by multiply 0.99
        if np.random.randint(100) < rate and explore :
            action = np.random.randint(0, self.a_dim)
        else:
            action = torch.max(actions_value, 1)[1].cpu().numpy()
            action = action[0]
                    
        return action
    
    def update_params(self, batch):
        self.epoch += 1
        # Get tensors from the batch
        batch_size = len(batch.state)
        
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).to(device)
        done_batch = torch.tensor(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)
        
        # Get the actions and the state values to compute the targets

        
        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_values, min_value, max_value)
    
        # Update the critic network
        # Compute the target
        
        q_next = self.critic_target(next_state_batch).detach()
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        q_target = reward_batch + (1.0 - done_batch) * self.gamma *\
                   q_next.max(1)[0].view(batch_size, 1)
        
        q_eval = self.critic(state_batch).gather(1, action_batch.unsqueeze(1))
        value_loss = F.mse_loss(q_eval, q_target)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward(retain_graph = False)
        self.critic_optimizer.step()
    
        def reject_outliers(data, m = 3):
            return np.asarray(data)[list(abs(data - np.mean(data)) < 3 * np.std(data))]
        
        # record losses to check early stopping.
        if self.epoch >= self.check_epoch:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = value_loss.item()
            self.position = (self.position + 1) % self.capacity
            
            if len(self.memory) == self.capacity:
                if max(reject_outliers(self.memory)) - min(reject_outliers(self.memory)) <= self.gap_stop:
                    self.value_stop = True
                    print('Online network has early stopped now.')
        
        # soft update target network
        # soft_update(self.critic_target, self.critic, self.tau)
        # hard update target network every 1000 optmized
        if (self.epoch % 1000 == 0) & (self.epoch <= 95000):
            hard_update(self.critic_target, self.critic)
        # for one instance
        # if (self.epoch % 1000 == 0) & (self.epoch <= 3000):
        #     hard_update(self.critic_target, self.critic)
        
        print(round(value_loss.item(),6))
    
        self.value_losses.append(round(value_loss.cpu().detach().item(), 6))
