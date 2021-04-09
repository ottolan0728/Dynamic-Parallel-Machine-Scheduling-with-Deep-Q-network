# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:16:46 2020

@author: Allen
"""


from torch import nn
import torch
import numpy as np
import random
from collections import namedtuple

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    if (len(np_array.shape) != 4) & (len(np_array) > 1):
        np_array = np.reshape(np_array, (1,1,np_array.shape[0], np_array.shape[1]))
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.1)

def record(global_ep_r, ep_r):
    discounting_rate = 0.05
    if global_ep_r == 0.:
        global_ep_r = ep_r
    else:
        global_ep_r = global_ep_r * (1-discounting_rate) + ep_r * discounting_rate
    return global_ep_r

def _active_job(list1, list2):
    output = []
    output2 = []
    for i, j in zip(list1, list2):
        if i == 0:
            output.append(j)
            output2.append(1)
        else:
            output.append(np.nan)
            output2.append(np.nan)
    return output, output2

def _nanargmin(arr):
    try:
        return np.nanargmin(arr)
    except ValueError:
        return np.nan

def _nanargmax(arr):
    try:
        return np.nanargmax(arr)
    except ValueError:
        return np.nan
    
def _max_min(dic, max_min = 'max'):
    if max_min == 'max':
        Max = dic[max(dic, key = dic.get)]
        max_ind = [i for i, j in dic.items() if j == Max]
        output = [dic[i] for i in max_ind]
    else:
        Min = dic[min(dic, key = dic.get)]
        min_ind = [i for i, j in dic.items() if j == Min]
    
        # then choose the candidate machine to select job by mini makespan with machine.
        # If have two candidate, we will random choose one candidate to assign.
        if len(min_ind) > 1 :
            min_ind = [random.choice(min_ind)]
        
        output = [dic[i] for i in min_ind]
    return output

# sort sequence of machine selection 
def _sort_by_value(d): 
    items = d.items() 
    backitems = [[v[1],v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0,len(backitems))] 


# calculate reward
def cal_reward_load(time, test):
    load = (sum([x / max(time) for x in time]) -1) / (len(time) -1)
    if test :
        rate = 0.98
        reward = 10. if load >= rate  else 0.
    else:
        rate = [0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9]
        reward = 10. if load >= rate[0] else 9. if load >= rate[1]\
                 else 8. if load >=rate[2] else 7. if load >=rate[3]\
                 else 6. if load >=rate[4] else 5. if load >=rate[5]\
                 else 4. if load >=rate[6] else 3. if load >=rate[7]\
                 else 1. if load >=rate[8] else 0.
    return float(reward)

def cal_reward_optimal(makespan, epoch, optimal, test):
    if test :
        rate = 1.05
    else:
        rate = 1.25 if epoch < 400 else 1.2 if epoch < 600 else 1.15\
                   if epoch < 800 else 1.10 if epoch <1000 else 1.05
    reward = 1. if makespan[0] <= round(rate * optimal, 0) else -1.
    return float(reward)

def cal_reward_Min(makespan, epoch, Min, test):
    if test :
        rate = 1.05
    else:
        rate = 1.25 if epoch < 400 else 1.2 if epoch < 600 else 1.15\
                   if epoch < 800 else 1.10 if epoch <1000 else 1.05
    
    reward = 1. if makespan <= int(rate * Min) else -1.
    return float(reward)

def record_result(makespan_gap, Testing_time, training_time, stop_epoch,Test_optimal_num):
    from statistics import stdev 
    result1 = 'Gap of proposed method with optimal is %.4f, std is %.4f ' %\
          (((sum(makespan_gap[0]) / len(makespan_gap[0]))), (stdev(makespan_gap[0])))
    result2 ='Gap of SPT with optimal is %.4f, std is %.4f  ' %\
          (((sum(makespan_gap[1]) / len(makespan_gap[0]))), (stdev(makespan_gap[1])))
    result3 ='Gap of Load balancing with optimal is %.4f, std is %.4f  ' %\
          (((sum(makespan_gap[2]) / len(makespan_gap[0]))), (stdev(makespan_gap[2])))
    
    result4 = 'Average instance cost %.2f seconds' % ((Testing_time) /10)
    result5 = 'Total training instance cost %.2f minutes' % ((training_time))
    result7 = 'Got optimal solutions: %.0f' % (Test_optimal_num)
    try:
        result6 ='Stop in %.0f th epoch' % (stop_epoch)
    except NameError:
        result6 = 'This experinmemt does not early stop'
    print(result7 + '\n' + result1 + '\n' + result2 + '\n' + result3 + '\n' +\
          result4 + '\n' + result5 + '\n' + result6 + '\n')
    result_list = [result7, result1, result2, result3, result4, result5, result6]

    return result_list

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
