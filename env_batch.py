# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:45:02 2020

@author: Allen
"""

# TODO
# 1.Reward rollout

import copy
import random
import numpy as np
from utils import _active_job, _nanargmin, _nanargmax, _max_min
import math
import os

class pmsp_env():
    def __init__(self,ins = "3_5", dim = 0, a_dim = 5):
        f = open(os.getcwd() + "/instance/"+ins+".txt", 'r', encoding = 'utf-8-sig')
        x = f.readlines()
        f.close()
        y = [i.split() for i in x]
        z = [list(map(int,j)) for j in y][0]
        if len(z) == dim + 1:
            self.optimal = z[-1]
            del z[-1]
        elif len(z) > dim + 1:
            self.optimal = z[-3]
            self.SPT = z[-2]
            self.MLB = z[-1]
            del z[-1]
            del z[-1]
            del z[-1]
        else:
            print('error')
        
        m_array = np.array(z)
        
        ## define the variable
        self.a_dim = a_dim
        self.m_array = m_array
        self.machine_num = m_array[0]
        self.machine = [0]*m_array[0].astype(int)
        self.machine_time = [0]*m_array[0].astype(int)
        self.job_status = [0]*(m_array.shape[0]-1) # if job has not assigned, is 0
        self.job_num = len(self.job_status)
        self.dim = 1 + (self.job_num + self.machine_num) *2
        self.processing_time = m_array[1:]
        self.makespan = 0
        self.max_makespan = 0
        self.min_makespan = 0
        self.done = False
        self.machine_ind = 0
        self.total_job_time = sum(self.processing_time)
        self.worst = sum(self.processing_time)
        self.best = int(sum(self.processing_time) / self.machine_num)
        self.subbest = int(sum(self.processing_time) / (self.machine_num - 1 ))
        self.load = 0
        self.load_ = 0
        
        ## define the record of job and time
        self.job_dic = {}
        self.proceesing_time_dic = {}
        self.makespan_dic = {}
        
        for i in range(self.machine_num):
            self.job_dic[i] = [] # prcoess job each machine
            self.proceesing_time_dic[i] = [] # prcoessing time each machine
            self.makespan_dic[i] = 0 # makesapn each machine
        
        self.state = []
        self.state.append([0] + self.machine + self.machine_time + self.job_status + self.processing_time.tolist())
        ## t ~ t-4
        for i in range(4):
            state = [0] * self.dim
            self.state.append(state)
        self.state = np.array(self.state)
        
        self.state_ = []
        
        self.state_view = []
        self.state_view.append(['machine_status'] + self.machine)
        self.state_view.append(['machine_proceesing_time'] + self.machine_time) 
        self.state_view.append(['job_status'] + self.job_status)
        self.state_view.append(['processing_time'] + self.processing_time.tolist())
    
    def step(self, action):        
        act = action
        reward = [0] * self.machine_num
        self.state = self.state_ if len(self.state_) >0 else self.state
        
        if np.nansum(_active_job(self.job_status, self.processing_time)[0]) == 0 :
            print('done')
            self.done = True

        else:
            ## calculate each machine of longest makepsan
            for i in self.proceesing_time_dic:
                self.makespan_dic[i] = sum(self.proceesing_time_dic[i])
                
            ## sort sequence of machine selection 
            # machine_seq = _sort_by_value(self.makespan_dic)
            
            ## Take action, sequence based on lodaing balance
            chosen_job = 9999
            m = (self.state[0][0] + 1) % self.machine_num

            self.max_makespan = _max_min(self.makespan_dic)
            self.min_makespan = _max_min(self.makespan_dic, 'min')
            if self.max_makespan[0] == 0:
                self.load = 0
            else:
                self.load = (sum([x / self.max_makespan[0] for x in self.machine_time]) - 1) / (self.machine_num-1)

            active_job, active_ind  = _active_job(self.job_status, self.processing_time)
            # ===================================================
            # Take action
            # SPT
            if act == 0:
                self.machine[m] += 1
                chosen_job = _nanargmin(active_job)

            # LPT
            elif act == 1:
                self.machine[m] += 1
                chosen_job = _nanargmax(active_job)
                
            # Load Balancing
            elif act == 2:
                self.machine[m] += 1
                if self.max_makespan[0] != 0:
                    cand = [np.nan] * self.job_num
                    for count, i in enumerate(active_job):
                        if np.isnan(i) == False:
                            machine_time = copy.deepcopy(self.machine_time)
                            makespan_dic = copy.deepcopy(self.makespan_dic)
                            machine_time[m] += i
                            makespan_dic[m] += i
                            max_makespan = _max_min(makespan_dic)
                            cand[count] = (sum([x / max_makespan[0] for x in machine_time]) - 1) / (self.machine_num-1)
                        else:
                            cand[count] = -1
                    chosen_job = _nanargmax(cand)
                else:
                    chosen_job = random.choice(range(self.job_num))
            
            # PIRO process in random order
            
            elif act == 3:
                self.machine[m] += 1
                value = np.nan
                active = copy.deepcopy(active_job)
                if len(active) - sum(math.isnan(x) for x in active) >=3:
                    active[_nanargmax(active_job)] = np.nan
                    active[_nanargmin(active_job)] = np.nan
                while(np.isnan(value)):
                    job = random.sample(list(enumerate(active)), 1)[0]
                    value = job[1]
                    chosen_job = job[0]
            
            # Idle 0
            elif act == 4:
                pass
            # ===================================================
            if chosen_job != 9999:
                self.job_status[chosen_job] = m+1
                self.job_dic[m].append(chosen_job)
                self.proceesing_time_dic[m].append(self.processing_time[chosen_job])
                self.makespan_dic[m] += self.processing_time[chosen_job]
                self.machine_time[m] += self.processing_time[chosen_job]
            self.max_makespan = _max_min(self.makespan_dic)
            
            self.makespan = _max_min(self.makespan_dic)
            
            self.state_ = []
            self.state_.append([m] + self.machine + self.machine_time + self.job_status + self.processing_time.tolist())
            ## t ~ t-4
            for t in range(4):
                self.state_.append(self.state[t])
            self.state_ = np.array(self.state_)
            
            self.state_view = []
            self.state_view.append(['machine_status'] + self.machine)
            self.state_view.append(['machine_proceesing_time'] + self.machine_time) 
            self.state_view.append(['job_status'] + self.job_status)
            self.state_view.append(['processing_time'] + self.processing_time.tolist())
            if np.nansum(_active_job(self.job_status, self.processing_time)[0]) == 0 :
                self.done = True
            # ===================================================
            ## reward calcualted by whthere improve LBRM
            
            # if sum([self.state[x][0] for x in range(len(self.state))]) == 0:
            #     reward = 0
            # else:
            #     self.load_ = (sum([x / self.max_makespan[0] for x in self.machine_time]) - 1) / (self.machine_num-1)
            #     reward = 1. if (self.load_ - self.load) >= 0. else -1
                
            ## reward calculdated by LBRM
            
            if sum([self.state[x][0] for x in range(len(self.state))]) == 0:
                reward = 0
            else:
                self.load_ = (sum([x / self.max_makespan[0] for x in self.machine_time]) - 1) / (self.machine_num-1)
                reward = self.load_
            
        return self.state_ , reward , self.done
# =============================================================================
# 
# =============================================================================