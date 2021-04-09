# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:14:24 2020

@author: Allen
"""
import numpy as np
import matplotlib.pyplot as plt
import configparser

def plot_result(data, yname, size, group, ini_path, save_path, xname = 'Epoch'):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    plt.plot(data)
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.ylim(min(data), max(data))
    if group == 'test':
        plt.xticks(np.array(range(10)))
    plt.yticks(np.linspace(min(data), max(data), 5))
    plt.tight_layout()
    plt.savefig(save_path +\
                size + "_" + Serial + "_" + yname + "_" + group +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()
    
def plot_result_loss(data, yname, size, group, ini_path, save_path, xname = 'Epoch', value = True):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    plt.plot(data)
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(yname)
    plt.xlabel(xname)
    if value == False:
        plt.ylim(-1.1, 1.1)
        plt.yticks(np.linspace(-1.1, 1.1, 10))
    else:
        plt.ylim(0, max(data)*1.1)
        plt.yticks(np.linspace(0, max(data)*1.1, 10))
    plt.tight_layout()
    plt.savefig(save_path +\
                size + "_" + Serial + "_" + yname + "_" + group +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()
    
def plot_result_bar(data, yname, size, group, ini_path, save_path, xname = 'Epoch'):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    single_width = 0.7
    plt.rcParams.update({'font.size': 20})
    plt.bar(list(range(len(data))) , data,\
            width = single_width)
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.ylim(min(data) - 0.05, 1)
    if group == 'test':
        plt.xticks(np.array(range(10)))
    plt.yticks(np.linspace(min(data) - 0.05, 1, 5))
    plt.tight_layout()
    plt.savefig(save_path +\
                size + "_" + Serial + "_" + yname + "_" + group +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()
    
def plot_result_makespan(data, name, size, ini_path, save_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    title = ['Optimal', 'DQN', 'SPT', 'Machine load balancing']
    Max = 0
    Min = 999999
    marker = ['v', '^', 'o', (8,2,0)]
    for i in range(len(data)):
        plt.plot(range(len(data[i])), data[i], linewidth = 2, color = 'C' + str(i),\
                 label = title[i], marker = marker[i], markersize = 8)
        Max = max(Max, max(data[i]))
        Min = min(Min, min(data[i]))
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(name)
    plt.xlabel('Instance')
    plt.ylim(Min // 5 * 5 - 5, Max // 5 * 5 + 5)
    plt.xticks(np.array(range(0,10)), list(range(1,11)))
    plt.yticks(np.linspace(Min // 5 * 5 - 5, Max // 5 * 5 +5, 6))
    plt.legend(prop={'size': 20},loc='upper center',  bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig(save_path +\
                size + "_" + Serial + "_makespan_test" +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()
    
def plot_result_makespan_bar(data, name, size, ini_path, save_path, single = False):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(18,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    title = ['Optimal', 'DQN', 'SPT', 'Machine load balancing']
    n_bars = len(title)
    total_width, single_width = .8, .75
    bar_width = total_width / n_bars
    Max = 0
    Min = 999999
    for i in range(len(data)):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        plt.bar([x + x_offset for x in list(range(len(data[0])))] , data[i],\
                width = bar_width * single_width, color = 'C' + str(i), label = title[i])
        Max = max(Max, max(data[i]))
        Min = min(Min, min(data[i]))
    
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(name)
    plt.xlabel('Instance')
    if single == False:
        plt.xticks(np.array(range(0,10)), list(range(1,11)))
    else:
        plt.xticks(np.array(range(1)))
    plt.ylim(Min // 5 * 5 - 5, Max // 5 * 5 + 5)
    plt.yticks(np.linspace(Min // 5 * 5 - 5, Max // 5 * 5 +5, 6))
    plt.legend(prop={'size': 20},loc='upper center',  bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=5)
    if single == False:
        plt.savefig(save_path +\
                    size + "_" + Serial + "_makespan_bar_test" +  ".png",\
                    format = "png" ,dpi=300, bbox_inches = "tight")
    else:
        plt.savefig(save_path +\
                    size + "_" + Serial + "_makespan_bar_test_1" +  ".png",\
                    format = "png" ,dpi=300, bbox_inches = "tight")
        
    plt.show()

def plot_result_multi(data, name, size, group, ini_path, save_path, Range = 100, state = 'state'):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    action = ['SPT', 'LPT', 'Machine load balancing', 'MPIRO', 'Idle']
    
    Max = 0
    marker = ['v', '^', 'o', (8,2,0), 's']
    for i in range(len(data)):
        plt.plot(data[i], color = 'C' + str(i), label = action[i],\
                linewidth = 2, marker = marker[i], markersize = 8)
        Max = max(Max, max(data[i]))
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(name)
    if state == 'state':
        plt.xlabel('Time period (t)')
        plt.xticks(list(range(0,len(data[0]) +1, 3)), rotation = 90)
        plt.legend(prop={'size': 20},loc='upper center',  bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, shadow=True, ncol=5)
    else:
        xtick = []
        for i in range(len(data[0])):
            xtick.append(str(i*Range +1 ) + '-' + str((i+1)*Range))
        plt.xlabel('Number of instances')
        plt.xticks(list(range(0, len(data[0]), 1)),\
                   xtick ,rotation = 90)
        group = 'train_number'
        plt.legend(prop={'size': 20},loc='upper center',  bbox_to_anchor=(0.5, -0.45),
                   fancybox=True, shadow=True, ncol=5)
        
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0., 1., 11), ['0%', '10%','20%','30%','40%',\
               '50%','60%','70%','80%','90%','100%',])
    plt.savefig(save_path +\
                size + "_" + Serial + "_action_per_" + group +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()

def plot_result_multi_line(data, name, size, group, ini_path, save_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    Serial = config['Version'][size]
    
    plt.figure(figsize=(10,6))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    action = ['SPT', 'LPT', 'Machine load balancing', 'MPIRO', 'Idle']

    Max = 0
    marker = ['v', '^', 'o', (8,2,0), 's']
    for i in range(len(data)):
        plt.plot(data[i], color = 'C' + str(i), label = action[i],\
                linewidth = 2, marker = marker[i], markersize = 8)
        Max = max(Max, max(data[i]))
    size_ = size.split('_')
    plt.suptitle('Number of jobs = %0.f, number of machines = %0.f' % ((int(size_[1]), int(size[0]))),\
                 color = 'dimgrey', y = 1.03)
    plt.ylabel(name)
    plt.xlabel('Time period (t)')
    plt.xticks(list(range(0,len(data[0]) +1, 3)), rotation = 90)
    if group == 'train':
        plt.yticks(np.linspace(0, Max // 5 * 5 +5, 6))
    else:
        plt.yticks([0,2,4,6,8,10,11])
    plt.legend(prop={'size': 20},loc='upper center',  bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig(save_path +\
                size + "_" + Serial + "_action_" + group +  ".png",\
                format = "png" ,dpi=300, bbox_inches = "tight")
    plt.show()