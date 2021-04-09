# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:39:24 2020

@author: Allen
"""

# TODO
## 1.While schedule testing instance, schedule sample iteratly. Then choose the best one. 
##   Moreover, one of the instance will not randomly explore at each instance.
## 2.Early stopping. check losses wheter increase for 100 consecutive rounds
## 3.Bacth instance, one hundred epoch an instance. 
# 4.Final state give extra reward to each state.


### import
import random
import configparser
from utils import v_wrap, record, record_result
import torch
from env_batch import pmsp_env
import os
from plot import plot_result, plot_result_multi, plot_result_makespan,\
                 plot_result_multi_line, plot_result_makespan_bar,\
                 plot_result_bar, plot_result_loss
import time
from utils import ReplayMemory, Transition
from DQN import DQN
import copy, pickle
import math

os.chdir(os.getcwd())
# store or load file path setting
ini_path = os.getcwd() +'/figure/version.ini'
pkl_path = os.getcwd() +'/parameter/'
fig_path = os.getcwd() +'/figure/'

Score_list, Load_list = [], []
Reward_index_list, Reward_list = [], []
round_index_list, round_list = [], []
global_reward_list, global_reward = [], 0
action_list, state_list, Reinitialize = [], [], []
optimal_num, action_stat = 0, {}
early_stop = False
action_100 = {}

### Parameter setting

epoch = 100
action = 4
batch_size = 32 # OpenAI 128
explore_rate = 90 # (%)
replay_size = 1e4 # OpenAI 1e6
round_max = 100
test_round = 100 # testing exploration round
check_round = 100 # check early round

### DDPG parameter setting
gamma, tau = 0.99, 1e-3

### Create batch data
machine_num, job_num = 4, 30
dim = 1 + machine_num * 2 + job_num * 2
start, end = 1, 100
size = str(machine_num) + '_' + str(job_num)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQN(gamma, tau, dim, action)

reward_pos, reward_neg = [0.] * 4, [0.] * 4
stop_epoch = epoch * round_max

config = configparser.ConfigParser()
config.read(ini_path)
Serial = config['Version'][size]

# ============================================================================
### Training
count, num_idle, num_idle_instance, round_ins, opti_num = 0,0,0,0,0
tStart = time.time()

# Initialize replay memory
#memory = ReplayMemory(int(replay_size))
replay_memory = ReplayMemory(int(replay_size))

while((round_ins <= 100) and (early_stop != True)):
    round_ins +=1
    ins = round_ins
    _tStart = time.time()   
    print("=======================================")
    print('Instance', ins, 'start. And now round is ', round_ins)
    print("=======================================")
    for i in range(epoch):
        if explore_rate > 10:
            explore_rate *= 0.99 if count % 10 == 0 else 1
        else:
            explore_rate = 10
        action_l = []
        j = 0 # represent number of rounds
        Time_L = []
        cumu_reward = 0
        job = pmsp_env(size + '_' + str(ins), dim = job_num +1, a_dim = action)
        s = job.state
        
        while(True):
            j += 1
            a = agent.choose_action(v_wrap(s).to(device), explore_rate)
            action_l.append(a)
    
            s_,r,done =  job.step(a)
    
            if done :
                Score_list.append(round(1-(job.makespan[0] - job.optimal) / job.optimal, 2))
                Load_list.append((sum([x / job.max_makespan[0] for x in job.machine_time]) - 1) / (job.machine_num-1))
                ## push memory, all reward r(t) = final reward r(T) if action is not idle.
                for t in range(len(agent.batch_memory)):
                    replay_memory.push(v_wrap(agent.batch_memory[t][0]),\
                                       agent.batch_memory[t][1],\
                                       float(agent.batch_memory[t][2]),\
                                       v_wrap(agent.batch_memory[t][3]),\
                                       agent.batch_memory[t][4])
                        
                    if agent.batch_memory[t][4] == 1:
                        reward_pos[int(agent.batch_memory[t][1])] += 1
                    else:
                        reward_neg[int(agent.batch_memory[t][1])] += 1
                    
                Time_L.append(job.makespan)
                if job.makespan[0] == job.optimal:
                    optimal_num +=1
                num_idle = 0
                break
            else:
                num_idle_instance, num_idle = 0, 0
                agent.rewards.append(r)
                cumu_reward += r
                agent.batch_memory.append([s, a, done, s_, r])
                s = s_
                
        ## Record result
        state_list.append(s_)
        action_list.append(['index_' + str(count)] + [action_l])
        round_index_list.append(['index_' + str(count)] + [j])
        round_list.append(j)
        Reward_index_list.append(['index_' + str(count)] + [round(cumu_reward, 3)])
        Reward_list.append(round(cumu_reward, 3))
        global_reward = record(global_reward, cumu_reward)
        global_reward_list.append(global_reward)
            
        del agent.batch_memory[:]
        if (len(replay_memory) > batch_size * 20) &\
            (i % 1 == 0):
            transitions = replay_memory.sample(int(batch_size))
            batch = Transition(*zip(*transitions))
            agent.update_params(batch)
    
### check whether early stop after 200 optimized.
    if round_ins > check_round:
        if ((agent.value_stop == True)):
            early_stop = True
            stop_epoch = round_ins * epoch
            print("=======================================")
            print('In epoch %.0f early stop.' % (stop_epoch))
            print("=======================================")
    
    print('Instance', ins, 'end')
    tEnd = time.time()
    print('This instance cost %.2f minutes' % ((tEnd - _tStart) / 60))

print('Total instance cost %.2f minutes' % ((tEnd - tStart) / 60))

training_time =  round((tEnd - tStart) / 60, 2)

# record the information of action
for a in range(action):
    action_stat[a] = [0.] * job_num * 50
for i in range(len(action_list)):
    for count, s in enumerate(action_list[i][1]):
        action_stat[s][count] +=1

for c in reversed(range(len(action_stat[0]))):
    # based on action num
    if (action_stat[0][c] == 0.) & (action_stat[1][c] == 0.) &\
       (action_stat[2][c] == 0.) & (action_stat[3][c] == 0.):
        action_stat[0].pop(c), action_stat[1].pop(c), action_stat[2].pop(c),\
        action_stat[3].pop(c)
    else:
        continue

action_per = copy.deepcopy(action_stat)
total_num_act = []

for i in range(len(action_per[0])):
    num = 0
    for m in range(action):
        num += action_per[m][i]
    total_num_act.append(num)

for i in range(len(action_per)):
    action_per[i] = [count / total_num_act[x] for x, count in enumerate(action_per[i])]

Range = 100 if len(action_list) <= 1000 else 200 if len(action_list) <= 2000 \
        else 300 if len(action_list) <= 3000 else 400 if len(action_list) <= 4000 \
        else 500 if len(action_list) <= 5000 else 600 if len(action_list) <= 6000 \
        else 700 if len(action_list) <= 7000 else 800 if len(action_list) <= 8000 \
        else 900 if len(action_list) <= 9000 else 1000 if len(action_list) <= 11000 \
        else 10000
        
for a in range(action):
    action_100[a] = [] * math.ceil(len(action_list) / Range)
bound = len(action_list) // Range
for i, n in enumerate(range(0, bound * Range, Range)):
    total_num = 0
    this_action = [0] * action
    if i!= bound -1:
        endding = n + Range
    else:
        endding = len(action_list)
    for l in range(n, endding):
        total_num += len(action_list[l][1])
        for count, s in enumerate(action_list[l][1]):
            this_action[s] +=1
    this_action = [x / total_num for x in this_action]
    for a in range(action):
        action_100[a].append(this_action[a])

# record both of loss
value_losses = copy.deepcopy(agent.value_losses)

### training plot

print('Got optimal solutions: %.0f' %(optimal_num))
plot_result(global_reward_list,yname = 'Reward', size = size, group = 'train', ini_path = ini_path, save_path = fig_path)
plot_result(Score_list,yname = 'Makepsan (Cmax) score', size = size, group = 'train', ini_path = ini_path, save_path = fig_path)
plot_result_loss(value_losses,yname = 'value_losses', size = size, group = 'train', ini_path = ini_path, save_path = fig_path)
plot_result_multi(action_per, name = 'Selected percentage', size = size, group = 'train', ini_path = ini_path, save_path = fig_path)
plot_result_multi(action_100, name = 'Selected percentage', size = size, group = 'train',\
                  state = 'number', Range = Range, ini_path = ini_path, save_path = fig_path)
plot_result_multi_line(action_stat, name = 'Selected frequency', size = size, group = 'train', ini_path = ini_path, save_path = fig_path)


torch.save(agent.critic.state_dict(), pkl_path +\
           size + '_' + Serial + '_Critic_net_params.pkl')

# ============================================================================
### Testing
Test_score_list, Test_state_list, Test_load_list = [],[],[]
Test_action_list, Test_action_stat = [], {}
Test_Cmax_list, Test_optimal_list = [] ,[]
Test_SPT_list, Test_MLB_list = [], []
Test_Reward_index_list, Test_Reward_list = [] ,[]
Test_round_index_list, Test_round_list = [] ,[]
Test_global_reward_list, Test_global_reward, Test_optimal_num = [], 0, 0
Testing_time = 0

index = list(range(101, 111))
index.insert(0, 1)
agent = DQN(gamma, tau, dim, action)
agent.critic.load_state_dict(torch.load(pkl_path +\
                                        size + '_' + Serial + '_Critic_net_params.pkl'))

for ins in index:
    print('ins ', ins, ' start')
    tStart = time.time()
    Test_action_l, Test_Time_L = [], []
    j = 0
    best_makespan = 99999.
    job = pmsp_env(size + '_' + str(ins), dim = job_num +1, a_dim = action)
    Test_optimal_list.append(job.optimal)
    for i in range(test_round):
        cumu_reward = 0
        job = pmsp_env(size + '_' + str(ins), dim = job_num +1, a_dim = action)
        s = job.state
        action_l = []
        rounds = 0
        while(True):
            j += 1
            if i != epoch -1:
                a = agent.choose_action(v_wrap(s).to(device), explore_rate, True)
            else:
                a = agent.choose_action(v_wrap(s).to(device), explore_rate, True, explore = False)
            action_l.append(a)
    
            s_,r,done =  job.step(a)
            rounds += 1
    
            if done :
                cumu_reward += r
                if job.makespan[0] < best_makespan:
                    best_makespan = copy.deepcopy(job.makespan[0])
                    Test_Time_L.append(job.makespan[0])
                    best_reward = copy.deepcopy(cumu_reward)
                    best_action = action_l
                break
            elif rounds > job_num * machine_num:
                break
            else:
                agent.rewards.append(r)
                cumu_reward += r
                s = s_
                
                
    if best_makespan == job.optimal:
        Test_optimal_num +=1
    Test_SPT_list.append(job.SPT)
    Test_MLB_list.append(job.MLB)
    Test_score_list.append(round(1-(best_makespan - job.optimal) / job.optimal, 2))
    Test_load_list.append(round((sum([x / best_makespan for x in job.machine_time]) - 1) / (job.machine_num-1),2))
    
    Test_state_list.append(s_)
    Test_action_list.append(['index_' + str(ins)] + [best_action])
    Test_round_index_list.append(['index_' + str(ins)] + [j])
    Test_round_list.append(j)
    Test_Cmax_list.append(best_makespan)
    Test_Reward_index_list.append(['index_' + str(ins)] + [round(best_reward, 3)])
    Test_Reward_list.append(round(best_reward, 3))
    Test_global_reward = record(Test_global_reward, best_reward)
    Test_global_reward_list.append(Test_global_reward)
    tEnd = time.time()
    print('This instance cost %.2f seconds' % ((tEnd - tStart)))
    Testing_time += tEnd - tStart
    
# save result to pickle
result = [Test_optimal_list, Test_Cmax_list, Test_SPT_list, Test_MLB_list]
result_1 = [[result[x][0]] for x in range(len(result))]
result_test = [result[x][1:] for x in range(len(result))]
makespan_gap = []
for L in result[1:]:
    gap = [round((L[count] - result[0][count]) / result[0][count], 4) for count in range(len(L))]
    makespan_gap.append(gap)

file = open(fig_path +\
            size + "_" + Serial + "_result_test.pickle" '.pickle', 'wb')
pickle.dump(result, file)
file.close()

file = open(fig_path +\
            size + "_" + Serial + "_gap_test.pickle" '.pickle', 'wb')
pickle.dump(makespan_gap, file)
file.close()

# record the information of action
Test_action_list.pop(0)

for a in range(action):
    Test_action_stat[a] = [0.] * job_num * 50
for i in range(len(Test_action_list)):
    for count, s in enumerate(Test_action_list[i][1]):
        Test_action_stat[s][count] +=1

for c in reversed(range(len(Test_action_stat[0]))):
    # based on action num
    if (Test_action_stat[0][c] == 0.) & (Test_action_stat[1][c] == 0.) &\
       (Test_action_stat[2][c] == 0.) & (Test_action_stat[3][c] == 0.):
        Test_action_stat[0].pop(c), Test_action_stat[1].pop(c), Test_action_stat[2].pop(c),\
        Test_action_stat[3].pop(c)
    else:
        continue

Test_action_per = copy.deepcopy(Test_action_stat)
total_num_act = []

for i in range(len(Test_action_per[0])):
    num = 0
    for m in range(action):
        num += Test_action_per[m][i]
    total_num_act.append(num)

for i in range(len(action_per)):
    Test_action_per[i] = [count / total_num_act[x] for x, count in enumerate(Test_action_per[i])]

### testing plot
plot_result(Test_Reward_list, yname = 'Reward', xname = 'Instance', size = size, group = 'test', ini_path = ini_path, save_path = fig_path)

plot_result_makespan_bar(result_1, name = 'Makepsan (Cmax)', size = size, ini_path = ini_path, save_path = fig_path, single = True)
plot_result_makespan_bar(result_test, name = 'Makepsan (Cmax)', size = size, ini_path = ini_path, save_path = fig_path)
plot_result_makespan(result_test, name = 'Makepsan (Cmax)', size = size, ini_path = ini_path, save_path = fig_path)

plot_result_bar(Test_score_list,yname = 'Makepsan score', xname = 'Instance', size = size, group = 'test', ini_path = ini_path, save_path = fig_path)
plot_result_bar(Test_load_list,yname = 'Load balancing score',xname = 'Instance', size = size, group = 'test', ini_path = ini_path, save_path = fig_path)
plot_result_multi(Test_action_per, name = 'Selected percentage', size = size, group = 'test', ini_path = ini_path, save_path = fig_path)
plot_result_multi_line(Test_action_stat, name = 'Selected frequency', size = size, group = 'test', ini_path = ini_path, save_path = fig_path)

# print result
result_list = record_result(makespan_gap, Testing_time, training_time,\
                            stop_epoch, Test_optimal_num)
with open(fig_path + size + "_" + Serial + "_result_test.txt", 'wb') as f:
    for item in result_list:
        f.write(('%s\n' % (item)).encode())
file.close()

print('The size is %s, and trained by multi instance' % (size))
# ============================================================================
