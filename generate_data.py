# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:49:21 2020

@author: Allen
"""
import os
import numpy as np

'''
number of machine = {2,4,6,8,10,20,50}
number of jobs = {20,30,40,50,60,70,90,100,120,150}
processing time = U[10, 80]
'''

def generate_data(machine, job, number):
    job_L = np.random.uniform(10,80, job).astype(int).tolist()
    output = [machine] + job_L
    
    with open(os.path.join(os.getcwd() + "/instance",\
               str(machine) + '_' + str(job) + '_' + str(number) + '.txt'), 'w') as f:
        for item in output:
            f.write('%s ' % item)
        f.close()

num_machines = 6
num_jobs = 40
start = 1
num_instance = 128

for i in range(start, num_instance+1):
    generate_data(num_machines, num_jobs, i)

generate_data(num_machines, num_jobs, 4)
