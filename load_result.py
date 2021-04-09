# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:24:47 2020

@author: Allen
"""

from plot import plot_result, plot_result_multi, plot_result_makespan,\
                 plot_result_multi_line, plot_result_makespan_bar,\
                 plot_result_bar, plot_result_loss

import copy, pickle
import os
folder = ['batch_early', 'batch_noearly', 'single_early']

size = '6_30'
ind = '4'
folder_index = 2

path = os.getcwd() +'/good_parameter'
file = '/' + size + '/' + folder[folder_index] +'/' + size + '_' + ind +'_result_test.pickle.pickle'



ini_path = os.getcwd() +'/Figure/version.ini'
pkl_path = os.getcwd() +'/parameter/'
fig_path = os.getcwd() +'/figure/'


with open(path + file, 'rb') as file:
    result_test = pickle.load(file)

# plot_result_makespan_bar(result_test, name = 'Makepsan (Cmax)', size = size, ini_path = ini_path, save_path = fig_path)
# plot_result_makespan(result_test, name = 'Makepsan (Cmax)', size = size, ini_path = ini_path, save_path = fig_path)

from scipy import stats
from scipy.stats import ttest_ind_from_stats
import numpy as np

def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1)
    std2 = np.std(group2)
    nobs1 = len(group1)
    nobs2 = len(group2)

    modified_std1 = np.sqrt(np.float32(nobs1)/
                    np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/
                    np.float32(nobs2-1)) * std2
    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=nobs1, mean2=mean2, std2=modified_std2, nobs2=nobs2)
    return statistic, pvalue

SPT_tvalue, SPT_pvaule = t_test(result_test[1], result_test[2])
MLB_tvalue, MLB_pvaule = t_test(result_test[1], result_test[3])

print('DQN t-test with SPT t-value = %.3f, p-value = %.3f' % (SPT_tvalue, SPT_pvaule))
print('DQN t-test with MLB t-value = %.3f, p-value = %.3f' % (MLB_tvalue, MLB_pvaule))

