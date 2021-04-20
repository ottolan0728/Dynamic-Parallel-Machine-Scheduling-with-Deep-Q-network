# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``env_batch.py`` - The code of DQN environment | <strong> State Reward Action </strong>

    *There are no parameters to be modified in this file. The generation of states, job assignments, and calculation of rewards are all included in the operation of this file.

``net_batch.py`` - The code of DQN network | <strong> Agent </strong>

    *There are two functions in this file, including Actor network and Critic network. It mainly uses Critic network, which means that Online and Target networks are currently constructed with convolutional layers.

``DQN.py`` - The code of DQN optimization | <strong> Agent Optimization </strong>

    This file is to optimize DQN agents. After the agent interacts with an instance (finished assigning all jobs), the optimization of the agent is executed in this file. The parameters that need to be adjusted are the ones in 'early stopping' and the number of steps in line 134, which represents how many epochs to update Target network.

``util.py`` - The code of other functions

    All the functions used by other codes are encoded in this file. For example, 'env_batch' calculates which jobs that are active, and is written in this file.

``plot.py`` - The code for drawing the pictures of the results

    The results of the "main_batch" will be drawn and presented by "plot.py", so if you want to redraw the pictures, please modify this code.

``main_single`` - Optimized by one instance

    Almost the same as "main_batch", only the instances given in the training phase are different.

``main_batch`` - Optimized by multiple instances

    This is the main file for executing this code. The main parameters are also set in this file, such as ###Parameter setting### which are adjustable parameters in line 47, and the ones which change the sizes of the instances in line 62. The other one is the code used to draw the graphs of the action distribution in line 203, which needs to be modified according to the epoch size so that the results will be more intuitive.

``generate.py`` - The code for generating the instances

    This file is the code that generates the instances of different sizes.



## Folder

``Cplex`` - The code for running MIP and GA by executing "GA.py" inside

``parameter`` - The storage location of 'main_single' and 'main_batch's network parameters

``figure`` - The storage location of the figures of 'main_single' and 'main_batch's results

``instance`` - The folder for storing different scale instances

``optimal`` - The code for obtaining optimal solutions with OR-tools

``good_parameter`` - Figures, parameters and results of the experiments of the paper that are distinguished by different instance sizes
