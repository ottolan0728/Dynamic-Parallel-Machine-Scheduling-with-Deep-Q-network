# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``"env_batch.py`` - The code of DQN environment | <strong> State Reward Action </strong>

    *There are no parameters to be modified in this file. The generation of states, job assignments, and calculation of rewards are all included in the operation of this file.

``net_batch.py`` - The code of DQN network | <strong> Agent </strong>

    *There are two functions in this file, including Actor network and Critic network. It mainly uses Critic network, which means that Online and Target networks are currently constructed with convolutional layers.

``DQN.py`` - The code of DQN optimization | <strong> Agent Optimization </strong>

    This file is to optimize DQN agents. After the agent interacts with an instance (finished assigning all jobs), the optimization of the agent is executed in this file. The parameters to be adjusted are the parameters in ``early stopping'', and 134 The number of steps in the row, that is, how many epochs to update the target

``util.py`` - The code of other functions

    ��Lcode�|�Ψ쪺Function�X�G�m�g�b�̭��A���Oenv_batch�p�⨺��job�Oactive���N�b���ɮפ����g�C

``plot.py`` - �e���G����code

    main_batch�]�������G�N�|��plot��function�e���Ϩçe�{�A�ҥH�n���Ϫ��e�b��code�ק�

``main_single`` - �]�u�Q1��instance optimize

    �Pmain_batch�X�G�ۦP�A�u�b�V�m���q������instance���P

``main_batch`` - �]�Q�h��instance optimize

    �������榹��code��main�ɡA�D�n�ѼƤ]�O�b���ɮפ��]�w�A62����scale���j�p�A47��###Parameter setting�]�O�i�H�վ㪺�ѼơC�t�~203��O�Ψӵe�ʧ@�����Ϫ�code�A�ݭn�ھ�epoch�j�p���ק�A�e�{�����G�~�|������[�C

``generate.py`` - ����instance��code

    ����code�����ͤ��Pscale�j�p��instances��code


## Folder

``Cplex`` - �]MIP�PGA��code�A�̭���GA.py����

``parameter`` - main_single�Pmain_batch�����Ѽ��x�s��m

``figure`` - main_single�Pmain_batch���G���x�s��m

``instance`` - �x�sinstance����Ƨ�

``optimal`` - ��OR-tools�]optimal ��code�A����Cplex�N�n

``good_parameter`` - �פ���絲�G���ϡB�ѼƻP���G�A�����Psize
