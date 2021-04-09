# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``env_batch.py`` - ?���?code | <strong> State Reward Action </strong>

    *�o���ɮרS���n�ק諸�ѼơA�D�n��State���ͦ��B�����u�@(job)�H�έp��reward�m�b�o���ɮ׹B�@

``net_batch.py`` - DQN����code | <strong> Agent </strong>

    *���ɮצ�actor��critic���Function�A�D�n�ϥ�Critic�������A�N��online�Ptarget network��͡A�ثe�Hconvolutional layer�غc

``DQN.py`` - DQN optimize code | <strong> Agent Optimize </strong>

    ���ɮץD�n�O�B�@optimized���Ʊ��Aagent�C�P�@��instance���ʧ���(�����������u�@)�|����optimize�O�b���ɮפ��i��A�n�վ㪺�ѼƬ�``early stopping``�����ѼơA�H��134�檺step���ơA�]�N�O�h��epoch�n��s�@��target

``util.py`` - �@��function code

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
