# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``"env_batch.py`` - The code of DQN environment | <strong> State Reward Action </strong>

    *There are no parameters to be modified in this file. The generation of states, job assignments, and calculation of rewards are all included in the operation of this file.

``net_batch.py`` - The code of DQN network | <strong> Agent </strong>

    *There are two functions in this file, including Actor network and Critic network. It mainly uses Critic network, which means that Online and Target networks are currently constructed with convolutional layers.

``DQN.py`` - The code of DQN optimization | <strong> Agent Optimization </strong>

    This file is to optimize DQN agents. After the agent interacts with an instance (finished assigning all jobs), the optimization of the agent is executed in this file. The parameters to be adjusted are the parameters in ``early stopping'', and 134 The number of steps in the row, that is, how many epochs to update the target

``util.py`` - The code of other functions

    ¨ä¥Lcode·|¥Î¨ìªºFunction´X¥Gm¼g¦b¸Ì­±¡A¹³¬Oenv_batch­pºâ¨º¨Çjob¬Oactiveªº´N¦b¦¹ÀÉ®×¤¤¼¶¼g¡C

``plot.py`` - µeµ²ªGªº¹Ïcode

    main_batch¶]§¹ªºµ²ªG±N·|¥Ñplotªºfunctionµe¦¨¹Ï¨Ã§e²{¡A©Ò¥H­n§ó§ï¹Ïªºµe¦b¦¹code­×§ï

``main_single`` - ¶]¥u³Q1­Óinstance optimize

    »Pmain_batch´X¥G¬Û¦P¡A¥u¦b°V½m¶¥¬qµ¹¤©ªºinstance¤£¦P

``main_batch`` - ¶]³Q¦h­Óinstance optimize

    ¦¹¬°°õ¦æ¦¹¤ÀcodeªºmainÀÉ¡A¥D­n°Ñ¼Æ¤]¬O¦b¦¹ÀÉ®×¤¤³]©w¡A62¦æ§ó§ïscaleªº¤j¤p¡A47¦æ###Parameter setting¤]¬O¥i¥H½Õ¾ãªº°Ñ¼Æ¡C¥t¥~203¦æ¬O¥Î¨Óµe°Ê§@¤À¥¬¹Ïªºcode¡A»İ­n®Ú¾Úepoch¤j¤p°µ­×§ï¡A§e²{ªºµ²ªG¤~·|¤ñ¸ûª½Æ[¡C

``generate.py`` - ²£¥Íinstanceªºcode

    ¦¹¥÷code¬°²£¥Í¤£¦Pscale¤j¤pªºinstancesªºcode


## Folder

``Cplex`` - ¶]MIP»PGAªºcode¡A¸Ì­±ªºGA.py°õ¦æ

``parameter`` - main_single»Pmain_batchºô¸ô°Ñ¼ÆÀx¦s¦ì¸m

``figure`` - main_single»Pmain_batchµ²ªG¹ÏÀx¦s¦ì¸m

``instance`` - Àx¦sinstanceªº¸ê®Æ§¨

``optimal`` - ¥ÎOR-tools¶]optimal ªºcode¡A¦ı¥ÎCplex´N¦n

``good_parameter`` - ½×¤å¹êÅçµ²ªGªº¹Ï¡B°Ñ¼Æ»Pµ²ªG¡A¤À¤£¦Psize
