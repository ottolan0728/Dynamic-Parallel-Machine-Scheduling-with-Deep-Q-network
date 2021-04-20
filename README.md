# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``"env_batch.py`` - The code of DQN environment | <strong> State Reward Action </strong>

    *There are no parameters to be modified in this file. The generation of states, job assignments, and calculation of rewards are all included in the operation of this file.

``net_batch.py`` - The code of DQN network | <strong> Agent </strong>

    *There are two functions in this file, including Actor network and Critic network. It mainly uses Critic network, which means that Online and Target networks are currently constructed with convolutional layers.

``DQN.py`` - The code of DQN optimization | <strong> Agent Optimization </strong>

    This file is to optimize DQN agents. After the agent interacts with an instance (finished assigning all jobs), the optimization of the agent is executed in this file. The parameters to be adjusted are the parameters in ``early stopping'', and 134 The number of steps in the row, that is, how many epochs to update the target

``util.py`` - The code of other functions

    其他code會用到的Function幾乎�m寫在裡面，像是env_batch計算那些job是active的就在此檔案中撰寫。

``plot.py`` - 畫結果的圖code

    main_batch跑完的結果將會由plot的function畫成圖並呈現，所以要更改圖的畫在此code修改

``main_single`` - 跑只被1個instance optimize

    與main_batch幾乎相同，只在訓練階段給予的instance不同

``main_batch`` - 跑被多個instance optimize

    此為執行此分code的main檔，主要參數也是在此檔案中設定，62行更改scale的大小，47行###Parameter setting也是可以調整的參數。另外203行是用來畫動作分布圖的code，需要根據epoch大小做修改，呈現的結果才會比較直觀。

``generate.py`` - 產生instance的code

    此份code為產生不同scale大小的instances的code


## Folder

``Cplex`` - 跑MIP與GA的code，裡面的GA.py執行

``parameter`` - main_single與main_batch網路參數儲存位置

``figure`` - main_single與main_batch結果圖儲存位置

``instance`` - 儲存instance的資料夾

``optimal`` - 用OR-tools跑optimal 的code，但用Cplex就好

``good_parameter`` - 論文實驗結果的圖、參數與結果，分不同size
