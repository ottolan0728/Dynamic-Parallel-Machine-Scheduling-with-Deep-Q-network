# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``env_batch.py`` - The code of DQN environment | <strong> State Reward Action </strong>

    *There are no parameters to be modified in this file. The generation of states, job assignments, and calculation of rewards are all included in the operation of this file.

``net_batch.py`` - DQN網路code | <strong> Agent </strong>

    *此檔案有actor跟critic兩個Function，主要使用Critic的網路，代表online與target network兩，目前以convolutional layer建構

``DQN.py`` - DQN optimize code | <strong> Agent Optimize </strong>

    此檔案主要是運作optimized的事情，agent每與一個instance互動完後(指派完全部工作)會執行optimize是在此檔案中進行，要調整的參數為``early stopping``內的參數，以及134行的step次數，也就是多少epoch要更新一次target

``util.py`` - 一些function code

    其他code會用到的Function幾乎寫在裡面，像是env_batch計算那些job是active的就在此檔案中撰寫。

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
