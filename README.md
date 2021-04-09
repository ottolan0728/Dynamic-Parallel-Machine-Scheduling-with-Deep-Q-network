# Parallel-Machine-Scheduling-with-Deep-Q-network-Algorithm

## Code

``env_batch.py`` - ?挵澧?code | <strong> State Reward Action </strong>

    *硂郎⊿Τ璶э把计璶рStateネΘ(job)の璸衡reward恗硂郎笲

``net_batch.py`` - DQN呼隔code | <strong> Agent </strong>

    *郎Τactor蛤criticㄢFunction璶ㄏノCritic呼隔online籔target networkㄢ幫ヘ玡convolutional layer篶

``DQN.py`` - DQN optimize code | <strong> Agent Optimize </strong>

    郎璶琌笲optimizedㄆ薄agent–籔instanceが笆Ч(Ч场)穦磅︽optimize琌郎い秈︽璶秸俱把计``early stopping``ず把计の134︽stepΩ计碞琌ぶepoch璶穝Ωtarget

``util.py`` - ㄇfunction code

    ㄤcode穦ノFunction碭恗糶柑钩琌env_batch璸衡êㄇjob琌active碞郎い级糶

``plot.py`` - 礶挡狦瓜code

    main_batch禲Ч挡狦盢穦パplotfunction礶Θ瓜瞷┮璶э瓜礶codeэ

``main_single`` - 禲砆1instance optimize

    籔main_batch碭癡絤顶琿倒ぉinstanceぃ

``main_batch`` - 禲砆instance optimize

    磅︽だcodemain郎璶把计琌郎い砞﹚62︽эscale47︽###Parameter setting琌秸俱把计203︽琌ノㄓ礶笆だガ瓜code惠璶沮epoch暗э瞷挡狦穦ゑ耕芠

``generate.py`` - 玻ネinstancecode

    code玻ネぃscaleinstancescode


## Folder

``Cplex`` - 禲MIP籔GAcode柑GA.py磅︽

``parameter`` - main_single籔main_batch呼隔把计纗竚

``figure`` - main_single籔main_batch挡狦瓜纗竚

``instance`` - 纗instance戈Ж

``optimal`` - ノOR-tools禲optimal codeノCplex碞

``good_parameter`` - 阶ゅ龟喷挡狦瓜把计籔挡狦だぃsize
