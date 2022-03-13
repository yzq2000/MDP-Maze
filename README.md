# MDP-Maze

## 操作介绍

### 运行Q-Learning算法

- 点击Q-Learning按钮，红色方框开始100次的Q-Learning
- 控制台会输出每次的步数和结果。
- 绿色数字为QTable
  - e.g. (1,1)位置右侧的数字表示从state(1,1)采取action向右走对应的Q值。

### 运行policy iteration算法

- 点击policy evaluation按钮，计算当前策略方案下的价值函数
- 点击policy update按钮，根据当前价值函数优化策略方案
- 交替点击这两个按钮，直至收敛，即可求得最优策略方案和最优价值函数
- 蓝色数字为价值函数，箭头表示策略方案

### 运行value iteration算法

- 点击value iteration按钮，开始迭代价值函数直至收敛，并由最优价值函数确定最优策略方案
- 蓝色数字为价值函数，箭头表示策略方案

## 相关资料

Q-Learning
- 视频学习 https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-q-learning/
- 用于参考的迷宫环境源码 https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/2_Q_Learning_maze

MDP
- 视频学习 https://www.bilibili.com/video/BV1g7411m7Ms/?spm_id_from=333.788.recommend_more_video.-1
- 相关课件见附件 MDP.pdf
