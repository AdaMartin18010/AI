# 16. 控制理论

## 16.1 控制理论概述

控制理论研究动态系统的建模、分析与调节方法，是自动化、工程、AI等领域的基础。

- **开环控制**
- **闭环控制（反馈控制）**

## 16.2 动态系统建模

- 微分方程模型
- 状态空间模型

## 16.3 线性系统理论

- 线性时不变系统（LTI）
- 可控性与可观性

### 16.3.1 主要定理

- 卡尔曼可控性判据
- 卡尔曼可观性判据

## 16.4 频域分析

- 拉普拉斯变换
- 传递函数
- 波特图、奈奎斯特图

## 16.5 现代控制方法

- 最优控制（LQR）
- 鲁棒控制（H∞）
- 自适应控制

## 16.6 非线性与智能控制

- 非线性系统
- 模糊控制、神经网络控制

## 16.7 编程实现

### 16.7.1 Python实现（LQR）

```python
import numpy as np
from scipy.linalg import solve_continuous_are
# LQR控制器设计
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.array([[1]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("LQR增益:", K)
```

### 16.7.2 Haskell实现（离散系统仿真）

```haskell
-- 离散时间一阶系统仿真
discreteSystem :: Double -> Double -> [Double] -> [Double]
discreteSystem a b us = scanl (\x u -> a*x + b*u) 0 us
```

## 16.8 学习资源

- **《自动控制原理》**（胡寿松）
- **Khan Academy**：控制系统基础
- **MIT OpenCourseWare**：控制理论

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：优化理论](14-OptimizationTheory.md)
- [下一章：图论](17-GraphTheory.md)
- [返回数学主目录](../README.md)
