# 8. 变分法

## 8.1 变分法概述

变分法研究函数极值问题，是最优控制、物理学、泛函分析等领域的重要工具。

- **泛函极值**
- **欧拉-拉格朗日方程**

## 8.2 基本问题

- 求使泛函$J[y]=\int_a^b F(x,y,y')dx$取极值的函数$y(x)$

## 8.3 欧拉-拉格朗日方程

$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0$

## 8.4 变分法的推广

- 多元变分法
- 有约束变分法
- 最优控制与庞特里亚金极大值原理

## 8.5 典型应用

- 最短路径问题
- 悬链线问题
- 物理中的最小作用量原理

## 8.6 编程实现

### 8.6.1 Python实现（欧拉-拉格朗日方程数值解）

```python
import numpy as np
from scipy.integrate import solve_bvp
# 示例：最短路径问题
# y'' = 0, y(0)=0, y(1)=1

def fun(x, y):
    return np.vstack((y[1], np.zeros_like(x)))

def bc(ya, yb):
    return np.array([ya[0], yb[0]-1])

x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
sol = solve_bvp(fun, bc, x, y)
print(sol.sol(x)[0])
```

### 8.6.2 Haskell实现（欧拉-拉格朗日方程结构）

```haskell
-- 欧拉-拉格朗日方程结构定义
data EulerLagrange = EL { f :: Double -> Double -> Double -> Double }
-- 具体数值解略
```

## 8.7 学习资源

- **《变分法与最优控制》**（李承治）
- **Khan Academy**：变分法
- **Coursera**：最优控制与变分法

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：泛函分析](09-FunctionalAnalysis.md)
- [下一章：调和分析](10-HarmonicAnalysis.md)
- [返回数学主目录](../README.md)
