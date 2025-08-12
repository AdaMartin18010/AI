# 13. 数值分析

## 13.1 数值分析概述

数值分析研究用计算机近似求解数学问题的方法与理论，是科学计算、工程仿真等领域的基础。

- **误差分析**
- **数值稳定性**

## 13.2 插值与逼近

- 拉格朗日插值、多项式插值
- 最小二乘逼近

## 13.3 数值积分与微分

- 牛顿-柯特斯公式
- 龙贝格积分、辛普森法
- 差分法

## 13.4 线性方程组的数值解法

- 高斯消元法
- LU分解、QR分解
- 共轭梯度法

## 13.5 非线性方程与优化

- 牛顿法、割线法
- 最速下降法

## 13.6 常微分方程数值解

- 欧拉法、龙格-库塔法
- 刚性方程与自适应步长

## 13.7 编程实现

### 13.7.1 Python实现（插值与线性方程组）

```python
import numpy as np
from scipy.interpolate import lagrange
# 拉格朗日插值
x = np.array([0, 1, 2])
y = np.array([1, 3, 2])
poly = lagrange(x, y)
print(poly(1.5))

# 高斯消元法
A = np.array([[2,1],[1,3]])
b = np.array([8,13])
x = np.linalg.solve(A, b)
print(x)
```

### 13.7.2 Haskell实现（欧拉法）

```haskell
-- 欧拉法求常微分方程数值解
euler :: (Double -> Double -> Double) -> Double -> Double -> Double -> Int -> [(Double, Double)]
euler f x0 y0 h n = take (n+1) $ iterate step (x0, y0)
  where step (x, y) = (x+h, y + h * f x y)
```

## 13.8 学习资源

- **《数值分析》**（徐树方）
- **Khan Academy**：数值方法
- **Coursera**：科学计算与数值分析

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：数论](19-NumberTheory.md)
- [下一章：泛函分析](09-FunctionalAnalysis.md)
- [返回数学主目录](../README.md)
