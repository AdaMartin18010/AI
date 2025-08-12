# 14. 优化理论

## 14.1 优化理论概述

优化理论研究在约束条件下使目标函数达到极值的方法。

- **线性规划**
- **非线性规划**
- **凸优化**

## 14.2 线性规划

- 标准形式：$
\begin{align}
\text{min}\quad & c^T x \\
\text{s.t.}\quad & Ax \leq b
\end{align}$
- 单纯形法

## 14.3 非线性规划

- 拉格朗日乘子法
- KKT条件

## 14.4 凸优化

- 凸集、凸函数
- 最优性条件

## 14.5 组合优化

- 整数规划
- 图优化

## 14.6 编程实现

### 14.6.1 Python实现（线性规划）

```python
from scipy.optimize import linprog
c = [1, 2]
A = [[-1, 1], [3, 2]]
b = [1, 12]
res = linprog(c, A_ub=A, b_ub=b)
print(res.x)
```

### 14.6.2 Haskell实现（梯度下降）

```haskell
-- 梯度下降法
gradientDescent :: (Double -> Double) -> Double -> Double -> Double
gradientDescent f x0 alpha =
    let grad x = (f (x + 1e-6) - f x) / 1e-6
        go x | abs (grad x) < 1e-6 = x
             | otherwise = go (x - alpha * grad x)
    in go x0
```

## 14.7 学习资源

- **《最优化理论与方法》**（施加兵）
- **Khan Academy**：优化理论
- **Coursera**：凸优化

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：信息论](15-InformationTheory.md)
- [下一章：控制理论](16-ControlTheory.md)
- [返回数学主目录](../README.md)
