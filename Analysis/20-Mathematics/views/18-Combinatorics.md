# 18. 组合数学

## 18.1 组合数学概述

组合数学研究有限或可数离散结构的计数、构造与分析方法。

- **计数原理**：加法原理、乘法原理
- **排列与组合**

## 18.2 计数原理

- 加法原理、乘法原理
- 容斥原理

### 18.2.1 典型公式

- $C_n^k = \frac{n!}{k!(n-k)!}$
- $P_n^k = \frac{n!}{(n-k)!}$

## 18.3 递推与生成函数

- 递推关系
- 生成函数

## 18.4 组合设计与构造

- 拉丁方、区组设计
- 图的着色

## 18.5 组合优化与极值

- 匹配、覆盖、独立集
- 极值问题

## 18.6 编程实现

### 18.6.1 Python实现（组合数与生成函数）

```python
import math
# 组合数
print(math.comb(5, 2))
# 生成函数示例
from sympy import symbols, expand
x = symbols('x')
print(expand((1 + x)**5))
```

### 18.6.2 Haskell实现（递推关系）

```haskell
-- 斐波那契数列递推
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)
```

## 18.7 学习资源

- **《组合数学》**（刘培德）
- **Khan Academy**：组合数学
- **Coursera**：组合学与离散数学

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：数理逻辑](11-ProbabilityTheory.md)
- [下一章：图论](17-GraphTheory.md)
- [返回数学主目录](../README.md)
