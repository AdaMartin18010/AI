# 15. 信息论

## 15.1 信息论概述

信息论研究信息的度量、传输、编码与压缩等基本规律。

- **信息量**：度量不确定性的减少
- **熵**：信息的不确定性度量

## 15.2 香农熵

$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$

## 15.3 互信息与条件熵

- **互信息**：$I(X;Y) = H(X) + H(Y) - H(X,Y)$
- **条件熵**：$H(Y|X) = H(X,Y) - H(X)$

## 15.4 相对熵与KL散度

$D_{KL}(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$

## 15.5 信道容量

$C = \max_{p(x)} I(X;Y)$

## 15.6 典型编码理论

- 哈夫曼编码
- 香农-范诺编码
- 纠错码

## 15.7 编程实现

### 15.7.1 Python实现

```python
import numpy as np
def shannon_entropy(p):
    return -np.sum(p * np.log2(p))
p = np.array([0.5, 0.5])
print(f"香农熵: {shannon_entropy(p)}")
```

### 15.7.2 Haskell实现

```haskell
shannonEntropy :: [Double] -> Double
shannonEntropy ps = negate $ sum [p * logBase 2 p | p <- ps, p > 0]
```

## 15.8 学习资源

- **《信息论基础》**（Cover & Thomas）
- **Khan Academy**：信息论
- **Coursera**：信息论与编码

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：控制理论](16-ControlTheory.md)
- [下一章：优化理论](14-OptimizationTheory.md)
- [返回数学主目录](../README.md)
