# 9. 泛函分析

## 9.1 泛函分析概述

泛函分析研究无穷维向量空间及其线性算子的理论，是现代分析、量子力学、最优化等领域的基础。

- **巴拿赫空间**
- **希尔伯特空间**

## 9.2 赋范空间与内积空间

- 赋范空间、巴拿赫空间
- 内积空间、希尔伯特空间

## 9.3 有界线性算子

- 有界性、连续性
- 谱理论

## 9.4 重要定理

- 巴拿赫不动点定理
- 哈恩-巴拿赫定理
- 开映射定理、闭图定理

## 9.5 常见应用

- 积分方程
- 偏微分方程
- 最优化理论

## 9.6 编程实现

### 9.6.1 Python实现（幂法求矩阵主特征值）

```python
import numpy as np
A = np.array([[2,1],[1,3]])
x = np.random.rand(2)
for _ in range(10):
    x = A @ x
    x = x / np.linalg.norm(x)
print("主特征值近似：", x @ A @ x / (x @ x))
```

### 9.6.2 Haskell实现（巴拿赫不动点迭代）

```haskell
-- 巴拿赫不动点迭代法
banach :: (Double -> Double) -> Double -> Double -> Int -> Double
banach f x0 tol n = go x0 n
  where go x 0 = x
        go x k = let x1 = f x in if abs (x1 - x) < tol then x1 else go x1 (k-1)
```

## 9.7 学习资源

- **《泛函分析引论》**（袁亚湘）
- **Khan Academy**：泛函分析基础
- **MIT OpenCourseWare**：泛函分析

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：数值分析](13-NumericalAnalysis.md)
- [下一章：变分法](08-CalculusOfVariations.md)
- [返回数学主目录](../README.md)
