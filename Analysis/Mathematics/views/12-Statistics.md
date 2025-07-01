# 12. 统计学

## 12.1 统计学概述

统计学研究数据的收集、分析、解释与推断，是科学研究、工程、社会科学等领域的基础。

- **描述统计**
- **推断统计**

## 12.2 描述统计

- 均值、中位数、众数
- 方差、标准差、偏度、峰度
- 相关系数、协方差

## 12.3 推断统计

- 参数估计（点估计、区间估计）
- 假设检验（t检验、卡方检验、方差分析）
- 回归分析

## 12.4 概率分布

- 正态分布、二项分布、泊松分布
- 指数分布、t分布、卡方分布

## 12.5 多元统计与高级方法

- 主成分分析（PCA）
- 因子分析、聚类分析
- 判别分析

## 12.6 编程实现

### 12.6.1 Python实现（描述统计与回归）

```python
import numpy as np
from scipy import stats
# 描述统计
x = np.random.normal(0, 1, 100)
print("均值：", np.mean(x), "标准差：", np.std(x))
# 线性回归
from sklearn.linear_model import LinearRegression
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.randn(100) * 0.1
model = LinearRegression().fit(X, y)
print("回归系数：", model.coef_[0])
```

### 12.6.2 Haskell实现（均值与方差）

```haskell
mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

variance :: [Double] -> Double
variance xs = let m = mean xs in sum [(x - m)^2 | x <- xs] / fromIntegral (length xs)
```

## 12.7 学习资源

- **《统计学习方法》**（李航）
- **Khan Academy**：统计学基础
- **Coursera**：统计推断与数据分析

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：调和分析](10-HarmonicAnalysis.md)
- [下一章：形式语言理论](22-FormalLanguageTheory.md)
- [返回数学主目录](../README.md)
