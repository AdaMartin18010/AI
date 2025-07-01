# 7. 回归分析

## 7.1 回归分析概述

回归分析用于研究自变量与因变量之间的数量关系。

- **线性回归**：因变量与自变量线性关系
- **非线性回归**：因变量与自变量非线性关系

## 7.2 线性回归模型

### 7.2.1 一元线性回归

$y = \beta_0 + \beta_1 x + \varepsilon$

- $\beta_0$：截距
- $\beta_1$：斜率
- $\varepsilon$：误差项

#### 最小二乘法估计

$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$
$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$

### 7.2.2 多元线性回归

$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$

## 7.3 回归诊断与假设

- 残差分析
- 多重共线性
- 异方差性
- 自相关性

## 7.4 非线性回归与正则化

- 多项式回归
- 岭回归（Ridge）
- Lasso回归

## 7.5 编程实现

### 7.5.1 Python实现

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# 一元线性回归
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression().fit(X, y)
print(f"截距: {model.intercept_}, 斜率: {model.coef_[0]}")

# 岭回归
ridge = Ridge(alpha=1.0).fit(X, y)
print(f"Ridge回归斜率: {ridge.coef_[0]}")

# Lasso回归
lasso = Lasso(alpha=0.1).fit(X, y)
print(f"Lasso回归斜率: {lasso.coef_[0]}")
```

### 7.5.2 R实现

```r
# 一元线性回归
x <- c(1,2,3,4,5)
y <- c(2,4,5,4,5)
model <- lm(y ~ x)
summary(model)

# 岭回归
library(MASS)
ridge <- lm.ridge(y ~ x, lambda=1)
print(ridge)

# Lasso回归
library(glmnet)
lasso <- glmnet(as.matrix(x), y, alpha=1, lambda=0.1)
print(lasso)
```

## 7.6 学习资源

- **《应用回归分析》**（Draper & Smith）
- **Khan Academy**：回归分析
- **Coursera**：机器学习与回归

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：假设检验](06-HypothesisTesting.md)
- [下一章：时间序列分析](08-TimeSeries.md)
- [返回数学主目录](../README.md)
