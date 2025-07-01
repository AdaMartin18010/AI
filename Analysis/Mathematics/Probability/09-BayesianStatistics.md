# 9. 贝叶斯统计

## 9.1 贝叶斯统计概述

贝叶斯统计以概率为度量不确定性的工具，通过先验知识与观测数据结合进行推断。

- **先验分布**：反映观测前对参数的认识
- **后验分布**：结合数据后的参数分布

## 9.2 贝叶斯公式

$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$

- $P(\theta)$：先验分布
- $P(X|\theta)$：似然函数
- $P(\theta|X)$：后验分布
- $P(X)$：边缘似然

## 9.3 典型贝叶斯模型

### 9.3.1 共轭先验

- 二项分布的Beta先验
- 泊松分布的Gamma先验

### 9.3.2 贝叶斯线性回归

- 先验：$\beta \sim N(\mu_0, \Sigma_0)$
- 后验：$\beta|X,y \sim N(\mu_n, \Sigma_n)$

### 9.3.3 贝叶斯分类器

- 朴素贝叶斯分类器

## 9.4 MCMC与采样方法

- 马尔可夫链蒙特卡洛（MCMC）
- 吉布斯采样、Metropolis-Hastings算法

## 9.5 编程实现

### 9.5.1 Python实现（PyMC3）

```python
import pymc3 as pm
import numpy as np
# 贝叶斯线性回归示例
X = np.linspace(0, 1, 100)
y = 1.5 * X + np.random.normal(0, 0.1, 100)
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha + beta * X
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
    trace = pm.sample(1000, tune=1000, cores=1)
print(pm.summary(trace))
```

### 9.5.2 R实现

```r
# 贝叶斯线性回归（rstanarm包）
library(rstanarm)
X <- 1:100 / 100
y <- 1.5 * X + rnorm(100, 0, 0.1)
model <- stan_glm(y ~ X)
print(summary(model))
```

## 9.6 学习资源

- **《贝叶斯数据分析》**（Gelman等）
- **Khan Academy**：贝叶斯推断
- **Coursera**：贝叶斯统计与建模

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：时间序列分析](08-TimeSeries.md)
- [下一章：机器学习统计](10-MachineLearningStats.md)
- [返回数学主目录](../README.md)
