# 5. 统计推断

## 5.1 统计推断概述

统计推断是利用样本数据对总体参数进行估计和假设检验的方法论。

- **参数估计**：点估计与区间估计
- **假设检验**：对总体参数的假设进行判断

## 5.2 参数估计

### 5.2.1 点估计

- 样本均值 $\bar{x}$ 估计总体均值 $\mu$
- 样本方差 $s^2$ 估计总体方差 $\sigma^2$

### 5.2.2 区间估计

$P(\theta_L < \theta < \theta_U) = 1-\alpha$

#### 置信区间示例

- 均值置信区间（正态分布，方差已知）：
  $\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$
- 均值置信区间（方差未知）：
  $\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$

## 5.3 假设检验

### 5.3.1 基本流程

1. 提出原假设 $H_0$ 与备择假设 $H_1$
2. 选择检验统计量
3. 确定显著性水平 $\alpha$
4. 计算P值或临界值
5. 做出决策

### 5.3.2 常见检验

- 单样本/双样本t检验
- 方差分析（ANOVA）
- 卡方检验

## 5.4 极大似然估计（MLE）

$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)$

## 5.5 贝叶斯推断简介

- 先验分布、后验分布
- 贝叶斯公式：$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$

## 5.6 编程实现

### 5.6.1 Python实现

```python
import numpy as np
from scipy import stats
# 样本均值与置信区间
sample = np.random.normal(0, 1, 100)
mean = np.mean(sample)
conf_int = stats.norm.interval(0.95, loc=mean, scale=stats.sem(sample))
print(f"均值: {mean}, 置信区间: {conf_int}")

# 单样本t检验
result = stats.ttest_1samp(sample, 0)
print(f"t检验p值: {result.pvalue}")
```

### 5.6.2 R实现

```r
# 均值与置信区间
sample <- rnorm(100)
mean_val <- mean(sample)
conf_int <- t.test(sample)$conf.int
print(mean_val)
print(conf_int)

# 单样本t检验
t.test(sample, mu=0)
```

## 5.7 学习资源

- **《统计推断》**（Casella & Berger）
- **Khan Academy**：统计推断
- **MIT OpenCourseWare**：统计学

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：概率分布](04-ProbabilityDistributions.md)
- [下一章：假设检验](06-HypothesisTesting.md)
- [返回数学主目录](../README.md)
