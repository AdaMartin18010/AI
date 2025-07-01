# 6. 假设检验

## 6.1 假设检验概述

假设检验是利用样本数据对总体参数的假设进行判断的统计方法。

- **原假设 $H_0$**：待检验的假设
- **备择假设 $H_1$**：与原假设相对立的假设

## 6.2 检验流程

1. 提出 $H_0$ 和 $H_1$
2. 选择检验统计量
3. 确定显著性水平 $\alpha$
4. 计算P值或临界值
5. 做出决策（拒绝或不拒绝 $H_0$）

## 6.3 常见假设检验方法

### 6.3.1 t检验

- 单样本t检验：检验样本均值与总体均值是否有显著差异
- 双样本t检验：检验两组样本均值是否有显著差异

### 6.3.2 方差分析（ANOVA）

检验多组均值是否有显著差异

### 6.3.3 卡方检验

- 拟合优度检验
- 列联表独立性检验

### 6.3.4 非参数检验

- 秩和检验（Mann-Whitney U检验）
- 符号检验

## 6.4 第一类与第二类错误

- **第一类错误（Type I error）**：错误拒绝 $H_0$
- **第二类错误（Type II error）**：错误接受 $H_0$
- 显著性水平 $\alpha$：第一类错误概率
- 检验功效 $1-\beta$：正确拒绝 $H_0$ 的概率

## 6.5 P值与置信区间

- **P值**：观察到的样本结果在 $H_0$ 成立下出现的概率
- **置信区间**：参数可能取值的区间估计

## 6.6 编程实现

### 6.6.1 Python实现

```python
import numpy as np
from scipy import stats
# 单样本t检验
sample = np.random.normal(0, 1, 30)
t_stat, p_value = stats.ttest_1samp(sample, 0)
print(f"t统计量: {t_stat}, p值: {p_value}")

# 双样本t检验
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)
t_stat2, p_value2 = stats.ttest_ind(group1, group2)
print(f"双样本t检验p值: {p_value2}")

# 卡方检验
obs = np.array([[10, 20], [20, 20]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print(f"卡方统计量: {chi2}, p值: {p}")
```

### 6.6.2 R实现

```r
# 单样本t检验
sample <- rnorm(30)
t.test(sample, mu=0)

# 双样本t检验
group1 <- rnorm(30)
group2 <- rnorm(30, mean=0.5)
t.test(group1, group2)

# 卡方检验
obs <- matrix(c(10, 20, 20, 20), nrow=2)
chisq.test(obs)
```

## 6.7 学习资源

- **《统计推断》**（Casella & Berger）
- **Khan Academy**：假设检验
- **MIT OpenCourseWare**：统计学

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：统计推断](05-StatisticalInference.md)
- [下一章：回归分析](07-RegressionAnalysis.md)
- [返回数学主目录](../README.md)
