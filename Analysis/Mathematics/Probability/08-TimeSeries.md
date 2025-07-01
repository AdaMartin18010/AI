# 8. 时间序列分析

## 8.1 时间序列分析概述

时间序列分析研究随时间变化的数据序列的统计特性与建模方法。

- **平稳性**：均值和方差不随时间变化
- **自相关性**：序列自身的相关性

## 8.2 主要模型

### 8.2.1 自回归模型（AR）

$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \varepsilon_t$

### 8.2.2 移动平均模型（MA）

$X_t = \mu + \sum_{i=1}^q \theta_i \varepsilon_{t-i} + \varepsilon_t$

### 8.2.3 ARMA模型

$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \sum_{j=1}^q \theta_j \varepsilon_{t-j} + \varepsilon_t$

### 8.2.4 ARIMA模型

适用于非平稳序列，包含差分操作。

### 8.2.5 季节性模型（SARIMA）

处理季节性波动的时间序列。

## 8.3 时间序列分析步骤

1. 数据可视化与预处理
2. 平稳性检验（ADF检验等）
3. 模型识别与参数估计
4. 模型诊断
5. 预测与评估

## 8.4 编程实现

### 8.4.1 Python实现

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# 构造时间序列数据
ts = pd.Series(np.random.randn(100))
# ARIMA建模
model = ARIMA(ts, order=(1,1,1))
result = model.fit()
print(result.summary())
# 预测
forecast = result.forecast(steps=5)
print(forecast)
```

### 8.4.2 R实现

```r
# 构造时间序列数据
ts <- ts(rnorm(100))
# ARIMA建模
model <- arima(ts, order=c(1,1,1))
summary(model)
# 预测
forecast <- predict(model, n.ahead=5)
print(forecast)
```

## 8.5 学习资源

- **《时间序列分析》**（Box & Jenkins）
- **Khan Academy**：时间序列基础
- **Coursera**：时间序列与预测

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：回归分析](07-RegressionAnalysis.md)
- [下一章：贝叶斯统计](09-BayesianStatistics.md)
- [返回数学主目录](../README.md)
