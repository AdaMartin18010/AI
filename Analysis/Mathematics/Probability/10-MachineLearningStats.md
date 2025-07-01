# 10. 机器学习统计

## 10.1 机器学习中的统计基础

机器学习大量依赖统计学原理进行模型设计、训练与评估。

- **概率建模**：描述数据分布与不确定性
- **参数估计**：最大似然、贝叶斯估计
- **模型评估**：交叉验证、置信区间、假设检验

## 10.2 典型统计模型

### 10.2.1 线性回归与逻辑回归

- 线性回归：$y = \beta_0 + \beta_1 x + \varepsilon$
- 逻辑回归：$P(y=1|x) = \frac{1}{1+e^{-w^Tx}}$

### 10.2.2 判别分析

- 线性判别分析（LDA）
- 二次判别分析（QDA）

### 10.2.3 生成模型与判别模型

- 生成模型：朴素贝叶斯、高斯混合模型（GMM）
- 判别模型：支持向量机（SVM）、神经网络

## 10.3 统计学习理论

- 经验风险最小化（ERM）
- 结构风险最小化（SRM）
- VC维与泛化能力

## 10.4 模型评估与选择

- 交叉验证（K-fold）
- 混淆矩阵、ROC曲线、AUC
- 假设检验与置信区间

## 10.5 编程实现

### 10.5.1 Python实现

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score

# 线性回归
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.randn(100) * 0.1
lr = LinearRegression().fit(X, y)
print(f"线性回归系数: {lr.coef_[0]}")

# 逻辑回归
X_bin = np.random.rand(100, 2)
y_bin = (X_bin[:, 0] + X_bin[:, 1] > 1).astype(int)
clf = LogisticRegression().fit(X_bin, y_bin)
print(f"逻辑回归权重: {clf.coef_}")

# 交叉验证
scores = cross_val_score(lr, X, y, cv=5)
print(f"交叉验证得分: {scores}")

# 混淆矩阵与AUC
preds = clf.predict(X_bin)
cm = confusion_matrix(y_bin, preds)
auc = roc_auc_score(y_bin, clf.predict_proba(X_bin)[:,1])
print(f"混淆矩阵:\n{cm}\nAUC: {auc}")
```

### 10.5.2 R实现

```r
# 线性回归
X <- matrix(runif(100), ncol=1)
y <- 2 * X + rnorm(100, 0, 0.1)
lm_model <- lm(y ~ X)
summary(lm_model)

# 逻辑回归
X_bin <- matrix(runif(200), ncol=2)
y_bin <- as.integer(X_bin[,1] + X_bin[,2] > 1)
glm_model <- glm(y_bin ~ X_bin, family=binomial)
summary(glm_model)

# 交叉验证
library(boot)
cv.glm(data.frame(X, y), lm_model, K=5)$delta
```

## 10.6 学习资源

- **《统计学习方法》**（李航）
- **Khan Academy**：统计与机器学习
- **Coursera**：机器学习专项课程

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：贝叶斯统计](09-BayesianStatistics.md)
- [返回数学主目录](../README.md)
