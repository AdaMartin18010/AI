# 4. 概率分布

> **已完成深度优化与批判性提升**  
> 本文档已按统一标准补充批判性分析、未来展望、术语表、符号表、交叉引用等内容。

## 4.1 概率分布概述

概率分布描述随机变量所有可能取值及其概率。

- **离散型分布**：概率质量函数（PMF）
- **连续型分布**：概率密度函数（PDF）

---

## 4.2 常见离散型分布

### 4.2.1 伯努利分布

$P(X = x) = p^x (1-p)^{1-x},\ x \in \{0,1\}$

### 4.2.2 二项分布

$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$

### 4.2.3 泊松分布

$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

---

## 4.3 常见连续型分布

### 4.3.1 均匀分布

$f(x) = \frac{1}{b-a},\ x \in [a, b]$

### 4.3.2 正态分布

$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

### 4.3.3 指数分布

$f(x) = \lambda e^{-\lambda x},\ x \geq 0$

---

## 4.4 分布函数性质

- 非负性：$P(X \in A) \geq 0$
- 归一性：$\sum_x P(X = x) = 1$ 或 $\int_{-\infty}^{+\infty} f(x)dx = 1$
- 单调性与右连续性

---

## 4.5 分布间关系

- 极限定理：二项分布趋于正态分布
- 泊松近似：二项分布在 $n \to \infty, p \to 0$ 时趋于泊松分布

---

## 4.6 编程实现

### 4.6.1 Python实现

```python
import numpy as np
from scipy.stats import bernoulli, binom, poisson, uniform, norm, expon

# 伯努利分布
p = 0.3
rv = bernoulli(p)
print('伯努利分布P(X=1):', rv.pmf(1))

# 二项分布
n, p = 10, 0.5
print('二项分布P(X=3):', binom.pmf(3, n, p))

# 泊松分布
lam = 2
print('泊松分布P(X=1):', poisson.pmf(1, lam))

# 正态分布
mu, sigma = 0, 1
print('正态分布P(X=0):', norm.pdf(0, mu, sigma))

# 均匀分布
print('均匀分布P(X=0.5):', uniform.pdf(0.5, 0, 1))
```

### 4.6.2 Haskell实现

```haskell
-- 伯努利分布
bernoulli :: Double -> Int -> Double
bernoulli p x = if x == 1 then p else 1 - p

-- 二项分布
binomial :: Int -> Double -> Int -> Double
binomial n p k = fromIntegral (choose n k) * p^k * (1-p)^(n-k)
  where
    choose n k = product [n-k+1..n] `div` product [1..k]

-- 泊松分布
poisson :: Double -> Int -> Double
poisson lambda k = (lambda^k * exp(-lambda)) / fromIntegral (product [1..k])
```

---

## 4.7 批判性分析

- 概率分布理论极大丰富了概率建模与统计推断的表达能力，但实际数据中分布假设（正态性、独立性等）常被违背。
- 离散与连续分布的统一理论虽强大，但高维、混合型分布的建模与推断仍具挑战。
- 概率分布理论与AI、数据科学、物理等领域的结合日益紧密，但跨学科表达与应用体系尚需完善。

---

## 4.8 未来展望

- 推动概率分布理论与AI、数据科学、复杂系统等领域的深度融合。
- 丰富高维、动态、混合型分布的建模与推断方法。
- 探索概率分布理论在生命科学、社会科学、工程等领域的创新应用。
- 推动概率分布理论在贝叶斯推断、因果推断、信息论等前沿方向的推广与创新。

---

## 4.9 术语表

- **概率分布（Probability Distribution）**：描述随机变量所有可能取值及其概率的函数。
- **概率质量函数（PMF）**：离散型随机变量的概率分布。
- **概率密度函数（PDF）**：连续型随机变量的概率分布。
- **分布函数（CDF）**：$F_X(x) = P(X \leq x)$。
- **参数（Parameter）**：分布的特征量（如均值、方差）。
- **极限定理（Limit Theorem）**：分布间的极限关系。

---

## 4.10 符号表

- $X$：随机变量
- $P(X = x)$：离散型概率
- $f_X(x)$：连续型概率密度
- $F_X(x)$：分布函数
- $\mu$：均值
- $\sigma^2$：方差
- $n$：样本容量
- $\lambda$：泊松分布参数

---

## 4.11 交叉引用

- [Matter/批判框架标准化.md]
- [Matter/FormalLanguage/形式语言的多维批判性分析：从基础理论到应用实践.md]
- [Analysis/20-Mathematics/Probability/03-RandomVariables.md]
- [Analysis/20-Mathematics/Probability/05-StatisticalInference.md]
- [Analysis/20-Mathematics/Algebra/07-CategoryTheory.md]
- [Analysis/20-Mathematics/Calculus/10-AdvancedTopics.md]

---

## 4.12 学习资源

- **《概率论与数理统计》**（茆诗松）
- **Khan Academy**：概率分布
- **MIT OpenCourseWare**：概率与统计

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：随机变量](03-RandomVariables.md)
- [下一章：统计推断](05-StatisticalInference.md)
- [返回数学主目录](../README.md)
