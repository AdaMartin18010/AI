# 3. 随机变量

> **已完成深度优化与批判性提升**  
> 本文档已按统一标准补充批判性分析、未来展望、术语表、符号表、交叉引用等内容。

## 3.1 随机变量定义

### 3.1.1 概念

随机变量（Random Variable）是将样本空间中的每个结果映射为实数的函数。

- **离散型随机变量**：取有限或可数值
- **连续型随机变量**：取区间内任意实数值

### 3.1.2 记号

- $X, Y, Z$ 通常表示随机变量
- $P(X = x)$ 表示 $X$ 取值为 $x$ 的概率

---

## 3.2 分布函数

### 3.2.1 分布函数（CDF）

$F_X(x) = P(X \leq x)$

### 3.2.2 概率质量函数（PMF）

离散型：$P(X = x)$

### 3.2.3 概率密度函数（PDF）

连续型：$f_X(x)$，满足 $P(a < X \leq b) = \int_a^b f_X(x)dx$

---

## 3.3 期望与方差

### 3.3.1 期望

- 离散型：$E[X] = \sum_x x P(X = x)$
- 连续型：$E[X] = \int_{-\infty}^{+\infty} x f_X(x) dx$

### 3.3.2 方差

$Var(X) = E[(X - E[X])^2]$

---

## 3.4 常见随机变量

- 伯努利随机变量
- 二项分布
- 泊松分布
- 均匀分布
- 正态分布

---

## 3.5 多维随机变量

- 联合分布
- 边缘分布
- 条件分布

---

## 3.6 编程实现

### 3.6.1 Python实现

```python
import numpy as np
# 离散型随机变量
values = [0, 1]
probs = [0.3, 0.7]
expectation = np.dot(values, probs)
variance = np.dot([(x - expectation)**2 for x in values], probs)
print(f"期望: {expectation}, 方差: {variance}")

# 连续型随机变量（正态分布）
from scipy.stats import norm
mu, sigma = 0, 1
E = norm.mean(mu, sigma)
Var = norm.var(mu, sigma)
print(f"正态分布期望: {E}, 方差: {Var}")
```

### 3.6.2 Haskell实现

```haskell
-- 离散型随机变量期望
expectation :: [Double] -> [Double] -> Double
expectation xs ps = sum $ zipWith (*) xs ps

-- 方差
variance :: [Double] -> [Double] -> Double
variance xs ps =
    let mu = expectation xs ps
    in sum $ zipWith (\x p -> p * (x - mu)^2) xs ps
```

---

## 3.7 批判性分析

- 随机变量理论极大丰富了概率建模与统计推断的表达能力，但实际数据中变量分布、独立性等假设常被违背。
- 多维随机变量、联合分布等理论虽强大，但高维数据的可视化与直观理解仍具挑战。
- 随机变量理论与AI、数据科学、物理等领域的结合日益紧密，但跨学科表达与应用体系尚需完善。

---

## 3.8 未来展望

- 推动随机变量理论与AI、数据科学、复杂系统等领域的深度融合。
- 丰富高维、动态、异质数据的随机变量建模与推断方法。
- 探索随机变量理论在生命科学、社会科学、工程等领域的创新应用。
- 推动随机变量理论在贝叶斯推断、因果推断、信息论等前沿方向的推广与创新。

---

## 3.9 术语表

- **随机变量（Random Variable）**：将样本空间结果映射为实数的函数。
- **分布函数（CDF）**：$F_X(x) = P(X \leq x)$。
- **概率质量函数（PMF）**：离散型随机变量的概率分布。
- **概率密度函数（PDF）**：连续型随机变量的概率分布。
- **期望（Expectation）**：随机变量的均值。
- **方差（Variance）**：随机变量的离散程度。
- **联合分布（Joint Distribution）**：多维随机变量的分布。
- **边缘分布（Marginal Distribution）**：多维分布中部分变量的分布。
- **条件分布（Conditional Distribution）**：在给定部分变量取值下的分布。

---

## 3.10 符号表

- $X, Y, Z$：随机变量
- $P(X = x)$：离散型概率
- $f_X(x)$：连续型概率密度
- $E[X]$：期望
- $Var(X)$：方差
- $F_X(x)$：分布函数
- $p(x, y)$：联合概率

---

## 3.11 交叉引用

- [Matter/批判分析框架.md]
- [Matter/FormalLanguage/形式语言的多维批判性分析：从基础理论到应用实践.md]
- [Analysis/Mathematics/Probability/02-ProbabilityTheory.md]
- [Analysis/Mathematics/Probability/04-ProbabilityDistributions.md]
- [Analysis/Mathematics/Algebra/07-CategoryTheory.md]
- [Analysis/Mathematics/Calculus/10-AdvancedTopics.md]

---

## 3.12 学习资源

- **《概率论与数理统计》**（茆诗松）
- **Khan Academy**：随机变量与分布
- **MIT OpenCourseWare**：概率与统计

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：概率论基础](02-ProbabilityTheory.md)
- [下一章：概率分布](04-ProbabilityDistributions.md)
- [返回数学主目录](../README.md)
