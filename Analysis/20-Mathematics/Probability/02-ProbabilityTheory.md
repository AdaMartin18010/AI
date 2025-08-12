# 2. 概率论基础

> **已完成深度优化与批判性提升**  
> 本文档已按统一标准补充批判性分析、未来展望、术语表、符号表、交叉引用等内容。

## 2.1 概率空间与事件

### 2.1.1 概率空间定义

概率空间 $(\Omega, \mathcal{F}, P)$ 包含：

- $\Omega$：样本空间
- $\mathcal{F}$：事件的集合（$\sigma$-代数）
- $P$：概率测度，满足概率公理

### 2.1.2 事件运算

- 并：$A \cup B$
- 交：$A \cap B$
- 补集：$A^c$

---

## 2.2 概率的基本性质

### 2.2.1 概率公理

```latex
\begin{align}
& P(\Omega) = 1 \\
& P(A) \geq 0 \\
& A \cap B = \emptyset \implies P(A \cup B) = P(A) + P(B)
\end{align}
```

### 2.2.2 加法公式

```latex
P(A \cup B) = P(A) + P(B) - P(A \cap B)
```

### 2.2.3 全概率公式

```latex
P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)
```

### 2.2.4 贝叶斯公式

```latex
P(B_j|A) = \frac{P(A|B_j)P(B_j)}{\sum_{i=1}^n P(A|B_i)P(B_i)}
```

---

## 2.3 条件概率与独立性

### 2.3.1 条件概率

$P(A|B) = \frac{P(A \cap B)}{P(B)}$

### 2.3.2 事件独立性

若 $P(A \cap B) = P(A)P(B)$，则 $A$ 与 $B$ 独立。

---

## 2.4 古典概率与几何概率

### 2.4.1 古典概率

等可能事件：$P(A) = \frac{|A|}{|\Omega|}$

### 2.4.2 几何概率

基于度量：$P(A) = \frac{\mu(A)}{\mu(\Omega)}$

---

## 2.5 随机过程简介

- 伯努利过程
- 泊松过程
- 马尔可夫链

---

## 2.6 编程实现

### 2.6.1 Python实现

```python
import random
# 古典概率：抛硬币
sample_space = ['H', 'T']
event = ['H']
prob = len(event) / len(sample_space)
print(f"抛出正面的概率: {prob}")

# 条件概率
p_a_and_b = 0.2
p_b = 0.5
p_a_given_b = p_a_and_b / p_b
print(f"P(A|B) = {p_a_given_b}")
```

### 2.6.2 Haskell实现

```haskell
-- 古典概率
classicalProbability :: (Eq a) => [a] -> [a] -> Double
classicalProbability sampleSpace event =
    fromIntegral (length event) / fromIntegral (length sampleSpace)

-- 条件概率
conditionalProbability :: Double -> Double -> Double
conditionalProbability pAB pB = pAB / pB
```

---

## 2.7 批判性分析

- 概率空间与测度论公理化极大提升了理论严密性，但实际问题中事件独立性、等可能性等假设常被违背。
- 古典概率、几何概率等模型虽直观，但对复杂系统、连续空间的刻画有限。
- 随机过程理论与实际数据建模、AI等领域的结合日益紧密，但高维、动态系统的理论与算法仍在发展。

---

## 2.8 未来展望

- 推动概率空间、随机过程理论与AI、数据科学、复杂系统等领域的深度融合。
- 丰富高维、动态、异质数据的概率建模与推断方法。
- 探索概率理论在生命科学、社会科学、工程等领域的创新应用。
- 推动概率理论在贝叶斯推断、因果推断、信息论等前沿方向的推广与创新。

---

## 2.9 术语表

- **概率空间（Probability Space）**：$(\Omega, \mathcal{F}, P)$三元组。
- **事件（Event）**：样本空间的子集。
- **概率测度（Probability Measure）**：满足概率公理的函数。
- **条件概率（Conditional Probability）**：在已知某事件发生条件下另一个事件发生的概率。
- **独立性（Independence）**：事件之间互不影响。
- **古典概率（Classical Probability）**：等可能事件概率模型。
- **几何概率（Geometric Probability）**：基于度量的概率模型。
- **随机过程（Stochastic Process）**：随时间变化的随机现象。

---

## 2.10 符号表

- $P(A)$：事件$A$的概率
- $\Omega$：样本空间
- $A, B$：事件
- $\mathcal{F}$：事件集合
- $\mu$：度量
- $X_t$：随机过程在时刻$t$的状态

---

## 2.11 交叉引用

- [Matter/批判框架标准化.md]
- [Matter/FormalLanguage/形式语言的多维批判性分析：从基础理论到应用实践.md]
- [Analysis/20-Mathematics/Probability/01-Overview.md]
- [Analysis/20-Mathematics/Probability/03-RandomVariables.md]
- [Analysis/20-Mathematics/Algebra/07-CategoryTheory.md]
- [Analysis/20-Mathematics/Calculus/10-AdvancedTopics.md]

---

## 2.12 学习资源

- **《概率论基础》**（William Feller）
- **MIT 6.041SC**：概率与随机过程公开课
- **Khan Academy**：概率基础

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：概率论与统计学概述](01-Overview.md)
- [下一章：随机变量](03-RandomVariables.md)
- [返回数学主目录](../README.md)
