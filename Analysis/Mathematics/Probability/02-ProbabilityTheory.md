# 2. 概率论基础

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

## 2.3 条件概率与独立性

### 2.3.1 条件概率

$P(A|B) = \frac{P(A \cap B)}{P(B)}$

### 2.3.2 事件独立性

若 $P(A \cap B) = P(A)P(B)$，则 $A$ 与 $B$ 独立。

## 2.4 古典概率与几何概率

### 2.4.1 古典概率

等可能事件：$P(A) = \frac{|A|}{|\Omega|}$

### 2.4.2 几何概率

基于度量：$P(A) = \frac{\mu(A)}{\mu(\Omega)}$

## 2.5 随机过程简介

- 伯努利过程
- 泊松过程
- 马尔可夫链

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

## 2.7 学习资源

- **《概率论基础》**（William Feller）
- **MIT 6.041SC**：概率与随机过程公开课
- **Khan Academy**：概率基础

---
**导航：**

- [返回概率论目录](README.md)
- [上一章：概率论与统计学概述](01-Overview.md)
- [下一章：随机变量](03-RandomVariables.md)
- [返回数学主目录](../README.md)
