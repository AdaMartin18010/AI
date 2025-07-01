# 1. 概率论与统计学概述

## 1.1 基本概念

### 1.1.1 概率论

概率论是研究随机现象数量规律的数学分支，为统计学提供理论基础。

**核心概念：**

- **随机事件**：可能发生也可能不发生的事件
- **概率**：事件发生的可能性度量
- **样本空间**：所有可能结果的集合
- **事件**：样本空间的子集

### 1.1.2 统计学

统计学是收集、分析、解释和呈现数据的科学。

**主要分支：**

- **描述统计**：数据汇总和可视化
- **推断统计**：从样本推断总体
- **贝叶斯统计**：基于先验知识的统计推断

## 1.2 数学表示

### 1.2.1 概率公理

```latex
\begin{align}
& P(\Omega) = 1 \quad \text{(样本空间概率为1)} \\
& P(A) \geq 0 \quad \text{(非负性)} \\
& P(A \cup B) = P(A) + P(B) - P(A \cap B) \quad \text{(加法公式)}
\end{align}
```

### 1.2.2 条件概率

```latex
P(A|B) = \frac{P(A \cap B)}{P(B)}
```

### 1.2.3 贝叶斯定理

```latex
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
```

## 1.3 核心分支

### 1.3.1 概率论分支

- **古典概率**：等可能事件
- **几何概率**：基于几何度量的概率
- **公理化概率**：基于测度论的概率
- **随机过程**：随时间变化的随机现象

### 1.3.2 统计学分支

- **参数统计**：假设总体分布形式
- **非参数统计**：不假设分布形式
- **稳健统计**：对异常值不敏感的统计方法
- **序贯分析**：逐步收集数据的统计方法

## 1.4 应用领域

### 1.4.1 自然科学

- **物理学**：量子力学、统计力学
- **生物学**：遗传学、生态学
- **化学**：反应动力学、分子动力学

### 1.4.2 工程科学

- **可靠性工程**：系统失效分析
- **质量控制**：生产过程监控
- **信号处理**：噪声分析

### 1.4.3 社会科学

- **经济学**：金融风险、计量经济学
- **心理学**：心理测量、实验设计
- **社会学**：人口统计、社会调查

### 1.4.4 人工智能

- **机器学习**：模型训练、验证
- **深度学习**：神经网络优化
- **强化学习**：策略优化

## 1.5 编程实现

### 1.5.1 Python实现

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 基本概率计算
def basic_probability():
    # 古典概率：掷骰子
    sample_space = [1, 2, 3, 4, 5, 6]
    event = [2, 4, 6]  # 偶数
    probability = len(event) / len(sample_space)
    print(f"掷出偶数的概率: {probability}")
    
    # 条件概率
    def conditional_probability(p_a_and_b, p_b):
        return p_a_and_b / p_b
    
    # 贝叶斯定理
    def bayes_theorem(p_b_given_a, p_a, p_b):
        return (p_b_given_a * p_a) / p_b

# 统计描述
def descriptive_statistics(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    var = np.var(data)
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'variance': var
    }
```

### 1.5.2 R实现

```r
# 基本统计函数
basic_stats <- function(data) {
  mean_val <- mean(data)
  median_val <- median(data)
  sd_val <- sd(data)
  var_val <- var(data)
  
  list(
    mean = mean_val,
    median = median_val,
    sd = sd_val,
    variance = var_val
  )
}

# 概率分布
normal_dist <- function(x, mu = 0, sigma = 1) {
  dnorm(x, mean = mu, sd = sigma)
}

# 假设检验
t_test_example <- function(data1, data2) {
  t.test(data1, data2)
}
```

### 1.5.3 Haskell实现

```haskell
-- 概率计算
module Probability where

-- 古典概率
classicalProbability :: (Eq a) => [a] -> [a] -> Double
classicalProbability sampleSpace event = 
    fromIntegral (length event) / fromIntegral (length sampleSpace)

-- 条件概率
conditionalProbability :: Double -> Double -> Double
conditionalProbability pAB pB = pAB / pB

-- 贝叶斯定理
bayesTheorem :: Double -> Double -> Double -> Double
bayesTheorem pBA pA pB = (pBA * pA) / pB

-- 统计函数
mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

variance :: [Double] -> Double
variance xs = 
    let m = mean xs
        n = fromIntegral (length xs)
    in sum [(x - m)^2 | x <- xs] / n

standardDeviation :: [Double] -> Double
standardDeviation = sqrt . variance
```

## 1.6 学习资源

### 1.6.1 经典教材

- **《概率论与数理统计》** - 茆诗松
- **《Statistical Inference》** - Casella & Berger
- **《Probability and Statistics》** - DeGroot & Schervish

### 1.6.2 在线资源

- **Khan Academy**：概率统计基础课程
- **MIT OpenCourseWare**：概率论公开课
- **Coursera**：统计学专项课程

### 1.6.3 实践项目

- **Kaggle竞赛**：数据科学项目
- **R语言项目**：统计分析实践
- **Python项目**：机器学习应用

## 1.7 与其他数学分支的关系

### 1.7.1 与微积分的关系

- **积分**：连续随机变量的概率密度
- **微分**：概率密度函数的导数
- **级数**：离散分布的期望计算

### 1.7.2 与线性代数的关系

- **矩阵**：协方差矩阵、相关矩阵
- **向量**：随机向量、特征向量
- **特征值**：主成分分析

### 1.7.3 与信息论的关系

- **熵**：信息熵、交叉熵
- **互信息**：变量间的依赖关系
- **KL散度**：分布间的距离度量

---

**导航：**

- [返回概率论目录](README.md)
- [下一章：概率论基础](02-ProbabilityTheory.md)
- [返回数学主目录](../README.md)
