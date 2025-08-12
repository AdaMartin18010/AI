# 19. 数论

## 19.1 数论概述

数论研究整数的性质及其规律，是纯数学、密码学、计算机科学等领域的重要基础。

- **整除性**
- **素数与因数分解**

## 19.2 基本定理

- 算术基本定理
- 欧几里得算法
- 中国剩余定理
- 费马小定理、欧拉定理

## 19.3 同余与模运算

- 同余关系
- 模加、模乘、模逆元

## 19.4 素数分布与筛法

- 埃拉托斯特尼筛法
- 梅森素数

## 19.5 代数数论与应用

- 代数整数、理想
- RSA加密、椭圆曲线密码

## 19.6 编程实现

### 19.6.1 Python实现（素数筛与模逆元）

```python
def sieve(n):
    is_prime = [True]*(n+1)
    is_prime[0:2] = [False, False]
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [x for x, p in enumerate(is_prime) if p]
print(sieve(30))

# 模逆元
def modinv(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None
print(modinv(3, 7))
```

### 19.6.2 Haskell实现（欧几里得算法）

```haskell
-- 欧几里得算法求最大公约数
gcd' :: Integer -> Integer -> Integer
gcd' a 0 = a
gcd' a b = gcd' b (a `mod` b)
```

## 19.7 学习资源

- **《初等数论》**（华罗庚）
- **Khan Academy**：数论
- **Coursera**：数论与密码学

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：图论](17-GraphTheory.md)
- [下一章：数值分析](13-NumericalAnalysis.md)
- [返回数学主目录](../README.md)
