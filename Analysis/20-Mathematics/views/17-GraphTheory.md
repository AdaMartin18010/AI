# 17. 图论

## 17.1 图论概述

图论研究点与边组成的离散结构及其性质，是组合数学、计算机科学、网络科学等领域的基础。

- **无向图、有向图**
- **加权图、连通图**

## 17.2 基本概念

- 顶点、边、度
- 路径、回路、连通性
- 子图、生成树

## 17.3 重要定理与性质

- 欧拉定理、哈密顿定理
- 图的染色定理
- 图的矩阵表示

## 17.4 典型模型

- 完全图、二分图、树、平面图
- 网络流模型

## 17.5 算法与应用

- 最短路径（Dijkstra、Floyd）
- 最小生成树（Kruskal、Prim）
- 最大流（Ford-Fulkerson）

## 17.6 编程实现

### 17.6.1 Python实现（最短路径与最小生成树）

```python
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([(0,1,2),(1,2,1),(0,2,4)])
# Dijkstra最短路径
dist = nx.shortest_path_length(G, 0, 2, weight='weight')
print("最短路径长度:", dist)
# 最小生成树
mst = nx.minimum_spanning_tree(G)
print("最小生成树边:", list(mst.edges()))
```

### 17.6.2 Haskell实现（DFS遍历）

```haskell
-- 图的深度优先遍历
dfs :: Eq a => [(a, [a])] -> a -> [a]
dfs graph v = v : concat [dfs graph u | u <- neighbors v, u `notElem` visited]
  where neighbors x = maybe [] id (lookup x graph)
        visited = []
```

## 17.7 学习资源

- **《图论及其应用》**（Bondy & Murty）
- **Khan Academy**：图论
- **Coursera**：图论与网络

---
**导航：**

- [返回数学views目录](README.md)
- [上一章：控制理论](16-ControlTheory.md)
- [下一章：数论](19-NumberTheory.md)
- [返回数学主目录](../README.md)
