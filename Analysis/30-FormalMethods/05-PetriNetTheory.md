# Petri网理论基础与应用

## 目录

- [Petri网理论基础与应用](#petri网理论基础与应用)
  - [目录](#目录)
  - [1. Petri网基础](#1-petri网基础)
    - [1.1 Petri网定义](#11-petri网定义)
    - [1.2 变迁规则](#12-变迁规则)
    - [1.3 基本性质](#13-基本性质)
  - [2. 可达性分析](#2-可达性分析)
    - [2.1 可达性关系](#21-可达性关系)
    - [2.2 状态空间分析](#22-状态空间分析)
    - [2.3 可达性算法](#23-可达性算法)
  - [3. 并发性分析](#3-并发性分析)
    - [3.1 并发变迁](#31-并发变迁)
    - [3.2 冲突分析](#32-冲突分析)
    - [3.3 并发控制](#33-并发控制)
  - [4. 结构性质](#4-结构性质)
    - [4.1 有界性](#41-有界性)
    - [4.2 活性](#42-活性)
    - [4.3 可逆性](#43-可逆性)
  - [5. 高级Petri网](#5-高级petri网)
    - [5.1 时间Petri网](#51-时间petri网)
    - [5.2 着色Petri网](#52-着色petri网)
    - [5.3 层次Petri网](#53-层次petri网)
  - [6. 实际应用](#6-实际应用)
    - [6.1 并发系统建模](#61-并发系统建模)
    - [6.2 工作流建模](#62-工作流建模)
    - [6.3 协议验证](#63-协议验证)
  - [7. 形式化验证](#7-形式化验证)
    - [7.1 模型检查](#71-模型检查)
    - [7.2 性质验证](#72-性质验证)
    - [7.3 工具支持](#73-工具支持)
  - [总结](#总结)

---

## 1. Petri网基础

### 1.1 Petri网定义

**定义 1.1 (Petri网)**
Petri网是一个四元组 $N = (P, T, F, M_0)$，其中：

- $P$ 是有限的位置集合（places）
- $T$ 是有限的变迁集合（transitions）
- $F \subseteq (P \times T) \cup (T \times P)$ 是流关系（flow relation）
- $M_0 : P \rightarrow \mathbb{N}$ 是初始标识（initial marking）

**形式化表示**：

```math
\text{Petri Net} = \langle \text{Places}, \text{Transitions}, \text{Flow Relation}, \text{Initial Marking} \rangle
```

**定义 1.2 (标识)**
标识 $M : P \rightarrow \mathbb{N}$ 表示每个位置中的托肯（token）数量。

**定义 1.3 (前集和后集)**
对于 $x \in P \cup T$：

- $^\bullet x = \{y \mid (y, x) \in F\}$ 是 $x$ 的前集
- $x^\bullet = \{y \mid (x, y) \in F\}$ 是 $x$ 的后集

**算法 1.1 (Petri网构造)**:

```haskell
data PetriNet = PetriNet {
  places :: Set Place,
  transitions :: Set Transition,
  flowRelation :: Set (Place, Transition),
  initialMarking :: Marking
}

constructPetriNet :: [Place] -> [Transition] -> [(Place, Transition)] -> Marking -> PetriNet
constructPetriNet places transitions flow initialMarking = 
  PetriNet {
    places = Set.fromList places,
    transitions = Set.fromList transitions,
    flowRelation = Set.fromList flow,
    initialMarking = initialMarking
  }
```

### 1.2 变迁规则

**定义 1.4 (变迁使能)**
变迁 $t \in T$ 在标识 $M$ 下使能，记作 $M[t\rangle$，当且仅当：

```math
\forall p \in ^\bullet t : M(p) \geq F(p, t)
```

**定义 1.5 (变迁发生)**
如果 $M[t\rangle$，则变迁 $t$ 可以发生，产生新标识 $M'$，记作 $M[t\rangle M'$，其中：

```math
M'(p) = M(p) - F(p, t) + F(t, p)
```

**定理 1.1 (变迁发生保持托肯守恒)**
对于任何变迁 $t$ 和标识 $M$，如果 $M[t\rangle M'$，则：

```math
\sum_{p \in P} M'(p) = \sum_{p \in P} M(p)
```

**证明：** 通过流关系的定义：

```math
\sum_{p \in P} M'(p) = \sum_{p \in P} (M(p) - F(p, t) + F(t, p)) = \sum_{p \in P} M(p)
```

**算法 1.2 (变迁发生)**:

```haskell
fireTransition :: PetriNet -> Marking -> Transition -> Maybe Marking
fireTransition net marking transition = 
  if isEnabled net marking transition
  then Just (computeNewMarking net marking transition)
  else Nothing

isEnabled :: PetriNet -> Marking -> Transition -> Bool
isEnabled net marking transition = 
  let preSet = getPreSet net transition
      flowFunction = flowRelation net
  in all (\place -> 
    let requiredTokens = getFlowWeight flowFunction place transition
        availableTokens = marking place
    in availableTokens >= requiredTokens) preSet

computeNewMarking :: PetriNet -> Marking -> Transition -> Marking
computeNewMarking net marking transition = 
  let places = places net
      flowFunction = flowRelation net
      newMarking place = 
        let inputWeight = getFlowWeight flowFunction place transition
            outputWeight = getFlowWeight flowFunction transition place
            currentTokens = marking place
        in currentTokens - inputWeight + outputWeight
  in Marking newMarking
```

### 1.3 基本性质

**定义 1.6 (Petri网性质)**:

- **有界性**：所有位置的托肯数量有上界
- **安全性**：所有位置最多只有一个托肯
- **活性**：每个变迁都能无限次发生
- **可逆性**：可以从任何可达标识回到初始标识

**算法 1.3 (性质检查)**:

```haskell
checkProperties :: PetriNet -> PropertyReport
checkProperties net = 
  let boundedness = checkBoundedness net
      safety = checkSafety net
      liveness = checkLiveness net
      reversibility = checkReversibility net
  in PropertyReport {
    bounded = boundedness,
    safe = safety,
    live = liveness,
    reversible = reversibility
  }
```

---

## 2. 可达性分析

### 2.1 可达性关系

**定义 2.1 (可达性)**
标识 $M'$ 从标识 $M$ 可达，记作 $M \rightarrow^* M'$，如果存在变迁序列 $\sigma = t_1 t_2 \cdots t_n$ 使得：

```math
M[t_1\rangle M_1[t_2\rangle M_2 \cdots [t_n\rangle M'
```

**定义 2.2 (可达集)**
从标识 $M$ 可达的所有标识集合：

```math
R(M) = \{M' \mid M \rightarrow^* M'\}
```

**定理 2.1 (可达性保持)**
如果 $M \rightarrow^* M'$ 且 $M'[t\rangle M''$，则 $M \rightarrow^* M''$。

**证明：** 通过可达性的传递性：

1. $M \rightarrow^* M'$ 存在变迁序列 $\sigma$
2. $M'[t\rangle M''$ 表示 $t$ 在 $M'$ 下使能
3. 因此 $M \rightarrow^* M''$ 通过序列 $\sigma t$

### 2.2 状态空间分析

**定义 2.3 (状态空间)**
Petri网的状态空间是图 $G = (V, E)$，其中：

- $V = R(M_0)$ 是可达标识集合
- $E = \{(M, M') \mid \exists t : M[t\rangle M'\}$ 是变迁关系

**算法 2.1 (可达性分析)**:

```haskell
reachabilityAnalysis :: PetriNet -> [Marking]
reachabilityAnalysis net = 
  let initial = initialMarking net
      reachable = bfs initial [initial]
  in reachable
  where
    bfs :: Marking -> [Marking] -> [Marking]
    bfs current visited = 
      let enabled = enabledTransitions net current
          newMarkings = [fireTransition net current t | t <- enabled]
          unvisited = filter (`notElem` visited) newMarkings
      in if null unvisited 
         then visited
         else bfs (head unvisited) (visited ++ unvisited)
```

### 2.3 可达性算法

**算法 2.2 (深度优先搜索可达性)**:

```haskell
dfsReachability :: PetriNet -> [Marking]
dfsReachability net = 
  let initial = initialMarking net
      reachable = dfs initial Set.empty
  in Set.toList reachable
  where
    dfs :: Marking -> Set Marking -> Set Marking
    dfs current visited = 
      if Set.member current visited
      then visited
      else 
        let visited' = Set.insert current visited
            enabled = enabledTransitions net current
            newMarkings = [fireTransition net current t | t <- enabled]
            reachable = foldl (\acc marking -> 
              case marking of
                Just m -> dfs m acc
                Nothing -> acc) visited' newMarkings
        in reachable
```

**算法 2.3 (符号可达性分析)**:

```haskell
symbolicReachability :: PetriNet -> SymbolicStateSpace
symbolicReachability net = 
  let -- 使用决策图表示状态空间
      initialState = createSymbolicState (initialMarking net)
      reachableStates = computeSymbolicReachability net initialState
  in reachableStates

computeSymbolicReachability :: PetriNet -> SymbolicState -> SymbolicStateSpace
computeSymbolicReachability net initialState = 
  let -- 使用符号方法计算可达状态
      reachable = iterateSymbolicReachability net [initialState]
  in SymbolicStateSpace reachable
```

---

## 3. 并发性分析

### 3.1 并发变迁

**定义 3.1 (并发变迁)**
两个变迁 $t_1, t_2 \in T$ 在标识 $M$ 下并发，记作 $M[t_1, t_2\rangle$，当且仅当：

1. $M[t_1\rangle$ 且 $M[t_2\rangle$
2. $^\bullet t_1 \cap ^\bullet t_2 = \emptyset$（无共享输入位置）

**定理 3.1 (并发交换性)**
如果 $M[t_1, t_2\rangle$，则 $M[t_1\rangle M_1[t_2\rangle M'$ 且 $M[t_2\rangle M_2[t_1\rangle M'$，其中 $M_1 \neq M_2$ 但 $M'$ 相同。

**证明：** 通过并发变迁的定义：

1. 无共享输入位置确保独立性
2. 变迁发生顺序不影响最终结果
3. 中间标识可能不同但最终标识相同

**算法 3.1 (并发检测)**:

```haskell
detectConcurrency :: PetriNet -> Marking -> [Transition] -> [Transition]
detectConcurrency net marking transitions = 
  let enabled = filter (isEnabled net marking) transitions
      concurrent = filter (\t1 -> 
        all (\t2 -> t1 /= t2 && areConcurrent net marking t1 t2) enabled) enabled
  in concurrent

areConcurrent :: PetriNet -> Marking -> Transition -> Transition -> Bool
areConcurrent net marking t1 t2 = 
  let preSet1 = getPreSet net t1
      preSet2 = getPreSet net t2
      sharedPlaces = Set.intersection preSet1 preSet2
  in Set.null sharedPlaces
```

### 3.2 冲突分析

**定义 3.2 (冲突)**
两个变迁 $t_1, t_2 \in T$ 在标识 $M$ 下冲突，当且仅当：

1. $M[t_1\rangle$ 且 $M[t_2\rangle$
2. $^\bullet t_1 \cap ^\bullet t_2 \neq \emptyset$（有共享输入位置）

**定理 3.2 (冲突解决)**
如果 $t_1, t_2$ 在 $M$ 下冲突，则最多只能发生其中一个变迁。

**证明：** 通过冲突定义：

1. 共享输入位置限制托肯数量
2. 一个变迁的发生会消耗共享托肯
3. 另一个变迁将不再使能

**算法 3.2 (冲突检测)**:

```haskell
detectConflicts :: PetriNet -> Marking -> [Transition] -> [(Transition, Transition)]
detectConflicts net marking transitions = 
  let enabled = filter (isEnabled net marking) transitions
      conflicts = [(t1, t2) | t1 <- enabled, t2 <- enabled, t1 < t2, 
                              areConflicting net marking t1 t2]
  in conflicts

areConflicting :: PetriNet -> Marking -> Transition -> Transition -> Bool
areConflicting net marking t1 t2 = 
  let preSet1 = getPreSet net t1
      preSet2 = getPreSet net t2
      sharedPlaces = Set.intersection preSet1 preSet2
  in not (Set.null sharedPlaces)
```

### 3.3 并发控制

**定义 3.3 (并发控制)**
管理并发变迁发生的策略。

**算法 3.3 (并发控制策略)**:

```haskell
concurrentControl :: PetriNet -> Marking -> [Transition] -> [Transition]
concurrentControl net marking transitions = 
  let -- 检测并发变迁
      concurrent = detectConcurrency net marking transitions
      
      -- 应用并发控制策略
      controlled = applyConcurrencyControl net marking concurrent
  in controlled

applyConcurrencyControl :: PetriNet -> Marking -> [Transition] -> [Transition]
applyConcurrencyControl net marking concurrent = 
  case concurrencyPolicy net of
    MaximalConcurrency -> concurrent
    PriorityBased priorities -> 
      sortByPriority priorities concurrent
    ResourceBased resources -> 
      filterByResources resources concurrent
```

---

## 4. 结构性质

### 4.1 有界性

**定义 4.1 (有界性)**
位置 $p \in P$ 是 $k$-有界的，如果对于所有可达标识 $M \in R(M_0)$，都有 $M(p) \leq k$。

**定义 4.2 (安全Petri网)**
Petri网是安全的，如果所有位置都是1-有界的。

**定理 4.1 (有界性判定)**
位置 $p$ 是 $k$-有界的当且仅当在状态空间中 $M(p) \leq k$ 对所有可达标识 $M$ 成立。

**算法 4.1 (有界性检查)**:

```haskell
checkBoundedness :: PetriNet -> BoundednessReport
checkBoundedness net = 
  let reachable = reachabilityAnalysis net
      boundedness = map (\place -> 
        let maxTokens = maximum (map (\marking -> marking place) reachable)
        in (place, maxTokens)) (places net)
  in BoundednessReport boundedness

isSafe :: PetriNet -> Bool
isSafe net = 
  let boundedness = checkBoundedness net
  in all (\(_, maxTokens) -> maxTokens <= 1) (boundednessReport boundedness)
```

### 4.2 活性

**定义 4.3 (活性)**
变迁 $t \in T$ 是活的，如果对于每个可达标识 $M \in R(M_0)$，都存在标识 $M' \in R(M)$ 使得 $M'[t\rangle$。

**定义 4.4 (死锁)**
标识 $M$ 是死锁，如果没有变迁在 $M$ 下使能。

**定理 4.2 (活性保持)**
如果所有变迁都是活的，则Petri网不会出现死锁。

**证明：** 通过活性定义：

1. 每个变迁在任何可达标识后都能再次使能
2. 不存在所有变迁都无法使能的标识
3. 因此不会出现死锁

**算法 4.2 (活性检查)**:

```haskell
checkLiveness :: PetriNet -> LivenessReport
checkLiveness net = 
  let reachable = reachabilityAnalysis net
      liveness = map (\transition -> 
        let isLive = all (\marking -> 
          existsReachableMarking net marking transition) reachable
        in (transition, isLive)) (transitions net)
  in LivenessReport liveness

existsReachableMarking :: PetriNet -> Marking -> Transition -> Bool
existsReachableMarking net marking transition = 
  let reachableFromMarking = reachabilityAnalysisFrom net marking
  in any (\m -> isEnabled net m transition) reachableFromMarking
```

### 4.3 可逆性

**定义 4.5 (可逆性)**
Petri网是可逆的，如果对于每个可达标识 $M \in R(M_0)$，都有 $M \rightarrow^* M_0$。

**定理 4.3 (可逆性判定)**
Petri网是可逆的当且仅当初始标识 $M_0$ 在状态空间中是强连通的。

**算法 4.3 (可逆性检查)**:

```haskell
checkReversibility :: PetriNet -> Bool
checkReversibility net = 
  let reachable = reachabilityAnalysis net
      initial = initialMarking net
      reversible = all (\marking -> 
        isReachableFrom net marking initial) reachable
  in reversible

isReachableFrom :: PetriNet -> Marking -> Marking -> Bool
isReachableFrom net target source = 
  let reachableFromSource = reachabilityAnalysisFrom net source
  in elem target reachableFromSource
```

---

## 5. 高级Petri网

### 5.1 时间Petri网

**定义 5.1 (时间Petri网)**
时间Petri网是六元组 $N = (P, T, F, M_0, I, D)$，其中：

- $(P, T, F, M_0)$ 是基础Petri网
- $I : T \rightarrow \mathbb{R}^+ \times \mathbb{R}^+$ 是时间间隔函数
- $D : T \rightarrow \mathbb{R}^+$ 是持续时间函数

**定义 5.2 (时间变迁发生)**
时间变迁 $t$ 在时间 $\tau$ 发生，如果：

1. $M[t\rangle$
2. $\tau \in I(t)$
3. 变迁持续时间为 $D(t)$

**算法 5.1 (时间Petri网模拟)**:

```haskell
data TimedPetriNet = TimedPetriNet {
  baseNet :: PetriNet,
  timeIntervals :: Map Transition (Double, Double),
  durations :: Map Transition Double
}

simulateTimedPetriNet :: TimedPetriNet -> Double -> [TimedEvent]
simulateTimedPetriNet net simulationTime = 
  let initialState = TimedState (initialMarking (baseNet net)) 0.0
      events = simulateTimedEvents net initialState simulationTime
  in events

simulateTimedEvents :: TimedPetriNet -> TimedState -> Double -> [TimedEvent]
simulateTimedEvents net currentState endTime = 
  if currentTime currentState >= endTime
  then []
  else 
    let enabled = getEnabledTransitions net currentState
        nextEvent = selectNextEvent net currentState enabled
        newState = fireTimedTransition net currentState nextEvent
    in nextEvent : simulateTimedEvents net newState endTime
```

### 5.2 着色Petri网

**定义 5.3 (着色Petri网)**
着色Petri网是五元组 $N = (P, T, F, M_0, C)$，其中：

- $(P, T, F, M_0)$ 是基础Petri网
- $C : P \cup T \rightarrow \text{Type}$ 是颜色函数

**定义 5.4 (着色标识)**
着色标识 $M : P \rightarrow \text{Multiset}(C(p))$ 表示每个位置中的有色托肯。

**算法 5.2 (着色Petri网构造)**:

```haskell
data ColoredPetriNet = ColoredPetriNet {
  baseNet :: PetriNet,
  colorFunction :: Map (Either Place Transition) ColorType,
  colorTokens :: Map Place (Multiset Color)
}

constructColoredPetriNet :: PetriNet -> ColorFunction -> ColoredPetriNet
constructColoredPetriNet baseNet colorFunc = 
  let initialColoredMarking = createColoredMarking baseNet colorFunc
  in ColoredPetriNet {
    baseNet = baseNet,
    colorFunction = colorFunc,
    colorTokens = initialColoredMarking
  }
```

### 5.3 层次Petri网

**定义 5.5 (层次Petri网)**
层次Petri网允许将子网作为变迁，形成层次结构。

**算法 5.3 (层次Petri网分析)**:

```haskell
data HierarchicalPetriNet = HierarchicalPetriNet {
  topLevel :: PetriNet,
  subnets :: Map Transition PetriNet,
  hierarchy :: HierarchyStructure
}

analyzeHierarchicalNet :: HierarchicalPetriNet -> AnalysisResult
analyzeHierarchicalNet hnet = 
  let -- 分析顶层网络
      topLevelAnalysis = analyzePetriNet (topLevel hnet)
      
      -- 分析子网络
      subnetAnalyses = map (\(transition, subnet) -> 
        analyzePetriNet subnet) (Map.toList (subnets hnet))
      
      -- 综合分析结果
      combinedAnalysis = combineAnalyses topLevelAnalysis subnetAnalyses
  in combinedAnalysis
```

---

## 6. 实际应用

### 6.1 并发系统建模

**定义 6.1 (生产者-消费者模型)**:

```haskell
data ProducerConsumer = ProducerConsumer {
  buffer :: Place,
  producer :: Transition,
  consumer :: Transition,
  produce :: Transition,
  consume :: Transition
}

modelProducerConsumer :: ProducerConsumer
modelProducerConsumer = 
  let buffer = Place "Buffer"
      producer = Transition "Producer"
      consumer = Transition "Consumer"
      produce = Transition "Produce"
      consume = Transition "Consume"
      
      places = [buffer]
      transitions = [producer, consumer, produce, consume]
      flow = [(producer, produce), (produce, buffer), 
              (buffer, consume), (consume, consumer)]
      initialMarking = Marking (\p -> if p == buffer then 0 else 0)
  in ProducerConsumer {
    buffer = buffer,
    producer = producer,
    consumer = consumer,
    produce = produce,
    consume = consume
  }
```

### 6.2 工作流建模

**定义 6.2 (工作流Petri网)**
工作流Petri网用于建模业务流程。

**算法 6.1 (工作流分析)**:

```haskell
analyzeWorkflow :: WorkflowPetriNet -> WorkflowAnalysis
analyzeWorkflow workflow = 
  let -- 可达性分析
      reachability = reachabilityAnalysis (baseNet workflow)
      
      -- 性能分析
      performance = analyzePerformance workflow
      
      -- 死锁检测
      deadlocks = detectDeadlocks workflow
      
      -- 瓶颈分析
      bottlenecks = analyzeBottlenecks workflow
  in WorkflowAnalysis {
    reachability = reachability,
    performance = performance,
    deadlocks = deadlocks,
    bottlenecks = bottlenecks
  }
```

### 6.3 协议验证

**定义 6.3 (协议Petri网)**
协议Petri网用于验证通信协议的正确性。

**算法 6.2 (协议验证)**:

```haskell
verifyProtocol :: ProtocolPetriNet -> ProtocolVerification
verifyProtocol protocol = 
  let -- 安全性验证
      safety = verifySafety protocol
      
      -- 活性验证
      liveness = verifyLiveness protocol
      
      -- 公平性验证
      fairness = verifyFairness protocol
      
      -- 一致性验证
      consistency = verifyConsistency protocol
  in ProtocolVerification {
    safety = safety,
    liveness = liveness,
    fairness = fairness,
    consistency = consistency
  }
```

---

## 7. 形式化验证

### 7.1 模型检查

**定义 7.1 (模型检查)**
使用算法验证Petri网是否满足给定的性质。

**算法 7.1 (CTL模型检查)**:

```haskell
modelCheckCTL :: PetriNet -> CTLFormula -> Bool
modelCheckCTL net formula = 
  case formula of
    Atomic prop -> checkAtomicProperty net prop
    Not phi -> not (modelCheckCTL net phi)
    And phi psi -> modelCheckCTL net phi && modelCheckCTL net psi
    Or phi psi -> modelCheckCTL net phi || modelCheckCTL net psi
    EX phi -> checkEX net phi
    AX phi -> checkAX net phi
    EF phi -> checkEF net phi
    AF phi -> checkAF net phi
    EG phi -> checkEG net phi
    AG phi -> checkAG net phi
    EU phi psi -> checkEU net phi psi
    AU phi psi -> checkAU net phi psi

checkEF :: PetriNet -> CTLFormula -> Bool
checkEF net phi = 
  let reachable = reachabilityAnalysis net
      satisfyingStates = filter (\marking -> 
        modelCheckCTL net phi) reachable
  in not (null satisfyingStates)
```

### 7.2 性质验证

**定义 7.2 (性质验证)**
验证Petri网是否满足特定的性质。

**算法 7.2 (性质验证)**:

```haskell
verifyProperties :: PetriNet -> [Property] -> VerificationResult
verifyProperties net properties = 
  let results = map (\prop -> verifyProperty net prop) properties
  in VerificationResult results

verifyProperty :: PetriNet -> Property -> PropertyResult
verifyProperty net property = 
  case property of
    Boundedness k -> verifyBoundedness net k
    Safety -> verifySafety net
    Liveness -> verifyLiveness net
    Reversibility -> verifyReversibility net
    DeadlockFreedom -> verifyDeadlockFreedom net
```

### 7.3 工具支持

**定义 7.3 (Petri网工具)**
支持Petri网分析和验证的工具。

**算法 7.3 (工具集成)**:

```haskell
integrateTools :: PetriNet -> ToolIntegration
integrateTools net = 
  let -- CPN Tools集成
      cpnTools = integrateCPNTools net
      
      -- Snoopy集成
      snoopy = integrateSnoopy net
      
      -- TINA集成
      tina = integrateTINA net
      
      -- 自定义工具
      customTools = integrateCustomTools net
  in ToolIntegration {
    cpnTools = cpnTools,
    snoopy = snoopy,
    tina = tina,
    customTools = customTools
  }
```

---

## 总结

Petri网理论基础与应用专题系统梳理了Petri网的数学基础、可达性分析、并发性分析等内容，通过形式化分析和算法实现，揭示了Petri网背后的深层原理。

通过系统化的Petri网理论体系，我们能够理解从基础Petri网到高级时间Petri网、着色Petri网的完整发展脉络，为并发系统建模和验证提供了坚实的理论基础。

这种系统化的理解不仅有助于我们更好地设计和分析并发系统，也为Petri网理论的研究和发展提供了重要的理论指导。

---

**相关链接**：

- [形式化语言理论](../../30-FormalMethods/02-FormalLanguages.md)
- [类型理论基础](../../30-FormalMethods/03-TypeTheory.md)
- [控制论理论](../../30-FormalMethods/04-ControlTheory.md)
- [数学基础理论](../../20-Mathematics/02-CoreConcepts.md)
