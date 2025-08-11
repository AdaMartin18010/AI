# 控制论理论基础与高级理论

## 目录

- [控制论理论基础与高级理论](#控制论理论基础与高级理论)
  - [目录](#目录)
  - [1. 控制系统基础架构](#1-控制系统基础架构)
    - [1.1 系统分类与层次结构](#11-系统分类与层次结构)
    - [1.2 状态空间表示](#12-状态空间表示)
    - [1.3 系统线性化](#13-系统线性化)
  - [2. 高级稳定性理论](#2-高级稳定性理论)
    - [2.1 李雅普诺夫稳定性深化](#21-李雅普诺夫稳定性深化)
    - [2.2 输入输出稳定性](#22-输入输出稳定性)
    - [2.3 鲁棒稳定性](#23-鲁棒稳定性)
  - [3. 控制算法与设计](#3-控制算法与设计)
    - [3.1 线性控制设计](#31-线性控制设计)
    - [3.2 非线性控制方法](#32-非线性控制方法)
    - [3.3 自适应控制](#33-自适应控制)
  - [4. 最优控制理论](#4-最优控制理论)
    - [4.1 变分法基础](#41-变分法基础)
    - [4.2 动态规划](#42-动态规划)
    - [4.3 哈密顿-雅可比方程](#43-哈密顿-雅可比方程)
  - [5. 鲁棒控制理论](#5-鲁棒控制理论)
    - [5.1 H∞控制](#51-h控制)
    - [5.2 μ综合](#52-μ综合)
    - [5.3 不确定性建模](#53-不确定性建模)
  - [6. 分布式控制系统](#6-分布式控制系统)
    - [6.1 多智能体系统](#61-多智能体系统)
    - [6.2 网络化控制](#62-网络化控制)
    - [6.3 协同控制](#63-协同控制)
  - [7. 应用与前沿发展](#7-应用与前沿发展)
    - [7.1 机器人控制](#71-机器人控制)
    - [7.2 航空航天控制](#72-航空航天控制)
    - [7.3 智能控制系统](#73-智能控制系统)
  - [总结](#总结)

---

## 1. 控制系统基础架构

### 1.1 系统分类与层次结构

**定义 1.1 (系统分类)**
控制系统按特性分类：

1. **线性系统**：满足叠加原理
2. **非线性系统**：不满足叠加原理
3. **时变系统**：参数随时间变化
4. **时不变系统**：参数不随时间变化
5. **连续时间系统**：状态连续变化
6. **离散时间系统**：状态离散变化

**定义 1.2 (系统层次)**
控制系统按复杂度分层：

- **单输入单输出(SISO)**：$\mathbb{R} \rightarrow \mathbb{R}$
- **多输入多输出(MIMO)**：$\mathbb{R}^m \rightarrow \mathbb{R}^p$
- **分布式系统**：多个子系统协同
- **网络化系统**：通过网络连接

**定理 1.1 (系统分解)**
任何复杂系统都可以分解为基本子系统的组合。

**证明：** 通过结构分解：

1. 将系统分解为可控和不可控部分
2. 将可控部分分解为可观和不可观部分
3. 每个部分都可以独立分析和设计

### 1.2 状态空间表示

**定义 1.3 (广义状态空间)**
广义状态空间表示：

```math
\dot{x}(t) = f(x(t), u(t), t)
y(t) = h(x(t), u(t), t)
```

其中 $x(t) \in \mathbb{R}^n$, $u(t) \in \mathbb{R}^m$, $y(t) \in \mathbb{R}^p$。

**定义 1.4 (线性化)**
非线性系统在平衡点 $(x_e, u_e)$ 附近的线性化：

```math
\delta \dot{x}(t) = A \delta x(t) + B \delta u(t)
\delta y(t) = C \delta x(t) + D \delta u(t)
```

其中：

```math
A = \frac{\partial f}{\partial x}\bigg|_{(x_e, u_e)}, \quad B = \frac{\partial f}{\partial u}\bigg|_{(x_e, u_e)}
C = \frac{\partial h}{\partial x}\bigg|_{(x_e, u_e)}, \quad D = \frac{\partial h}{\partial u}\bigg|_{(x_e, u_e)}
```

**算法 1.1 (系统线性化)**:

```haskell
data NonlinearSystem = NonlinearSystem {
  stateDimension :: Int,
  inputDimension :: Int,
  outputDimension :: Int,
  stateFunction :: Vector Double -> Vector Double -> Double -> Vector Double,
  outputFunction :: Vector Double -> Vector Double -> Double -> Vector Double
}

linearizeSystem :: NonlinearSystem -> Vector Double -> Vector Double -> LinearSystem
linearizeSystem sys xEquilibrium uEquilibrium = 
  let -- 计算雅可比矩阵
      aMatrix = computeJacobian (stateFunction sys) xEquilibrium uEquilibrium 0.0
      bMatrix = computeJacobian (stateFunction sys) xEquilibrium uEquilibrium 0.0
      cMatrix = computeJacobian (outputFunction sys) xEquilibrium uEquilibrium 0.0
      dMatrix = computeJacobian (outputFunction sys) xEquilibrium uEquilibrium 0.0
  in LinearSystem {
    a = aMatrix,
    b = bMatrix,
    c = cMatrix,
    d = dMatrix
  }

computeJacobian :: (Vector Double -> Vector Double -> Double -> Vector Double) 
                -> Vector Double -> Vector Double -> Double -> Matrix Double
computeJacobian f x u t = 
  let n = length x
      epsilon = 1e-8
      jacobian = matrix n n (\(i, j) -> 
        let xPlus = x + (unitVector n j * epsilon)
            xMinus = x - (unitVector n j * epsilon)
            derivative = (f xPlus u t - f xMinus u t) / (2 * epsilon)
        in derivative `atIndex` i)
  in jacobian
```

### 1.3 系统线性化

**定义 1.5 (线性化条件)**
系统在平衡点附近可线性化的条件：

1. 系统在平衡点连续可微
2. 平衡点附近系统行为主要由线性项决定
3. 非线性项相对于线性项为高阶小量

**算法 1.2 (自动线性化)**:

```haskell
autoLinearize :: NonlinearSystem -> [LinearSystem]
autoLinearize sys = 
  let -- 找到所有平衡点
      equilibria = findEquilibria sys
      
      -- 在每个平衡点进行线性化
      linearized = map (\eq -> linearizeSystem sys (fst eq) (snd eq)) equilibria
  in linearized

findEquilibria :: NonlinearSystem -> [(Vector Double, Vector Double)]
findEquilibria sys = 
  let -- 使用数值方法找到平衡点
      initialGuesses = generateInitialGuesses sys
      equilibria = map (findEquilibrium sys) initialGuesses
  in filter isValidEquilibrium equilibria
```

---

## 2. 高级稳定性理论

### 2.1 李雅普诺夫稳定性深化

**定义 2.1 (李雅普诺夫函数)**
函数 $V : \mathbb{R}^n \rightarrow \mathbb{R}$ 是系统 $\dot{x} = f(x)$ 的李雅普诺夫函数，如果：

1. $V(0) = 0$
2. $V(x) > 0$ 对于 $x \neq 0$
3. $\dot{V}(x) = \nabla V(x)^T f(x) \leq 0$ 对于 $x \neq 0$

**定义 2.2 (全局渐近稳定性)**
平衡点 $x_e = 0$ 是全局渐近稳定的，如果：

1. 它是李雅普诺夫稳定的
2. $\lim_{t \rightarrow \infty} x(t) = 0$ 对于所有初始条件

**定理 2.1 (全局渐近稳定性判据)**
如果存在径向无界的李雅普诺夫函数 $V(x)$ 使得 $\dot{V}(x) < 0$ 对于 $x \neq 0$，则平衡点是全局渐近稳定的。

**证明：** 通过李雅普诺夫直接法：

1. 径向无界性确保所有轨迹有界
2. $\dot{V}(x) < 0$ 确保 $V(x)$ 严格递减
3. 结合李雅普诺夫稳定性得到全局渐近稳定性

**算法 2.1 (李雅普诺夫函数构造)**:

```haskell
data LyapunovFunction = LyapunovFunction {
  function :: Vector Double -> Double,
  gradient :: Vector Double -> Vector Double
}

constructLyapunovFunction :: Matrix Double -> LyapunovFunction
constructLyapunovFunction aMatrix = 
  let -- 求解李雅普诺夫方程 A^T P + P A = -Q
      qMatrix = identity (rows aMatrix)
      pMatrix = solveLyapunovEquation aMatrix qMatrix
      
      -- 构造二次型李雅普诺夫函数
      lyapunovFunc x = x `dot` (pMatrix `multiply` x)
      lyapunovGrad x = 2 * (pMatrix `multiply` x)
  in LyapunovFunction {
    function = lyapunovFunc,
    gradient = lyapunovGrad
  }

solveLyapunovEquation :: Matrix Double -> Matrix Double -> Matrix Double
solveLyapunovEquation a q = 
  let n = rows a
      -- 将李雅普诺夫方程转换为线性系统
      vecP = solve (kroneckerProduct (transpose a) (identity n) + 
                   kroneckerProduct (identity n) a) (vectorize q)
  in reshape n n vecP
```

### 2.2 输入输出稳定性

**定义 2.3 (L2稳定性)**
系统是L2稳定的，如果存在常数 $\gamma > 0$ 使得：

```math
\|y\|_2 \leq \gamma \|u\|_2
```

对于所有 $u \in L_2$。

**定义 2.4 (L∞稳定性)**
系统是L∞稳定的，如果存在常数 $\gamma > 0$ 使得：

```math
\|y\|_\infty \leq \gamma \|u\|_\infty
```

对于所有 $u \in L_\infty$。

**定理 2.2 (小增益定理)**
如果两个L2稳定系统 $G_1$ 和 $G_2$ 的增益分别为 $\gamma_1$ 和 $\gamma_2$，且 $\gamma_1 \gamma_2 < 1$，则反馈连接系统是L2稳定的。

**证明：** 通过增益分析：

1. 反馈系统的输入输出关系
2. 利用三角不等式
3. 应用小增益条件

**算法 2.2 (L2增益计算)**:

```haskell
computeL2Gain :: LinearSystem -> Double
computeL2Gain sys = 
  let -- 计算H∞范数
      hInfinityNorm = computeHInfinityNorm sys
  in hInfinityNorm

computeHInfinityNorm :: LinearSystem -> Double
computeHInfinityNorm sys = 
  let -- 通过求解Riccati方程计算H∞范数
      gamma = binarySearch 0.0 1000.0 (\g -> 
        let riccatiSolution = solveHInfinityRiccati sys g
        in isPositiveDefinite riccatiSolution)
  in gamma

solveHInfinityRiccati :: LinearSystem -> Double -> Matrix Double
solveHInfinityRiccati sys gamma = 
  let a = aMatrix sys
      b = bMatrix sys
      c = cMatrix sys
      d = dMatrix sys
      
      -- H∞ Riccati方程
      riccatiMatrix = a `multiply` x + x `multiply` (transpose a) +
                     x `multiply` ((1/gamma^2) `scale` (b `multiply` (transpose b))) `multiply` x +
                     c `multiply` (transpose c)
  in solveRiccatiEquation riccatiMatrix
```

### 2.3 鲁棒稳定性

**定义 2.5 (鲁棒稳定性)**
系统在不确定性存在下保持稳定性的能力。

**定义 2.6 (结构不确定性)**
系统参数的不确定性可以用结构化模型表示：

```math
\dot{x} = (A + \Delta A)x + (B + \Delta B)u
```

其中 $\Delta A$ 和 $\Delta B$ 是未知但有界的扰动。

**定理 2.3 (鲁棒稳定性判据)**
如果系统在标称参数下稳定，且不确定性满足小增益条件，则系统鲁棒稳定。

---

## 3. 控制算法与设计

### 3.1 线性控制设计

**定义 3.1 (状态反馈控制)**
状态反馈控制律：

```math
u(t) = -Kx(t)
```

其中 $K$ 是反馈增益矩阵。

**算法 3.1 (极点配置)**:

```haskell
polePlacement :: LinearSystem -> [Complex Double] -> Matrix Double
polePlacement sys desiredPoles = 
  let a = aMatrix sys
      b = bMatrix sys
      
      -- 检查可控性
      controllable = isControllable a b
      
      if controllable
      then
        -- 使用Ackermann公式
        let n = rows a
            phi = characteristicPolynomial desiredPoles
            k = computeAckermannGain a b phi
        in k
      else error "System not controllable"

computeAckermannGain :: Matrix Double -> Matrix Double -> [Double] -> Matrix Double
computeAckermannGain a b coefficients = 
  let n = rows a
      -- 计算可控性矩阵
      controllabilityMatrix = buildControllabilityMatrix a b
      
      -- 计算特征多项式
      phiA = evaluatePolynomial coefficients a
      
      -- Ackermann公式
      k = (lastRow controllabilityMatrix) `multiply` (inverse controllabilityMatrix) `multiply` phiA
  in k
```

### 3.2 非线性控制方法

**定义 3.2 (反馈线性化)**
通过状态反馈将非线性系统转换为线性系统。

**算法 3.2 (反馈线性化)**:

```haskell
feedbackLinearization :: NonlinearSystem -> LinearSystem
feedbackLinearization sys = 
  let -- 计算相对度
      relativeDegree = computeRelativeDegree sys
      
      -- 构造坐标变换
      transformation = constructTransformation sys relativeDegree
      
      -- 计算反馈律
      feedbackLaw = computeFeedbackLaw sys transformation
      
      -- 应用反馈线性化
      linearized = applyFeedbackLinearization sys feedbackLaw
  in linearized

computeRelativeDegree :: NonlinearSystem -> Int
computeRelativeDegree sys = 
  let -- 计算Lie导数直到输出导数包含输入
      lieDerivatives = computeLieDerivatives sys
      relativeDegree = findRelativeDegree lieDerivatives
  in relativeDegree
```

### 3.3 自适应控制

**定义 3.3 (自适应控制)**
控制器参数根据系统行为自动调整的控制方法。

**算法 3.3 (模型参考自适应控制)**:

```haskell
data AdaptiveController = AdaptiveController {
  referenceModel :: LinearSystem,
  adaptationLaw :: Vector Double -> Vector Double -> Vector Double,
  controllerGains :: Vector Double
}

modelReferenceAdaptiveControl :: AdaptiveController -> NonlinearSystem -> ControlSignal
modelReferenceAdaptiveControl controller plant = 
  let -- 计算跟踪误差
      trackingError = computeTrackingError controller plant
      
      -- 更新控制器参数
      updatedGains = adaptationLaw controller trackingError (controllerGains controller)
      
      -- 生成控制信号
      controlSignal = generateControlSignal updatedGains plant
  in controlSignal

computeTrackingError :: AdaptiveController -> NonlinearSystem -> Vector Double
computeTrackingError controller plant = 
  let -- 参考模型输出
      referenceOutput = simulateSystem (referenceModel controller)
      
      -- 实际系统输出
      actualOutput = simulateSystem plant
      
      -- 计算误差
      error = referenceOutput - actualOutput
  in error
```

---

## 4. 最优控制理论

### 4.1 变分法基础

**定义 4.1 (最优控制问题)**
最小化性能指标：

```math
J = \int_{t_0}^{t_f} L(x(t), u(t), t) dt + \phi(x(t_f))
```

**定义 4.2 (哈密顿函数)**
哈密顿函数：

```math
H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u, t)
```

**定理 4.1 (最优性必要条件)**
如果 $u^*(t)$ 是最优控制，则存在协态变量 $\lambda^*(t)$ 使得：

1. 状态方程：$\dot{x} = \frac{\partial H}{\partial \lambda}$
2. 协态方程：$\dot{\lambda} = -\frac{\partial H}{\partial x}$
3. 控制方程：$\frac{\partial H}{\partial u} = 0$

### 4.2 动态规划

**定义 4.3 (值函数)**
值函数 $V(x, t)$ 表示从状态 $x$ 和时间 $t$ 开始的最优成本。

**贝尔曼方程**：

```math
V(x, t) = \min_{u} \left\{ L(x, u, t) + V(f(x, u, t), t + \Delta t) \right\}
```

**算法 4.1 (动态规划求解)**:

```haskell
solveDynamicProgramming :: OptimalControlProblem -> ValueFunction
solveDynamicProgramming problem = 
  let -- 离散化状态空间
      discretizedStates = discretizeStateSpace problem
      
      -- 初始化值函数
      initialValueFunction = initializeValueFunction discretizedStates
      
      -- 迭代求解
      finalValueFunction = iterateValueIteration problem initialValueFunction
  in finalValueFunction

valueIteration :: OptimalControlProblem -> ValueFunction -> ValueFunction
valueIteration problem currentValue = 
  let -- 对每个状态计算最优值
      updatedValues = map (\state -> 
        let -- 对每个控制输入计算成本
            costs = map (\control -> 
              let nextState = systemDynamics problem state control
                  immediateCost = immediateCostFunction problem state control
                  futureCost = currentValue nextState
              in immediateCost + futureCost) (controlSpace problem)
            
            -- 选择最小成本
            optimalCost = minimum costs
        in optimalCost) (stateSpace problem)
  in ValueFunction updatedValues
```

### 4.3 哈密顿-雅可比方程

**定义 4.4 (哈密顿-雅可比方程)**
哈密顿-雅可比方程：

```math
\frac{\partial V}{\partial t} + \min_{u} H(x, u, \frac{\partial V}{\partial x}, t) = 0
```

**算法 4.2 (哈密顿-雅可比求解)**:

```haskell
solveHamiltonJacobi :: OptimalControlProblem -> ValueFunction
solveHamiltonJacobi problem = 
  let -- 使用有限差分方法
      grid = createGrid problem
      
      -- 初始化边界条件
      boundaryConditions = setBoundaryConditions problem
      
      -- 迭代求解
      solution = solvePDE grid boundaryConditions hamiltonJacobiOperator
  in solution

hamiltonJacobiOperator :: Grid -> ValueFunction -> ValueFunction
hamiltonJacobiOperator grid valueFunction = 
  let -- 计算空间导数
      spatialDerivatives = computeSpatialDerivatives grid valueFunction
      
      -- 计算时间导数
      timeDerivative = computeTimeDerivative grid valueFunction
      
      -- 应用哈密顿-雅可比算子
      updatedValue = applyHamiltonJacobiOperator spatialDerivatives timeDerivative
  in updatedValue
```

---

## 5. 鲁棒控制理论

### 5.1 H∞控制

**定义 5.1 (H∞控制问题)**
设计控制器使得闭环系统的H∞范数小于给定阈值。

**定义 5.2 (H∞性能指标)**
性能指标：

```math
\|T_{zw}\|_\infty < \gamma
```

其中 $T_{zw}$ 是从干扰 $w$ 到性能输出 $z$ 的传递函数。

**算法 5.1 (H∞控制器设计)**:

```haskell
designHInfinityController :: LinearSystem -> Double -> Controller
designHInfinityController sys gamma = 
  let -- 构造广义对象
      generalizedPlant = constructGeneralizedPlant sys
      
      -- 求解H∞ Riccati方程
      riccatiSolution = solveHInfinityRiccati generalizedPlant gamma
      
      -- 构造控制器
      controller = constructController riccatiSolution
  in controller

solveHInfinityRiccati :: GeneralizedPlant -> Double -> RiccatiSolution
solveHInfinityRiccati plant gamma = 
  let -- 构造Riccati方程
      riccatiEquation = constructRiccatiEquation plant gamma
      
      -- 求解方程
      solution = solveRiccatiEquation riccatiEquation
  in solution
```

### 5.2 μ综合

**定义 5.3 (μ综合)**
μ综合是一种处理结构不确定性的鲁棒控制方法。

**定义 5.4 (μ函数)**
μ函数：

```math
\mu(M) = \frac{1}{\min\{\sigma(\Delta) : \det(I - M\Delta) = 0\}}
```

**算法 5.2 (μ综合设计)**:

```haskell
muSynthesis :: UncertainSystem -> Controller
muSynthesis uncertainSys = 
  let -- 构造不确定性模型
      uncertaintyModel = constructUncertaintyModel uncertainSys
      
      -- 迭代μ综合
      controller = iterateMuSynthesis uncertaintyModel
  in controller

iterateMuSynthesis :: UncertaintyModel -> Controller
iterateMuSynthesis model = 
  let -- D-K迭代
      (dScaling, controller) = dkIteration model
      
      -- 检查收敛性
      if isConverged dScaling
      then controller
      else iterateMuSynthesis (updateModel model controller)
```

### 5.3 不确定性建模

**定义 5.5 (不确定性类型)**:

- **参数不确定性**：系统参数的不确定性
- **非结构不确定性**：未建模动态
- **结构不确定性**：已知结构但未知参数

**算法 5.3 (不确定性建模)**:

```haskell
modelUncertainty :: System -> UncertaintyModel
modelUncertainty sys = 
  let -- 参数不确定性
      parametricUncertainty = modelParametricUncertainty sys
      
      -- 非结构不确定性
      unstructuredUncertainty = modelUnstructuredUncertainty sys
      
      -- 结构不确定性
      structuredUncertainty = modelStructuredUncertainty sys
  in UncertaintyModel {
    parametric = parametricUncertainty,
    unstructured = unstructuredUncertainty,
    structured = structuredUncertainty
  }
```

---

## 6. 分布式控制系统

### 6.1 多智能体系统

**定义 6.1 (多智能体系统)**
多智能体系统由多个相互作用的智能体组成。

**定义 6.2 (一致性)**
多智能体系统达到一致性，如果：

```math
\lim_{t \rightarrow \infty} \|x_i(t) - x_j(t)\| = 0, \quad \forall i, j
```

**算法 6.1 (一致性控制)**:

```haskell
consensusControl :: MultiAgentSystem -> ControlLaw
consensusControl mas = 
  let -- 构造拉普拉斯矩阵
      laplacianMatrix = constructLaplacianMatrix mas
      
      -- 设计一致性控制器
      controller = designConsensusController laplacianMatrix
  in controller

designConsensusController :: Matrix Double -> ControlLaw
designConsensusController laplacian = 
  let -- 使用拉普拉斯矩阵设计控制律
      controlLaw agentId state = 
        let neighbors = getNeighbors agentId
            neighborStates = map (\n -> getState n) neighbors
            consensusError = sum (map (\s -> s - state) neighborStates)
        in -k * consensusError
  in controlLaw
```

### 6.2 网络化控制

**定义 6.3 (网络化控制系统)**
通过网络连接的控制系统。

**定义 6.4 (网络诱导延迟)**
网络传输引起的延迟。

**算法 6.2 (网络化控制设计)**:

```haskell
networkedControlDesign :: NetworkedSystem -> Controller
networkedControlDesign netSys = 
  let -- 建模网络延迟
      delayModel = modelNetworkDelay netSys
      
      -- 设计鲁棒控制器
      robustController = designRobustController delayModel
  in robustController
```

### 6.3 协同控制

**定义 6.5 (协同控制)**
多个系统协同完成复杂任务的控制方法。

**算法 6.3 (协同控制算法)**:

```haskell
cooperativeControl :: CooperativeSystem -> ControlStrategy
cooperativeControl coopSys = 
  let -- 任务分解
      subtasks = decomposeTask coopSys
      
      -- 分配子任务
      taskAssignment = assignSubtasks subtasks
      
      -- 设计协同控制器
      cooperativeController = designCooperativeController taskAssignment
  in cooperativeController
```

---

## 7. 应用与前沿发展

### 7.1 机器人控制

**机器人动力学控制**：

- 逆动力学控制
- 自适应控制
- 鲁棒控制

**算法 7.1 (机器人控制)**:

```haskell
robotControl :: RobotSystem -> Trajectory -> ControlLaw
robotControl robot trajectory = 
  let -- 计算期望轨迹
      desiredTrajectory = computeDesiredTrajectory trajectory
      
      -- 设计跟踪控制器
      trackingController = designTrackingController robot desiredTrajectory
  in trackingController
```

### 7.2 航空航天控制

**飞行控制系统**：

- 姿态控制
- 轨迹跟踪
- 鲁棒控制

### 7.3 智能控制系统

**智能控制方法**：

- 模糊控制
- 神经网络控制
- 遗传算法控制

---

## 总结

控制论理论基础与高级理论专题系统梳理了控制系统的数学基础、稳定性理论、控制算法等内容，通过形式化分析和算法实现，揭示了控制系统背后的深层原理。

通过系统化的控制理论体系，我们能够理解从基础线性控制到高级鲁棒控制的完整发展脉络，为现代控制系统设计提供了坚实的理论基础。

这种系统化的理解不仅有助于我们更好地设计和实现控制系统，也为控制理论的研究和发展提供了重要的理论指导。

---

**相关链接**：

- [形式化语言理论](../../FormalMethods/02-FormalLanguages.md)
- [类型理论基础](../../FormalMethods/03-TypeTheory.md)
- [数学基础理论](../../Mathematics/02-CoreConcepts.md)
- [AI理论基础](../../AI/03-Theory.md)
