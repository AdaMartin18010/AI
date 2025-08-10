# 04-SoftwareEngineering 软件工程主题导航

> 本主题为软件工程知识体系自动递归精炼输出，所有内容严格编号、交叉引用、进度持续跟踪。

## 目录结构与知识映射

```tree
SoftwareEngineering/
├── Architecture/00-Overview.md         # 架构
├── DesignPattern/00-Overview.md        # 设计模式
├── Microservices/00-Overview.md        # 微服务
├── Workflow-01-Basics.md               # 工作流基础
├── Workflow-02-Modeling.md             # 工作流建模
├── Workflow-03-Engine.md               # 工作流引擎
├── Workflow-04-Relation.md             # 工作流关系
├── Workflow-05-Application.md          # 工作流应用
├── Workflow-06-Prospect.md             # 工作流展望
├── Workflow.md                         # 工作流总览
└── README.md                           # 分支导航与进度梳理
```

### 知识源映射与交叉引用

- **设计模式**: 对应 [Matter/Software/DesignPattern/](../../Matter/Software/DesignPattern/)，详见 [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md)
- **微服务**: 对应 [Matter/Software/Microservice/](../../Matter/Software/Microservice/)，详见 [Microservices/00-Overview.md](Microservices/00-Overview.md)
- **工作流**: 对应 [Matter/Software/WorkflowDomain/](../../Matter/Software/WorkflowDomain/)、[Matter/FormalModel/Software/WorkFlow/](../../Matter/FormalModel/Software/WorkFlow/)，详见 [Workflow-01-Basics.md](Workflow-01-Basics.md) 等
- **架构**: 对应 [Matter/Software/Architecture/](../../Matter/Software/Architecture/)（如有）

## 跨学科交叉引用网络

### 与形式化方法的验证融合

- **设计模式验证**: [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md) ←→ [FormalMethods/02-FormalLanguages.md](../FormalMethods/02-FormalLanguages.md) 规范建模
- **微服务架构**: [Microservices/00-Overview.md](Microservices/00-Overview.md) ←→ [FormalMethods/04-ModelChecking.md](../FormalMethods/04-ModelChecking.md) 服务验证
- **工作流建模**: [Workflow-02-Modeling.md](Workflow-02-Modeling.md) ←→ [FormalMethods/05-PetriNetTheory.md](../FormalMethods/05-PetriNetTheory.md) Petri网建模
- **分布式系统**: [Architecture/00-Overview.md](Architecture/00-Overview.md) ←→ [FormalMethods/07-DistributedModel.md](../FormalMethods/07-DistributedModel.md) 分布式验证

### 与数学理论的基础支撑

- **算法复杂度**: [Architecture/00-Overview.md](Architecture/00-Overview.md) ←→ [Mathematics/Probability/10-MachineLearningStats.md](../Mathematics/Probability/10-MachineLearningStats.md) 性能分析
- **群论对称**: [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md) ←→ [Mathematics/Algebra/02-Groups.md](../Mathematics/Algebra/02-Groups.md) 结构模式
- **图论优化**: [Microservices/00-Overview.md](Microservices/00-Overview.md) ←→ [Mathematics/Geometry/08-Topology.md](../Mathematics/Geometry/08-Topology.md) 拓扑设计
- **微分方程**: [Workflow-03-Engine.md](Workflow-03-Engine.md) ←→ [Mathematics/Calculus/07-DifferentialEquations.md](../Mathematics/Calculus/07-DifferentialEquations.md) 动态系统

### 与AI系统的智能化整合

- **AI架构模式**: [Architecture/00-Overview.md](Architecture/00-Overview.md) ←→ [AI/04-DesignPattern.md](../AI/04-DesignPattern.md) 智能架构
- **智能微服务**: [Microservices/00-Overview.md](Microservices/00-Overview.md) ←→ [AI/06-Applications.md](../AI/06-Applications.md) 分布式AI
- **AI工作流**: [Workflow-05-Application.md](Workflow-05-Application.md) ←→ [AI/05-Model.md](../AI/05-Model.md) AI流程自动化
- **认知架构**: [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md) ←→ [AI/04-MetaModel.md](../AI/04-MetaModel.md) 认知模式

### 与哲学理念的价值对齐

- **软件伦理**: [Architecture/00-Overview.md](Architecture/00-Overview.md) ←→ [Philosophy/03-Ethics.md](../Philosophy/03-Ethics.md) 技术伦理
- **系统本体**: [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md) ←→ [Philosophy/01-Ontology.md](../Philosophy/01-Ontology.md) 设计本体论
- **工程认识论**: [Workflow-01-Basics.md](Workflow-01-Basics.md) ←→ [Philosophy/02-Epistemology.md](../Philosophy/02-Epistemology.md) 过程认知
- **技术哲学**: [Workflow-06-Prospect.md](Workflow-06-Prospect.md) ←→ [Philosophy/06-Interdisciplinary.md](../Philosophy/06-Interdisciplinary.md) 技术哲学

## 进度与自动化追踪

- 本分支已纳入自动化递归修订与进度跟踪体系，所有核心文档均已与 Matter 知识库建立映射与交叉引用。
- 进度与质量标准详见 [../PROGRESS.md](../PROGRESS.md)
- 后续修订、批判、整合、表达规范等均按统一标准自动推进。

## 交叉引用与进度标记

- [DesignPattern/00-Overview.md](DesignPattern/00-Overview.md) ←→ [Matter/Software/DesignPattern/](../../Matter/Software/DesignPattern/)
- [Microservices/00-Overview.md](Microservices/00-Overview.md) ←→ [Matter/Software/Microservice/](../../Matter/Software/Microservice/)
- [Workflow-01-Basics.md](Workflow-01-Basics.md) 等 ←→ [Matter/Software/WorkflowDomain/](../../Matter/Software/WorkflowDomain/)、[Matter/FormalModel/Software/WorkFlow/](../../Matter/FormalModel/Software/WorkFlow/)
- 架构 ←→ [Matter/Software/Architecture/](../../Matter/Software/Architecture/)（如有）

---

> 本README及各子文档将持续自动递归梳理、交叉引用、进度跟踪与表达规范化，所有修订与进展均记录于 [../PROGRESS.md](../PROGRESS.md)。
