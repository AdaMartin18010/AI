# 06-ComputerScience 计算机科学主题导航

> 本主题为计算机科学知识体系自动递归精炼输出，所有内容严格编号、交叉引用、进度持续跟踪。

## 目录结构与知识映射

```tree
ComputerScience/
├── 01-Overview.md           # 总览
├── 02-Computability.md      # 可计算性
├── 03-Algorithms.md         # 算法
└── README.md                # 分支导航与进度梳理
```

### 知识源映射与交叉引用

- **理论与批判分析**: 对应 [Matter/FormalLanguage/CS/](../../Matter/FormalLanguage/CS/)，详见 [01-Overview.md](01-Overview.md)、[02-Computability.md](02-Computability.md)
- **科学模型与交叉**: 对应 [Matter/FormalModel/Model/Science/](../../Matter/FormalModel/Model/Science/)，详见 [01-Overview.md](01-Overview.md)
- **算法与应用**: 对应 [Matter/FormalLanguage/CS/](../../Matter/FormalLanguage/CS/)，详见 [03-Algorithms.md](03-Algorithms.md)

## 跨学科交叉引用网络

### 与数学理论的深度融合

- **可计算性理论**: [02-Computability.md](02-Computability.md) ←→ [Mathematics/Algebra/07-CategoryTheory.md](../Mathematics/Algebra/07-CategoryTheory.md) 范畴论基础
- **算法复杂度**: [03-Algorithms.md](03-Algorithms.md) ←→ [Mathematics/Probability/10-MachineLearningStats.md](../Mathematics/Probability/10-MachineLearningStats.md) 统计学习理论
- **图论算法**: [03-Algorithms.md](03-Algorithms.md) ←→ [Mathematics/Geometry/08-Topology.md](../Mathematics/Geometry/08-Topology.md) 拓扑结构
- **数值算法**: [03-Algorithms.md](03-Algorithms.md) ←→ [Mathematics/Calculus/08-FunctionalAnalysis.md](../Mathematics/Calculus/08-FunctionalAnalysis.md) 泛函分析

### 与AI系统的理论支撑

- **计算理论**: [01-Overview.md](01-Overview.md) ←→ [AI/02-MetaTheory.md](../AI/02-MetaTheory.md) AI元理论基础
- **算法设计**: [03-Algorithms.md](03-Algorithms.md) ←→ [AI/05-Model.md](../AI/05-Model.md) AI模型算法
- **复杂度分析**: [02-Computability.md](02-Computability.md) ←→ [AI/03-Theory.md](../AI/03-Theory.md) AI理论复杂度
- **分布式计算**: [01-Overview.md](01-Overview.md) ←→ [AI/06-Applications.md](../AI/06-Applications.md) 分布式AI系统

### 与形式化方法的验证融合

- **类型系统**: [01-Overview.md](01-Overview.md) ←→ [FormalMethods/03-TypeTheory.md](../FormalMethods/03-TypeTheory.md) 程序验证
- **模型检查**: [02-Computability.md](02-Computability.md) ←→ [FormalMethods/04-ModelChecking.md](../FormalMethods/04-ModelChecking.md) 算法正确性
- **Petri网**: [01-Overview.md](01-Overview.md) ←→ [FormalMethods/05-PetriNetTheory.md](../FormalMethods/05-PetriNetTheory.md) 并发计算
- **认知模型**: [02-Computability.md](02-Computability.md) ←→ [FormalMethods/06-CognitiveModel.md](../FormalMethods/06-CognitiveModel.md) 计算认知

### 与哲学思辨的理论对话

- **计算哲学**: [01-Overview.md](01-Overview.md) ←→ [Philosophy/01-Ontology.md](../Philosophy/01-Ontology.md) 计算本体论
- **算法伦理**: [03-Algorithms.md](03-Algorithms.md) ←→ [Philosophy/03-Ethics.md](../Philosophy/03-Ethics.md) 算法伦理学
- **认知计算**: [02-Computability.md](02-Computability.md) ←→ [Philosophy/02-Epistemology.md](../Philosophy/02-Epistemology.md) 计算认识论
- **逻辑基础**: [01-Overview.md](01-Overview.md) ←→ [Philosophy/04-Logic.md](../Philosophy/04-Logic.md) 计算逻辑

### 与软件工程的实践应用

- **设计模式**: [03-Algorithms.md](03-Algorithms.md) ←→ [SoftwareEngineering/DesignPattern/00-Overview.md](../SoftwareEngineering/DesignPattern/00-Overview.md) 算法模式
- **架构设计**: [01-Overview.md](01-Overview.md) ←→ [SoftwareEngineering/Architecture/00-Overview.md](../SoftwareEngineering/Architecture/00-Overview.md) 系统架构
- **微服务**: [01-Overview.md](01-Overview.md) ←→ [SoftwareEngineering/Microservices/00-Overview.md](../SoftwareEngineering/Microservices/00-Overview.md) 分布式算法
- **工作流**: [03-Algorithms.md](03-Algorithms.md) ←→ [SoftwareEngineering/Workflow-02-Modeling.md](../SoftwareEngineering/Workflow-02-Modeling.md) 流程算法

### 与编程语言的实现融合

- **语言理论**: [01-Overview.md](01-Overview.md) ←→ [ProgrammingLanguage/02-LanguageTheory.md](../ProgrammingLanguage/02-LanguageTheory.md) 计算语言
- **编程范式**: [03-Algorithms.md](03-Algorithms.md) ←→ [ProgrammingLanguage/03-ProgrammingParadigms.md](../ProgrammingLanguage/03-ProgrammingParadigms.md) 算法范式
- **异步编程**: [02-Computability.md](02-Computability.md) ←→ [ProgrammingLanguage/06-AsyncProgramming.md](../ProgrammingLanguage/06-AsyncProgramming.md) 异步计算
- **并发模型**: [01-Overview.md](01-Overview.md) ←→ [ProgrammingLanguage/06-AsyncProgramming.md](../ProgrammingLanguage/06-AsyncProgramming.md) 异步编程

## 进度与自动化追踪

- 本分支已纳入自动化递归修订与进度跟踪体系，所有核心文档均已与 Matter 知识库建立映射与交叉引用。
- 进度与质量标准详见 [../PROGRESS.md](../PROGRESS.md)
- 后续修订、批判、整合、表达规范等均按统一标准自动推进。

## 交叉引用与进度标记

- [01-Overview.md](01-Overview.md) ←→ [Matter/FormalLanguage/CS/](../../Matter/FormalLanguage/CS/)、[Matter/FormalModel/Model/Science/](../../Matter/FormalModel/Model/Science/)
- [02-Computability.md](02-Computability.md) ←→ [Matter/FormalLanguage/CS/](../../Matter/FormalLanguage/CS/)
- [03-Algorithms.md](03-Algorithms.md) ←→ [Matter/FormalLanguage/CS/](../../Matter/FormalLanguage/CS/)

---

> 本README及各子文档将持续自动递归梳理、交叉引用、进度跟踪与表达规范化，所有修订与进展均记录于 [../PROGRESS.md](../PROGRESS.md)。
