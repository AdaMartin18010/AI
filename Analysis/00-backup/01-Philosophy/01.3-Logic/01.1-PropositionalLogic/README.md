# 命题逻辑理论 (Propositional Logic Theory)

## 概述 (Overview)

命题逻辑是逻辑学的基础分支，研究由原子命题通过逻辑联结词构成的复合命题的推理规律。它是现代逻辑学的起点，为更复杂的逻辑系统提供了基础框架。

## 目录结构 (Table of Contents)

1. [语法理论 (Syntax Theory)](#语法理论-syntax-theory)
2. [语义理论 (Semantic Theory)](#语义理论-semantic-theory)
3. [公理化系统 (Axiomatic System)](#公理化系统-axiomatic-system)
4. [自然演绎系统 (Natural Deduction System)](#自然演绎系统-natural-deduction-system)
5. [表列演算 (Tableau Calculus)](#表列演算-tableau-calculus)
6. [可满足性问题 (Satisfiability Problem)](#可满足性问题-satisfiability-problem)
7. [AI应用 (AI Applications)](#ai应用-ai-applications)
8. [技术实现 (Technical Implementation)](#技术实现-technical-implementation)

## 语法理论 (Syntax Theory)

### 1.1 命题语言 (Propositional Language)

**定义 1.1.1** (命题语言)
命题语言 $\mathcal{L}_P = (P, \mathcal{F}, \mathcal{F}_0)$ 包含：

- $P$: 命题变元集合 (Propositional Variables)
- $\mathcal{F}$: 公式集合 (Formulas)
- $\mathcal{F}_0$: 原子公式集合 (Atomic Formulas)

**定义 1.1.2** (公式归纳定义)
公式集合 $\mathcal{F}$ 是满足以下条件的最小集合：

1. **基础**: $P \subseteq \mathcal{F}$ (原子命题是公式)
2. **归纳**:
   - 如果 $\phi \in \mathcal{F}$，则 $\neg\phi \in \mathcal{F}$ (否定)
   - 如果 $\phi, \psi \in \mathcal{F}$，则 $(\phi \land \psi) \in \mathcal{F}$ (合取)
   - 如果 $\phi, \psi \in \mathcal{F}$，则 $(\phi \lor \psi) \in \mathcal{F}$ (析取)
   - 如果 $\phi, \psi \in \mathcal{F}$，则 $(\phi \rightarrow \psi) \in \mathcal{F}$ (蕴含)
   - 如果 $\phi, \psi \in \mathcal{F}$，则 $(\phi \leftrightarrow \psi) \in \mathcal{F}$ (等价)
3. **闭包**: 只有通过上述规则构造的表达式才是公式

### 1.2 语法结构 (Syntactic Structure)

**定义 1.1.3** (公式复杂度)
公式 $\phi$ 的复杂度 $|\phi|$ 定义为：

- $|p| = 1$ (原子命题)
- $|\neg\phi| = |\phi| + 1$
- $|(\phi \circ \psi)| = |\phi| + |\psi| + 1$ (其中 $\circ \in \{\land, \lor, \rightarrow, \leftrightarrow\}$)

**定义 1.1.4** (子公式)
公式 $\phi$ 的子公式集合 $Sub(\phi)$ 定义为：

- $Sub(p) = \{p\}$ (原子命题)
- $Sub(\neg\phi) = \{\neg\phi\} \cup Sub(\phi)$
- $Sub((\phi \circ \psi)) = \{(\phi \circ \psi)\} \cup Sub(\phi) \cup Sub(\psi)$

## 语义理论 (Semantic Theory)

### 2.1 真值赋值 (Truth Assignment)

**定义 2.1.1** (真值赋值)
真值赋值是从命题变元到真值集合 $\{T, F\}$ 的函数：
$\sigma: P \rightarrow \{T, F\}$

**定义 2.1.2** (真值函数)
真值函数 $\overline{\sigma}: \mathcal{F} \rightarrow \{T, F\}$ 定义为：

1. $\overline{\sigma}(p) = \sigma(p)$ (原子命题)
2. $\overline{\sigma}(\neg\phi) = T$ 当且仅当 $\overline{\sigma}(\phi) = F$
3. $\overline{\sigma}(\phi \land \psi) = T$ 当且仅当 $\overline{\sigma}(\phi) = T$ 且 $\overline{\sigma}(\psi) = T$
4. $\overline{\sigma}(\phi \lor \psi) = T$ 当且仅当 $\overline{\sigma}(\phi) = T$ 或 $\overline{\sigma}(\psi) = T$
5. $\overline{\sigma}(\phi \rightarrow \psi) = T$ 当且仅当 $\overline{\sigma}(\phi) = F$ 或 $\overline{\sigma}(\psi) = T$
6. $\overline{\sigma}(\phi \leftrightarrow \psi) = T$ 当且仅当 $\overline{\sigma}(\phi) = \overline{\sigma}(\psi)$

### 2.2 语义关系 (Semantic Relations)

**定义 2.2.1** (满足关系)

- $\sigma \models \phi$ 表示 $\overline{\sigma}(\phi) = T$
- $\sigma \not\models \phi$ 表示 $\overline{\sigma}(\phi) = F$

**定义 2.2.2** (逻辑蕴含)
$\Gamma \models \phi$ 表示对于所有满足 $\Gamma$ 的真值赋值 $\sigma$，都有 $\sigma \models \phi$

**定义 2.2.3** (逻辑等价)
$\phi \equiv \psi$ 表示 $\phi \models \psi$ 且 $\psi \models \phi$

### 2.3 语义性质 (Semantic Properties)

**定理 2.3.1** (德摩根律)

1. $\neg(\phi \land \psi) \equiv \neg\phi \lor \neg\psi$
2. $\neg(\phi \lor \psi) \equiv \neg\phi \land \neg\psi$

**定理 2.3.2** (分配律)

1. $\phi \land (\psi \lor \chi) \equiv (\phi \land \psi) \lor (\phi \land \chi)$
2. $\phi \lor (\psi \land \chi) \equiv (\phi \lor \psi) \land (\phi \lor \chi)$

**定理 2.3.3** (双重否定律)
$\neg\neg\phi \equiv \phi$

## 公理化系统 (Axiomatic System)

### 3.1 希尔伯特系统 (Hilbert System)

**定义 3.1.1** (命题逻辑公理)
命题逻辑的公理模式：

1. **A1**: $\phi \rightarrow (\psi \rightarrow \phi)$
2. **A2**: $(\phi \rightarrow (\psi \rightarrow \chi)) \rightarrow ((\phi \rightarrow \psi) \rightarrow (\phi \rightarrow \chi))$
3. **A3**: $(\neg\phi \rightarrow \neg\psi) \rightarrow (\psi \rightarrow \phi)$

**定义 3.1.2** (推理规则)
**分离规则 (Modus Ponens)**: 从 $\phi$ 和 $\phi \rightarrow \psi$ 推出 $\psi$

**定义 3.1.3** (形式证明)
从假设集 $\Gamma$ 到公式 $\phi$ 的形式证明是公式序列 $\phi_1, \phi_2, \ldots, \phi_n = \phi$，其中每个 $\phi_i$ 要么是公理，要么属于 $\Gamma$，要么通过分离规则从前面的公式得到。

**定义 3.1.4** (可证性)
$\Gamma \vdash \phi$ 表示存在从 $\Gamma$ 到 $\phi$ 的形式证明。

### 3.2 元理论性质 (Metatheoretical Properties)

**定理 3.2.1** (可靠性定理)
如果 $\Gamma \vdash \phi$，则 $\Gamma \models \phi$

**证明**: 通过归纳法证明：

1. **基础**: 公理都是重言式
2. **归纳**: 分离规则保持有效性

**定理 3.2.2** (完备性定理)
如果 $\Gamma \models \phi$，则 $\Gamma \vdash \phi$

**证明**: 通过极大一致集构造模型

**定理 3.2.3** (紧致性定理)
如果 $\Gamma \models \phi$，则存在 $\Gamma$ 的有限子集 $\Delta$ 使得 $\Delta \models \phi$

## 自然演绎系统 (Natural Deduction System)

### 4.1 推理规则 (Inference Rules)

**定义 4.1.1** (引入规则)

1. **$\land$-引入**: 从 $\phi$ 和 $\psi$ 推出 $\phi \land \psi$
2. **$\lor$-引入**: 从 $\phi$ 推出 $\phi \lor \psi$ 或 $\psi \lor \phi$
3. **$\rightarrow$-引入**: 从假设 $\phi$ 推出 $\psi$ 后，可以推出 $\phi \rightarrow \psi$
4. **$\neg$-引入**: 从假设 $\phi$ 推出矛盾后，可以推出 $\neg\phi$

**定义 4.1.2** (消除规则)

1. **$\land$-消除**: 从 $\phi \land \psi$ 推出 $\phi$ 或 $\psi$
2. **$\lor$-消除**: 从 $\phi \lor \psi$、$\phi \rightarrow \chi$ 和 $\psi \rightarrow \chi$ 推出 $\chi$
3. **$\rightarrow$-消除**: 从 $\phi$ 和 $\phi \rightarrow \psi$ 推出 $\psi$
4. **$\neg$-消除**: 从 $\phi$ 和 $\neg\phi$ 推出任意公式

### 4.2 证明结构 (Proof Structure)

**定义 4.2.1** (自然演绎证明)
自然演绎证明是使用引入和消除规则的证明树，其中：

- 叶子节点是假设或公理
- 内部节点是应用推理规则的结果
- 根节点是要证明的结论

## 表列演算 (Tableau Calculus)

### 5.1 表列规则 (Tableau Rules)

**定义 5.1.1** (表列规则)
表列演算的规则：

1. **$\land$-规则**: 如果 $\phi \land \psi$ 在分支中，添加 $\phi$ 和 $\psi$
2. **$\lor$-规则**: 如果 $\phi \lor \psi$ 在分支中，创建两个分支分别添加 $\phi$ 和 $\psi$
3. **$\rightarrow$-规则**: 如果 $\phi \rightarrow \psi$ 在分支中，创建两个分支分别添加 $\neg\phi$ 和 $\psi$
4. **$\neg$-规则**: 如果 $\neg\neg\phi$ 在分支中，添加 $\phi$

### 5.2 表列方法 (Tableau Method)

**定义 5.2.1** (表列)
表列是树形结构，其中：

- 每个节点包含公式集合
- 分支表示析取
- 节点表示合取

**定义 5.2.2** (封闭分支)
分支是封闭的，如果它包含 $\phi$ 和 $\neg\phi$

**定义 5.2.3** (表列证明)
$\phi$ 的表列证明是 $\neg\phi$ 的封闭表列

## 可满足性问题 (Satisfiability Problem)

### 6.1 SAT问题 (SAT Problem)

**定义 6.1.1** (SAT问题)
给定命题逻辑公式 $\phi$，判断是否存在真值赋值 $\sigma$ 使得 $\sigma \models \phi$

**定理 6.1.1** (SAT复杂性)
SAT问题是NP完全的

### 6.2 SAT求解算法 (SAT Solving Algorithms)

**定义 6.1.2** (DPLL算法)
DPLL (Davis-Putnam-Logemann-Loveland) 算法：

1. **单元传播**: 如果某个子句只有一个文字，则赋值使其为真
2. **纯文字消除**: 如果某个文字在所有子句中都是正或负，则赋值使其为真
3. **分支**: 选择未赋值的变量进行分支

**定义 6.1.3** (CDCL算法)
CDCL (Conflict-Driven Clause Learning) 算法：

1. 使用DPLL作为基础
2. 当发生冲突时，学习新的子句
3. 使用非时序回溯

## AI应用 (AI Applications)

### 7.1 知识表示 (Knowledge Representation)

**定义 7.1.1** (知识库)
知识库是命题逻辑公式的集合，用于表示领域知识。

**示例 7.1.1** (专家系统规则)

```text
如果 患者有发烧 且 患者有咳嗽，则 患者可能感冒
如果 患者可能感冒，则 建议休息
```

### 7.2 自动推理 (Automated Reasoning)

**定义 7.2.1** (推理引擎)
推理引擎是基于逻辑规则的自动推理系统。

**算法 7.2.1** (前向推理)

```python
def forward_chaining(kb, query):
    agenda = kb.facts
    inferred = set()
    
    while agenda:
        fact = agenda.pop()
        if fact == query:
            return True
        if fact not in inferred:
            inferred.add(fact)
            for rule in kb.rules:
                if fact in rule.premises:
                    rule.premises.remove(fact)
                    if not rule.premises:
                        agenda.append(rule.conclusion)
    return False
```

### 7.3 模型检查 (Model Checking)

**定义 7.3.1** (模型检查)
模型检查是验证系统是否满足规范的过程。

**算法 7.3.1** (CTL模型检查)

```python
def check_ctl(model, formula):
    if formula.type == 'atomic':
        return model.states_with_prop(formula.prop)
    elif formula.type == 'not':
        return model.states - check_ctl(model, formula.arg)
    elif formula.type == 'and':
        return check_ctl(model, formula.left) & check_ctl(model, formula.right)
    elif formula.type == 'EX':
        return model.predecessors(check_ctl(model, formula.arg))
    # ... 其他情况
```

## 技术实现 (Technical Implementation)

### 8.1 Rust实现 (Rust Implementation)

```rust
// 命题逻辑核心模块
pub mod syntax;
pub mod semantics;
pub mod proof;
pub mod sat_solver;

// 语法模块
pub mod syntax {
    use std::fmt;
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum Formula {
        Atom(String),
        Not(Box<Formula>),
        And(Box<Formula>, Box<Formula>),
        Or(Box<Formula>, Box<Formula>),
        Implies(Box<Formula>, Box<Formula>),
        Iff(Box<Formula>, Box<Formula>),
    }
    
    impl Formula {
        pub fn complexity(&self) -> usize {
            match self {
                Formula::Atom(_) => 1,
                Formula::Not(f) => f.complexity() + 1,
                Formula::And(f1, f2) | Formula::Or(f1, f2) |
                Formula::Implies(f1, f2) | Formula::Iff(f1, f2) => {
                    f1.complexity() + f2.complexity() + 1
                }
            }
        }
        
        pub fn subformulas(&self) -> std::collections::HashSet<Formula> {
            let mut set = std::collections::HashSet::new();
            self.collect_subformulas(&mut set);
            set
        }
        
        fn collect_subformulas(&self, set: &mut std::collections::HashSet<Formula>) {
            set.insert(self.clone());
            match self {
                Formula::Atom(_) => {},
                Formula::Not(f) => f.collect_subformulas(set),
                Formula::And(f1, f2) | Formula::Or(f1, f2) |
                Formula::Implies(f1, f2) | Formula::Iff(f1, f2) => {
                    f1.collect_subformulas(set);
                    f2.collect_subformulas(set);
                }
            }
        }
    }
    
    impl fmt::Display for Formula {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Formula::Atom(name) => write!(f, "{}", name),
                Formula::Not(formula) => write!(f, "¬({})", formula),
                Formula::And(f1, f2) => write!(f, "({} ∧ {})", f1, f2),
                Formula::Or(f1, f2) => write!(f, "({} ∨ {})", f1, f2),
                Formula::Implies(f1, f2) => write!(f, "({} → {})", f1, f2),
                Formula::Iff(f1, f2) => write!(f, "({} ↔ {})", f1, f2),
            }
        }
    }
}

// 语义模块
pub mod semantics {
    use std::collections::HashMap;
    use super::syntax::Formula;
    
    #[derive(Debug, Clone)]
    pub struct TruthAssignment {
        assignment: HashMap<String, bool>,
    }
    
    impl TruthAssignment {
        pub fn new() -> Self {
            Self {
                assignment: HashMap::new(),
            }
        }
        
        pub fn set(&mut self, var: &str, value: bool) {
            self.assignment.insert(var.to_string(), value);
        }
        
        pub fn get(&self, var: &str) -> Option<bool> {
            self.assignment.get(var).copied()
        }
        
        pub fn evaluate(&self, formula: &Formula) -> Option<bool> {
            match formula {
                Formula::Atom(name) => self.get(name),
                Formula::Not(f) => self.evaluate(f).map(|v| !v),
                Formula::And(f1, f2) => {
                    let v1 = self.evaluate(f1)?;
                    let v2 = self.evaluate(f2)?;
                    Some(v1 && v2)
                }
                Formula::Or(f1, f2) => {
                    let v1 = self.evaluate(f1)?;
                    let v2 = self.evaluate(f2)?;
                    Some(v1 || v2)
                }
                Formula::Implies(f1, f2) => {
                    let v1 = self.evaluate(f1)?;
                    let v2 = self.evaluate(f2)?;
                    Some(!v1 || v2)
                }
                Formula::Iff(f1, f2) => {
                    let v1 = self.evaluate(f1)?;
                    let v2 = self.evaluate(f2)?;
                    Some(v1 == v2)
                }
            }
        }
        
        pub fn satisfies(&self, formula: &Formula) -> bool {
            self.evaluate(formula).unwrap_or(false)
        }
    }
    
    pub fn is_tautology(formula: &Formula) -> bool {
        let vars = collect_variables(formula);
        check_all_assignments(formula, &vars, 0, &mut TruthAssignment::new())
    }
    
    fn collect_variables(formula: &Formula) -> Vec<String> {
        let mut vars = std::collections::HashSet::new();
        collect_vars_recursive(formula, &mut vars);
        vars.into_iter().collect()
    }
    
    fn collect_vars_recursive(formula: &Formula, vars: &mut std::collections::HashSet<String>) {
        match formula {
            Formula::Atom(name) => { vars.insert(name.clone()); }
            Formula::Not(f) => collect_vars_recursive(f, vars),
            Formula::And(f1, f2) | Formula::Or(f1, f2) |
            Formula::Implies(f1, f2) | Formula::Iff(f1, f2) => {
                collect_vars_recursive(f1, vars);
                collect_vars_recursive(f2, vars);
            }
        }
    }
    
    fn check_all_assignments(
        formula: &Formula,
        vars: &[String],
        index: usize,
        assignment: &mut TruthAssignment,
    ) -> bool {
        if index >= vars.len() {
            return assignment.satisfies(formula);
        }
        
        assignment.set(&vars[index], true);
        let result1 = check_all_assignments(formula, vars, index + 1, assignment);
        
        assignment.set(&vars[index], false);
        let result2 = check_all_assignments(formula, vars, index + 1, assignment);
        
        result1 && result2
    }
}

// 证明系统模块
pub mod proof {
    use super::syntax::Formula;
    use std::collections::HashSet;
    
    #[derive(Debug)]
    pub struct ProofStep {
        pub formula: Formula,
        pub rule: String,
        pub premises: Vec<usize>,
    }
    
    #[derive(Debug)]
    pub struct Proof {
        pub steps: Vec<ProofStep>,
        pub assumptions: HashSet<Formula>,
    }
    
    impl Proof {
        pub fn new() -> Self {
            Self {
                steps: Vec::new(),
                assumptions: HashSet::new(),
            }
        }
        
        pub fn add_assumption(&mut self, formula: Formula) {
            self.assumptions.insert(formula);
        }
        
        pub fn add_step(&mut self, formula: Formula, rule: &str, premises: Vec<usize>) {
            self.steps.push(ProofStep {
                formula,
                rule: rule.to_string(),
                premises,
            });
        }
        
        pub fn is_valid(&self) -> bool {
            // 验证证明的有效性
            for (i, step) in self.steps.iter().enumerate() {
                if !self.validate_step(i, step) {
                    return false;
                }
            }
            true
        }
        
        fn validate_step(&self, index: usize, step: &ProofStep) -> bool {
            // 验证每个证明步骤的有效性
            match step.rule.as_str() {
                "assumption" => self.assumptions.contains(&step.formula),
                "modus_ponens" => {
                    if step.premises.len() == 2 {
                        let premise1 = &self.steps[step.premises[0]].formula;
                        let premise2 = &self.steps[step.premises[1]].formula;
                        self.is_implication(premise1, premise2, &step.formula)
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }
        
        fn is_implication(&self, premise1: &Formula, premise2: &Formula, conclusion: &Formula) -> bool {
            // 检查是否为有效的蕴含推理
            if let Formula::Implies(ant, cons) = premise1 {
                ant.as_ref() == premise2 && cons.as_ref() == conclusion
            } else {
                false
            }
        }
    }
}

// SAT求解器模块
pub mod sat_solver {
    use super::syntax::Formula;
    use super::semantics::TruthAssignment;
    use std::collections::HashMap;
    
    pub struct SATSolver {
        clauses: Vec<Vec<i32>>, // CNF形式的子句
        var_map: HashMap<String, usize>, // 变量名到索引的映射
        var_names: Vec<String>, // 索引到变量名的映射
    }
    
    impl SATSolver {
        pub fn new() -> Self {
            Self {
                clauses: Vec::new(),
                var_map: HashMap::new(),
                var_names: Vec::new(),
            }
        }
        
        pub fn add_formula(&mut self, formula: &Formula) {
            let cnf = self.to_cnf(formula);
            self.add_cnf_clauses(&cnf);
        }
        
        pub fn solve(&self) -> Option<TruthAssignment> {
            let mut assignment = vec![None; self.var_names.len()];
            if self.dpll(&mut assignment) {
                let mut truth_assignment = TruthAssignment::new();
                for (i, value) in assignment.iter().enumerate() {
                    if let Some(val) = value {
                        truth_assignment.set(&self.var_names[i], *val);
                    }
                }
                Some(truth_assignment)
            } else {
                None
            }
        }
        
        fn to_cnf(&mut self, formula: &Formula) -> Vec<Vec<i32>> {
            // 将公式转换为CNF形式
            let nnf = self.to_nnf(formula);
            self.distribute_or_over_and(&nnf)
        }
        
        fn to_nnf(&mut self, formula: &Formula) -> Formula {
            // 转换为否定范式
            match formula {
                Formula::Atom(_) => formula.clone(),
                Formula::Not(f) => match f.as_ref() {
                    Formula::Atom(_) => formula.clone(),
                    Formula::Not(g) => self.to_nnf(g),
                    Formula::And(f1, f2) => {
                        Formula::Or(
                            Box::new(self.to_nnf(&Formula::Not(Box::new(f1.clone())))),
                            Box::new(self.to_nnf(&Formula::Not(Box::new(f2.clone())))),
                        )
                    }
                    Formula::Or(f1, f2) => {
                        Formula::And(
                            Box::new(self.to_nnf(&Formula::Not(Box::new(f1.clone())))),
                            Box::new(self.to_nnf(&Formula::Not(Box::new(f2.clone())))),
                        )
                    }
                    _ => formula.clone(),
                },
                Formula::And(f1, f2) => {
                    Formula::And(
                        Box::new(self.to_nnf(f1)),
                        Box::new(self.to_nnf(f2)),
                    )
                }
                Formula::Or(f1, f2) => {
                    Formula::Or(
                        Box::new(self.to_nnf(f1)),
                        Box::new(self.to_nnf(f2)),
                    )
                }
                Formula::Implies(f1, f2) => {
                    self.to_nnf(&Formula::Or(
                        Box::new(Formula::Not(Box::new(f1.clone()))),
                        Box::new(f2.clone()),
                    ))
                }
                Formula::Iff(f1, f2) => {
                    let imp1 = Formula::Implies(Box::new(f1.clone()), Box::new(f2.clone()));
                    let imp2 = Formula::Implies(Box::new(f2.clone()), Box::new(f1.clone()));
                    self.to_nnf(&Formula::And(Box::new(imp1), Box::new(imp2)))
                }
            }
        }
        
        fn distribute_or_over_and(&mut self, formula: &Formula) -> Vec<Vec<i32>> {
            // 分配或运算到合取运算上
            match formula {
                Formula::Atom(name) => {
                    let var_id = self.get_var_id(name);
                    vec![vec![var_id as i32]]
                }
                Formula::Not(f) => {
                    if let Formula::Atom(name) = f.as_ref() {
                        let var_id = self.get_var_id(name);
                        vec![vec![-(var_id as i32)]]
                    } else {
                        vec![]
                    }
                }
                Formula::And(f1, f2) => {
                    let mut clauses = self.distribute_or_over_and(f1);
                    clauses.extend(self.distribute_or_over_and(f2));
                    clauses
                }
                Formula::Or(f1, f2) => {
                    let clauses1 = self.distribute_or_over_and(f1);
                    let clauses2 = self.distribute_or_over_and(f2);
                    self.cross_product(clauses1, clauses2)
                }
                _ => vec![],
            }
        }
        
        fn cross_product(&self, clauses1: Vec<Vec<i32>>, clauses2: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
            let mut result = Vec::new();
            for clause1 in &clauses1 {
                for clause2 in &clauses2 {
                    let mut new_clause = clause1.clone();
                    new_clause.extend(clause2.clone());
                    result.push(new_clause);
                }
            }
            result
        }
        
        fn get_var_id(&mut self, name: &str) -> usize {
            if let Some(&id) = self.var_map.get(name) {
                id
            } else {
                let id = self.var_names.len();
                self.var_map.insert(name.to_string(), id);
                self.var_names.push(name.to_string());
                id
            }
        }
        
        fn add_cnf_clauses(&mut self, clauses: &[Vec<i32>]) {
            self.clauses.extend(clauses.to_vec());
        }
        
        fn dpll(&self, assignment: &mut [Option<bool>]) -> bool {
            // 单元传播
            if let Some(unit_literal) = self.find_unit_clause(assignment) {
                let var_id = unit_literal.abs() as usize - 1;
                let value = unit_literal > 0;
                assignment[var_id] = Some(value);
                return self.dpll(assignment);
            }
            
            // 纯文字消除
            if let Some(pure_literal) = self.find_pure_literal(assignment) {
                let var_id = pure_literal.abs() as usize - 1;
                let value = pure_literal > 0;
                assignment[var_id] = Some(value);
                return self.dpll(assignment);
            }
            
            // 检查是否所有子句都满足
            if self.all_clauses_satisfied(assignment) {
                return true;
            }
            
            // 检查是否有冲突
            if self.has_conflict(assignment) {
                return false;
            }
            
            // 分支
            if let Some(unassigned_var) = self.find_unassigned_variable(assignment) {
                assignment[unassigned_var] = Some(true);
                if self.dpll(assignment) {
                    return true;
                }
                
                assignment[unassigned_var] = Some(false);
                if self.dpll(assignment) {
                    return true;
                }
                
                assignment[unassigned_var] = None;
                return false;
            }
            
            false
        }
        
        fn find_unit_clause(&self, assignment: &[Option<bool>]) -> Option<i32> {
            for clause in &self.clauses {
                let mut unassigned_literal = None;
                let mut clause_satisfied = false;
                
                for &literal in clause {
                    let var_id = literal.abs() as usize - 1;
                    if let Some(value) = assignment[var_id] {
                        let literal_value = literal > 0;
                        if value == literal_value {
                            clause_satisfied = true;
                            break;
                        }
                    } else if unassigned_literal.is_none() {
                        unassigned_literal = Some(literal);
                    }
                }
                
                if !clause_satisfied && unassigned_literal.is_some() {
                    let mut other_unassigned = false;
                    for &literal in clause {
                        let var_id = literal.abs() as usize - 1;
                        if assignment[var_id].is_none() && literal != unassigned_literal.unwrap() {
                            other_unassigned = true;
                            break;
                        }
                    }
                    if !other_unassigned {
                        return unassigned_literal;
                    }
                }
            }
            None
        }
        
        fn find_pure_literal(&self, assignment: &[Option<bool>]) -> Option<i32> {
            let mut literal_counts: HashMap<i32, i32> = HashMap::new();
            
            for clause in &self.clauses {
                let mut clause_satisfied = false;
                for &literal in clause {
                    let var_id = literal.abs() as usize - 1;
                    if let Some(value) = assignment[var_id] {
                        let literal_value = literal > 0;
                        if value == literal_value {
                            clause_satisfied = true;
                            break;
                        }
                    }
                }
                
                if !clause_satisfied {
                    for &literal in clause {
                        let var_id = literal.abs() as usize - 1;
                        if assignment[var_id].is_none() {
                            *literal_counts.entry(literal).or_insert(0) += 1;
                        }
                    }
                }
            }
            
            for (literal, count) in literal_counts {
                let neg_literal = -literal;
                if !literal_counts.contains_key(&neg_literal) {
                    return Some(literal);
                }
            }
            
            None
        }
        
        fn all_clauses_satisfied(&self, assignment: &[Option<bool>]) -> bool {
            for clause in &self.clauses {
                let mut clause_satisfied = false;
                for &literal in clause {
                    let var_id = literal.abs() as usize - 1;
                    if let Some(value) = assignment[var_id] {
                        let literal_value = literal > 0;
                        if value == literal_value {
                            clause_satisfied = true;
                            break;
                        }
                    }
                }
                if !clause_satisfied {
                    return false;
                }
            }
            true
        }
        
        fn has_conflict(&self, assignment: &[Option<bool>]) -> bool {
            for clause in &self.clauses {
                let mut all_false = true;
                let mut has_unassigned = false;
                
                for &literal in clause {
                    let var_id = literal.abs() as usize - 1;
                    if let Some(value) = assignment[var_id] {
                        let literal_value = literal > 0;
                        if value == literal_value {
                            all_false = false;
                            break;
                        }
                    } else {
                        has_unassigned = true;
                    }
                }
                
                if all_false && !has_unassigned {
                    return true;
                }
            }
            false
        }
        
        fn find_unassigned_variable(&self, assignment: &[Option<bool>]) -> Option<usize> {
            assignment.iter().position(|&x| x.is_none())
        }
    }
}
```

### 8.2 Haskell形式化证明 (Haskell Formal Proofs)

```haskell
-- 命题逻辑类型定义
module PropositionalLogic where

import Data.Set (Set)
import qualified Data.Set as Set
import Data.Map (Map)
import qualified Data.Map as Map

-- 公式类型
data Formula = Atom String
             | Not Formula
             | And Formula Formula
             | Or Formula Formula
             | Implies Formula Formula
             | Iff Formula Formula
             deriving (Eq, Ord, Show)

-- 真值赋值类型
type TruthAssignment = Map String Bool

-- 语义评估
evaluate :: TruthAssignment -> Formula -> Maybe Bool
evaluate assignment formula = case formula of
    Atom name -> Map.lookup name assignment
    Not f -> not <$> evaluate assignment f
    And f1 f2 -> do
        v1 <- evaluate assignment f1
        v2 <- evaluate assignment f2
        return (v1 && v2)
    Or f1 f2 -> do
        v1 <- evaluate assignment f1
        v2 <- evaluate assignment f2
        return (v1 || v2)
    Implies f1 f2 -> do
        v1 <- evaluate assignment f1
        v2 <- evaluate assignment f2
        return (not v1 || v2)
    Iff f1 f2 -> do
        v1 <- evaluate assignment f1
        v2 <- evaluate assignment f2
        return (v1 == v2)

-- 重言式检查
isTautology :: Formula -> Bool
isTautology formula = all (\assignment -> 
    case evaluate assignment formula of
        Just True -> True
        _ -> False
    ) allAssignments
  where
    vars = collectVariables formula
    allAssignments = generateAllAssignments vars

-- 收集变量
collectVariables :: Formula -> Set String
collectVariables = Set.fromList . collectVars
  where
    collectVars (Atom name) = [name]
    collectVars (Not f) = collectVars f
    collectVars (And f1 f2) = collectVars f1 ++ collectVars f2
    collectVars (Or f1 f2) = collectVars f1 ++ collectVars f2
    collectVars (Implies f1 f2) = collectVars f1 ++ collectVars f2
    collectVars (Iff f1 f2) = collectVars f1 ++ collectVars f2

-- 生成所有真值赋值
generateAllAssignments :: Set String -> [TruthAssignment]
generateAllAssignments vars = 
    map (Map.fromList . zip (Set.toList vars)) (generateBoolLists (Set.size vars))

generateBoolLists :: Int -> [[Bool]]
generateBoolLists 0 = [[]]
generateBoolLists n = 
    [True:bs | bs <- generateBoolLists (n-1)] ++
    [False:bs | bs <- generateBoolLists (n-1)]

-- 逻辑等价
equivalent :: Formula -> Formula -> Bool
equivalent f1 f2 = isTautology (Iff f1 f2)

-- 德摩根律证明
deMorgan1 :: Formula -> Formula -> Bool
deMorgan1 f1 f2 = equivalent 
    (Not (And f1 f2)) 
    (Or (Not f1) (Not f2))

deMorgan2 :: Formula -> Formula -> Bool
deMorgan2 f1 f2 = equivalent 
    (Not (Or f1 f2)) 
    (And (Not f1) (Not f2))

-- 分配律证明
distributive1 :: Formula -> Formula -> Formula -> Bool
distributive1 f1 f2 f3 = equivalent
    (And f1 (Or f2 f3))
    (Or (And f1 f2) (And f1 f3))

distributive2 :: Formula -> Formula -> Formula -> Bool
distributive2 f1 f2 f3 = equivalent
    (Or f1 (And f2 f3))
    (And (Or f1 f2) (Or f1 f3))

-- 双重否定律证明
doubleNegation :: Formula -> Bool
doubleNegation f = equivalent f (Not (Not f))

-- 证明系统
data ProofStep = ProofStep {
    formula :: Formula,
    rule :: String,
    premises :: [Int]
} deriving Show

data Proof = Proof {
    steps :: [ProofStep],
    assumptions :: Set Formula
} deriving Show

-- 验证证明
validateProof :: Proof -> Bool
validateProof proof = all (validateStep proof) (zip [0..] (steps proof))

validateStep :: Proof -> (Int, ProofStep) -> Bool
validateStep proof (index, step) = case rule step of
    "assumption" -> Set.member (formula step) (assumptions proof)
    "modus_ponens" -> 
        length (premises step) == 2 &&
        let premise1 = formula (steps proof !! (premises step !! 0))
            premise2 = formula (steps proof !! (premises step !! 1))
        in isValidImplication premise1 premise2 (formula step)
    _ -> False

isValidImplication :: Formula -> Formula -> Formula -> Bool
isValidImplication premise1 premise2 conclusion = 
    case premise1 of
        Implies ant cons -> ant == premise2 && cons == conclusion
        _ -> False

-- SAT求解器
type Clause = [Int]  -- 正数表示正文字，负数表示负文字
type CNF = [Clause]

solveSAT :: CNF -> Maybe TruthAssignment
solveSAT cnf = 
    let vars = Set.fromList [abs literal | clause <- cnf, literal <- clause]
        initialAssignment = Map.empty
    in dpll cnf vars initialAssignment

dpll :: CNF -> Set Int -> TruthAssignment -> Maybe TruthAssignment
dpll cnf vars assignment
    | allClausesSatisfied cnf assignment = Just assignment
    | hasConflict cnf assignment = Nothing
    | otherwise = case findUnitClause cnf assignment of
        Just literal -> 
            let var = abs literal
                value = literal > 0
                newAssignment = Map.insert (show var) value assignment
            in dpll cnf vars newAssignment
        Nothing -> case findPureLiteral cnf assignment of
            Just literal ->
                let var = abs literal
                    value = literal > 0
                    newAssignment = Map.insert (show var) value assignment
                in dpll cnf vars newAssignment
            Nothing -> case findUnassignedVar vars assignment of
                Just var ->
                    let assignment1 = Map.insert (show var) True assignment
                        assignment2 = Map.insert (show var) False assignment
                    in case dpll cnf vars assignment1 of
                        Just result -> Just result
                        Nothing -> dpll cnf vars assignment2
                Nothing -> Nothing

allClausesSatisfied :: CNF -> TruthAssignment -> Bool
allClausesSatisfied cnf assignment = all (clauseSatisfied assignment) cnf

clauseSatisfied :: TruthAssignment -> Clause -> Bool
clauseSatisfied assignment clause = any (literalSatisfied assignment) clause

literalSatisfied :: TruthAssignment -> Int -> Bool
literalSatisfied assignment literal = 
    case Map.lookup (show (abs literal)) assignment of
        Just value -> (literal > 0) == value
        Nothing -> False

hasConflict :: CNF -> TruthAssignment -> Bool
hasConflict cnf assignment = any (clauseConflict assignment) cnf

clauseConflict :: TruthAssignment -> Clause -> Bool
clauseConflict assignment clause = 
    all (\literal -> 
        case Map.lookup (show (abs literal)) assignment of
            Just value -> (literal > 0) /= value
            Nothing -> False
    ) clause

findUnitClause :: CNF -> TruthAssignment -> Maybe Int
findUnitClause cnf assignment = 
    case [literal | clause <- cnf, 
                   not (clauseSatisfied assignment clause),
                   let unassigned = [l | l <- clause, 
                                       Map.notMember (show (abs l)) assignment],
                   length unassigned == 1,
                   let literal = head unassigned] of
        (literal:_) -> Just literal
        [] -> Nothing

findPureLiteral :: CNF -> TruthAssignment -> Maybe Int
findPureLiteral cnf assignment = 
    let literalCounts = Map.fromListWith (+) 
        [(literal, 1) | clause <- cnf,
                        not (clauseSatisfied assignment clause),
                        literal <- clause,
                        Map.notMember (show (abs literal)) assignment]
        pureLiterals = [literal | (literal, count) <- Map.toList literalCounts,
                                 not (Map.member (-literal) literalCounts)]
    in case pureLiterals of
        (literal:_) -> Just literal
        [] -> Nothing

findUnassignedVar :: Set Int -> TruthAssignment -> Maybe Int
findUnassignedVar vars assignment = 
    case [var | var <- Set.toList vars, 
                Map.notMember (show var) assignment] of
        (var:_) -> Just var
        [] -> Nothing
```

## 总结 (Summary)

命题逻辑作为逻辑学的基础，为形式化推理提供了严格的数学框架。通过语法理论、语义理论、公理化系统和各种证明方法，命题逻辑建立了完整的理论体系。在AI应用中，命题逻辑为知识表示、自动推理、模型检查等核心技术提供了理论基础。

随着SAT求解器的发展，命题逻辑在软件验证、人工智能、电子设计自动化等领域发挥着越来越重要的作用。通过Rust和Haskell的实现，我们可以看到命题逻辑理论在实际应用中的强大能力。

---

**相关链接**:

- [逻辑学基础理论](../README.md)
- [谓词逻辑理论](../01.2-PredicateLogic/README.md)
- [模态逻辑理论](../01.3-ModalLogic/README.md)
- [时态逻辑理论](../01.4-TemporalLogic/README.md)
