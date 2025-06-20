//! 逻辑学基础理论模块
//! 
//! 本模块提供了逻辑学的基础理论实现，包括：
//! - 形式逻辑系统
//! - 模态逻辑与可能世界语义
//! - 时态逻辑与时间推理
//! - 直觉逻辑与构造性证明
//! - 线性逻辑与资源管理
//! - 高阶逻辑与类型理论

pub mod formal_logic;
pub mod modal_logic;
pub mod temporal_logic;
pub mod intuitionistic_logic;
pub mod linear_logic;
pub mod higher_order_logic;
pub mod reasoning_system;
pub mod knowledge_representation;

use std::collections::HashMap;
use std::fmt;

/// 逻辑系统核心结构
#[derive(Debug, Clone)]
pub struct LogicSystem {
    /// 语法系统
    pub syntax: Syntax,
    /// 语义系统
    pub semantics: Semantics,
    /// 证明系统
    pub proof_system: ProofSystem,
    /// 解释器
    pub interpreter: Interpreter,
}

/// 语法系统
#[derive(Debug, Clone)]
pub struct Syntax {
    /// 原子命题集合
    pub atoms: Vec<String>,
    /// 逻辑连接词
    pub connectives: Vec<Connective>,
    /// 量词
    pub quantifiers: Vec<Quantifier>,
}

/// 语义系统
#[derive(Debug, Clone)]
pub struct Semantics {
    /// 论域
    pub domain: Vec<String>,
    /// 解释函数
    pub interpretation: HashMap<String, String>,
    /// 赋值函数
    pub valuation: HashMap<String, bool>,
}

/// 证明系统
#[derive(Debug, Clone)]
pub struct ProofSystem {
    /// 公理集合
    pub axioms: Vec<Formula>,
    /// 推理规则
    pub rules: Vec<InferenceRule>,
    /// 证明序列
    pub proofs: Vec<Proof>,
}

/// 解释器
#[derive(Debug, Clone)]
pub struct Interpreter {
    /// 环境
    pub environment: HashMap<String, String>,
    /// 解释函数
    pub interpret_function: fn(&Formula) -> String,
}

/// 逻辑连接词
#[derive(Debug, Clone, PartialEq)]
pub enum Connective {
    Not,
    And,
    Or,
    Implies,
    Iff,
    Necessity,  // 必然性
    Possibility, // 可能性
    Next,       // 下一个
    Always,     // 总是
    Eventually, // 最终
    Until,      // 直到
}

/// 量词
#[derive(Debug, Clone, PartialEq)]
pub enum Quantifier {
    Forall,  // 全称量词
    Exists,  // 存在量词
}

/// 公式
#[derive(Debug, Clone, PartialEq)]
pub enum Formula {
    Atom(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Iff(Box<Formula>, Box<Formula>),
    Forall(String, Box<Formula>),
    Exists(String, Box<Formula>),
    Necessity(Box<Formula>),
    Possibility(Box<Formula>),
    Next(Box<Formula>),
    Always(Box<Formula>),
    Eventually(Box<Formula>),
    Until(Box<Formula>, Box<Formula>),
}

/// 推理规则
#[derive(Debug, Clone)]
pub struct InferenceRule {
    /// 规则名称
    pub name: String,
    /// 前提
    pub premises: Vec<Formula>,
    /// 结论
    pub conclusion: Formula,
}

/// 证明
#[derive(Debug, Clone)]
pub struct Proof {
    /// 证明步骤
    pub steps: Vec<ProofStep>,
    /// 结论
    pub conclusion: Formula,
}

/// 证明步骤
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// 步骤编号
    pub number: usize,
    /// 公式
    pub formula: Formula,
    /// 理由
    pub justification: String,
    /// 依赖的步骤
    pub dependencies: Vec<usize>,
}

impl LogicSystem {
    /// 创建新的逻辑系统
    pub fn new() -> Self {
        LogicSystem {
            syntax: Syntax::new(),
            semantics: Semantics::new(),
            proof_system: ProofSystem::new(),
            interpreter: Interpreter::new(),
        }
    }

    /// 检查系统一致性
    pub fn check_consistency(&self) -> bool {
        self.syntax.is_consistent() &&
        self.semantics.is_consistent() &&
        self.proof_system.is_consistent()
    }

    /// 证明公式
    pub fn prove(&self, formula: &Formula) -> Option<Proof> {
        self.proof_system.prove(formula)
    }

    /// 解释公式
    pub fn interpret(&self, formula: &Formula) -> String {
        self.interpreter.interpret(formula)
    }

    /// 验证公式
    pub fn validate(&self, formula: &Formula) -> bool {
        self.syntax.is_well_formed(formula) &&
        self.semantics.is_satisfiable(formula)
    }
}

impl Syntax {
    /// 创建新的语法系统
    pub fn new() -> Self {
        Syntax {
            atoms: Vec::new(),
            connectives: vec![
                Connective::Not,
                Connective::And,
                Connective::Or,
                Connective::Implies,
                Connective::Iff,
            ],
            quantifiers: vec![
                Quantifier::Forall,
                Quantifier::Exists,
            ],
        }
    }

    /// 检查语法一致性
    pub fn is_consistent(&self) -> bool {
        !self.atoms.is_empty() && !self.connectives.is_empty()
    }

    /// 检查公式是否良构
    pub fn is_well_formed(&self, formula: &Formula) -> bool {
        match formula {
            Formula::Atom(name) => self.atoms.contains(name),
            Formula::Not(f) => self.is_well_formed(f),
            Formula::And(f1, f2) => self.is_well_formed(f1) && self.is_well_formed(f2),
            Formula::Or(f1, f2) => self.is_well_formed(f1) && self.is_well_formed(f2),
            Formula::Implies(f1, f2) => self.is_well_formed(f1) && self.is_well_formed(f2),
            Formula::Iff(f1, f2) => self.is_well_formed(f1) && self.is_well_formed(f2),
            Formula::Forall(_, f) => self.is_well_formed(f),
            Formula::Exists(_, f) => self.is_well_formed(f),
            Formula::Necessity(f) => self.is_well_formed(f),
            Formula::Possibility(f) => self.is_well_formed(f),
            Formula::Next(f) => self.is_well_formed(f),
            Formula::Always(f) => self.is_well_formed(f),
            Formula::Eventually(f) => self.is_well_formed(f),
            Formula::Until(f1, f2) => self.is_well_formed(f1) && self.is_well_formed(f2),
        }
    }
}

impl Semantics {
    /// 创建新的语义系统
    pub fn new() -> Self {
        Semantics {
            domain: Vec::new(),
            interpretation: HashMap::new(),
            valuation: HashMap::new(),
        }
    }

    /// 检查语义一致性
    pub fn is_consistent(&self) -> bool {
        true // 简化实现
    }

    /// 检查公式是否可满足
    pub fn is_satisfiable(&self, formula: &Formula) -> bool {
        // 简化实现，实际应该进行模型检查
        match formula {
            Formula::Atom(name) => self.valuation.get(name).copied().unwrap_or(true),
            Formula::Not(f) => !self.is_satisfiable(f),
            Formula::And(f1, f2) => self.is_satisfiable(f1) && self.is_satisfiable(f2),
            Formula::Or(f1, f2) => self.is_satisfiable(f1) || self.is_satisfiable(f2),
            Formula::Implies(f1, f2) => !self.is_satisfiable(f1) || self.is_satisfiable(f2),
            Formula::Iff(f1, f2) => self.is_satisfiable(f1) == self.is_satisfiable(f2),
            _ => true, // 简化处理
        }
    }
}

impl ProofSystem {
    /// 创建新的证明系统
    pub fn new() -> Self {
        ProofSystem {
            axioms: vec![
                // 命题逻辑公理
                Formula::Implies(
                    Box::new(Formula::Atom("p".to_string())),
                    Box::new(Formula::Atom("p".to_string())),
                ),
            ],
            rules: vec![
                // 假言推理规则
                InferenceRule {
                    name: "Modus Ponens".to_string(),
                    premises: vec![
                        Formula::Implies(
                            Box::new(Formula::Atom("p".to_string())),
                            Box::new(Formula::Atom("q".to_string())),
                        ),
                        Formula::Atom("p".to_string()),
                    ],
                    conclusion: Formula::Atom("q".to_string()),
                },
            ],
            proofs: Vec::new(),
        }
    }

    /// 检查证明系统一致性
    pub fn is_consistent(&self) -> bool {
        !self.axioms.is_empty() && !self.rules.is_empty()
    }

    /// 证明公式
    pub fn prove(&self, formula: &Formula) -> Option<Proof> {
        // 简化实现，实际应该进行自动定理证明
        if self.axioms.contains(formula) {
            Some(Proof {
                steps: vec![ProofStep {
                    number: 1,
                    formula: formula.clone(),
                    justification: "Axiom".to_string(),
                    dependencies: vec![],
                }],
                conclusion: formula.clone(),
            })
        } else {
            None
        }
    }
}

impl Interpreter {
    /// 创建新的解释器
    pub fn new() -> Self {
        Interpreter {
            environment: HashMap::new(),
            interpret_function: |_| "Interpretation".to_string(),
        }
    }

    /// 解释公式
    pub fn interpret(&self, formula: &Formula) -> String {
        match formula {
            Formula::Atom(name) => format!("原子命题: {}", name),
            Formula::Not(f) => format!("非({})", self.interpret(f)),
            Formula::And(f1, f2) => format!("({} 且 {})", self.interpret(f1), self.interpret(f2)),
            Formula::Or(f1, f2) => format!("({} 或 {})", self.interpret(f1), self.interpret(f2)),
            Formula::Implies(f1, f2) => format!("({} 蕴含 {})", self.interpret(f1), self.interpret(f2)),
            Formula::Iff(f1, f2) => format!("({} 当且仅当 {})", self.interpret(f1), self.interpret(f2)),
            Formula::Forall(var, f) => format!("对所有{}，{}", var, self.interpret(f)),
            Formula::Exists(var, f) => format!("存在{}，{}", var, self.interpret(f)),
            Formula::Necessity(f) => format!("必然({})", self.interpret(f)),
            Formula::Possibility(f) => format!("可能({})", self.interpret(f)),
            Formula::Next(f) => format!("下一个({})", self.interpret(f)),
            Formula::Always(f) => format!("总是({})", self.interpret(f)),
            Formula::Eventually(f) => format!("最终({})", self.interpret(f)),
            Formula::Until(f1, f2) => format!("({} 直到 {})", self.interpret(f1), self.interpret(f2)),
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::Atom(name) => write!(f, "{}", name),
            Formula::Not(formula) => write!(f, "¬{}", formula),
            Formula::And(f1, f2) => write!(f, "({} ∧ {})", f1, f2),
            Formula::Or(f1, f2) => write!(f, "({} ∨ {})", f1, f2),
            Formula::Implies(f1, f2) => write!(f, "({} → {})", f1, f2),
            Formula::Iff(f1, f2) => write!(f, "({} ↔ {})", f1, f2),
            Formula::Forall(var, formula) => write!(f, "∀{}. {}", var, formula),
            Formula::Exists(var, formula) => write!(f, "∃{}. {}", var, formula),
            Formula::Necessity(formula) => write!(f, "□{}", formula),
            Formula::Possibility(formula) => write!(f, "◇{}", formula),
            Formula::Next(formula) => write!(f, "○{}", formula),
            Formula::Always(formula) => write!(f, "□{}", formula),
            Formula::Eventually(formula) => write!(f, "◇{}", formula),
            Formula::Until(f1, f2) => write!(f, "({} U {})", f1, f2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_system_creation() {
        let system = LogicSystem::new();
        assert!(system.check_consistency());
    }

    #[test]
    fn test_formula_parsing() {
        let atom = Formula::Atom("p".to_string());
        let not_atom = Formula::Not(Box::new(atom.clone()));
        let and_formula = Formula::And(Box::new(atom.clone()), Box::new(not_atom.clone()));
        
        assert_eq!(format!("{}", atom), "p");
        assert_eq!(format!("{}", not_atom), "¬p");
        assert_eq!(format!("{}", and_formula), "(p ∧ ¬p)");
    }

    #[test]
    fn test_syntax_well_formed() {
        let mut syntax = Syntax::new();
        syntax.atoms.push("p".to_string());
        syntax.atoms.push("q".to_string());
        
        let formula = Formula::And(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("q".to_string())),
        );
        
        assert!(syntax.is_well_formed(&formula));
    }

    #[test]
    fn test_semantics_satisfiability() {
        let semantics = Semantics::new();
        let formula = Formula::Atom("p".to_string());
        
        assert!(semantics.is_satisfiable(&formula));
    }

    #[test]
    fn test_proof_system() {
        let proof_system = ProofSystem::new();
        let axiom = Formula::Implies(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("p".to_string())),
        );
        
        let proof = proof_system.prove(&axiom);
        assert!(proof.is_some());
    }

    #[test]
    fn test_interpreter() {
        let interpreter = Interpreter::new();
        let formula = Formula::And(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("q".to_string())),
        );
        
        let interpretation = interpreter.interpret(&formula);
        assert!(interpretation.contains("且"));
    }
} 