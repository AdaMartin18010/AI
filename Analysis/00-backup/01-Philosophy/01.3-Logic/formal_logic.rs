//! 形式逻辑实现
//! 
//! 本模块提供了形式逻辑的完整实现，包括：
//! - 命题逻辑
//! - 一阶逻辑
//! - 真值表方法
//! - 归结推理
//! - 模型检查

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::{Formula, Connective, Quantifier};

/// 命题逻辑系统
#[derive(Debug, Clone)]
pub struct PropositionalLogic {
    /// 原子命题集合
    pub atoms: HashSet<String>,
    /// 真值赋值
    pub valuation: HashMap<String, bool>,
}

/// 一阶逻辑系统
#[derive(Debug, Clone)]
pub struct FirstOrderLogic {
    /// 常量符号
    pub constants: HashSet<String>,
    /// 函数符号
    pub functions: HashMap<String, usize>, // 函数名 -> 参数个数
    /// 谓词符号
    pub predicates: HashMap<String, usize>, // 谓词名 -> 参数个数
    /// 论域
    pub domain: Vec<String>,
    /// 解释函数
    pub interpretation: HashMap<String, String>,
}

/// 项（一阶逻辑中的表达式）
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Variable(String),
    Constant(String),
    Function(String, Vec<Term>),
}

/// 原子公式（一阶逻辑中的基本公式）
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicFormula {
    pub predicate: String,
    pub terms: Vec<Term>,
}

/// 真值表
#[derive(Debug, Clone)]
pub struct TruthTable {
    /// 变量列表
    pub variables: Vec<String>,
    /// 真值行
    pub rows: Vec<TruthTableRow>,
}

/// 真值表行
#[derive(Debug, Clone)]
pub struct TruthTableRow {
    /// 变量赋值
    pub assignment: HashMap<String, bool>,
    /// 公式真值
    pub truth_value: bool,
}

/// 归结推理系统
#[derive(Debug, Clone)]
pub struct ResolutionSystem {
    /// 子句集合
    pub clauses: Vec<Clause>,
    /// 归结规则
    pub resolution_rules: Vec<ResolutionRule>,
}

/// 子句（析取范式）
#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    /// 文字集合
    pub literals: Vec<Literal>,
}

/// 文字（原子命题或其否定）
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Positive(String),
    Negative(String),
}

/// 归结规则
#[derive(Debug, Clone)]
pub struct ResolutionRule {
    /// 规则名称
    pub name: String,
    /// 前提子句
    pub premises: Vec<Clause>,
    /// 结论子句
    pub conclusion: Clause,
}

impl PropositionalLogic {
    /// 创建新的命题逻辑系统
    pub fn new() -> Self {
        PropositionalLogic {
            atoms: HashSet::new(),
            valuation: HashMap::new(),
        }
    }

    /// 添加原子命题
    pub fn add_atom(&mut self, atom: String) {
        self.atoms.insert(atom.clone());
        self.valuation.insert(atom, true); // 默认真值
    }

    /// 设置原子命题的真值
    pub fn set_valuation(&mut self, atom: &str, value: bool) {
        self.valuation.insert(atom.to_string(), value);
    }

    /// 计算公式的真值
    pub fn evaluate(&self, formula: &Formula) -> Option<bool> {
        match formula {
            Formula::Atom(name) => self.valuation.get(name).copied(),
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
            _ => None, // 命题逻辑不支持量词和模态算子
        }
    }

    /// 生成真值表
    pub fn generate_truth_table(&self, formula: &Formula) -> TruthTable {
        let mut variables = Vec::new();
        self.collect_variables(formula, &mut variables);
        variables.sort();

        let mut rows = Vec::new();
        let num_vars = variables.len();
        let num_combinations = 1 << num_vars;

        for i in 0..num_combinations {
            let mut assignment = HashMap::new();
            for (j, var) in variables.iter().enumerate() {
                assignment.insert(var.clone(), (i >> j) & 1 == 1);
            }

            // 创建临时逻辑系统进行求值
            let mut temp_logic = self.clone();
            for (var, value) in &assignment {
                temp_logic.set_valuation(var, *value);
            }

            let truth_value = temp_logic.evaluate(formula).unwrap_or(false);

            rows.push(TruthTableRow {
                assignment,
                truth_value,
            });
        }

        TruthTable { variables, rows }
    }

    /// 收集公式中的变量
    fn collect_variables(&self, formula: &Formula, variables: &mut Vec<String>) {
        match formula {
            Formula::Atom(name) => {
                if !variables.contains(name) {
                    variables.push(name.clone());
                }
            }
            Formula::Not(f) => self.collect_variables(f, variables),
            Formula::And(f1, f2) => {
                self.collect_variables(f1, variables);
                self.collect_variables(f2, variables);
            }
            Formula::Or(f1, f2) => {
                self.collect_variables(f1, variables);
                self.collect_variables(f2, variables);
            }
            Formula::Implies(f1, f2) => {
                self.collect_variables(f1, variables);
                self.collect_variables(f2, variables);
            }
            Formula::Iff(f1, f2) => {
                self.collect_variables(f1, variables);
                self.collect_variables(f2, variables);
            }
            _ => {} // 忽略量词和模态算子
        }
    }

    /// 检查公式是否为重言式
    pub fn is_tautology(&self, formula: &Formula) -> bool {
        let truth_table = self.generate_truth_table(formula);
        truth_table.rows.iter().all(|row| row.truth_value)
    }

    /// 检查公式是否为矛盾式
    pub fn is_contradiction(&self, formula: &Formula) -> bool {
        let truth_table = self.generate_truth_table(formula);
        truth_table.rows.iter().all(|row| !row.truth_value)
    }

    /// 检查公式是否为可满足式
    pub fn is_satisfiable(&self, formula: &Formula) -> bool {
        let truth_table = self.generate_truth_table(formula);
        truth_table.rows.iter().any(|row| row.truth_value)
    }
}

impl FirstOrderLogic {
    /// 创建新的一阶逻辑系统
    pub fn new() -> Self {
        FirstOrderLogic {
            constants: HashSet::new(),
            functions: HashMap::new(),
            predicates: HashMap::new(),
            domain: Vec::new(),
            interpretation: HashMap::new(),
        }
    }

    /// 添加常量
    pub fn add_constant(&mut self, constant: String) {
        self.constants.insert(constant);
    }

    /// 添加函数符号
    pub fn add_function(&mut self, name: String, arity: usize) {
        self.functions.insert(name, arity);
    }

    /// 添加谓词符号
    pub fn add_predicate(&mut self, name: String, arity: usize) {
        self.predicates.insert(name, arity);
    }

    /// 设置论域
    pub fn set_domain(&mut self, domain: Vec<String>) {
        self.domain = domain;
    }

    /// 设置解释
    pub fn set_interpretation(&mut self, symbol: String, interpretation: String) {
        self.interpretation.insert(symbol, interpretation);
    }

    /// 检查项是否良构
    pub fn is_well_formed_term(&self, term: &Term) -> bool {
        match term {
            Term::Variable(_) => true,
            Term::Constant(name) => self.constants.contains(name),
            Term::Function(name, args) => {
                if let Some(&arity) = self.functions.get(name) {
                    args.len() == arity && args.iter().all(|arg| self.is_well_formed_term(arg))
                } else {
                    false
                }
            }
        }
    }

    /// 检查原子公式是否良构
    pub fn is_well_formed_atomic(&self, atomic: &AtomicFormula) -> bool {
        if let Some(&arity) = self.predicates.get(&atomic.predicate) {
            atomic.terms.len() == arity && 
            atomic.terms.iter().all(|term| self.is_well_formed_term(term))
        } else {
            false
        }
    }

    /// 检查公式是否良构
    pub fn is_well_formed_formula(&self, formula: &Formula) -> bool {
        match formula {
            Formula::Atom(name) => {
                // 简化处理，实际应该解析为原子公式
                self.predicates.contains_key(name)
            }
            Formula::Not(f) => self.is_well_formed_formula(f),
            Formula::And(f1, f2) => {
                self.is_well_formed_formula(f1) && self.is_well_formed_formula(f2)
            }
            Formula::Or(f1, f2) => {
                self.is_well_formed_formula(f1) && self.is_well_formed_formula(f2)
            }
            Formula::Implies(f1, f2) => {
                self.is_well_formed_formula(f1) && self.is_well_formed_formula(f2)
            }
            Formula::Iff(f1, f2) => {
                self.is_well_formed_formula(f1) && self.is_well_formed_formula(f2)
            }
            Formula::Forall(_, f) => self.is_well_formed_formula(f),
            Formula::Exists(_, f) => self.is_well_formed_formula(f),
            _ => false, // 一阶逻辑不支持模态算子
        }
    }
}

impl ResolutionSystem {
    /// 创建新的归结推理系统
    pub fn new() -> Self {
        ResolutionSystem {
            clauses: Vec::new(),
            resolution_rules: vec![
                ResolutionRule {
                    name: "Binary Resolution".to_string(),
                    premises: vec![
                        Clause { literals: vec![Literal::Positive("p".to_string())] },
                        Clause { literals: vec![Literal::Negative("p".to_string())] },
                    ],
                    conclusion: Clause { literals: vec![] },
                },
            ],
        }
    }

    /// 添加子句
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// 归结推理
    pub fn resolve(&self, clause1: &Clause, clause2: &Clause) -> Option<Clause> {
        for literal1 in &clause1.literals {
            for literal2 in &clause2.literals {
                if self.can_resolve(literal1, literal2) {
                    let mut new_literals = Vec::new();
                    
                    // 添加clause1中除了literal1之外的所有文字
                    for lit in &clause1.literals {
                        if lit != literal1 {
                            new_literals.push(lit.clone());
                        }
                    }
                    
                    // 添加clause2中除了literal2之外的所有文字
                    for lit in &clause2.literals {
                        if lit != literal2 {
                            new_literals.push(lit.clone());
                        }
                    }
                    
                    // 去重
                    new_literals.sort();
                    new_literals.dedup();
                    
                    return Some(Clause { literals: new_literals });
                }
            }
        }
        None
    }

    /// 检查两个文字是否可以归结
    fn can_resolve(&self, literal1: &Literal, literal2: &Literal) -> bool {
        match (literal1, literal2) {
            (Literal::Positive(name1), Literal::Negative(name2)) => name1 == name2,
            (Literal::Negative(name1), Literal::Positive(name2)) => name1 == name2,
            _ => false,
        }
    }

    /// 归结反驳
    pub fn resolution_refutation(&self, goal: &Clause) -> Option<Vec<Clause>> {
        let mut clauses = self.clauses.clone();
        clauses.push(goal.clone());
        
        let mut new_clauses = Vec::new();
        let mut proof = Vec::new();
        
        loop {
            let mut found_new = false;
            
            for i in 0..clauses.len() {
                for j in (i + 1)..clauses.len() {
                    if let Some(resolvent) = self.resolve(&clauses[i], &clauses[j]) {
                        if !clauses.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent.clone());
                            proof.push(resolvent.clone());
                            found_new = true;
                            
                            // 检查是否得到空子句
                            if resolvent.literals.is_empty() {
                                return Some(proof);
                            }
                        }
                    }
                }
            }
            
            if !found_new {
                break;
            }
            
            clauses.extend(new_clauses.drain(..));
        }
        
        None
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(name) => write!(f, "{}", name),
            Term::Constant(name) => write!(f, "{}", name),
            Term::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for AtomicFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.predicate)?;
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", term)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Positive(name) => write!(f, "{}", name),
            Literal::Negative(name) => write!(f, "¬{}", name),
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.literals.is_empty() {
            write!(f, "⊥")
        } else {
            for (i, literal) in self.literals.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∨ ")?;
                }
                write!(f, "{}", literal)?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for TruthTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 打印表头
        for var in &self.variables {
            write!(f, "{:>8} ", var)?;
        }
        writeln!(f, "{:>8}", "Result")?;
        
        // 打印分隔线
        for _ in &self.variables {
            write!(f, "{:>8} ", "--------")?;
        }
        writeln!(f, "{:>8}", "--------")?;
        
        // 打印数据行
        for row in &self.rows {
            for var in &self.variables {
                let value = row.assignment.get(var).unwrap_or(&false);
                write!(f, "{:>8} ", if *value { "T" } else { "F" })?;
            }
            writeln!(f, "{:>8}", if row.truth_value { "T" } else { "F" })?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propositional_logic() {
        let mut logic = PropositionalLogic::new();
        logic.add_atom("p".to_string());
        logic.add_atom("q".to_string());
        
        let formula = Formula::And(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("q".to_string())),
        );
        
        logic.set_valuation("p", true);
        logic.set_valuation("q", true);
        
        assert_eq!(logic.evaluate(&formula), Some(true));
    }

    #[test]
    fn test_truth_table() {
        let mut logic = PropositionalLogic::new();
        logic.add_atom("p".to_string());
        logic.add_atom("q".to_string());
        
        let formula = Formula::Implies(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("q".to_string())),
        );
        
        let truth_table = logic.generate_truth_table(&formula);
        assert_eq!(truth_table.rows.len(), 4);
    }

    #[test]
    fn test_tautology() {
        let mut logic = PropositionalLogic::new();
        logic.add_atom("p".to_string());
        
        let formula = Formula::Or(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Not(Box::new(Formula::Atom("p".to_string())))),
        );
        
        assert!(logic.is_tautology(&formula));
    }

    #[test]
    fn test_first_order_logic() {
        let mut logic = FirstOrderLogic::new();
        logic.add_constant("a".to_string());
        logic.add_function("f".to_string(), 1);
        logic.add_predicate("P".to_string(), 1);
        
        let term = Term::Function("f".to_string(), vec![Term::Constant("a".to_string())]);
        assert!(logic.is_well_formed_term(&term));
    }

    #[test]
    fn test_resolution() {
        let mut system = ResolutionSystem::new();
        
        let clause1 = Clause {
            literals: vec![Literal::Positive("p".to_string()), Literal::Positive("q".to_string())],
        };
        let clause2 = Clause {
            literals: vec![Literal::Negative("p".to_string())],
        };
        
        let resolvent = system.resolve(&clause1, &clause2);
        assert!(resolvent.is_some());
        
        if let Some(res) = resolvent {
            assert_eq!(res.literals.len(), 1);
            assert_eq!(res.literals[0], Literal::Positive("q".to_string()));
        }
    }
} 