// 定理证明器 - 简单一阶逻辑推理系统实现
// 本实现演示了合一算法和归结原理的基本实现

use std::collections::{HashMap, HashSet};
use std::fmt;

/// 项 - 表示一阶逻辑中的项
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Term {
    Variable(String),       // 变量，如 x, y, z
    Constant(String),       // 常量，如 a, b, c
    Function(String, Vec<Term>), // 函数应用，如 f(x, g(y))
}

/// 文字 - 表示一阶逻辑中的原子公式或其否定
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Literal {
    predicate: String,      // 谓词名称
    args: Vec<Term>,        // 谓词参数
    polarity: bool,         // true 表示肯定，false 表示否定
}

/// 子句 - 文字的析取
#[derive(Clone, PartialEq, Eq, Debug)]
struct Clause {
    literals: HashSet<Literal>,
}

/// 代入 - 变量到项的映射
type Substitution = HashMap<String, Term>;

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Term::Variable(name) => write!(f, "{}", name),
            Term::Constant(name) => write!(f, "{}", name),
            Term::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")? }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.polarity {
            write!(f, "¬")?;
        }
        write!(f, "{}(", self.predicate)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 { write!(f, ", ")? }
            write!(f, "{}", arg)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.literals.is_empty() {
            return write!(f, "□"); // 空子句表示矛盾
        }
        
        let mut first = true;
        for literal in &self.literals {
            if !first { write!(f, " ∨ ")? }
            write!(f, "{}", literal)?;
            first = false;
        }
        Ok(())
    }
}

impl Term {
    /// 应用代入到项
    fn apply_substitution(&self, subst: &Substitution) -> Term {
        match self {
            Term::Variable(name) => {
                if let Some(term) = subst.get(name) {
                    term.clone()
                } else {
                    self.clone()
                }
            },
            Term::Constant(_) => self.clone(),
            Term::Function(name, args) => {
                let new_args = args.iter()
                    .map(|arg| arg.apply_substitution(subst))
                    .collect();
                Term::Function(name.clone(), new_args)
            }
        }
    }

    /// 检查变量是否在项中出现
    fn contains_var(&self, var_name: &str) -> bool {
        match self {
            Term::Variable(name) => name == var_name,
            Term::Constant(_) => false,
            Term::Function(_, args) => {
                args.iter().any(|arg| arg.contains_var(var_name))
            }
        }
    }
}

impl Literal {
    /// 应用代入到文字
    fn apply_substitution(&self, subst: &Substitution) -> Literal {
        let new_args = self.args.iter()
            .map(|arg| arg.apply_substitution(subst))
            .collect();
        
        Literal {
            predicate: self.predicate.clone(),
            args: new_args,
            polarity: self.polarity,
        }
    }
    
    /// 文字的否定
    fn negate(&self) -> Literal {
        Literal {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            polarity: !self.polarity,
        }
    }
}

impl Clause {
    /// 创建新子句
    fn new(literals: Vec<Literal>) -> Self {
        Clause {
            literals: literals.into_iter().collect(),
        }
    }
    
    /// 应用代入到子句
    fn apply_substitution(&self, subst: &Substitution) -> Clause {
        let new_literals = self.literals.iter()
            .map(|lit| lit.apply_substitution(subst))
            .collect();
        
        Clause {
            literals: new_literals,
        }
    }
}

/// 定理证明器
struct TheoremProver {
    clauses: Vec<Clause>,
    derived_clauses: HashSet<Clause>,
}

impl TheoremProver {
    /// 创建新的定理证明器
    fn new() -> Self {
        TheoremProver {
            clauses: Vec::new(),
            derived_clauses: HashSet::new(),
        }
    }
    
    /// 添加子句到知识库
    fn add_clause(&mut self, clause: Clause) {
        if !self.derived_clauses.contains(&clause) {
            println!("添加子句: {}", clause);
            self.clauses.push(clause.clone());
            self.derived_clauses.insert(clause);
        }
    }
    
    /// 尝试合一两个项
    fn unify(&self, t1: &Term, t2: &Term) -> Option<Substitution> {
        self.unify_with_subst(t1, t2, HashMap::new())
    }
    
    /// 带有当前代入的合一算法
    fn unify_with_subst(&self, t1: &Term, t2: &Term, mut subst: Substitution) -> Option<Substitution> {
        match (t1, t2) {
            // 变量与任意项的合一
            (Term::Variable(name), _) => {
                if let Some(term) = subst.get(name) {
                    // 如果变量已被绑定，则递归合一
                    self.unify_with_subst(term, t2, subst)
                } else if let Term::Variable(name2) = t2 {
                    if name == name2 {
                        // 相同变量
                        Some(subst)
                    } else {
                        // 绑定变量
                        subst.insert(name.clone(), t2.clone());
                        Some(subst)
                    }
                } else if t2.contains_var(name) {
                    // 出现检测失败
                    None
                } else {
                    // 绑定变量到项
                    subst.insert(name.clone(), t2.clone());
                    Some(subst)
                }
            },
            // 项与变量的合一（交换参数重用上面的逻辑）
            (_, Term::Variable(_)) => {
                self.unify_with_subst(t2, t1, subst)
            },
            // 常量与常量的合一
            (Term::Constant(name1), Term::Constant(name2)) => {
                if name1 == name2 {
                    Some(subst)
                } else {
                    None
                }
            },
            // 函数与函数的合一
            (Term::Function(name1, args1), Term::Function(name2, args2)) => {
                if name1 != name2 || args1.len() != args2.len() {
                    return None;
                }
                
                // 逐个合一参数
                let mut current_subst = subst;
                for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                    if let Some(new_subst) = self.unify_with_subst(
                        &arg1.apply_substitution(&current_subst), 
                        &arg2.apply_substitution(&current_subst), 
                        current_subst
                    ) {
                        current_subst = new_subst;
                    } else {
                        return None;
                    }
                }
                Some(current_subst)
            },
            // 其他组合无法合一
            _ => None,
        }
    }
    
    /// 尝试合一两个文字
    fn unify_literals(&self, lit1: &Literal, lit2: &Literal) -> Option<Substitution> {
        // 文字必须具有相同谓词和相反极性
        if lit1.predicate != lit2.predicate || lit1.polarity == lit2.polarity {
            return None;
        }
        
        // 参数数量必须相同
        if lit1.args.len() != lit2.args.len() {
            return None;
        }
        
        // 尝试合一所有参数
        let mut subst = HashMap::new();
        for (arg1, arg2) in lit1.args.iter().zip(lit2.args.iter()) {
            if let Some(new_subst) = self.unify_with_subst(
                &arg1.apply_substitution(&subst), 
                &arg2.apply_substitution(&subst), 
                subst
            ) {
                subst = new_subst;
            } else {
                return None;
            }
        }
        
        Some(subst)
    }
    
    /// 使用归结原理从两个子句生成新子句
    fn resolve(&self, clause1: &Clause, clause2: &Clause) -> Vec<Clause> {
        let mut results = Vec::new();
        
        // 尝试每对文字的归结
        for lit1 in &clause1.literals {
            for lit2 in &clause2.literals {
                if let Some(subst) = self.unify_literals(lit1, lit2) {
                    // 创建新子句，移除合一的文字
                    let mut new_literals = HashSet::new();
                    
                    // 添加第一个子句中的非合一文字
                    for l in &clause1.literals {
                        if l != lit1 {
                            new_literals.insert(l.apply_substitution(&subst));
                        }
                    }
                    
                    // 添加第二个子句中的非合一文字
                    for l in &clause2.literals {
                        if l != lit2 {
                            new_literals.insert(l.apply_substitution(&subst));
                        }
                    }
                    
                    // 创建新子句
                    let new_clause = Clause { literals: new_literals };
                    results.push(new_clause);
                }
            }
        }
        
        results
    }
    
    /// 使用归结原理进行推理
    fn prove(&mut self) -> bool {
        let mut i = 0;
        
        while i < self.clauses.len() {
            let mut j = 0;
            while j < self.clauses.len() {
                if i != j {
                    // 尝试归结两个子句
                    let new_clauses = self.resolve(&self.clauses[i], &self.clauses[j]);
                    
                    // 处理每个新生成的子句
                    for clause in new_clauses {
                        // 检查是否找到空子句(矛盾)
                        if clause.literals.is_empty() {
                            println!("发现空子句，证明成功！");
                            return true;
                        }
                        
                        // 添加新子句到知识库
                        self.add_clause(clause);
                    }
                }
                j += 1;
            }
            i += 1;
        }
        
        println!("无法证明目标命题。");
        false
    }
}

/// 示例：证明定理 P(a) ∧ (∀x. P(x) → Q(x)) → Q(a)
fn main() {
    let mut prover = TheoremProver::new();
    
    // 添加前提 P(a)
    prover.add_clause(Clause::new(vec![
        Literal { predicate: "P".to_string(), args: vec![Term::Constant("a".to_string())], polarity: true }
    ]));
    
    // 添加前提 ∀x. P(x) → Q(x)，转换为子句 ¬P(x) ∨ Q(x)
    prover.add_clause(Clause::new(vec![
        Literal { predicate: "P".to_string(), args: vec![Term::Variable("x".to_string())], polarity: false },
        Literal { predicate: "Q".to_string(), args: vec![Term::Variable("x".to_string())], polarity: true }
    ]));
    
    // 添加否定的结论 ¬Q(a)
    prover.add_clause(Clause::new(vec![
        Literal { predicate: "Q".to_string(), args: vec![Term::Constant("a".to_string())], polarity: false }
    ]));
    
    // 进行证明
    println!("开始证明...");
    let result = prover.prove();
    println!("证明结果：{}", if result { "成功" } else { "失败" });
} 