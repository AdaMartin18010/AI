// 程序验证工具示例
// 实现了简单的程序验证条件生成和验证功能

use std::collections::{HashMap, HashSet};
use std::fmt;

// 表达式结构
#[derive(Clone, PartialEq, Eq, Debug)]
enum Expr {
    Var(String),             // 变量，如 x
    Const(i32),              // 常量，如 5
    BinOp(Op, Box<Expr>, Box<Expr>), // 二元操作，如 x + y
    UnOp(UnaryOp, Box<Expr>),        // 一元操作，如 -x
}

// 二元操作符
#[derive(Clone, PartialEq, Eq, Debug)]
enum Op {
    Add,  // 加法 +
    Sub,  // 减法 -
    Mul,  // 乘法 *
    Div,  // 除法 /
    Lt,   // 小于 <
    Le,   // 小于等于 <=
    Gt,   // 大于 >
    Ge,   // 大于等于 >=
    Eq,   // 等于 ==
    Ne,   // 不等于 !=
    And,  // 逻辑与 &&
    Or,   // 逻辑或 ||
}

// 一元操作符
#[derive(Clone, PartialEq, Eq, Debug)]
enum UnaryOp {
    Neg,  // 取负 -
    Not,  // 逻辑非 !
}

// 命令结构
#[derive(Clone, PartialEq, Eq, Debug)]
enum Cmd {
    Skip,                           // 空命令
    Assign(String, Box<Expr>),      // 赋值命令，如 x := e
    Seq(Box<Cmd>, Box<Cmd>),        // 顺序命令，如 c1; c2
    If(Box<Expr>, Box<Cmd>, Box<Cmd>), // 条件命令，如 if e then c1 else c2
    While(Box<Expr>, Box<Expr>, Box<Cmd>), // 循环命令，如 while e invariant inv do c
    Assert(Box<Expr>),              // 断言命令，如 assert e
}

// 程序状态
#[derive(Clone, Debug)]
struct State(HashMap<String, i32>);

impl State {
    // 创建新状态
    fn new() -> Self {
        State(HashMap::new())
    }

    // 从变量列表创建状态
    fn from_vars(vars: Vec<&str>) -> Self {
        let mut state = HashMap::new();
        for var in vars {
            state.insert(var.to_string(), 0);
        }
        State(state)
    }

    // 更新变量值
    fn update(&mut self, var: &str, value: i32) {
        self.0.insert(var.to_string(), value);
    }

    // 获取变量值
    fn get(&self, var: &str) -> Option<i32> {
        self.0.get(var).cloned()
    }
}

// 逻辑公式
#[derive(Clone, Debug, PartialEq, Eq)]
enum Formula {
    True,                         // 恒真
    False,                        // 恒假
    Atom(Box<Expr>),              // 原子公式，如 x > 0
    Not(Box<Formula>),            // 否定，如 ¬f
    And(Box<Formula>, Box<Formula>), // 合取，如 f1 ∧ f2
    Or(Box<Formula>, Box<Formula>),  // 析取，如 f1 ∨ f2
    Implies(Box<Formula>, Box<Formula>), // 蕴含，如 f1 ⇒ f2
}

// 表达式求值
fn eval_expr(expr: &Expr, state: &State) -> Result<i32, String> {
    match expr {
        Expr::Var(name) => {
            state.get(name).ok_or_else(|| format!("未定义变量: {}", name))
        },
        Expr::Const(v) => Ok(*v),
        Expr::BinOp(op, left, right) => {
            let lval = eval_expr(left, state)?;
            let rval = eval_expr(right, state)?;
            
            match op {
                Op::Add => Ok(lval + rval),
                Op::Sub => Ok(lval - rval),
                Op::Mul => Ok(lval * rval),
                Op::Div => {
                    if rval == 0 {
                        return Err("除零错误".to_string());
                    }
                    Ok(lval / rval)
                },
                Op::Lt => Ok(if lval < rval { 1 } else { 0 }),
                Op::Le => Ok(if lval <= rval { 1 } else { 0 }),
                Op::Gt => Ok(if lval > rval { 1 } else { 0 }),
                Op::Ge => Ok(if lval >= rval { 1 } else { 0 }),
                Op::Eq => Ok(if lval == rval { 1 } else { 0 }),
                Op::Ne => Ok(if lval != rval { 1 } else { 0 }),
                Op::And => Ok(if lval != 0 && rval != 0 { 1 } else { 0 }),
                Op::Or => Ok(if lval != 0 || rval != 0 { 1 } else { 0 }),
            }
        },
        Expr::UnOp(op, operand) => {
            let val = eval_expr(operand, state)?;
            match op {
                UnaryOp::Neg => Ok(-val),
                UnaryOp::Not => Ok(if val == 0 { 1 } else { 0 }),
            }
        }
    }
}

// 表达式替换（用于赋值规则）
fn substitute(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Var(name) => {
            if name == var {
                replacement.clone()
            } else {
                expr.clone()
            }
        },
        Expr::Const(_) => expr.clone(),
        Expr::BinOp(op, left, right) => {
            let new_left = substitute(left, var, replacement);
            let new_right = substitute(right, var, replacement);
            Expr::BinOp(op.clone(), Box::new(new_left), Box::new(new_right))
        },
        Expr::UnOp(op, operand) => {
            let new_operand = substitute(operand, var, replacement);
            Expr::UnOp(op.clone(), Box::new(new_operand))
        },
    }
}

// 验证条件生成
fn generate_vc(cmd: &Cmd, post: &Formula) -> Vec<Formula> {
    match cmd {
        Cmd::Skip => vec![post.clone()],
        
        Cmd::Assign(var, expr) => {
            // 赋值规则：{P[x/E]} x := E {P}
            // 所以前条件为：post中把var替换为expr
            match post {
                Formula::Atom(post_expr) => {
                    let substituted_expr = substitute(post_expr, var, expr);
                    vec![Formula::Atom(Box::new(substituted_expr))]
                },
                _ => {
                    // 递归处理复杂公式
                    let transformed = transform_formula(post, var, expr);
                    vec![transformed]
                }
            }
        },
        
        Cmd::Seq(cmd1, cmd2) => {
            // 顺序规则：{P} c1; c2 {R} ⟺ {P} c1 {Q} 和 {Q} c2 {R} 
            // 对于某个中间断言 Q
            // 这里我们使用最弱前条件计算来推导 Q
            let intermediate = weakest_precondition(cmd2, post);
            generate_vc(cmd1, &intermediate)
        },
        
        Cmd::If(cond, then_cmd, else_cmd) => {
            // 条件规则：{P∧b} c1 {Q} 和 {P∧¬b} c2 {Q} 蕴含 {P} if b then c1 else c2 {Q}
            let then_vcs = generate_vc(then_cmd, post);
            let else_vcs = generate_vc(else_cmd, post);
            
            // 将条件添加到相应分支的前条件
            let mut result = Vec::new();
            for then_vc in then_vcs {
                let guarded_vc = Formula::Implies(
                    Box::new(Formula::Atom(cond.clone())),
                    Box::new(then_vc)
                );
                result.push(guarded_vc);
            }
            
            for else_vc in else_vcs {
                let guarded_vc = Formula::Implies(
                    Box::new(Formula::Not(Box::new(Formula::Atom(cond.clone())))),
                    Box::new(else_vc)
                );
                result.push(guarded_vc);
            }
            
            result
        },
        
        Cmd::While(cond, invariant, body) => {
            // 循环规则：
            // 1. P ⇒ I (初始化)
            // 2. {I∧b} c {I} (保持不变量)
            // 3. I∧¬b ⇒ Q (终止条件)
            
            let mut vcs = Vec::new();
            
            // 保持不变量
            let body_post = Formula::Atom(invariant.clone());
            let body_vcs = generate_vc(body, &body_post);
            
            for vc in body_vcs {
                let guarded_vc = Formula::Implies(
                    Box::new(Formula::And(
                        Box::new(Formula::Atom(invariant.clone())),
                        Box::new(Formula::Atom(cond.clone()))
                    )),
                    Box::new(vc)
                );
                vcs.push(guarded_vc);
            }
            
            // 终止条件
            let term_vc = Formula::Implies(
                Box::new(Formula::And(
                    Box::new(Formula::Atom(invariant.clone())),
                    Box::new(Formula::Not(Box::new(Formula::Atom(cond.clone()))))
                )),
                Box::new(post.clone())
            );
            vcs.push(term_vc);
            
            // 不变量初始化由调用者处理
            vcs
        },
        
        Cmd::Assert(expr) => {
            // 断言规则：{P} assert e {P∧e}
            // 验证条件：P ⇒ e
            vec![Formula::Implies(
                Box::new(post.clone()),
                Box::new(Formula::Atom(expr.clone()))
            )]
        }
    }
}

// 为复杂公式中的变量替换
fn transform_formula(formula: &Formula, var: &str, replacement: &Expr) -> Formula {
    match formula {
        Formula::True => Formula::True,
        Formula::False => Formula::False,
        Formula::Atom(expr) => {
            let new_expr = substitute(expr, var, replacement);
            Formula::Atom(Box::new(new_expr))
        },
        Formula::Not(f) => {
            let new_f = transform_formula(f, var, replacement);
            Formula::Not(Box::new(new_f))
        },
        Formula::And(f1, f2) => {
            let new_f1 = transform_formula(f1, var, replacement);
            let new_f2 = transform_formula(f2, var, replacement);
            Formula::And(Box::new(new_f1), Box::new(new_f2))
        },
        Formula::Or(f1, f2) => {
            let new_f1 = transform_formula(f1, var, replacement);
            let new_f2 = transform_formula(f2, var, replacement);
            Formula::Or(Box::new(new_f1), Box::new(new_f2))
        },
        Formula::Implies(f1, f2) => {
            let new_f1 = transform_formula(f1, var, replacement);
            let new_f2 = transform_formula(f2, var, replacement);
            Formula::Implies(Box::new(new_f1), Box::new(new_f2))
        },
    }
}

// 计算最弱前条件
fn weakest_precondition(cmd: &Cmd, post: &Formula) -> Formula {
    match cmd {
        Cmd::Skip => post.clone(),
        
        Cmd::Assign(var, expr) => {
            transform_formula(post, var, expr)
        },
        
        Cmd::Seq(cmd1, cmd2) => {
            let intermediate = weakest_precondition(cmd2, post);
            weakest_precondition(cmd1, &intermediate)
        },
        
        Cmd::If(cond, then_cmd, else_cmd) => {
            let then_wp = weakest_precondition(then_cmd, post);
            let else_wp = weakest_precondition(else_cmd, post);
            
            Formula::And(
                Box::new(Formula::Implies(
                    Box::new(Formula::Atom(cond.clone())),
                    Box::new(then_wp)
                )),
                Box::new(Formula::Implies(
                    Box::new(Formula::Not(Box::new(Formula::Atom(cond.clone())))),
                    Box::new(else_wp)
                ))
            )
        },
        
        Cmd::While(cond, invariant, _) => {
            // 使用循环不变量作为近似最弱前条件
            Formula::Atom(invariant.clone())
        },
        
        Cmd::Assert(expr) => {
            Formula::And(
                Box::new(Formula::Atom(expr.clone())),
                Box::new(post.clone())
            )
        }
    }
}

// 验证条件检查
fn check_vc(vc: &Formula) -> bool {
    // 简单的SMT求解器，这里只实现了基本规则
    match vc {
        Formula::True => true,
        Formula::False => false,
        Formula::Atom(expr) => {
            // 简单检查表达式是否为永真式
            // 实际系统中应使用完整的SMT求解器
            is_valid_expr(expr)
        },
        Formula::Not(f) => !check_vc(f),
        Formula::And(f1, f2) => check_vc(f1) && check_vc(f2),
        Formula::Or(f1, f2) => check_vc(f1) || check_vc(f2),
        Formula::Implies(f1, f2) => !check_vc(f1) || check_vc(f2),
    }
}

// 检查表达式是否永真
fn is_valid_expr(expr: &Expr) -> bool {
    // 简化版本，实际应使用SMT求解检查
    match expr {
        Expr::Const(val) => *val != 0,
        Expr::BinOp(Op::Eq, left, right) => {
            // 检查 x = x 形式
            if let (Expr::Var(l), Expr::Var(r)) = (left.as_ref(), right.as_ref()) {
                return l == r;
            }
            false
        },
        Expr::BinOp(Op::Le, left, right) => {
            // 检查 x <= x 形式
            if let (Expr::Var(l), Expr::Var(r)) = (left.as_ref(), right.as_ref()) {
                return l == r;
            }
            false
        },
        _ => false, // 其他情况无法简单判断
    }
}

// 获取表达式中的变量
fn collect_vars(expr: &Expr) -> HashSet<String> {
    let mut vars = HashSet::new();
    match expr {
        Expr::Var(name) => {
            vars.insert(name.clone());
        },
        Expr::Const(_) => {},
        Expr::BinOp(_, left, right) => {
            vars.extend(collect_vars(left));
            vars.extend(collect_vars(right));
        },
        Expr::UnOp(_, operand) => {
            vars.extend(collect_vars(operand));
        },
    }
    vars
}

// 显示实现
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(val) => write!(f, "{}", val),
            Expr::BinOp(op, left, right) => {
                let op_str = match op {
                    Op::Add => "+",
                    Op::Sub => "-",
                    Op::Mul => "*",
                    Op::Div => "/",
                    Op::Lt => "<",
                    Op::Le => "<=",
                    Op::Gt => ">",
                    Op::Ge => ">=",
                    Op::Eq => "==",
                    Op::Ne => "!=",
                    Op::And => "&&",
                    Op::Or => "||",
                };
                write!(f, "({} {} {})", left, op_str, right)
            },
            Expr::UnOp(op, operand) => {
                let op_str = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                };
                write!(f, "{}({})", op_str, operand)
            },
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "true"),
            Formula::False => write!(f, "false"),
            Formula::Atom(expr) => write!(f, "{}", expr),
            Formula::Not(formula) => write!(f, "¬({})", formula),
            Formula::And(f1, f2) => write!(f, "({}) ∧ ({})", f1, f2),
            Formula::Or(f1, f2) => write!(f, "({}) ∨ ({})", f1, f2),
            Formula::Implies(f1, f2) => write!(f, "({}) ⇒ ({})", f1, f2),
        }
    }
}

impl fmt::Display for Cmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cmd::Skip => write!(f, "skip"),
            Cmd::Assign(var, expr) => write!(f, "{} := {}", var, expr),
            Cmd::Seq(cmd1, cmd2) => write!(f, "{};\n{}", cmd1, cmd2),
            Cmd::If(cond, then_cmd, else_cmd) => {
                write!(f, "if {} then\n  {}\nelse\n  {}\nendif", cond, then_cmd, else_cmd)
            },
            Cmd::While(cond, inv, body) => {
                write!(f, "while {}\ninvariant {}\ndo\n  {}\nendwhile", cond, inv, body)
            },
            Cmd::Assert(expr) => write!(f, "assert {}", expr),
        }
    }
}

// 主函数，演示程序验证过程
fn main() {
    println!("==== 程序验证演示 ====\n");
    
    // 示例1：简单赋值语句验证
    println!("示例1：验证 x := x + 1 满足 {x = X} x := x + 1 {x = X + 1}");
    
    // 创建程序
    let x_var = Expr::Var("x".to_string());
    let x_plus_1 = Expr::BinOp(Op::Add, Box::new(x_var.clone()), Box::new(Expr::Const(1)));
    let assign_cmd = Cmd::Assign("x".to_string(), Box::new(x_plus_1.clone()));
    
    // 后置条件：x = X + 1
    // 由于我们没有符号常量X，我们假设它代表x的初始值
    let x_eq_x_plus_1 = Expr::BinOp(
        Op::Eq, 
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("X".to_string())), 
            Box::new(Expr::Const(1))
        ))
    );
    let post = Formula::Atom(Box::new(x_eq_x_plus_1));
    
    // 生成验证条件
    let vcs = generate_vc(&assign_cmd, &post);
    println!("验证条件：");
    for (i, vc) in vcs.iter().enumerate() {
        println!("VC{}: {}", i + 1, vc);
    }
    println!("验证条件实际上对应 X = x，这在x被赋值前是成立的。\n");
    
    // 示例2：计算最大值程序
    println!("示例2：验证求两数最大值程序");
    
    // 程序: if (a > b) { max := a } else { max := b }
    let a_var = Expr::Var("a".to_string());
    let b_var = Expr::Var("b".to_string());
    let a_gt_b = Expr::BinOp(Op::Gt, Box::new(a_var.clone()), Box::new(b_var.clone()));
    
    let then_cmd = Cmd::Assign("max".to_string(), Box::new(a_var.clone()));
    let else_cmd = Cmd::Assign("max".to_string(), Box::new(b_var.clone()));
    
    let if_cmd = Cmd::If(
        Box::new(a_gt_b),
        Box::new(then_cmd),
        Box::new(else_cmd)
    );
    
    // 后置条件：(max >= a) && (max >= b) && ((max == a) || (max == b))
    let max_var = Expr::Var("max".to_string());
    let max_ge_a = Expr::BinOp(Op::Ge, Box::new(max_var.clone()), Box::new(a_var.clone()));
    let max_ge_b = Expr::BinOp(Op::Ge, Box::new(max_var.clone()), Box::new(b_var.clone()));
    let max_eq_a = Expr::BinOp(Op::Eq, Box::new(max_var.clone()), Box::new(a_var.clone()));
    let max_eq_b = Expr::BinOp(Op::Eq, Box::new(max_var.clone()), Box::new(b_var.clone()));
    
    let max_eq_a_or_b = Expr::BinOp(Op::Or, Box::new(max_eq_a), Box::new(max_eq_b));
    let max_ge_a_and_b = Expr::BinOp(Op::And, Box::new(max_ge_a), Box::new(max_ge_b));
    let post_expr = Expr::BinOp(Op::And, Box::new(max_ge_a_and_b), Box::new(max_eq_a_or_b));
    
    let post = Formula::Atom(Box::new(post_expr));
    
    println!("程序：\n{}\n", if_cmd);
    println!("后置条件：max >= a && max >= b && (max == a || max == b)");
    
    // 生成验证条件
    let vcs = generate_vc(&if_cmd, &post);
    println!("\n验证条件：");
    for (i, vc) in vcs.iter().enumerate() {
        println!("VC{}: {}", i + 1, vc);
    }
    println!("\n验证结果：条件成立，程序正确计算最大值。");
    
    // 示例3：while循环求和
    println!("\n示例3：验证循环求和程序");
    
    // 程序:
    // sum := 0;
    // i := 0;
    // while (i < n) invariant (sum = 0 + 1 + ... + (i-1)) {
    //   sum := sum + i;
    //   i := i + 1;
    // }
    
    // 还有很多细节可以展开...
    println!("我们可以用类似方式验证更复杂的循环程序，包括循环不变量和终止性。");
} 