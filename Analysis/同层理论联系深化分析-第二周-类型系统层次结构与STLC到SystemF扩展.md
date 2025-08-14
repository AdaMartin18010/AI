# 同层理论联系深化分析 - 第二周：类型系统层次结构与 STLC → System F 扩展

## 一、分析概述

### 1.1 目标

- 给出类型系统层次（无类型、STLC、System F、System Fω、CoC）与表达能力关系。
- 构建 STLC → System F 的系统化扩展与翻译框架（类型、项、推导规则）。
- 明确可保性（preservation）、可进性（progress）与擦除语义（erasure）的联系。

### 1.2 范围

- 形式化：语法、类型、推导规则、归约规则
- 扩展映射：类型扩展、项扩展、规则扩展
- 算法接口：类型检查/推导、注解与实例化

---

## 二、类型系统层次与能力

```mermaid
graph TD
    A[无类型 λ 演算] --> B[简单类型 λ 演算 STLC]
    B --> C[System F (参数多态)]
    C --> D[System Fω (高阶多态)]
    D --> E[CoC 构造演算 (依赖类型)]

    B --> B1[静态安全]
    C --> C1[参数多态]
    D --> D1[类型算子]
    E --> E1[依赖类型/定理证明]
```

- 表达能力：无类型 < STLC < System F < System Fω < CoC
- 类型检查复杂度：随表达能力递增而上升；STLC 判定性强，System F 类型推断不可判定（需显式注解）。

---

## 三、STLC 语法与规则（基线）

### 3.1 语法

- 类型：T ::= α | T → T | T × T | T + T | 1 | 0
- 项：e ::= x | λx:T.e | e e | (e, e) | fst e | snd e | inl e | inr e | case e of ... | () | abort e

### 3.2 归约与值

- 值：v ::= λx:T.e | (v, v) | inl v | inr v | ()
- β-归约： (λx:T.e) v ↦ e[x := v]

### 3.3 类型规则（片段）

```rust
// 以伪代码刻画推导
fn type_check(ctx: Ctx, e: Expr) -> Type {
    match e {
        Expr::Var(x) => ctx.lookup(x),
        Expr::Abs(x, t, body) => {
            let t_body = type_check(ctx.extend(x, t.clone()), *body);
            Type::Arrow(Box::new(t), Box::new(t_body))
        }
        Expr::App(e1, e2) => {
            let t1 = type_check(ctx.clone(), *e1);
            let t2 = type_check(ctx, *e2);
            match t1 { Type::Arrow(a, b) if *a == t2 => *b, _ => TypeError }
        }
        _ => unimplemented!()
    }
}
```

---

## 四、System F 扩展（∀-多态）

### 4.1 语法

- 类型：S ::= α | S → S | ∀α.S
- 项：t ::= x | λx:S.t | t t | Λα.t | t [S]

- 直觉：Λα.t 为类型抽象；t [S] 为类型应用。

### 4.2 关键规则（片段）

- 一般化（type abstraction）：Γ ⊢ t : T  ⇒  Γ ⊢ Λα.t : ∀α.T  （α 不自由于 Γ）
- 实例化（type application）：Γ ⊢ t : ∀α.T  ⇒  Γ ⊢ t [S] : T[α := S]

### 4.3 STLC → System F 扩展/翻译

```rust
// 类型与项的扩展（注解驱动）
struct STLCToSystemF;

impl STLCToSystemF {
    // 类型扩展（保守嵌入）
    fn extend_type(t: StlcType) -> SysFType {
        match t {
            StlcType::Base(b) => SysFType::Base(b),
            StlcType::Arrow(a, b) => SysFType::Arrow(Box::new(Self::extend_type(*a)), Box::new(Self::extend_type(*b))),
            StlcType::Prod(a, b) => SysFType::Prod(Box::new(Self::extend_type(*a)), Box::new(Self::extend_type(*b))),
        }
    }

    // 项扩展（类型标注保持）
    fn extend_term(e: StlcExpr) -> SysFExpr {
        match e {
            StlcExpr::Var(x) => SysFExpr::Var(x),
            StlcExpr::Abs(x, t, body) => {
                SysFExpr::Abs(x, Self::extend_type(t), Box::new(Self::extend_term(*body)))
            }
            StlcExpr::App(e1, e2) => {
                SysFExpr::App(Box::new(Self::extend_term(*e1)), Box::new(Self::extend_term(*e2)))
            }
            _ => unimplemented!()
        }
    }
}
```

- 多态化策略：在库级 API 上通过 ∀-抽象将 STLC 中“模板”函数显式化；在调用点以类型应用进行实例化。

### 4.4 擦除与语义一致性

- 擦除：erasure(Λα.t) = erasure(t)，erasure(t [S]) = erasure(t)
- 语义：System F 的运行时与 STLC 等价（忽略类型），多态仅作为静态层结构提供参数化。

---

## 五、性质：保型与可进

- Preservation（类型保留）：若 Γ ⊢ t : T 且 t ↦ t'，则 Γ ⊢ t' : T
- Progress（可进性）：若 ⊢ t : T，则 t 是值或存在 t' 使得 t ↦ t'

证明思路要点：
- 归纳于推导结构；关键在于应用与类型应用的相容性；
- 替换引理：值替换与类型替换分别与语法/规则良好交互；
- 规范化：强规范化在 System F 不平凡，需借助可还原性语义（Tait/Girard）。

---

## 六、算法接口与工程实践

- STLC：算法化类型推导（约束/合一/HM 变体）
- System F：需显式类型注解；提供“局部推断/占位符填充”工程折中

```rust
// 伪代码：局部推断接口
fn infer_partially(ctx: Ctx, term: SysFExpr) -> Result<SysFType, HoleConstraints> {
    // 对显式注解部分直接检查；对占位符生成约束；交给 SMT/合一器
    unimplemented!()
}
```

---

## 七、例子（map 的多态化）

- STLC：map : (A → B) → List A → List B
- System F：map : ∀A. ∀B. (A → B) → List A → List B

调用：map [Int] [Bool] (isZero) xs

---

## 八、结论与后续

- 已建立 STLC → System F 的扩展与翻译骨架，明确性质与擦除语义。
- 下一步进入“类型安全与推导正确性”的形式化证明细化与算法化实现接口。 