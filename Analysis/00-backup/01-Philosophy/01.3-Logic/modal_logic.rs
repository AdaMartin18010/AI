//! 模态逻辑实现
//! 
//! 本模块提供了模态逻辑的完整实现，包括：
//! - 可能世界语义
//! - Kripke模型
//! - 模态逻辑系统（K、T、S4、S5等）
//! - 模态公式求值
//! - 模型检查

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::Formula;

/// 可能世界
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct World {
    /// 世界标识符
    pub id: String,
    /// 世界描述
    pub description: String,
    /// 世界属性
    pub properties: HashMap<String, bool>,
}

/// 可达关系
#[derive(Debug, Clone)]
pub struct AccessibilityRelation {
    /// 关系名称
    pub name: String,
    /// 关系集合：从世界到世界集合的映射
    pub relation: HashMap<String, HashSet<String>>,
}

/// Kripke模型
#[derive(Debug, Clone)]
pub struct KripkeModel {
    /// 可能世界集合
    pub worlds: HashMap<String, World>,
    /// 可达关系
    pub accessibility: AccessibilityRelation,
    /// 赋值函数：世界 -> 原子命题 -> 真值
    pub valuation: HashMap<String, HashMap<String, bool>>,
}

/// 模态逻辑系统
#[derive(Debug, Clone, PartialEq)]
pub enum ModalSystem {
    K,   // 基本模态逻辑
    T,   // 自反模态逻辑
    S4,  // 传递自反模态逻辑
    S5,  // 等价关系模态逻辑
}

/// 模态公式求值器
#[derive(Debug, Clone)]
pub struct ModalEvaluator {
    /// Kripke模型
    pub model: KripkeModel,
    /// 当前世界
    pub current_world: String,
}

impl World {
    /// 创建新的可能世界
    pub fn new(id: String, description: String) -> Self {
        World {
            id,
            description,
            properties: HashMap::new(),
        }
    }

    /// 设置世界属性
    pub fn set_property(&mut self, property: String, value: bool) {
        self.properties.insert(property, value);
    }

    /// 获取世界属性
    pub fn get_property(&self, property: &str) -> Option<bool> {
        self.properties.get(property).copied()
    }
}

impl AccessibilityRelation {
    /// 创建新的可达关系
    pub fn new(name: String) -> Self {
        AccessibilityRelation {
            name,
            relation: HashMap::new(),
        }
    }

    /// 添加可达关系
    pub fn add_relation(&mut self, from: String, to: String) {
        self.relation.entry(from).or_insert_with(HashSet::new).insert(to);
    }

    /// 检查两个世界之间是否存在可达关系
    pub fn is_accessible(&self, from: &str, to: &str) -> bool {
        self.relation.get(from).map_or(false, |targets| targets.contains(to))
    }

    /// 获取从指定世界可达的所有世界
    pub fn get_accessible_worlds(&self, from: &str) -> HashSet<String> {
        self.relation.get(from).cloned().unwrap_or_default()
    }

    /// 检查关系是否自反
    pub fn is_reflexive(&self) -> bool {
        for world in self.relation.keys() {
            if !self.is_accessible(world, world) {
                return false;
            }
        }
        true
    }

    /// 检查关系是否对称
    pub fn is_symmetric(&self) -> bool {
        for (from, targets) in &self.relation {
            for to in targets {
                if !self.is_accessible(to, from) {
                    return false;
                }
            }
        }
        true
    }

    /// 检查关系是否传递
    pub fn is_transitive(&self) -> bool {
        for (from, targets) in &self.relation {
            for to in targets {
                if let Some(second_targets) = self.relation.get(to) {
                    for second_to in second_targets {
                        if !self.is_accessible(from, second_to) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// 检查关系是否等价（自反、对称、传递）
    pub fn is_equivalence(&self) -> bool {
        self.is_reflexive() && self.is_symmetric() && self.is_transitive()
    }
}

impl KripkeModel {
    /// 创建新的Kripke模型
    pub fn new() -> Self {
        KripkeModel {
            worlds: HashMap::new(),
            accessibility: AccessibilityRelation::new("default".to_string()),
            valuation: HashMap::new(),
        }
    }

    /// 添加可能世界
    pub fn add_world(&mut self, world: World) {
        let world_id = world.id.clone();
        self.worlds.insert(world_id.clone(), world);
        self.valuation.insert(world_id, HashMap::new());
    }

    /// 设置可达关系
    pub fn set_accessibility(&mut self, accessibility: AccessibilityRelation) {
        self.accessibility = accessibility;
    }

    /// 设置原子命题在世界中的真值
    pub fn set_valuation(&mut self, world_id: &str, atom: String, value: bool) {
        if let Some(world_valuation) = self.valuation.get_mut(world_id) {
            world_valuation.insert(atom, value);
        }
    }

    /// 获取原子命题在世界中的真值
    pub fn get_valuation(&self, world_id: &str, atom: &str) -> Option<bool> {
        self.valuation.get(world_id)?.get(atom).copied()
    }

    /// 检查模型是否满足特定系统公理
    pub fn satisfies_system(&self, system: ModalSystem) -> bool {
        match system {
            ModalSystem::K => true, // K系统没有特殊要求
            ModalSystem::T => self.accessibility.is_reflexive(),
            ModalSystem::S4 => self.accessibility.is_reflexive() && self.accessibility.is_transitive(),
            ModalSystem::S5 => self.accessibility.is_equivalence(),
        }
    }

    /// 创建满足特定系统的模型
    pub fn create_for_system(system: ModalSystem, worlds: Vec<World>) -> Self {
        let mut model = KripkeModel::new();
        
        for world in worlds {
            model.add_world(world);
        }
        
        let mut accessibility = AccessibilityRelation::new(format!("{:?}", system));
        
        // 根据系统要求设置可达关系
        match system {
            ModalSystem::K => {
                // K系统：无特殊要求
            }
            ModalSystem::T => {
                // T系统：自反关系
                for world_id in model.worlds.keys() {
                    accessibility.add_relation(world_id.clone(), world_id.clone());
                }
            }
            ModalSystem::S4 => {
                // S4系统：自反传递关系
                for world_id in model.worlds.keys() {
                    accessibility.add_relation(world_id.clone(), world_id.clone());
                }
                // 添加传递闭包
                for world_id in model.worlds.keys() {
                    for other_id in model.worlds.keys() {
                        if world_id != other_id {
                            accessibility.add_relation(world_id.clone(), other_id.clone());
                        }
                    }
                }
            }
            ModalSystem::S5 => {
                // S5系统：等价关系（全连通）
                for world_id in model.worlds.keys() {
                    for other_id in model.worlds.keys() {
                        accessibility.add_relation(world_id.clone(), other_id.clone());
                    }
                }
            }
        }
        
        model.set_accessibility(accessibility);
        model
    }
}

impl ModalEvaluator {
    /// 创建新的模态公式求值器
    pub fn new(model: KripkeModel, current_world: String) -> Self {
        ModalEvaluator {
            model,
            current_world,
        }
    }

    /// 在指定世界中求值公式
    pub fn evaluate_at_world(&self, formula: &Formula, world_id: &str) -> Option<bool> {
        match formula {
            Formula::Atom(name) => self.model.get_valuation(world_id, name),
            Formula::Not(f) => self.evaluate_at_world(f, world_id).map(|v| !v),
            Formula::And(f1, f2) => {
                let v1 = self.evaluate_at_world(f1, world_id)?;
                let v2 = self.evaluate_at_world(f2, world_id)?;
                Some(v1 && v2)
            }
            Formula::Or(f1, f2) => {
                let v1 = self.evaluate_at_world(f1, world_id)?;
                let v2 = self.evaluate_at_world(f2, world_id)?;
                Some(v1 || v2)
            }
            Formula::Implies(f1, f2) => {
                let v1 = self.evaluate_at_world(f1, world_id)?;
                let v2 = self.evaluate_at_world(f2, world_id)?;
                Some(!v1 || v2)
            }
            Formula::Iff(f1, f2) => {
                let v1 = self.evaluate_at_world(f1, world_id)?;
                let v2 = self.evaluate_at_world(f2, world_id)?;
                Some(v1 == v2)
            }
            Formula::Necessity(f) => {
                let accessible_worlds = self.model.accessibility.get_accessible_worlds(world_id);
                if accessible_worlds.is_empty() {
                    return Some(true); // 空集上的全称量词为真
                }
                for accessible_world in accessible_worlds {
                    if !self.evaluate_at_world(f, &accessible_world)? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            Formula::Possibility(f) => {
                let accessible_worlds = self.model.accessibility.get_accessible_worlds(world_id);
                for accessible_world in accessible_worlds {
                    if self.evaluate_at_world(f, &accessible_world)? {
                        return Some(true);
                    }
                }
                Some(false)
            }
            _ => None, // 不支持其他算子
        }
    }

    /// 在当前世界中求值公式
    pub fn evaluate(&self, formula: &Formula) -> Option<bool> {
        self.evaluate_at_world(formula, &self.current_world)
    }

    /// 检查公式在模型中是否有效（在所有世界中为真）
    pub fn is_valid(&self, formula: &Formula) -> bool {
        for world_id in self.model.worlds.keys() {
            if !self.evaluate_at_world(formula, world_id).unwrap_or(false) {
                return false;
            }
        }
        true
    }

    /// 检查公式在模型中是否可满足（在某个世界中为真）
    pub fn is_satisfiable(&self, formula: &Formula) -> bool {
        for world_id in self.model.worlds.keys() {
            if self.evaluate_at_world(formula, world_id).unwrap_or(false) {
                return true;
            }
        }
        false
    }

    /// 模型检查：检查公式在指定世界中是否成立
    pub fn model_check(&self, formula: &Formula, world_id: &str) -> bool {
        self.evaluate_at_world(formula, world_id).unwrap_or(false)
    }
}

/// 模态逻辑公理系统
#[derive(Debug, Clone)]
pub struct ModalAxiomSystem {
    /// 系统类型
    pub system: ModalSystem,
    /// 公理集合
    pub axioms: Vec<Formula>,
    /// 推理规则
    pub rules: Vec<String>,
}

impl ModalAxiomSystem {
    /// 创建新的模态公理系统
    pub fn new(system: ModalSystem) -> Self {
        let mut axiom_system = ModalAxiomSystem {
            system,
            axioms: Vec::new(),
            rules: vec!["Modus Ponens".to_string(), "Necessitation".to_string()],
        };

        // 添加系统特定的公理
        match system {
            ModalSystem::K => {
                // K公理：□(φ → ψ) → (□φ → □ψ)
                axiom_system.axioms.push(
                    Formula::Implies(
                        Box::new(Formula::Necessity(Box::new(
                            Formula::Implies(
                                Box::new(Formula::Atom("φ".to_string())),
                                Box::new(Formula::Atom("ψ".to_string())),
                            )
                        ))),
                        Box::new(Formula::Implies(
                            Box::new(Formula::Necessity(Box::new(Formula::Atom("φ".to_string())))),
                            Box::new(Formula::Necessity(Box::new(Formula::Atom("ψ".to_string())))),
                        )),
                    )
                );
            }
            ModalSystem::T => {
                // T公理：□φ → φ
                axiom_system.axioms.push(
                    Formula::Implies(
                        Box::new(Formula::Necessity(Box::new(Formula::Atom("φ".to_string())))),
                        Box::new(Formula::Atom("φ".to_string())),
                    )
                );
            }
            ModalSystem::S4 => {
                // 4公理：□φ → □□φ
                axiom_system.axioms.push(
                    Formula::Implies(
                        Box::new(Formula::Necessity(Box::new(Formula::Atom("φ".to_string())))),
                        Box::new(Formula::Necessity(Box::new(
                            Formula::Necessity(Box::new(Formula::Atom("φ".to_string())))
                        ))),
                    )
                );
            }
            ModalSystem::S5 => {
                // B公理：φ → □◇φ
                axiom_system.axioms.push(
                    Formula::Implies(
                        Box::new(Formula::Atom("φ".to_string())),
                        Box::new(Formula::Necessity(Box::new(
                            Formula::Possibility(Box::new(Formula::Atom("φ".to_string())))
                        ))),
                    )
                );
            }
        }

        axiom_system
    }

    /// 检查公式是否为系统定理
    pub fn is_theorem(&self, formula: &Formula) -> bool {
        // 简化实现，实际应该进行形式证明
        self.axioms.contains(formula)
    }

    /// 获取系统描述
    pub fn get_description(&self) -> String {
        match self.system {
            ModalSystem::K => "基本模态逻辑：无特殊公理".to_string(),
            ModalSystem::T => "自反模态逻辑：□φ → φ".to_string(),
            ModalSystem::S4 => "传递自反模态逻辑：□φ → □□φ".to_string(),
            ModalSystem::S5 => "等价关系模态逻辑：φ → □◇φ".to_string(),
        }
    }
}

impl fmt::Display for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "World({})", self.id)
    }
}

impl fmt::Display for KripkeModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Kripke Model:")?;
        writeln!(f, "Worlds:")?;
        for world in self.worlds.values() {
            writeln!(f, "  {}", world)?;
        }
        writeln!(f, "Accessibility Relations:")?;
        for (from, targets) in &self.accessibility.relation {
            write!(f, "  {} -> {{", from)?;
            for (i, target) in targets.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", target)?;
            }
            writeln!(f, "}}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation() {
        let mut world = World::new("w1".to_string(), "World 1".to_string());
        world.set_property("p".to_string(), true);
        world.set_property("q".to_string(), false);
        
        assert_eq!(world.get_property("p"), Some(true));
        assert_eq!(world.get_property("q"), Some(false));
    }

    #[test]
    fn test_accessibility_relation() {
        let mut relation = AccessibilityRelation::new("R".to_string());
        relation.add_relation("w1".to_string(), "w2".to_string());
        relation.add_relation("w1".to_string(), "w3".to_string());
        
        assert!(relation.is_accessible("w1", "w2"));
        assert!(relation.is_accessible("w1", "w3"));
        assert!(!relation.is_accessible("w2", "w1"));
    }

    #[test]
    fn test_kripke_model() {
        let mut model = KripkeModel::new();
        
        let world1 = World::new("w1".to_string(), "World 1".to_string());
        let world2 = World::new("w2".to_string(), "World 2".to_string());
        
        model.add_world(world1);
        model.add_world(world2);
        
        model.set_valuation("w1", "p".to_string(), true);
        model.set_valuation("w2", "p".to_string(), false);
        
        assert_eq!(model.get_valuation("w1", "p"), Some(true));
        assert_eq!(model.get_valuation("w2", "p"), Some(false));
    }

    #[test]
    fn test_modal_evaluation() {
        let mut model = KripkeModel::new();
        
        let world1 = World::new("w1".to_string(), "World 1".to_string());
        let world2 = World::new("w2".to_string(), "World 2".to_string());
        
        model.add_world(world1);
        model.add_world(world2);
        
        model.set_valuation("w1", "p".to_string(), true);
        model.set_valuation("w2", "p".to_string(), true);
        
        let mut accessibility = AccessibilityRelation::new("R".to_string());
        accessibility.add_relation("w1".to_string(), "w2".to_string());
        model.set_accessibility(accessibility);
        
        let evaluator = ModalEvaluator::new(model, "w1".to_string());
        
        let formula = Formula::Necessity(Box::new(Formula::Atom("p".to_string())));
        assert_eq!(evaluator.evaluate(&formula), Some(true));
    }

    #[test]
    fn test_modal_systems() {
        let worlds = vec![
            World::new("w1".to_string(), "World 1".to_string()),
            World::new("w2".to_string(), "World 2".to_string()),
        ];
        
        let s5_model = KripkeModel::create_for_system(ModalSystem::S5, worlds);
        assert!(s5_model.satisfies_system(ModalSystem::S5));
    }

    #[test]
    fn test_axiom_system() {
        let axiom_system = ModalAxiomSystem::new(ModalSystem::T);
        assert!(!axiom_system.axioms.is_empty());
        assert_eq!(axiom_system.rules.len(), 2);
    }
} 