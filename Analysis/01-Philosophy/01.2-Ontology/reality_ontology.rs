//! 现实本体论 (Reality Ontology)
//! 
//! 本模块实现了现实本体论的各种理论立场，包括实在论、反实在论、唯物论、唯心论和二元论。

#[derive(Debug, Clone)]
pub struct RealityOntology {
    pub realism: Realism,
    pub anti_realism: AntiRealism,
    pub materialism: Materialism,
    pub idealism: Idealism,
    pub dualism: Dualism,
}

impl RealityOntology {
    pub fn new() -> Self {
        Self {
            realism: Realism::new(),
            anti_realism: AntiRealism::new(),
            materialism: Materialism::new(),
            idealism: Idealism::new(),
            dualism: Dualism::new(),
        }
    }
    
    pub fn analyze_positions(&self) -> RealityAnalysis {
        RealityAnalysis {
            independent_reality: self.realism.independent_reality,
            mind_dependent: self.anti_realism.mind_dependent_reality,
            matter_fundamental: self.materialism.only_matter_exists,
            mind_fundamental: self.idealism.only_mind_exists,
            both_exist: self.dualism.both_matter_and_mind_exist,
        }
    }
}

/// 实在论 (Realism)
/// 
/// 核心观点：存在独立于心灵的客观实在
#[derive(Debug, Clone)]
pub struct Realism {
    pub independent_reality: bool,
    pub objective_truth: bool,
    pub mind_independence: bool,
}

impl Realism {
    pub fn new() -> Self {
        Self {
            independent_reality: true,
            objective_truth: true,
            mind_independence: true,
        }
    }
    
    pub fn reality_exists_independently(&self) -> bool {
        self.independent_reality
    }
    
    pub fn truth_is_objective(&self) -> bool {
        self.objective_truth
    }
    
    pub fn reality_is_mind_independent(&self) -> bool {
        self.mind_independence
    }
    
    pub fn formal_expression(&self) -> String {
        "∃x: Real(x) ∧ Independent(x, Mind)".to_string()
    }
}

/// 反实在论 (Anti-Realism)
/// 
/// 核心观点：实在依赖于心灵或语言
#[derive(Debug, Clone)]
pub struct AntiRealism {
    pub mind_dependent_reality: bool,
    pub language_dependent_truth: bool,
    pub constructivist_approach: bool,
}

impl AntiRealism {
    pub fn new() -> Self {
        Self {
            mind_dependent_reality: true,
            language_dependent_truth: true,
            constructivist_approach: true,
        }
    }
    
    pub fn reality_depends_on_mind(&self) -> bool {
        self.mind_dependent_reality
    }
    
    pub fn truth_depends_on_language(&self) -> bool {
        self.language_dependent_truth
    }
    
    pub fn is_constructivist(&self) -> bool {
        self.constructivist_approach
    }
    
    pub fn formal_expression(&self) -> String {
        "∀x: Real(x) → Dependent(x, Mind)".to_string()
    }
}

/// 唯物论 (Materialism)
/// 
/// 核心观点：物质是唯一实在，精神现象可还原为物质现象
#[derive(Debug, Clone)]
pub struct Materialism {
    pub only_matter_exists: bool,
    pub mental_reducible_to_physical: bool,
    pub causal_closure: bool,
}

impl Materialism {
    pub fn new() -> Self {
        Self {
            only_matter_exists: true,
            mental_reducible_to_physical: true,
            causal_closure: true,
        }
    }
    
    pub fn matter_is_fundamental(&self) -> bool {
        self.only_matter_exists
    }
    
    pub fn mental_is_reducible(&self) -> bool {
        self.mental_reducible_to_physical
    }
    
    pub fn physical_causation_is_closed(&self) -> bool {
        self.causal_closure
    }
    
    pub fn formal_expression(&self) -> String {
        "∀x: Exists(x) → Material(x)".to_string()
    }
}

/// 唯心论 (Idealism)
/// 
/// 核心观点：精神是唯一实在，物质现象可还原为精神现象
#[derive(Debug, Clone)]
pub struct Idealism {
    pub only_mind_exists: bool,
    pub physical_reducible_to_mental: bool,
    pub perception_dependent_reality: bool,
}

impl Idealism {
    pub fn new() -> Self {
        Self {
            only_mind_exists: true,
            physical_reducible_to_mental: true,
            perception_dependent_reality: true,
        }
    }
    
    pub fn mind_is_fundamental(&self) -> bool {
        self.only_mind_exists
    }
    
    pub fn physical_is_reducible(&self) -> bool {
        self.physical_reducible_to_mental
    }
    
    pub fn reality_depends_on_perception(&self) -> bool {
        self.perception_dependent_reality
    }
    
    pub fn formal_expression(&self) -> String {
        "∀x: Exists(x) → Mental(x)".to_string()
    }
}

/// 二元论 (Dualism)
/// 
/// 核心观点：物质和精神是两种不同的实在
#[derive(Debug, Clone)]
pub struct Dualism {
    pub both_matter_and_mind_exist: bool,
    pub irreducible_difference: bool,
    pub interaction_problem: bool,
}

impl Dualism {
    pub fn new() -> Self {
        Self {
            both_matter_and_mind_exist: true,
            irreducible_difference: true,
            interaction_problem: true,
        }
    }
    
    pub fn matter_and_mind_both_exist(&self) -> bool {
        self.both_matter_and_mind_exist
    }
    
    pub fn mind_is_irreducible(&self) -> bool {
        self.irreducible_difference
    }
    
    pub fn has_interaction_problem(&self) -> bool {
        self.interaction_problem
    }
    
    pub fn formal_expression(&self) -> String {
        "∃x ∃y: Material(x) ∧ Mental(y) ∧ x ≠ y".to_string()
    }
}

/// 现实分析报告
#[derive(Debug, Clone)]
pub struct RealityAnalysis {
    pub independent_reality: bool,
    pub mind_dependent: bool,
    pub matter_fundamental: bool,
    pub mind_fundamental: bool,
    pub both_exist: bool,
}

impl RealityAnalysis {
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== 现实本体论分析报告 ===\n\n");
        
        report.push_str(&format!("独立实在: {}\n", self.independent_reality));
        report.push_str(&format!("心灵依赖: {}\n", self.mind_dependent));
        report.push_str(&format!("物质根本: {}\n", self.matter_fundamental));
        report.push_str(&format!("心灵根本: {}\n", self.mind_fundamental));
        report.push_str(&format!("两者并存: {}\n", self.both_exist));
        
        report.push_str("\n=== 理论立场总结 ===\n");
        if self.independent_reality && !self.mind_dependent {
            report.push_str("• 实在论立场：承认独立于心灵的客观实在\n");
        }
        if self.mind_dependent {
            report.push_str("• 反实在论立场：实在依赖于心灵或语言\n");
        }
        if self.matter_fundamental && !self.mind_fundamental {
            report.push_str("• 唯物论立场：物质是唯一根本实在\n");
        }
        if self.mind_fundamental && !self.matter_fundamental {
            report.push_str("• 唯心论立场：心灵是唯一根本实在\n");
        }
        if self.both_exist {
            report.push_str("• 二元论立场：物质和心灵都是实在的\n");
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_realism() {
        let realism = Realism::new();
        assert!(realism.reality_exists_independently());
        assert!(realism.truth_is_objective());
        assert!(realism.reality_is_mind_independent());
    }
    
    #[test]
    fn test_anti_realism() {
        let anti_realism = AntiRealism::new();
        assert!(anti_realism.reality_depends_on_mind());
        assert!(anti_realism.truth_depends_on_language());
        assert!(anti_realism.is_constructivist());
    }
    
    #[test]
    fn test_materialism() {
        let materialism = Materialism::new();
        assert!(materialism.matter_is_fundamental());
        assert!(materialism.mental_is_reducible());
        assert!(materialism.physical_causation_is_closed());
    }
    
    #[test]
    fn test_idealism() {
        let idealism = Idealism::new();
        assert!(idealism.mind_is_fundamental());
        assert!(idealism.physical_is_reducible());
        assert!(idealism.reality_depends_on_perception());
    }
    
    #[test]
    fn test_dualism() {
        let dualism = Dualism::new();
        assert!(dualism.matter_and_mind_both_exist());
        assert!(dualism.mind_is_irreducible());
        assert!(dualism.has_interaction_problem());
    }
    
    #[test]
    fn test_reality_ontology_analysis() {
        let ontology = RealityOntology::new();
        let analysis = ontology.analyze_positions();
        let report = analysis.generate_report();
        
        assert!(report.contains("现实本体论分析报告"));
        assert!(report.contains("实在论立场"));
        assert!(report.contains("反实在论立场"));
        assert!(report.contains("唯物论立场"));
        assert!(report.contains("唯心论立场"));
        assert!(report.contains("二元论立场"));
    }
} 