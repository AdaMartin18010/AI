//! 知识表示与推理系统
//! Knowledge Representation and Reasoning System
//! 
//! 本模块实现了基于认识论的知识表示和推理系统，
//! 包括JTB理论、知识来源分类、推理方法等。

use std::collections::HashMap;
use std::fmt;

/// 知识主体 (Knowledge Subject)
#[derive(Debug, Clone, PartialEq)]
pub struct Subject {
    pub id: String,
    pub knowledge_base: KnowledgeBase,
    pub epistemic_method: EpistemicMethod,
}

/// 知识库 (Knowledge Base)
#[derive(Debug, Clone)]
pub struct KnowledgeBase {
    pub propositions: HashMap<String, Proposition>,
    pub beliefs: HashMap<String, Belief>,
    pub justifications: HashMap<String, Justification>,
}

/// 命题 (Proposition)
#[derive(Debug, Clone, PartialEq)]
pub struct Proposition {
    pub id: String,
    pub content: String,
    pub truth_value: Option<bool>,
    pub source: KnowledgeSource,
    pub confidence: f64,
}

/// 信念 (Belief)
#[derive(Debug, Clone)]
pub struct Belief {
    pub subject_id: String,
    pub proposition_id: String,
    pub strength: f64, // 信念强度 [0, 1]
    pub timestamp: u64,
}

/// 正当理由 (Justification)
#[derive(Debug, Clone)]
pub struct Justification {
    pub proposition_id: String,
    pub evidence: Vec<Evidence>,
    pub reasoning_chain: Vec<String>,
    pub reliability: f64,
}

/// 证据 (Evidence)
#[derive(Debug, Clone)]
pub struct Evidence {
    pub id: String,
    pub content: String,
    pub relevance: f64,
    pub reliability: f64,
    pub source: String,
}

/// 知识来源 (Knowledge Source)
#[derive(Debug, Clone, PartialEq)]
pub enum KnowledgeSource {
    Apriori,    // 先验知识
    Aposteriori, // 后验知识
    Inference,  // 推理
    Testimony,  // 证言
    Perception, // 感知
}

/// 认识论方法 (Epistemic Method)
#[derive(Debug, Clone)]
pub enum EpistemicMethod {
    Rationalism,    // 理性主义
    Empiricism,     // 经验主义
    CriticalRationalism, // 批判理性主义
    Pragmatism,     // 实用主义
}

/// 推理类型 (Reasoning Type)
#[derive(Debug, Clone)]
pub enum ReasoningType {
    Deductive,  // 演绎推理
    Inductive,  // 归纳推理
    Abductive,  // 溯因推理
}

impl Subject {
    /// 创建新的知识主体
    pub fn new(id: String, method: EpistemicMethod) -> Self {
        Self {
            id,
            knowledge_base: KnowledgeBase::new(),
            epistemic_method: method,
        }
    }

    /// 添加知识
    pub fn add_knowledge(&mut self, proposition: Proposition) {
        self.knowledge_base.propositions.insert(
            proposition.id.clone(),
            proposition.clone(),
        );
    }

    /// 形成信念
    pub fn form_belief(&mut self, proposition_id: String, strength: f64) {
        let belief = Belief {
            subject_id: self.id.clone(),
            proposition_id: proposition_id.clone(),
            strength,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.knowledge_base.beliefs.insert(proposition_id, belief);
    }

    /// 检查是否知道某个命题 (JTB理论)
    pub fn knows(&self, proposition_id: &str) -> bool {
        // 检查信念 (Belief)
        let has_belief = self.knowledge_base.beliefs.contains_key(proposition_id);
        if !has_belief {
            return false;
        }

        // 检查真理 (Truth)
        let is_true = self.knowledge_base
            .propositions
            .get(proposition_id)
            .and_then(|p| p.truth_value)
            .unwrap_or(false);

        if !is_true {
            return false;
        }

        // 检查正当理由 (Justification)
        let has_justification = self.knowledge_base.justifications.contains_key(proposition_id);

        has_belief && is_true && has_justification
    }

    /// 推理
    pub fn reason(&mut self, reasoning_type: ReasoningType, premises: Vec<String>, conclusion: String) -> bool {
        match reasoning_type {
            ReasoningType::Deductive => self.deductive_reasoning(premises, conclusion),
            ReasoningType::Inductive => self.inductive_reasoning(premises, conclusion),
            ReasoningType::Abductive => self.abductive_reasoning(premises, conclusion),
        }
    }

    /// 演绎推理
    fn deductive_reasoning(&mut self, premises: Vec<String>, conclusion: String) -> bool {
        // 检查所有前提是否为真
        let all_premises_true = premises.iter().all(|p| self.knows(p));
        
        if all_premises_true {
            // 创建结论命题
            let conclusion_prop = Proposition {
                id: conclusion.clone(),
                content: format!("Conclusion from premises: {:?}", premises),
                truth_value: Some(true),
                source: KnowledgeSource::Inference,
                confidence: 1.0,
            };
            
            self.add_knowledge(conclusion_prop);
            self.form_belief(conclusion, 1.0);
            
            // 添加正当理由
            let justification = Justification {
                proposition_id: conclusion.clone(),
                evidence: premises.iter().map(|p| Evidence {
                    id: p.clone(),
                    content: format!("Premise: {}", p),
                    relevance: 1.0,
                    reliability: 1.0,
                    source: "deductive_reasoning".to_string(),
                }).collect(),
                reasoning_chain: premises,
                reliability: 1.0,
            };
            
            self.knowledge_base.justifications.insert(conclusion, justification);
            true
        } else {
            false
        }
    }

    /// 归纳推理
    fn inductive_reasoning(&mut self, premises: Vec<String>, conclusion: String) -> bool {
        // 计算前提的支持度
        let support = premises.iter()
            .filter_map(|p| self.knowledge_base.propositions.get(p))
            .map(|p| p.confidence)
            .sum::<f64>() / premises.len() as f64;

        if support > 0.5 {
            let conclusion_prop = Proposition {
                id: conclusion.clone(),
                content: format!("Inductive conclusion from: {:?}", premises),
                truth_value: None, // 归纳推理不保证真理
                source: KnowledgeSource::Inference,
                confidence: support,
            };
            
            self.add_knowledge(conclusion_prop);
            self.form_belief(conclusion.clone(), support);
            
            let justification = Justification {
                proposition_id: conclusion.clone(),
                evidence: premises.iter().map(|p| Evidence {
                    id: p.clone(),
                    content: format!("Inductive premise: {}", p),
                    relevance: 0.8,
                    reliability: support,
                    source: "inductive_reasoning".to_string(),
                }).collect(),
                reasoning_chain: premises,
                reliability: support,
            };
            
            self.knowledge_base.justifications.insert(conclusion, justification);
            true
        } else {
            false
        }
    }

    /// 溯因推理
    fn abductive_reasoning(&mut self, premises: Vec<String>, conclusion: String) -> bool {
        // 溯因推理：从观察到的现象推断最佳解释
        let conclusion_prop = Proposition {
            id: conclusion.clone(),
            content: format!("Best explanation for: {:?}", premises),
            truth_value: None,
            source: KnowledgeSource::Inference,
            confidence: 0.7, // 溯因推理的置信度通常较低
        };
        
        self.add_knowledge(conclusion_prop);
        self.form_belief(conclusion.clone(), 0.7);
        
        let justification = Justification {
            proposition_id: conclusion.clone(),
            evidence: premises.iter().map(|p| Evidence {
                id: p.clone(),
                content: format!("Observation: {}", p),
                relevance: 0.9,
                reliability: 0.7,
                source: "abductive_reasoning".to_string(),
            }).collect(),
            reasoning_chain: premises,
            reliability: 0.7,
        };
        
        self.knowledge_base.justifications.insert(conclusion, justification);
        true
    }

    /// 贝叶斯更新
    pub fn bayesian_update(&mut self, proposition_id: &str, evidence: Evidence) {
        if let Some(prop) = self.knowledge_base.propositions.get_mut(proposition_id) {
            // 简化的贝叶斯更新
            let prior = prop.confidence;
            let likelihood = evidence.relevance * evidence.reliability;
            let posterior = (prior * likelihood) / (prior * likelihood + (1.0 - prior) * (1.0 - likelihood));
            
            prop.confidence = posterior;
            
            // 更新信念强度
            if let Some(belief) = self.knowledge_base.beliefs.get_mut(proposition_id) {
                belief.strength = posterior;
            }
        }
    }

    /// 获取知识统计
    pub fn knowledge_stats(&self) -> KnowledgeStats {
        let total_propositions = self.knowledge_base.propositions.len();
        let known_propositions = self.knowledge_base.propositions.values()
            .filter(|p| self.knows(&p.id))
            .count();
        
        let avg_confidence = self.knowledge_base.propositions.values()
            .map(|p| p.confidence)
            .sum::<f64>() / total_propositions as f64;

        KnowledgeStats {
            total_propositions,
            known_propositions,
            avg_confidence,
            epistemic_method: self.epistemic_method.clone(),
        }
    }
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            propositions: HashMap::new(),
            beliefs: HashMap::new(),
            justifications: HashMap::new(),
        }
    }
}

/// 知识统计
#[derive(Debug)]
pub struct KnowledgeStats {
    pub total_propositions: usize,
    pub known_propositions: usize,
    pub avg_confidence: f64,
    pub epistemic_method: EpistemicMethod,
}

impl fmt::Display for KnowledgeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Knowledge Stats:\n")?;
        write!(f, "  Total Propositions: {}\n", self.total_propositions)?;
        write!(f, "  Known Propositions: {}\n", self.known_propositions)?;
        write!(f, "  Average Confidence: {:.3}\n", self.avg_confidence)?;
        write!(f, "  Epistemic Method: {:?}", self.epistemic_method)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jtb_theory() {
        let mut subject = Subject::new("test_subject".to_string(), EpistemicMethod::Rationalism);
        
        // 添加一个真命题
        let prop = Proposition {
            id: "p1".to_string(),
            content: "2 + 2 = 4".to_string(),
            truth_value: Some(true),
            source: KnowledgeSource::Apriori,
            confidence: 1.0,
        };
        
        subject.add_knowledge(prop);
        subject.form_belief("p1".to_string(), 1.0);
        
        // 添加正当理由
        let justification = Justification {
            proposition_id: "p1".to_string(),
            evidence: vec![Evidence {
                id: "e1".to_string(),
                content: "Mathematical proof".to_string(),
                relevance: 1.0,
                reliability: 1.0,
                source: "mathematics".to_string(),
            }],
            reasoning_chain: vec!["mathematical_truth".to_string()],
            reliability: 1.0,
        };
        
        subject.knowledge_base.justifications.insert("p1".to_string(), justification);
        
        assert!(subject.knows("p1"));
    }

    #[test]
    fn test_deductive_reasoning() {
        let mut subject = Subject::new("test_subject".to_string(), EpistemicMethod::Rationalism);
        
        // 添加前提
        let premise1 = Proposition {
            id: "premise1".to_string(),
            content: "All humans are mortal".to_string(),
            truth_value: Some(true),
            source: KnowledgeSource::Apriori,
            confidence: 1.0,
        };
        
        let premise2 = Proposition {
            id: "premise2".to_string(),
            content: "Socrates is human".to_string(),
            truth_value: Some(true),
            source: KnowledgeSource::Aposteriori,
            confidence: 1.0,
        };
        
        subject.add_knowledge(premise1);
        subject.add_knowledge(premise2);
        subject.form_belief("premise1".to_string(), 1.0);
        subject.form_belief("premise2".to_string(), 1.0);
        
        // 演绎推理
        let success = subject.reason(
            ReasoningType::Deductive,
            vec!["premise1".to_string(), "premise2".to_string()],
            "conclusion".to_string(),
        );
        
        assert!(success);
        assert!(subject.knows("conclusion"));
    }
}

/// 示例：使用知识表示系统
pub fn example_usage() {
    println!("=== 知识表示与推理系统示例 ===");
    
    // 创建一个理性主义的知识主体
    let mut subject = Subject::new("philosopher".to_string(), EpistemicMethod::Rationalism);
    
    // 添加先验知识
    let mathematical_truth = Proposition {
        id: "math_truth".to_string(),
        content: "2 + 2 = 4".to_string(),
        truth_value: Some(true),
        source: KnowledgeSource::Apriori,
        confidence: 1.0,
    };
    
    subject.add_knowledge(mathematical_truth);
    subject.form_belief("math_truth".to_string(), 1.0);
    
    // 添加正当理由
    let justification = Justification {
        proposition_id: "math_truth".to_string(),
        evidence: vec![Evidence {
            id: "proof".to_string(),
            content: "Mathematical proof".to_string(),
            relevance: 1.0,
            reliability: 1.0,
            source: "mathematics".to_string(),
        }],
        reasoning_chain: vec!["axioms".to_string(), "deduction".to_string()],
        reliability: 1.0,
    };
    
    subject.knowledge_base.justifications.insert("math_truth".to_string(), justification);
    
    // 检查知识
    println!("主体知道数学真理: {}", subject.knows("math_truth"));
    
    // 演绎推理示例
    let premise1 = Proposition {
        id: "premise1".to_string(),
        content: "All A are B".to_string(),
        truth_value: Some(true),
        source: KnowledgeSource::Apriori,
        confidence: 1.0,
    };
    
    let premise2 = Proposition {
        id: "premise2".to_string(),
        content: "X is A".to_string(),
        truth_value: Some(true),
        source: KnowledgeSource::Aposteriori,
        confidence: 1.0,
    };
    
    subject.add_knowledge(premise1);
    subject.add_knowledge(premise2);
    subject.form_belief("premise1".to_string(), 1.0);
    subject.form_belief("premise2".to_string(), 1.0);
    
    let success = subject.reason(
        ReasoningType::Deductive,
        vec!["premise1".to_string(), "premise2".to_string()],
        "conclusion".to_string(),
    );
    
    println!("演绎推理成功: {}", success);
    println!("知道结论: {}", subject.knows("conclusion"));
    
    // 显示知识统计
    println!("{}", subject.knowledge_stats());
} 