//! 集合论操作实现
//! Set Theory Operations Implementation
//! 
//! 本模块实现了集合论的基本概念和操作，
//! 包括集合运算、关系、函数等。

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

/// 集合 (Set)
#[derive(Debug, Clone, PartialEq)]
pub struct Set<T: Eq + Hash> {
    elements: HashSet<T>,
}

/// 二元关系 (Binary Relation)
#[derive(Debug, Clone)]
pub struct Relation<A: Eq + Hash, B: Eq + Hash> {
    pairs: HashSet<(A, B)>,
}

/// 函数 (Function)
#[derive(Debug, Clone)]
pub struct Function<A: Eq + Hash, B: Eq + Hash> {
    mapping: HashMap<A, B>,
    domain: Set<A>,
    codomain: Set<B>,
}

/// 等价关系 (Equivalence Relation)
#[derive(Debug, Clone)]
pub struct EquivalenceRelation<T: Eq + Hash> {
    relation: Relation<T, T>,
}

/// 偏序关系 (Partial Order)
#[derive(Debug, Clone)]
pub struct PartialOrder<T: Eq + Hash> {
    relation: Relation<T, T>,
}

impl<T: Eq + Hash + Clone> Set<T> {
    /// 创建空集合
    pub fn new() -> Self {
        Self {
            elements: HashSet::new(),
        }
    }

    /// 从向量创建集合
    pub fn from_vec(elements: Vec<T>) -> Self {
        Self {
            elements: elements.into_iter().collect(),
        }
    }

    /// 添加元素
    pub fn insert(&mut self, element: T) -> bool {
        self.elements.insert(element)
    }

    /// 移除元素
    pub fn remove(&mut self, element: &T) -> bool {
        self.elements.remove(element)
    }

    /// 检查是否包含元素
    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    /// 集合大小
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// 并集
    pub fn union(&self, other: &Set<T>) -> Set<T> {
        let mut result = self.elements.clone();
        result.extend(other.elements.iter().cloned());
        Set { elements: result }
    }

    /// 交集
    pub fn intersection(&self, other: &Set<T>) -> Set<T> {
        let result: HashSet<T> = self.elements
            .intersection(&other.elements)
            .cloned()
            .collect();
        Set { elements: result }
    }

    /// 差集
    pub fn difference(&self, other: &Set<T>) -> Set<T> {
        let result: HashSet<T> = self.elements
            .difference(&other.elements)
            .cloned()
            .collect();
        Set { elements: result }
    }

    /// 对称差
    pub fn symmetric_difference(&self, other: &Set<T>) -> Set<T> {
        self.difference(other).union(&other.difference(self))
    }

    /// 子集检查
    pub fn is_subset(&self, other: &Set<T>) -> bool {
        self.elements.is_subset(&other.elements)
    }

    /// 真子集检查
    pub fn is_proper_subset(&self, other: &Set<T>) -> bool {
        self.elements.is_subset(&other.elements) && self.elements != other.elements
    }

    /// 幂集
    pub fn power_set(&self) -> Set<Set<T>> {
        let elements: Vec<T> = self.elements.iter().cloned().collect();
        let mut power_set = Set::new();
        
        // 生成所有子集
        for i in 0..(1 << elements.len()) {
            let mut subset = Set::new();
            for j in 0..elements.len() {
                if (i >> j) & 1 == 1 {
                    subset.insert(elements[j].clone());
                }
            }
            power_set.insert(subset);
        }
        
        power_set
    }

    /// 笛卡尔积
    pub fn cartesian_product<U: Eq + Hash + Clone>(&self, other: &Set<U>) -> Set<(T, U)> {
        let mut result = Set::new();
        for a in &self.elements {
            for b in &other.elements {
                result.insert((a.clone(), b.clone()));
            }
        }
        result
    }
}

impl<T: Eq + Hash + fmt::Display> fmt::Display for Set<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let elements: Vec<&T> = self.elements.iter().collect();
        for (i, element) in elements.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", element)?;
        }
        write!(f, "}}")
    }
}

impl<A: Eq + Hash + Clone, B: Eq + Hash + Clone> Relation<A, B> {
    /// 创建空关系
    pub fn new() -> Self {
        Self {
            pairs: HashSet::new(),
        }
    }

    /// 添加关系对
    pub fn add(&mut self, a: A, b: B) {
        self.pairs.insert((a, b));
    }

    /// 检查关系
    pub fn relates(&self, a: &A, b: &B) -> bool {
        self.pairs.contains(&(a.clone(), b.clone()))
    }

    /// 关系的定义域
    pub fn domain(&self) -> Set<A> {
        let domain_elements: HashSet<A> = self.pairs.iter()
            .map(|(a, _)| a.clone())
            .collect();
        Set { elements: domain_elements }
    }

    /// 关系的值域
    pub fn range(&self) -> Set<B> {
        let range_elements: HashSet<B> = self.pairs.iter()
            .map(|(_, b)| b.clone())
            .collect();
        Set { elements: range_elements }
    }

    /// 关系的逆
    pub fn inverse(&self) -> Relation<B, A> {
        let mut inverse_pairs = HashSet::new();
        for (a, b) in &self.pairs {
            inverse_pairs.insert((b.clone(), a.clone()));
        }
        Relation { pairs: inverse_pairs }
    }

    /// 关系复合
    pub fn compose<C: Eq + Hash + Clone>(&self, other: &Relation<B, C>) -> Relation<A, C> {
        let mut composition = HashSet::new();
        for (a, b1) in &self.pairs {
            for (b2, c) in &other.pairs {
                if b1 == b2 {
                    composition.insert((a.clone(), c.clone()));
                }
            }
        }
        Relation { pairs: composition }
    }
}

impl<A: Eq + Hash + Clone, B: Eq + Hash + Clone> Function<A, B> {
    /// 创建函数
    pub fn new(domain: Set<A>, codomain: Set<B>) -> Self {
        Self {
            mapping: HashMap::new(),
            domain,
            codomain,
        }
    }

    /// 定义函数值
    pub fn define(&mut self, a: A, b: B) -> Result<(), String> {
        if !self.domain.contains(&a) {
            return Err("Element not in domain".to_string());
        }
        if !self.codomain.contains(&b) {
            return Err("Element not in codomain".to_string());
        }
        self.mapping.insert(a, b);
        Ok(())
    }

    /// 函数应用
    pub fn apply(&self, a: &A) -> Option<&B> {
        self.mapping.get(a)
    }

    /// 检查是否为单射
    pub fn is_injective(&self) -> bool {
        let mut seen = HashSet::new();
        for value in self.mapping.values() {
            if !seen.insert(value) {
                return false;
            }
        }
        true
    }

    /// 检查是否为满射
    pub fn is_surjective(&self) -> bool {
        let image: HashSet<&B> = self.mapping.values().collect();
        image.len() == self.codomain.size()
    }

    /// 检查是否为双射
    pub fn is_bijective(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }

    /// 逆函数
    pub fn inverse(&self) -> Option<Function<B, A>> {
        if !self.is_bijective() {
            return None;
        }
        
        let mut inverse_mapping = HashMap::new();
        for (a, b) in &self.mapping {
            inverse_mapping.insert(b.clone(), a.clone());
        }
        
        Some(Function {
            mapping: inverse_mapping,
            domain: self.codomain.clone(),
            codomain: self.domain.clone(),
        })
    }
}

impl<T: Eq + Hash + Clone> EquivalenceRelation<T> {
    /// 创建等价关系
    pub fn new() -> Self {
        Self {
            relation: Relation::new(),
        }
    }

    /// 添加等价对
    pub fn add_equivalence(&mut self, a: T, b: T) {
        // 添加自反性
        self.relation.add(a.clone(), a.clone());
        self.relation.add(b.clone(), b.clone());
        
        // 添加对称性
        self.relation.add(a.clone(), b.clone());
        self.relation.add(b.clone(), a.clone());
    }

    /// 检查等价性
    pub fn are_equivalent(&self, a: &T, b: &T) -> bool {
        self.relation.relates(a, b)
    }

    /// 等价类
    pub fn equivalence_class(&self, element: &T) -> Set<T> {
        let mut class = Set::new();
        for (a, b) in &self.relation.pairs {
            if a == element {
                class.insert(b.clone());
            }
        }
        class
    }

    /// 商集
    pub fn quotient_set(&self, universe: &Set<T>) -> Set<Set<T>> {
        let mut quotient = Set::new();
        let mut processed = Set::new();
        
        for element in &universe.elements {
            if !processed.contains(element) {
                let class = self.equivalence_class(element);
                quotient.insert(class.clone());
                for member in &class.elements {
                    processed.insert(member.clone());
                }
            }
        }
        
        quotient
    }
}

impl<T: Eq + Hash + Clone> PartialOrder<T> {
    /// 创建偏序关系
    pub fn new() -> Self {
        Self {
            relation: Relation::new(),
        }
    }

    /// 添加偏序关系
    pub fn add_order(&mut self, a: T, b: T) {
        // 添加自反性
        self.relation.add(a.clone(), a.clone());
        self.relation.add(b.clone(), b.clone());
        
        // 添加传递性
        self.relation.add(a.clone(), b.clone());
    }

    /// 检查偏序关系
    pub fn is_less_than_or_equal(&self, a: &T, b: &T) -> bool {
        self.relation.relates(a, b)
    }

    /// 最小元素
    pub fn minimal_elements(&self, set: &Set<T>) -> Set<T> {
        let mut minimal = Set::new();
        
        for element in &set.elements {
            let mut is_minimal = true;
            for other in &set.elements {
                if other != element && self.relation.relates(other, element) {
                    is_minimal = false;
                    break;
                }
            }
            if is_minimal {
                minimal.insert(element.clone());
            }
        }
        
        minimal
    }

    /// 最大元素
    pub fn maximal_elements(&self, set: &Set<T>) -> Set<T> {
        let mut maximal = Set::new();
        
        for element in &set.elements {
            let mut is_maximal = true;
            for other in &set.elements {
                if other != element && self.relation.relates(element, other) {
                    is_maximal = false;
                    break;
                }
            }
            if is_maximal {
                maximal.insert(element.clone());
            }
        }
        
        maximal
    }
}

/// 基数计算
pub fn cardinality<T: Eq + Hash>(set: &Set<T>) -> usize {
    set.size()
}

/// 集合等势检查
pub fn are_equinumerous<A: Eq + Hash, B: Eq + Hash>(
    set_a: &Set<A>, 
    set_b: &Set<B>
) -> bool {
    set_a.size() == set_b.size()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_operations() {
        let mut set1 = Set::from_vec(vec![1, 2, 3, 4]);
        let set2 = Set::from_vec(vec![3, 4, 5, 6]);
        
        // 并集
        let union = set1.union(&set2);
        assert_eq!(union.size(), 6);
        
        // 交集
        let intersection = set1.intersection(&set2);
        assert_eq!(intersection.size(), 2);
        assert!(intersection.contains(&3));
        assert!(intersection.contains(&4));
        
        // 差集
        let difference = set1.difference(&set2);
        assert_eq!(difference.size(), 2);
        assert!(difference.contains(&1));
        assert!(difference.contains(&2));
    }

    #[test]
    fn test_relation() {
        let mut relation = Relation::new();
        relation.add(1, "one");
        relation.add(2, "two");
        
        assert!(relation.relates(&1, &"one"));
        assert!(!relation.relates(&1, &"two"));
        
        let domain = relation.domain();
        assert_eq!(domain.size(), 2);
        assert!(domain.contains(&1));
        assert!(domain.contains(&2));
    }

    #[test]
    fn test_function() {
        let domain = Set::from_vec(vec![1, 2, 3]);
        let codomain = Set::from_vec(vec!["one", "two", "three"]);
        
        let mut function = Function::new(domain, codomain);
        function.define(1, "one").unwrap();
        function.define(2, "two").unwrap();
        function.define(3, "three").unwrap();
        
        assert_eq!(function.apply(&1), Some(&"one"));
        assert!(function.is_bijective());
    }

    #[test]
    fn test_equivalence_relation() {
        let mut eq_rel = EquivalenceRelation::new();
        eq_rel.add_equivalence(1, 2);
        eq_rel.add_equivalence(2, 3);
        
        assert!(eq_rel.are_equivalent(&1, &3));
        
        let class = eq_rel.equivalence_class(&1);
        assert_eq!(class.size(), 3);
    }

    #[test]
    fn test_partial_order() {
        let mut partial_order = PartialOrder::new();
        partial_order.add_order(1, 2);
        partial_order.add_order(2, 3);
        partial_order.add_order(1, 3);
        
        assert!(partial_order.is_less_than_or_equal(&1, &3));
        
        let set = Set::from_vec(vec![1, 2, 3]);
        let minimal = partial_order.minimal_elements(&set);
        assert_eq!(minimal.size(), 1);
        assert!(minimal.contains(&1));
    }
}

/// 示例：集合论操作演示
pub fn example_usage() {
    println!("=== 集合论操作示例 ===");
    
    // 创建集合
    let set_a = Set::from_vec(vec![1, 2, 3, 4]);
    let set_b = Set::from_vec(vec![3, 4, 5, 6]);
    
    println!("集合 A: {}", set_a);
    println!("集合 B: {}", set_b);
    
    // 基本运算
    println!("A ∪ B: {}", set_a.union(&set_b));
    println!("A ∩ B: {}", set_a.intersection(&set_b));
    println!("A - B: {}", set_a.difference(&set_b));
    println!("A △ B: {}", set_a.symmetric_difference(&set_b));
    
    // 关系示例
    let mut relation = Relation::new();
    relation.add(1, "one");
    relation.add(2, "two");
    relation.add(3, "three");
    
    println!("关系: (1, one), (2, two), (3, three)");
    println!("1 与 'one' 有关系: {}", relation.relates(&1, &"one"));
    
    // 函数示例
    let domain = Set::from_vec(vec![1, 2, 3]);
    let codomain = Set::from_vec(vec!["one", "two", "three"]);
    
    let mut function = Function::new(domain, codomain);
    function.define(1, "one").unwrap();
    function.define(2, "two").unwrap();
    function.define(3, "three").unwrap();
    
    println!("函数 f(1) = {}", function.apply(&1).unwrap());
    println!("函数是双射: {}", function.is_bijective());
    
    // 等价关系示例
    let mut eq_rel = EquivalenceRelation::new();
    eq_rel.add_equivalence(1, 2);
    eq_rel.add_equivalence(2, 3);
    
    println!("1 与 3 等价: {}", eq_rel.are_equivalent(&1, &3));
    
    // 偏序关系示例
    let mut partial_order = PartialOrder::new();
    partial_order.add_order(1, 2);
    partial_order.add_order(2, 3);
    
    println!("1 ≤ 3: {}", partial_order.is_less_than_or_equal(&1, &3));
    
    // 幂集示例
    let small_set = Set::from_vec(vec![1, 2]);
    let power_set = small_set.power_set();
    println!("{{1, 2}} 的幂集大小: {}", power_set.size());
    
    // 基数
    println!("集合 A 的基数: {}", cardinality(&set_a));
} 