# 图谱可视化与交互

## 1. 概述

### 1.1 图谱可视化定义

图谱可视化（Graph Visualization）是将知识图谱以图形化方式展示的技术，通过节点、边、颜色、大小等视觉元素直观地呈现知识结构。交互式可视化允许用户与图谱进行动态交互，提升知识探索和理解的效率。

#### 1.1.1 可视化框架

```latex
\mathcal{GV} = \langle \mathcal{V}, \mathcal{L}, \mathcal{I}, \mathcal{A}, \mathcal{R} \rangle
```

其中：

- $\mathcal{V}$: 视觉元素集合（Visual Elements）
- $\mathcal{L}$: 布局算法集合（Layout Algorithms）
- $\mathcal{I}$: 交互功能集合（Interaction Functions）
- $\mathcal{A}$: 动画效果集合（Animation Effects）
- $\mathcal{R}$: 渲染引擎集合（Rendering Engines）

#### 1.1.2 可视化流程

```mermaid
graph LR
    A[数据预处理] --> B[布局计算]
    B --> C[视觉映射]
    C --> D[交互设计]
    D --> E[动画效果]
    E --> F[渲染输出]
```

### 1.2 可视化目标

#### 1.2.1 核心目标

1. **信息传达**：清晰准确地传达知识结构信息
2. **模式发现**：帮助用户发现数据中的模式和规律
3. **交互探索**：支持用户进行深度探索和查询
4. **美观性**：提供美观、舒适的视觉体验

#### 1.2.2 设计原则

1. **简洁性**：避免视觉混乱，突出重要信息
2. **一致性**：保持视觉元素的一致性
3. **层次性**：通过视觉层次引导用户注意力
4. **响应性**：快速响应用户交互操作

## 2. 布局算法

### 2.1 力导向布局

#### 2.1.1 经典力导向算法

```rust
pub struct ForceDirectedLayout {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub width: f64,
    pub height: f64,
    pub iterations: usize,
}

impl ForceDirectedLayout {
    pub fn new(width: f64, height: f64) -> Self {
        ForceDirectedLayout {
            nodes: Vec::new(),
            edges: Vec::new(),
            width,
            height,
            iterations: 100,
        }
    }
    
    pub fn add_node(&mut self, id: String, x: f64, y: f64) {
        self.nodes.push(Node {
            id,
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            mass: 1.0,
        });
    }
    
    pub fn add_edge(&mut self, source: String, target: String, weight: f64) {
        self.edges.push(Edge {
            source,
            target,
            weight,
        });
    }
    
    pub fn layout(&mut self) {
        for _ in 0..self.iterations {
            self.calculate_forces();
            self.update_positions();
            self.apply_boundaries();
        }
    }
    
    pub fn calculate_forces(&mut self) {
        // 重置力
        for node in &mut self.nodes {
            node.vx = 0.0;
            node.vy = 0.0;
        }
        
        // 计算斥力（节点间）
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                let dx = self.nodes[j].x - self.nodes[i].x;
                let dy = self.nodes[j].y - self.nodes[i].y;
                let distance = (dx * dx + dy * dy).sqrt();
                
                if distance > 0.0 {
                    let force = 1000.0 / (distance * distance);
                    let fx = force * dx / distance;
                    let fy = force * dy / distance;
                    
                    self.nodes[i].vx -= fx;
                    self.nodes[i].vy -= fy;
                    self.nodes[j].vx += fx;
                    self.nodes[j].vy += fy;
                }
            }
        }
        
        // 计算引力（边连接）
        for edge in &self.edges {
            let source_idx = self.find_node_index(&edge.source);
            let target_idx = self.find_node_index(&edge.target);
            
            if let (Some(i), Some(j)) = (source_idx, target_idx) {
                let dx = self.nodes[j].x - self.nodes[i].x;
                let dy = self.nodes[j].y - self.nodes[i].y;
                let distance = (dx * dx + dy * dy).sqrt();
                
                if distance > 0.0 {
                    let force = 0.1 * distance * edge.weight;
                    let fx = force * dx / distance;
                    let fy = force * dy / distance;
                    
                    self.nodes[i].vx += fx;
                    self.nodes[i].vy += fy;
                    self.nodes[j].vx -= fx;
                    self.nodes[j].vy -= fy;
                }
            }
        }
    }
    
    pub fn update_positions(&mut self) {
        for node in &mut self.nodes {
            node.x += node.vx;
            node.y += node.vy;
        }
    }
    
    pub fn apply_boundaries(&mut self) {
        for node in &mut self.nodes {
            if node.x < 0.0 { node.x = 0.0; node.vx = 0.0; }
            if node.x > self.width { node.x = self.width; node.vx = 0.0; }
            if node.y < 0.0 { node.y = 0.0; node.vy = 0.0; }
            if node.y > self.height { node.y = self.height; node.vy = 0.0; }
        }
    }
    
    pub fn find_node_index(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|node| node.id == id)
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub mass: f64,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f64,
}
```

#### 2.1.2 改进的力导向算法

```rust
pub struct ImprovedForceLayout {
    pub layout: ForceDirectedLayout,
    pub cooling_factor: f64,
    pub temperature: f64,
}

impl ImprovedForceLayout {
    pub fn new(width: f64, height: f64) -> Self {
        ImprovedForceLayout {
            layout: ForceDirectedLayout::new(width, height),
            cooling_factor: 0.95,
            temperature: 1000.0,
        }
    }
    
    pub fn layout_with_cooling(&mut self) {
        for iteration in 0..self.layout.iterations {
            self.layout.calculate_forces();
            
            // 应用温度控制
            for node in &mut self.layout.nodes {
                node.vx *= self.temperature / 1000.0;
                node.vy *= self.temperature / 1000.0;
            }
            
            self.layout.update_positions();
            self.layout.apply_boundaries();
            
            // 冷却
            self.temperature *= self.cooling_factor;
        }
    }
}
```

### 2.2 层次布局

#### 2.2.1 树形布局

```rust
pub struct TreeLayout {
    pub nodes: Vec<TreeNode>,
    pub width: f64,
    pub height: f64,
    pub node_spacing: f64,
    pub level_spacing: f64,
}

impl TreeLayout {
    pub fn new(width: f64, height: f64) -> Self {
        TreeLayout {
            nodes: Vec::new(),
            width,
            height,
            node_spacing: 100.0,
            level_spacing: 150.0,
        }
    }
    
    pub fn layout_tree(&mut self, root: &str) {
        if let Some(root_idx) = self.find_node_index(root) {
            self.calculate_tree_layout(root_idx, 0, 0.0, self.width);
        }
    }
    
    pub fn calculate_tree_layout(&mut self, node_idx: usize, level: usize, left: f64, right: f64) -> f64 {
        let node = &mut self.nodes[node_idx];
        let children = self.get_children(node_idx);
        
        if children.is_empty() {
            // 叶节点
            node.x = (left + right) / 2.0;
            node.y = level as f64 * self.level_spacing;
            return self.node_spacing;
        }
        
        // 计算子节点布局
        let mut total_width = 0.0;
        let mut child_positions = Vec::new();
        
        for child_idx in children {
            let child_width = self.calculate_tree_layout(child_idx, level + 1, left + total_width, right);
            child_positions.push((child_idx, left + total_width + child_width / 2.0));
            total_width += child_width;
        }
        
        // 定位当前节点
        node.x = (left + right) / 2.0;
        node.y = level as f64 * self.level_spacing;
        
        // 调整子节点位置
        for (child_idx, _) in child_positions {
            let child = &mut self.nodes[child_idx];
            child.x = node.x + (child.x - node.x) * 0.8; // 向父节点收缩
        }
        
        total_width.max(self.node_spacing)
    }
    
    pub fn get_children(&self, node_idx: usize) -> Vec<usize> {
        let node_id = &self.nodes[node_idx].id;
        self.nodes.iter()
            .enumerate()
            .filter(|(_, node)| node.parent.as_ref() == Some(node_id))
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub parent: Option<String>,
}
```

#### 2.2.2 分层布局

```rust
pub struct HierarchicalLayout {
    pub nodes: Vec<HierarchicalNode>,
    pub levels: Vec<Vec<usize>>,
    pub width: f64,
    pub height: f64,
}

impl HierarchicalLayout {
    pub fn new(width: f64, height: f64) -> Self {
        HierarchicalLayout {
            nodes: Vec::new(),
            levels: Vec::new(),
            width,
            height,
        }
    }
    
    pub fn assign_levels(&mut self) {
        // 使用拓扑排序分配层级
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        
        // 初始化
        for node in &self.nodes {
            in_degree.insert(node.id.clone(), 0);
            adjacency.insert(node.id.clone(), Vec::new());
        }
        
        // 构建邻接表和入度
        for node in &self.nodes {
            for edge in &node.outgoing_edges {
                adjacency.get_mut(&node.id).unwrap().push(edge.clone());
                *in_degree.get_mut(edge).unwrap() += 1;
            }
        }
        
        // 拓扑排序
        let mut queue: VecDeque<String> = VecDeque::new();
        for (id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(id.clone());
            }
        }
        
        let mut level = 0;
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut current_level = Vec::new();
            
            for _ in 0..level_size {
                let node_id = queue.pop_front().unwrap();
                let node_idx = self.find_node_index(&node_id).unwrap();
                current_level.push(node_idx);
                
                // 更新邻居的入度
                for neighbor in &adjacency[&node_id] {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
            
            self.levels.push(current_level);
            level += 1;
        }
    }
    
    pub fn layout_hierarchical(&mut self) {
        self.assign_levels();
        
        let level_height = self.height / self.levels.len() as f64;
        
        for (level_idx, level_nodes) in self.levels.iter().enumerate() {
            let y = level_idx as f64 * level_height + level_height / 2.0;
            let node_width = self.width / level_nodes.len() as f64;
            
            for (node_idx, &node_id) in level_nodes.iter().enumerate() {
                let x = node_idx as f64 * node_width + node_width / 2.0;
                self.nodes[node_id].x = x;
                self.nodes[node_id].y = y;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HierarchicalNode {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub outgoing_edges: Vec<String>,
}
```

### 2.3 圆形布局

#### 2.3.1 基础圆形布局

```rust
pub struct CircularLayout {
    pub nodes: Vec<CircularNode>,
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
}

impl CircularLayout {
    pub fn new(center_x: f64, center_y: f64, radius: f64) -> Self {
        CircularLayout {
            nodes: Vec::new(),
            center_x,
            center_y,
            radius,
        }
    }
    
    pub fn layout_circular(&mut self) {
        let node_count = self.nodes.len();
        if node_count == 0 {
            return;
        }
        
        let angle_step = 2.0 * std::f64::consts::PI / node_count as f64;
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = i as f64 * angle_step;
            node.x = self.center_x + self.radius * angle.cos();
            node.y = self.center_y + self.radius * angle.sin();
        }
    }
    
    pub fn layout_concentric(&mut self, groups: Vec<Vec<usize>>) {
        let group_count = groups.len();
        let radius_step = self.radius / group_count as f64;
        
        for (group_idx, group) in groups.iter().enumerate() {
            let current_radius = (group_idx + 1) as f64 * radius_step;
            let angle_step = 2.0 * std::f64::consts::PI / group.len() as f64;
            
            for (node_idx, &node_id) in group.iter().enumerate() {
                let angle = node_idx as f64 * angle_step;
                self.nodes[node_id].x = self.center_x + current_radius * angle.cos();
                self.nodes[node_id].y = self.center_y + current_radius * angle.sin();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircularNode {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub group: Option<String>,
}
```

## 3. 视觉映射

### 3.1 节点视觉映射

#### 3.1.1 大小映射

```rust
pub struct NodeVisualMapping {
    pub size_scale: f64,
    pub color_scale: ColorScale,
    pub shape_mapping: HashMap<String, NodeShape>,
}

impl NodeVisualMapping {
    pub fn map_node_size(&self, node: &Node, attribute: &str) -> f64 {
        let value = self.get_node_attribute(node, attribute);
        let normalized_value = self.normalize_value(value, attribute);
        normalized_value * self.size_scale
    }
    
    pub fn map_node_color(&self, node: &Node, attribute: &str) -> Color {
        let value = self.get_node_attribute(node, attribute);
        self.color_scale.map_value(value)
    }
    
    pub fn map_node_shape(&self, node: &Node, node_type: &str) -> NodeShape {
        self.shape_mapping.get(node_type)
            .cloned()
            .unwrap_or(NodeShape::Circle)
    }
    
    pub fn get_node_attribute(&self, node: &Node, attribute: &str) -> f64 {
        node.attributes.get(attribute)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }
    
    pub fn normalize_value(&self, value: f64, attribute: &str) -> f64 {
        // 基于属性范围进行归一化
        let (min_val, max_val) = self.get_attribute_range(attribute);
        if max_val > min_val {
            (value - min_val) / (max_val - min_val)
        } else {
            0.5
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Hexagon,
}

#[derive(Debug, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

pub struct ColorScale {
    pub colors: Vec<Color>,
}

impl ColorScale {
    pub fn new() -> Self {
        ColorScale {
            colors: vec![
                Color { r: 255, g: 0, b: 0, a: 255 },   // 红
                Color { r: 255, g: 165, b: 0, a: 255 }, // 橙
                Color { r: 255, g: 255, b: 0, a: 255 }, // 黄
                Color { r: 0, g: 255, b: 0, a: 255 },   // 绿
                Color { r: 0, g: 0, b: 255, a: 255 },   // 蓝
            ],
        }
    }
    
    pub fn map_value(&self, value: f64) -> Color {
        let normalized = value.max(0.0).min(1.0);
        let color_count = self.colors.len();
        let index = (normalized * (color_count - 1) as f64).round() as usize;
        
        if index >= color_count - 1 {
            self.colors[color_count - 1].clone()
        } else {
            let t = normalized * (color_count - 1) as f64 - index as f64;
            self.interpolate_color(&self.colors[index], &self.colors[index + 1], t)
        }
    }
    
    pub fn interpolate_color(&self, c1: &Color, c2: &Color, t: f64) -> Color {
        Color {
            r: (c1.r as f64 * (1.0 - t) + c2.r as f64 * t) as u8,
            g: (c1.g as f64 * (1.0 - t) + c2.g as f64 * t) as u8,
            b: (c1.b as f64 * (1.0 - t) + c2.b as f64 * t) as u8,
            a: (c1.a as f64 * (1.0 - t) + c2.a as f64 * t) as u8,
        }
    }
}
```

#### 3.1.2 标签映射

```rust
pub struct LabelMapping {
    pub font_size: f64,
    pub font_family: String,
    pub max_label_length: usize,
    pub label_position: LabelPosition,
}

impl LabelMapping {
    pub fn map_node_label(&self, node: &Node) -> String {
        let label = node.label.clone();
        if label.len() > self.max_label_length {
            format!("{}...", &label[..self.max_label_length])
        } else {
            label
        }
    }
    
    pub fn calculate_label_position(&self, node: &Node) -> (f64, f64) {
        match self.label_position {
            LabelPosition::Below => (node.x, node.y + node.size + 10.0),
            LabelPosition::Above => (node.x, node.y - node.size - 10.0),
            LabelPosition::Right => (node.x + node.size + 10.0, node.y),
            LabelPosition::Left => (node.x - node.size - 10.0, node.y),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LabelPosition {
    Above,
    Below,
    Left,
    Right,
}
```

### 3.2 边视觉映射

#### 3.2.1 边样式映射

```rust
pub struct EdgeVisualMapping {
    pub width_scale: f64,
    pub color_mapping: HashMap<String, Color>,
    pub style_mapping: HashMap<String, EdgeStyle>,
}

impl EdgeVisualMapping {
    pub fn map_edge_width(&self, edge: &Edge) -> f64 {
        let weight = edge.weight.unwrap_or(1.0);
        weight * self.width_scale
    }
    
    pub fn map_edge_color(&self, edge: &Edge) -> Color {
        self.color_mapping.get(&edge.relation_type)
            .cloned()
            .unwrap_or(Color { r: 128, g: 128, b: 128, a: 255 })
    }
    
    pub fn map_edge_style(&self, edge: &Edge) -> EdgeStyle {
        self.style_mapping.get(&edge.relation_type)
            .cloned()
            .unwrap_or(EdgeStyle::Solid)
    }
}

#[derive(Debug, Clone)]
pub enum EdgeStyle {
    Solid,
    Dashed,
    Dotted,
    Double,
}
```

## 4. 交互功能

### 4.1 基础交互

#### 4.1.1 缩放和平移

```rust
pub struct GraphInteraction {
    pub zoom_level: f64,
    pub pan_x: f64,
    pub pan_y: f64,
    pub min_zoom: f64,
    pub max_zoom: f64,
}

impl GraphInteraction {
    pub fn new() -> Self {
        GraphInteraction {
            zoom_level: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            min_zoom: 0.1,
            max_zoom: 10.0,
        }
    }
    
    pub fn zoom_in(&mut self, factor: f64) {
        self.zoom_level = (self.zoom_level * factor).min(self.max_zoom);
    }
    
    pub fn zoom_out(&mut self, factor: f64) {
        self.zoom_level = (self.zoom_level / factor).max(self.min_zoom);
    }
    
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.pan_x += dx;
        self.pan_y += dy;
    }
    
    pub fn transform_coordinates(&self, x: f64, y: f64) -> (f64, f64) {
        let transformed_x = (x + self.pan_x) * self.zoom_level;
        let transformed_y = (y + self.pan_y) * self.zoom_level;
        (transformed_x, transformed_y)
    }
    
    pub fn inverse_transform_coordinates(&self, x: f64, y: f64) -> (f64, f64) {
        let inverse_x = x / self.zoom_level - self.pan_x;
        let inverse_y = y / self.zoom_level - self.pan_y;
        (inverse_x, inverse_y)
    }
}
```

#### 4.1.2 节点选择

```rust
pub struct NodeSelection {
    pub selected_nodes: HashSet<String>,
    pub selection_mode: SelectionMode,
}

impl NodeSelection {
    pub fn new() -> Self {
        NodeSelection {
            selected_nodes: HashSet::new(),
            selection_mode: SelectionMode::Single,
        }
    }
    
    pub fn select_node(&mut self, node_id: &str) {
        match self.selection_mode {
            SelectionMode::Single => {
                self.selected_nodes.clear();
                self.selected_nodes.insert(node_id.to_string());
            },
            SelectionMode::Multiple => {
                self.selected_nodes.insert(node_id.to_string());
            },
            SelectionMode::Toggle => {
                if self.selected_nodes.contains(node_id) {
                    self.selected_nodes.remove(node_id);
                } else {
                    self.selected_nodes.insert(node_id.to_string());
                }
            },
        }
    }
    
    pub fn clear_selection(&mut self) {
        self.selected_nodes.clear();
    }
    
    pub fn get_neighbors(&self, graph: &Graph, node_id: &str) -> Vec<String> {
        graph.get_neighbors(node_id)
            .into_iter()
            .filter(|neighbor| self.selected_nodes.contains(neighbor))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum SelectionMode {
    Single,
    Multiple,
    Toggle,
}
```

### 4.2 高级交互

#### 4.2.1 搜索和过滤

```rust
pub struct GraphSearch {
    pub search_query: String,
    pub search_results: Vec<SearchResult>,
    pub search_type: SearchType,
}

impl GraphSearch {
    pub fn search(&mut self, graph: &Graph, query: &str) -> Vec<SearchResult> {
        self.search_query = query.to_string();
        self.search_results.clear();
        
        match self.search_type {
            SearchType::Exact => self.exact_search(graph, query),
            SearchType::Fuzzy => self.fuzzy_search(graph, query),
            SearchType::Regex => self.regex_search(graph, query),
        }
    }
    
    pub fn exact_search(&mut self, graph: &Graph, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        
        for node in graph.get_nodes() {
            if node.label.to_lowercase().contains(&query.to_lowercase()) {
                results.push(SearchResult {
                    node_id: node.id.clone(),
                    score: 1.0,
                    match_type: MatchType::Exact,
                });
            }
        }
        
        self.search_results = results;
        results
    }
    
    pub fn fuzzy_search(&mut self, graph: &Graph, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        
        for node in graph.get_nodes() {
            let distance = self.levenshtein_distance(&node.label, query);
            let max_len = node.label.len().max(query.len());
            let similarity = 1.0 - (distance as f64 / max_len as f64);
            
            if similarity > 0.7 {
                results.push(SearchResult {
                    node_id: node.id.clone(),
                    score: similarity,
                    match_type: MatchType::Fuzzy,
                });
            }
        }
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        self.search_results = results.clone();
        results
    }
    
    pub fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                    .min(matrix[i + 1][j] + 1)
                    .min(matrix[i][j] + cost);
            }
        }
        
        matrix[len1][len2]
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: String,
    pub score: f64,
    pub match_type: MatchType,
}

#[derive(Debug, Clone)]
pub enum SearchType {
    Exact,
    Fuzzy,
    Regex,
}

#[derive(Debug, Clone)]
pub enum MatchType {
    Exact,
    Fuzzy,
    Regex,
}
```

#### 4.2.2 路径高亮

```rust
pub struct PathHighlighting {
    pub highlighted_paths: Vec<Vec<String>>,
    pub highlight_color: Color,
    pub highlight_width: f64,
}

impl PathHighlighting {
    pub fn highlight_shortest_path(&mut self, graph: &Graph, source: &str, target: &str) -> Vec<String> {
        let path = graph.shortest_path(source, target);
        if !path.is_empty() {
            self.highlighted_paths.push(path.clone());
        }
        path
    }
    
    pub fn highlight_all_paths(&mut self, graph: &Graph, source: &str, target: &str, max_paths: usize) -> Vec<Vec<String>> {
        let paths = graph.all_paths(source, target, max_paths);
        self.highlighted_paths.extend(paths.clone());
        paths
    }
    
    pub fn clear_highlights(&mut self) {
        self.highlighted_paths.clear();
    }
    
    pub fn is_path_highlighted(&self, path: &[String]) -> bool {
        self.highlighted_paths.iter().any(|highlighted_path| {
            highlighted_path.len() == path.len() && 
            highlighted_path.iter().zip(path.iter()).all(|(a, b)| a == b)
        })
    }
}
```

## 5. 动画效果

### 5.1 布局动画

#### 5.1.1 平滑过渡

```rust
pub struct LayoutAnimation {
    pub animation_duration: f64,
    pub easing_function: EasingFunction,
    pub current_time: f64,
    pub is_animating: bool,
}

impl LayoutAnimation {
    pub fn new(duration: f64) -> Self {
        LayoutAnimation {
            animation_duration: duration,
            easing_function: EasingFunction::EaseInOut,
            current_time: 0.0,
            is_animating: false,
        }
    }
    
    pub fn start_animation(&mut self) {
        self.current_time = 0.0;
        self.is_animating = true;
    }
    
    pub fn update_animation(&mut self, delta_time: f64) -> bool {
        if !self.is_animating {
            return false;
        }
        
        self.current_time += delta_time;
        let progress = (self.current_time / self.animation_duration).min(1.0);
        
        if progress >= 1.0 {
            self.is_animating = false;
        }
        
        self.is_animating
    }
    
    pub fn interpolate_position(&self, start_pos: (f64, f64), end_pos: (f64, f64)) -> (f64, f64) {
        let progress = (self.current_time / self.animation_duration).min(1.0);
        let eased_progress = self.apply_easing(progress);
        
        let x = start_pos.0 + (end_pos.0 - start_pos.0) * eased_progress;
        let y = start_pos.1 + (end_pos.1 - start_pos.1) * eased_progress;
        
        (x, y)
    }
    
    pub fn apply_easing(&self, t: f64) -> f64 {
        match self.easing_function {
            EasingFunction::Linear => t,
            EasingFunction::EaseIn => t * t,
            EasingFunction::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            EasingFunction::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}
```

#### 5.1.2 粒子动画

```rust
pub struct ParticleAnimation {
    pub particles: Vec<Particle>,
    pub gravity: f64,
    pub friction: f64,
}

impl ParticleAnimation {
    pub fn new() -> Self {
        ParticleAnimation {
            particles: Vec::new(),
            gravity: 0.1,
            friction: 0.98,
        }
    }
    
    pub fn add_particle(&mut self, x: f64, y: f64, vx: f64, vy: f64) {
        self.particles.push(Particle {
            x,
            y,
            vx,
            vy,
            life: 1.0,
            decay: 0.02,
        });
    }
    
    pub fn update_particles(&mut self) {
        for particle in &mut self.particles {
            // 应用重力
            particle.vy += self.gravity;
            
            // 应用摩擦力
            particle.vx *= self.friction;
            particle.vy *= self.friction;
            
            // 更新位置
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // 更新生命周期
            particle.life -= particle.decay;
        }
        
        // 移除死亡粒子
        self.particles.retain(|p| p.life > 0.0);
    }
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub life: f64,
    pub decay: f64,
}
```

## 6. 跨学科交叉引用

### 6.1 与AI系统的深度融合

#### 6.1.1 智能可视化

- **机器学习**：[AI/03-Theory.md](../AI/03-Theory.md) ←→ 机器学习在布局优化中的应用
- **深度学习**：[AI/05-Model.md](../AI/05-Model.md) ←→ 深度学习在视觉特征提取中的应用
- **自然语言处理**：[AI/06-Applications.md](../AI/06-Applications.md) ←→ NLP在标签生成中的应用

#### 6.1.2 智能交互

- **强化学习**：[AI/02-MetaTheory.md](../AI/02-MetaTheory.md) ←→ 强化学习在交互优化中的应用
- **推荐系统**：[AI/04-DesignPattern.md](../AI/04-DesignPattern.md) ←→ 推荐系统在可视化推荐中的应用
- **智能搜索**：[AI/01-Overview.md](../AI/01-Overview.md) ←→ 智能搜索在知识探索中的应用

### 6.2 与计算机科学的理论支撑

#### 6.2.1 算法设计

- **图算法**：[ComputerScience/03-Algorithms.md](../ComputerScience/03-Algorithms.md) ←→ 图算法在布局计算中的应用
- **几何算法**：[ComputerScience/01-Overview.md](../ComputerScience/01-Overview.md) ←→ 几何算法在空间布局中的应用
- **优化算法**：[ComputerScience/02-Computability.md](../ComputerScience/02-Computability.md) ←→ 优化算法在布局优化中的应用

#### 6.2.2 图形学

- **计算机图形学**：[ComputerScience/01-Overview.md](../ComputerScience/01-Overview.md) ←→ 图形学在渲染中的应用
- **几何计算**：[ComputerScience/03-Algorithms.md](../ComputerScience/03-Algorithms.md) ←→ 几何计算在碰撞检测中的应用
- **实时渲染**：[ComputerScience/02-Computability.md](../ComputerScience/02-Computability.md) ←→ 实时渲染在交互中的应用

### 6.3 与数学理论的结合

#### 6.3.1 几何学

- **几何学**：[Mathematics/Geometry/01-Overview.md](../Mathematics/Geometry/01-Overview.md) ←→ 几何学在空间布局中的应用
- **拓扑学**：[Mathematics/Geometry/02-CoreConcepts.md](../Mathematics/Geometry/02-CoreConcepts.md) ←→ 拓扑学在图形变形中的应用
- **微分几何**：[Mathematics/Geometry/03-Euclidean.md](../Mathematics/Geometry/03-Euclidean.md) ←→ 微分几何在曲线绘制中的应用

#### 6.3.2 优化理论

- **线性规划**：[Mathematics/Algebra/04-LinearAlgebra.md](../Mathematics/Algebra/04-LinearAlgebra.md) ←→ 线性规划在布局优化中的应用
- **非线性优化**：[Mathematics/Calculus/03-DifferentialCalculus.md](../Mathematics/Calculus/03-DifferentialCalculus.md) ←→ 非线性优化在力导向布局中的应用
- **数值分析**：[Mathematics/Calculus/02-LimitsContinuity.md](../Mathematics/Calculus/02-LimitsContinuity.md) ←→ 数值分析在动画计算中的应用

### 6.4 与软件工程的实践应用

#### 6.4.1 系统架构

- **前端架构**：[SoftwareEngineering/Architecture/01-DistributedMicroservices.md](../SoftwareEngineering/Architecture/01-DistributedMicroservices.md) ←→ 前端架构在可视化系统中的应用
- **组件设计**：[SoftwareEngineering/DesignPattern/01-GoF.md](../SoftwareEngineering/DesignPattern/01-GoF.md) ←→ 组件设计在可视化组件中的应用
- **性能优化**：[SoftwareEngineering/DesignPattern/02-ConcurrentParallel.md](../SoftwareEngineering/DesignPattern/02-ConcurrentParallel.md) ←→ 性能优化在渲染中的应用

#### 6.4.2 工程实践

- **用户体验**：[SoftwareEngineering/Architecture/00-Overview.md](../SoftwareEngineering/Architecture/00-Overview.md) ←→ 用户体验在交互设计中的应用
- **响应式设计**：[SoftwareEngineering/Microservices/01-Basics.md](../SoftwareEngineering/Microservices/01-Basics.md) ←→ 响应式设计在自适应布局中的应用
- **测试策略**：[SoftwareEngineering/Workflow-01-Basics.md](../SoftwareEngineering/Workflow-01-Basics.md) ←→ 测试策略在可视化系统中的应用

## 7. 批判性分析与未来展望

### 7.1 理论假设与局限性

#### 7.1.1 布局算法局限性

1. **计算复杂度**：大规模图的布局计算复杂度高
2. **局部最优**：力导向算法容易陷入局部最优
3. **可扩展性**：传统算法难以处理超大规模图
4. **美学标准**：缺乏统一的美学评价标准

#### 7.1.2 交互设计挑战

1. **认知负荷**：复杂交互可能增加用户认知负荷
2. **学习成本**：高级交互功能需要用户学习
3. **性能问题**：实时交互对性能要求高
4. **可访问性**：需要考虑不同用户群体的可访问性

#### 7.1.3 可视化表达困难

1. **信息密度**：如何在有限空间内展示大量信息
2. **视觉混乱**：节点和边过多时容易产生视觉混乱
3. **语义表达**：如何准确表达复杂的语义关系
4. **文化差异**：不同文化背景下的视觉理解差异

### 7.2 技术挑战与创新方向

#### 7.2.1 技术挑战

1. **大规模可视化**：处理百万级节点和边的可视化
2. **实时渲染**：支持实时交互和动画效果
3. **多模态融合**：整合文本、图像、视频等多种模态
4. **跨平台兼容**：支持不同设备和平台的兼容性

#### 7.2.2 创新方向

1. **AI驱动的可视化**：利用AI技术优化可视化效果
2. **沉浸式体验**：VR/AR技术在可视化中的应用
3. **自适应布局**：根据用户行为自适应调整布局
4. **协作可视化**：支持多用户协作的可视化系统

### 7.3 未来发展趋势

#### 7.3.1 短期趋势（1-3年）

1. **WebGL普及**：基于WebGL的高性能可视化
2. **响应式设计**：自适应不同屏幕尺寸的可视化
3. **移动端优化**：针对移动设备的可视化优化
4. **实时协作**：支持实时协作的可视化平台

#### 7.3.2 中期趋势（3-5年）

1. **AI辅助设计**：AI辅助的可视化设计工具
2. **沉浸式可视化**：VR/AR在知识图谱可视化中的应用
3. **智能交互**：基于AI的智能交互系统
4. **多模态融合**：文本、图像、音频的多模态可视化

#### 7.3.3 长期趋势（5-10年）

1. **脑机接口**：基于脑机接口的可视化交互
2. **量子可视化**：利用量子计算的可视化技术
3. **全息显示**：全息技术在可视化中的应用
4. **自主可视化**：具备自主学习能力的可视化系统

## 8. 术语表

### 8.1 核心术语

- **图谱可视化（Graph Visualization）**：将知识图谱以图形化方式展示的技术
- **布局算法（Layout Algorithm）**：计算节点位置和边路径的算法
- **力导向布局（Force-Directed Layout）**：基于物理模拟的布局算法
- **视觉映射（Visual Mapping）**：将数据属性映射到视觉元素的过程
- **交互设计（Interaction Design）**：设计用户与可视化系统交互的方式

### 8.2 技术术语

- **WebGL**：基于Web的3D图形渲染技术
- **D3.js**：数据驱动的文档操作JavaScript库
- **Canvas**：HTML5的2D绘图API
- **SVG**：可缩放矢量图形格式
- **Three.js**：JavaScript 3D库

## 9. 符号表

### 9.1 数学符号

- $\mathcal{GV}$: 可视化框架
- $\mathcal{V}$: 视觉元素集合
- $\mathcal{L}$: 布局算法集合
- $\mathcal{I}$: 交互功能集合
- $\mathcal{A}$: 动画效果集合
- $\mathcal{R}$: 渲染引擎集合

### 9.2 算法符号

- $F$: 力向量
- $d$: 距离
- $k$: 弹簧常数
- $c$: 阻尼系数
- $t$: 时间
- $\alpha$: 学习率

---

**文档状态**: ✅ 已完成深度优化与批判性提升
**最后更新**: 2024-12-28
**下一步**: 07-IntelligentRecommendation.md 智能推荐与知识发现
**交叉引用**: 与所有分支建立完整的交叉引用网络
