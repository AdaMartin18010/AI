# 04.6 自然语言处理理论 (Natural Language Processing Theory)

## 概述

自然语言处理理论是AI形式科学理论体系的重要组成部分，研究计算机理解、生成和处理人类语言的理论基础。本模块涵盖语言模型、词嵌入、序列到序列模型、注意力机制等核心理论，以及现代NLP架构如Transformer、BERT、GPT等的理论基础。

## 目录结构

```text
04.6-NaturalLanguage/
├── README.md                    # 本文件
├── language_models.rs           # 语言模型实现
├── word_embeddings.rs           # 词嵌入实现
├── sequence_models.rs           # 序列模型实现
├── attention_mechanism.rs       # 注意力机制实现
├── transformer_architecture.rs  # Transformer架构实现
└── mod.rs                       # 模块主文件
```

## 核心理论框架

### 1. 语言模型理论

#### 1.1 N-gram语言模型

**概率定义**:
$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^n P(w_i | w_1, w_2, \ldots, w_{i-1})$$

**N-gram近似**:
$$P(w_i | w_1, w_2, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1})$$

**最大似然估计**:
$$P(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \frac{\text{count}(w_{i-n+1}, \ldots, w_i)}{\text{count}(w_{i-n+1}, \ldots, w_{i-1})}$$

#### 1.2 平滑技术

**拉普拉斯平滑**:
$$P(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \frac{\text{count}(w_{i-n+1}, \ldots, w_i) + 1}{\text{count}(w_{i-n+1}, \ldots, w_{i-1}) + |V|}$$

**插值平滑**:
$$P(w_i | w_{i-2}, w_{i-1}) = \lambda_1 P(w_i) + \lambda_2 P(w_i | w_{i-1}) + \lambda_3 P(w_i | w_{i-2}, w_{i-1})$$

其中 $\lambda_1 + \lambda_2 + \lambda_3 = 1$。

#### 1.3 神经网络语言模型

**前馈神经网络**:
$$P(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \text{softmax}(W \cdot \tanh(U \cdot [e_{i-n+1}, \ldots, e_{i-1}] + b_1) + b_2)$$

其中 $e_i$ 是词 $w_i$ 的嵌入向量。

### 2. 词嵌入理论

#### 2.1 Word2Vec

**Skip-gram模型**:
$$P(w_o | w_c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

其中 $v_c$ 是中心词的向量，$u_o$ 是上下文词的向量。

**目标函数**:
$$J = -\sum_{w_c, w_o} \log P(w_o | w_c)$$

**负采样**:
$$J = -\log \sigma(u_o^T v_c) - \sum_{k=1}^K \log \sigma(-u_k^T v_c)$$

其中 $u_k$ 是负样本词的向量。

#### 2.2 GloVe

**共现矩阵**:
$$X_{ij} = \text{count}(w_i, w_j)$$

**目标函数**:
$$J = \sum_{i,j=1}^V f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

其中 $f(x)$ 是权重函数：
$$f(x) = \begin{cases} 
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}$$

#### 2.3 FastText

**子词嵌入**:
$$s(w) = \sum_{g \in G_w} z_g$$

其中 $G_w$ 是词 $w$ 的n-gram集合，$z_g$ 是n-gram $g$ 的向量。

### 3. 序列模型理论

#### 3.1 循环神经网络 (RNN)

**基本RNN**:
$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b_h)$$
$$y_t = W_y h_t + b_y$$

**梯度消失问题**: 当序列很长时，梯度在反向传播过程中指数衰减。

#### 3.2 长短期记忆网络 (LSTM)

**遗忘门**:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**候选值**:
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**细胞状态**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**隐藏状态**:
$$h_t = o_t \odot \tanh(C_t)$$

#### 3.3 门控循环单元 (GRU)

**更新门**:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**重置门**:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**候选隐藏状态**:
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

**隐藏状态**:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 4. 序列到序列模型

#### 4.1 编码器-解码器架构

**编码器**:
$$h_t = \text{Encoder}(x_t, h_{t-1})$$

**解码器**:
$$s_t = \text{Decoder}(y_{t-1}, s_{t-1}, c_t)$$
$$P(y_t | y_{<t}, x) = \text{softmax}(W_o s_t + b_o)$$

其中 $c_t$ 是上下文向量。

#### 4.2 注意力机制

**注意力权重**:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}$$

其中 $e_{ij} = a(s_{i-1}, h_j)$ 是注意力分数。

**上下文向量**:
$$c_i = \sum_{j=1}^T \alpha_{ij} h_j$$

**注意力分数函数**:
$$a(s, h) = v^T \tanh(W_s s + W_h h + b)$$

### 5. Transformer架构理论

#### 5.1 自注意力机制

**查询、键、值**:
$$Q = XW_Q, K = XW_K, V = XW_V$$

**注意力输出**:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $d_k$ 是键的维度。

#### 5.2 多头注意力

**多头输出**:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, \ldots, head_h)W^O$$

其中：
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 5.3 位置编码

**正弦位置编码**:
$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

**相对位置编码**:
$$PE_{(pos,2i)} = \sin((pos - \text{rel\_pos})/10000^{2i/d_{model}})$$

### 6. 预训练语言模型

#### 6.1 BERT (Bidirectional Encoder Representations from Transformers)

**掩码语言模型 (MLM)**:
$$L_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$

其中 $M$ 是被掩码的位置集合。

**下一句预测 (NSP)**:
$$L_{NSP} = -\log P(\text{IsNext} | \text{segment}_1, \text{segment}_2)$$

**总损失**:
$$L = L_{MLM} + L_{NSP}$$

#### 6.2 GPT (Generative Pre-trained Transformer)

**自回归语言模型**:
$$P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n P(x_i | x_1, x_2, \ldots, x_{i-1})$$

**目标函数**:
$$L = -\sum_{i=1}^n \log P(x_i | x_1, x_2, \ldots, x_{i-1})$$

#### 6.3 T5 (Text-to-Text Transfer Transformer)

**统一框架**: 将所有NLP任务转换为文本到文本的格式。

**目标函数**:
$$L = -\sum_{i=1}^{|y|} \log P(y_i | x, y_{<i})$$

### 7. 机器翻译理论

#### 7.1 统计机器翻译

**贝叶斯框架**:
$$P(e|f) = \frac{P(f|e)P(e)}{P(f)}$$

其中 $e$ 是目标语言，$f$ 是源语言。

**噪声信道模型**:
$$e^* = \arg\max_e P(e|f) = \arg\max_e P(f|e)P(e)$$

#### 7.2 神经机器翻译

**编码器-解码器**:
$$h_t = \text{Encoder}(f_t, h_{t-1})$$
$$s_t = \text{Decoder}(e_{t-1}, s_{t-1}, c_t)$$

**注意力机制**:
$$c_t = \sum_{i=1}^{|f|} \alpha_{ti} h_i$$

### 8. 文本分类理论

#### 8.1 传统方法

**TF-IDF**:
$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

其中：
$$\text{IDF}(t) = \log \frac{N}{\text{DF}(t)}$$

#### 8.2 深度学习方法

**CNN文本分类**:
$$h_i = \text{ReLU}(W \cdot x_{i:i+k-1} + b)$$
$$h = \max_{i} h_i$$

**RNN文本分类**:
$$h_t = \text{RNN}(x_t, h_{t-1})$$
$$y = \text{softmax}(W \cdot h_T + b)$$

### 9. 命名实体识别 (NER)

#### 9.1 序列标注

**CRF层**:
$$P(y|x) = \frac{\exp(\sum_{i=1}^n \psi_i(y_{i-1}, y_i, x))}{\sum_{y'} \exp(\sum_{i=1}^n \psi_i(y'_{i-1}, y'_i, x))}$$

其中 $\psi_i$ 是势函数。

#### 9.2 BiLSTM-CRF

**双向LSTM**:
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

**CRF解码**:
$$y^* = \arg\max_y P(y|x)$$

### 10. 问答系统理论

#### 10.1 检索式问答

**TF-IDF检索**:
$$\text{score}(q,d) = \sum_{t \in q} \text{TF-IDF}(t,d)$$

**BM25**:
$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$$

#### 10.2 生成式问答

**Seq2Seq模型**:
$$P(a|q,c) = \prod_{i=1}^{|a|} P(a_i | q, c, a_{<i})$$

其中 $a$ 是答案，$q$ 是问题，$c$ 是上下文。

### 11. 文本生成理论

#### 11.1 语言模型生成

**贪婪解码**:
$$y_t = \arg\max_{y} P(y | x, y_{<t})$$

**束搜索**:
维护k个候选序列，选择概率最高的序列。

#### 11.2 条件生成

**条件语言模型**:
$$P(y|x) = \prod_{i=1}^{|y|} P(y_i | x, y_{<i})$$

**VAE文本生成**:
$$P(y) = \int P(y|z)P(z)dz$$

其中 $z$ 是潜在变量。

### 12. 对话系统理论

#### 12.1 任务导向对话

**状态跟踪**:
$$s_t = \text{StateTracker}(s_{t-1}, u_t)$$

**策略学习**:
$$a_t = \text{Policy}(s_t)$$

#### 12.2 开放域对话

**生成式对话**:
$$P(r|u) = \prod_{i=1}^{|r|} P(r_i | u, r_{<i})$$

其中 $r$ 是回复，$u$ 是用户输入。

## 实现示例

### Rust实现概览

```rust
// 语言模型
pub trait LanguageModel {
    fn probability(&self, sequence: &[String]) -> f64;
    fn generate(&self, length: usize) -> Vec<String>;
}

// 词嵌入
pub struct WordEmbedding {
    embeddings: HashMap<String, Vec<f64>>,
    dimension: usize,
}

impl WordEmbedding {
    pub fn similarity(&self, word1: &str, word2: &str) -> f64 {
        let emb1 = self.embeddings.get(word1).unwrap();
        let emb2 = self.embeddings.get(word2).unwrap();
        cosine_similarity(emb1, emb2)
    }
}

// 注意力机制
pub struct Attention {
    query_weight: Matrix,
    key_weight: Matrix,
    value_weight: Matrix,
    output_weight: Matrix,
}

impl Attention {
    pub fn forward(&self, query: &Matrix, key: &Matrix, value: &Matrix) -> Matrix {
        let q = query * &self.query_weight;
        let k = key * &self.key_weight;
        let v = value * &self.value_weight;
        
        let attention_weights = softmax(&(q * k.transpose()) / (k.cols() as f64).sqrt());
        attention_weights * v
    }
}

// Transformer编码器
pub struct TransformerEncoder {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl TransformerEncoder {
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let attn_output = self.attention.forward(input, input, input);
        let norm1_output = self.layer_norm1.forward(&(input + attn_output));
        
        let ff_output = self.feed_forward.forward(&norm1_output);
        self.layer_norm2.forward(&(norm1_output + ff_output))
    }
}
```

## 总结

自然语言处理理论为计算机理解和生成人类语言提供了完整的数学框架，从传统的统计方法到现代的深度学习模型，每个理论都为实际应用提供了重要指导。随着大语言模型的发展，NLP技术在机器翻译、问答系统、文本生成等领域取得了突破性进展。

## 参考文献

1. Jurafsky, D., & Martin, J. H. (2009). Speech and language processing. Pearson.
2. Goldberg, Y. (2017). Neural network methods for natural language processing. Synthesis lectures on human language technologies, 10(1), 1-309.
3. Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30.
4. Devlin, J., et al. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
5. Brown, T., et al. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901. 