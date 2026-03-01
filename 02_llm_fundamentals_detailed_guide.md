# LLM Fundamentals & Tokenization - Detailed Concept Guide

A comprehensive breakdown of Large Language Model concepts, Transformer architecture, Tokenization, and Embeddings with step-by-step explanations and visualizations.

---

## Table of Contents
1. [Core LLM Concepts](#core-llm-concepts)
2. [Transformer Architecture](#transformer-architecture)
3. [Attention Mechanism Deep Dive](#attention-mechanism-deep-dive)
4. [Tokenization & Embeddings](#tokenization--embeddings)
5. [Generation & Decoding Strategies](#generation--decoding-strategies)

---

## Core LLM Concepts

### What Are Large Language Models?

#### Concept Breakdown

**Definition**: Large Language Models (LLMs) are neural networks trained on vast text corpora to understand and generate human language. They learn statistical patterns in language to predict the next token.

**The Core Task: Next Token Prediction**
```
Input:  "The cat sat on the"
Output: "mat" (highest probability)
         "chair" (lower probability)
         "roof" (even lower probability)
```

**Key Characteristics:**

| Characteristic | Description | Example |
|----------------|-------------|---------|
| **Scale** | Billions of parameters | GPT-3: 175B, GPT-4: ~1.8T |
| **Pre-training** | Trained on massive text corpora | Internet, books, code |
| **Autoregressive** | Generate one token at a time | Left-to-right generation |
| **Few-shot capable** | Learn from examples in context | In-context learning |
| **Emergent abilities** | Capabilities not explicitly trained | Chain-of-thought reasoning |

#### How LLMs Work: Step-by-Step

**Step 1: Tokenization**
```python
# Raw text → Tokens
text = "Hello, world!"
tokens = ["Hello", ",", " world", "!"]
token_ids = [15496, 11, 995, 0]  # GPT-2 token IDs
```

**Step 2: Embedding Lookup**
```
Token ID 15496 → Embedding Vector [0.23, -0.45, 0.89, ..., 0.12] (768 dimensions)
```

**Step 3: Transformer Processing**
```
Input Embeddings → Self-Attention → Feed-Forward → Output Probabilities
```

**Step 4: Token Generation**
```
Output Logits → Softmax → Probability Distribution → Sample Next Token
```

#### Visualization: LLM Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│         (Vocabulary-sized logits → Softmax)                 │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCKS (×N)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Multi-Head  │ →  │  Add &      │ →  │ Feed        │      │
│  │ Attention   │    │  Norm       │    │ Forward     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ↓                 ↓                  ↓              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Add &      │ →  │  Layer      │    │  Add &      │      │
│  │  Norm       │    │  Norm       │    │  Norm       │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                   INPUT EMBEDDINGS                          │
│     Token Embeddings + Positional Encodings                 │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                     TOKEN IDS                               │
│              [15496, 11, 995, 0]                            │
└─────────────────────────────────────────────────────────────┘
```

---

### Question: Explain the Transformer Architecture

#### Concept Breakdown

**The Transformer Revolution (Vaswani et al., 2017)**

Before Transformers:
- **RNNs/LSTMs**: Sequential processing, slow, vanishing gradients
- **Problem**: "The cat sat on the mat... [50 words later] ...it was hungry"
  - LSTM struggles to connect "it" with "cat" over long distances

Transformers solved this with:
- **Self-Attention**: Direct connection between any two positions
- **Parallel Processing**: All tokens processed simultaneously
- **Scalable Training**: Can handle much larger datasets

#### Architecture Components

```
┌────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER ARCHITECTURE                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌──────────────────┐           ┌──────────────────┐          │
│   │     ENCODER      │           │     DECODER      │          │
│   │  (Understanding) │           │  (Generation)    │          │
│   └────────┬─────────┘           └────────┬─────────┘          │
│            │                              │                    │
│   ┌────────▼─────────┐           ┌────────▼─────────┐          │
│   │  Input Tokens    │           │  Output Tokens   │          │
│   │  (Source Lang)   │           │  (Target Lang)   │          │
│   └────────┬─────────┘           └────────┬─────────┘          │
│            │                              │                    │
│   ┌────────▼─────────┐           ┌────────▼─────────┐          │
│   │  Token + Pos     │           │  Token + Pos     │          │
│   │  Embedding       │           │  Embedding       │          │
│   └────────┬─────────┘           └────────┬─────────┘          │
│            │                              │                    │
│   ┌────────▼─────────┐           ┌────────▼─────────┐          │
│   │  × N Blocks      │           │  × N Blocks      │          │
│   │  ┌───────────┐   │           │  ┌───────────┐   │          │
│   │  │Self-Attn  │   │           │  │Masked     │   │          │
│   │  │           │   │           │  │Self-Attn  │   │          │
│   │  └─────┬─────┘   │           │  └─────┬─────┘   │          │
│   │  ┌─────▼─────┐   │           │  ┌─────▼─────┐   │          │
│   │  │Cross-Attn │◄──┼───────────┼──│ (Not in   │   │          │
│   │  │(For Dec)  │   │           │  │ Encoder)  │   │          │
│   │  └─────┬─────┘   │           │  └─────┬─────┘   │          │
│   │  ┌─────▼─────┐   │           │  ┌─────▼─────┐   │          │
│   │  │Feed-      │   │           │  │Feed-      │   │          │
│   │  │Forward    │   │           │  │Forward    │   │          │
│   │  └───────────┘   │           │  └───────────┘   │          │
│   └────────┬─────────┘           └────────┬─────────┘          │
│            │                              │                    │
│   ┌────────▼─────────┐           ┌────────▼─────────┐          │
│   │  Contextual      │           │  Output          │          │
│   │  Representations │           │  Probabilities   │          │
│   └──────────────────┘           └──────────────────┘          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### Encoder vs Decoder: Detailed Comparison

| Aspect | Encoder (BERT) | Decoder (GPT) | Encoder-Decoder (T5) |
|--------|---------------|---------------|---------------------|
| **Direction** | Bidirectional | Left-to-right | Both |
| **Attention** | Full self-attention | Masked self-attention | Cross-attention |
| **Best For** | Understanding | Generation | Translation |
| **Pre-training** | MLM (Mask prediction) | CLM (Next token) | Span corruption |
| **Masking** | No masking | Causal masking | Source: No, Target: Yes |
| **Example** | "The [MASK] sat" → "cat" | "The cat" → "sat" | English → French |

#### Step-by-Step: Data Flow Through Transformer

**Example Input**: "The cat sat"

**Step 1: Tokenization**
```
Text: "The cat sat"
Tokens: ["The", " cat", " sat"]
Token IDs: [1996, 4937, 2438]
```

**Step 2: Embedding Layer**
```python
# Each token ID → Dense vector
embedding_matrix.shape  # (vocab_size, hidden_dim) = (30000, 768)

token_embeddings = embedding_matrix[token_ids]
# Shape: (3, 768)
# [[0.2, -0.5, ..., 0.1],   # "The"
#  [0.3, 0.8, ..., -0.2],    # " cat"
#  [-0.1, 0.4, ..., 0.6]]    # " sat"
```

**Step 3: Add Positional Encoding**
```python
# Positional encoding adds position information
position_embeddings = [
    [0.0, 1.0, 0.0, ..., 0.5],   # Position 0
    [0.5, 0.8, 0.3, ..., 0.2],   # Position 1
    [0.9, 0.3, 0.7, ..., 0.1]    # Position 2
]

final_input = token_embeddings + position_embeddings
# Shape: (3, 768)
```

**Step 4: Self-Attention Processing**
```python
# Compute Query, Key, Value matrices
Q = final_input @ W_q  # (3, 768) @ (768, 768) = (3, 768)
K = final_input @ W_k  # (3, 768)
V = final_input @ W_v  # (3, 768)

# Attention scores
scores = Q @ K.T / sqrt(768)  # (3, 3)
# [[2.1, 0.5, 0.3],   # "The" attends most to itself
#  [0.4, 2.3, 1.8],   # "cat" attends to "cat" and "sat"
#  [0.2, 1.9, 2.0]]   # "sat" attends most to "sat"

attention_weights = softmax(scores)  # Row-wise softmax
output = attention_weights @ V  # (3, 768)
```

**Step 5: Feed-Forward Network**
```python
# Two linear transformations with ReLU
ff_output = relu(output @ W1 + b1) @ W2 + b2
# Expands to intermediate dim (3072), then back to 768
```

**Step 6: Layer Normalization & Residual**
```python
# Add residual connection and normalize
block_output = layer_norm(output + ff_output)
```

**Step 7: Repeat for N Layers**
```python
# Typically 12, 24, or 96 layers
for layer in range(num_layers):
    hidden_states = transformer_layer(hidden_states)
```

**Step 8: Output Projection**
```python
# Project to vocabulary size
logits = hidden_states @ W_output  # (3, 30000)
probabilities = softmax(logits, dim=-1)
```

---

## Attention Mechanism Deep Dive

### Question: Explain Self-Attention Step by Step

#### Concept Breakdown

**The Intuition**: When reading a sentence, your brain focuses on relevant words. Self-attention lets the model do the same - each token "looks at" other tokens to understand context.

**Example**: "The animal didn't cross the street because it was too tired"
- What does "it" refer to? "animal" (not "street")
- Self-attention helps "it" attend strongly to "animal"

#### Mathematical Formulation

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What information do I provide?
- **d_k**: Dimension of key vectors (for scaling)

#### Step-by-Step Calculation

**Sentence**: "The cat sat"

**Step 1: Create Input Representations**
```
Token      Embedding (simplified, d=4)
────────────────────────────────────
"The"  →  [1.0, 0.5, 0.2, 0.1]
"cat"  →  [0.3, 0.9, 0.4, 0.2]
"sat"  →  [0.2, 0.3, 0.8, 0.7]

X = [[1.0, 0.5, 0.2, 0.1],
     [0.3, 0.9, 0.4, 0.2],
     [0.2, 0.3, 0.8, 0.7]]  # Shape: (3, 4)
```

**Step 2: Initialize Weight Matrices** (Learned during training)
```
W_Q = [[0.1, 0.2],       W_K = [[0.3, 0.1],       W_V = [[0.2, 0.3],
       [0.3, 0.4],             [0.2, 0.5],             [0.1, 0.4],
       [0.2, 0.1],             [0.4, 0.2],             [0.3, 0.2],
       [0.4, 0.3]]             [0.1, 0.3]]             [0.4, 0.1]]
       
# Shape: (4, 2) for each
```

**Step 3: Compute Q, K, V**
```python
Q = X @ W_Q  # (3, 4) @ (4, 2) = (3, 2)
K = X @ W_K  # (3, 2)
V = X @ W_V  # (3, 2)

Q = [[0.32, 0.51],    # "The" Query
     [0.43, 0.72],    # "cat" Query
     [0.38, 0.49]]    # "sat" Query

K = [[0.44, 0.33],    # "The" Key
     [0.61, 0.59],    # "cat" Key
     [0.70, 0.47]]    # "sat" Key

V = [[0.33, 0.47],    # "The" Value
     [0.43, 0.58],    # "cat" Value
     [0.51, 0.45]]    # "sat" Value
```

**Step 4: Compute Attention Scores**
```python
# Q @ K^T
scores = Q @ K.T  # (3, 2) @ (2, 3) = (3, 3)

scores = [[0.31, 0.45, 0.38],   # "The" attending to ["The", "cat", "sat"]
          [0.43, 0.69, 0.60],   # "cat" attending to ["The", "cat", "sat"]
          [0.33, 0.52, 0.50]]   # "sat" attending to ["The", "cat", "sat"]

# Scale by sqrt(d_k) = sqrt(2) ≈ 1.41
scaled_scores = scores / 1.41

scaled_scores = [[0.22, 0.32, 0.27],
                 [0.30, 0.49, 0.43],
                 [0.23, 0.37, 0.35]]
```

**Step 5: Apply Softmax**
```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

attention_weights = [[0.31, 0.36, 0.33],   # "The": attends 31% to self, 36% to "cat"
                     [0.27, 0.40, 0.33],   # "cat": attends 40% to self
                     [0.29, 0.38, 0.33]]   # "sat": attends 38% to "cat"
```

**Step 6: Weighted Sum of Values**
```python
output = attention_weights @ V  # (3, 3) @ (3, 2) = (3, 2)

output = [[0.42, 0.50],   # Updated "The" representation
          [0.43, 0.51],   # Updated "cat" representation
          [0.43, 0.50]]   # Updated "sat" representation
```

#### Visualization: Attention Pattern

```
                    "The"   "cat"   "sat"
                   ┌─────┬───────┬───────┐
            "The"  │ 31% │  36%  │  33%  │
                   ├─────┼───────┼───────┤
Attention  "cat"   │ 27% │  40%  │  33%  │
Weights            ├─────┼───────┼───────┤
            "sat"  │ 29% │  38%  │  33%  │
                   └─────┴───────┴───────┘
```

**Interpretation**:
- "cat" attends most to itself (40%) - self-attention
- "sat" attends strongly to "cat" (38%) - verb-subject relationship
- Each token blends information from all tokens based on relevance

---

### Multi-Head Attention

#### Concept Breakdown

**Problem with Single Attention**: One perspective might miss different types of relationships

**Solution**: Run attention multiple times in parallel (heads), each learning different aspects:
- Head 1: Subject-verb relationships
- Head 2: Pronoun-antecedent relationships  
- Head 3: Adjective-noun modifications
- etc.

#### Mathematical Formulation

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
```

#### Step-by-Step: Multi-Head Attention

**Configuration**: num_heads = 2, d_model = 4, d_k = d_v = 2

**Step 1: Split Q, K, V for Each Head**
```python
# Original Q, K, V have shape (3, 4)
# Split into 2 heads: each gets (3, 2)

Q_head1 = Q[:, :2]   # First half
Q_head2 = Q[:, 2:]   # Second half

# Same for K and V
```

**Step 2: Compute Attention for Each Head**
```python
# Head 1: Focus on syntactic relationships
head1_output = attention(Q_head1, K_head1, V_head1)
# Shape: (3, 2)

# Head 2: Focus on semantic relationships
head2_output = attention(Q_head2, K_head2, V_head2)
# Shape: (3, 2)
```

**Step 3: Concatenate Heads**
```python
concatenated = [head1_output | head2_output]  # Shape: (3, 4)

# Example:
# [[0.42, 0.50, 0.38, 0.45],   # "The"
#  [0.43, 0.51, 0.40, 0.48],   # "cat"
#  [0.43, 0.50, 0.39, 0.46]]   # "sat"
```

**Step 4: Final Linear Transformation**
```python
W_O = [[0.1, 0.2, 0.3, 0.4],
       [0.2, 0.1, 0.4, 0.3],
       [0.3, 0.4, 0.1, 0.2],
       [0.4, 0.3, 0.2, 0.1]]  # Shape: (4, 4)

final_output = concatenated @ W_O  # Shape: (3, 4)
```

#### Visualization: Multi-Head Attention

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Q, K, V (shape: seq_len × d_model)                  │
│                      │                                      │
│         ┌────────────┼────────────┐                         │
│         ▼            ▼            ▼                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Head 1  │  │  Head 2  │  │  Head H  │                   │
│  │ d_k=64   │  │ d_k=64   │  │ d_k=64   │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                         │
│       ▼             ▼             ▼                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │Attention │  │Attention │  │Attention │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                         │
│       ▼             ▼             ▼                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Output 1 │  │ Output 2 │  │ Output H │                   │
│  │(seq, 64) │  │(seq, 64) │  │(seq, 64) │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                         │
│       └─────────────┼─────────────┘                         │
│                     ▼                                       │
│              ┌──────────┐                                   │
│              │ Concat   │ (seq_len × d_model)               │
│              └────┬─────┘                                   │
│                   ▼                                         │
│              ┌──────────┐                                   │
│              │  W_O     │ Linear projection                 │
│              └────┬─────┘                                   │
│                   ▼                                         │
│              Final Output                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tokenization & Embeddings

### Question: Explain Different Tokenization Methods

#### Concept Breakdown

**What is Tokenization?**
The process of converting raw text into tokens (smallest units) that models can process.

```
Raw Text → Tokenizer → Token IDs → Model
"Hello"  → ["Hello"] → [15496]  → Embedding
```

#### Tokenization Methods Comparison

| Method | Example | Vocab Size | Pros | Cons |
|--------|---------|------------|------|------|
| **Character** | H-e-l-l-o | ~100-256 | No OOV, small vocab | Very long sequences, loses word meaning |
| **Word** | Hello world | 10K-1M | Intuitive, meaningful | Large vocab, OOV problems, doesn't handle typos |
| **Subword (BPE)** | Hell-o world | 32K-100K | Balance, handles OOV | Can split words awkwardly |
| **SentencePiece** | ▁Hello ▁world | 32K-100K | Language agnostic | Requires pre-training |

#### Step-by-Step: Byte Pair Encoding (BPE)

**Training Process**

**Step 1: Initialize Vocabulary**
```
Corpus: "low lower lowest"
Initial vocabulary: {l, o, w, e, r, s, t}
Initial tokens: l o w _ l o w e r _ l o w e s t
```

**Step 2: Count Pairs**
```
Pairs and frequencies:
(l, o): 3
(o, w): 3
(w, _): 1
(w, e): 2
(e, r): 1
(e, s): 1
...
```

**Step 3: Merge Most Frequent Pair**
```
Most frequent: (l, o) = 3
New token: "lo"
Updated vocabulary: {l, o, w, e, r, s, t, lo}
Updated tokens: lo w _ lo w e r _ lo w e s t
```

**Step 4: Repeat Merging**
```
Iteration 2: Merge (lo, w) → "low"
Vocabulary: {l, o, w, e, r, s, t, lo, low}
Tokens: low _ low e r _ low e s t

Iteration 3: Merge (low, e) → "lowe"
Vocabulary: {l, o, w, e, r, s, t, lo, low, lowe}
Tokens: low _ lowe r _ lowe s t

Iteration 4: Merge (lowe, r) → "lower"
Vocabulary adds: lower
Tokens: low _ lower _ lowe s t

Iteration 5: Merge (lowe, s) → " lowes"
Continue until desired vocab size...
```

**Final Vocabulary (simplified)**
```
{l, o, w, e, r, s, t, lo, low, lowe, lower, lowes, lowest}
```

**Encoding Example**
```
Input: "lower"
Tokens: ["lower"] → Single token (in vocabulary)

Input: "lowering"
Tokens: ["lower", "ing"] → Split into known subwords

Input: "xyz"
Tokens: ["x", "y", "z"] → Character fallback for unknown
```

#### Real Example: GPT-2 Tokenization

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Tokenization is important!"
tokens = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token strings: {tokenizer.convert_ids_to_tokens(tokens)}")
```

**Output:**
```
Text: Tokenization is important!
Tokens: [11241, 318, 281, 1672, 318, 281, 362, 1023, 0]
Token strings: ['Token', 'ization', ' is', ' important', '!']
```

**Breakdown:**
| Token ID | Token | Note |
|----------|-------|------|
| 11241 | "Token" | Base word |
| 318 | "ization" | Suffix (common pattern) |
| 281 | " is" | Space + word |
| 1672 | " important" | Space + base |
| 0 | "!" | Punctuation |

#### Visualization: Tokenization Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOKENIZATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "ChatGPT is amazing!"                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │  Normalization│  Lowercase, Unicode NFKC                     │
│  │              │  "chatgpt is amazing!"                        │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Pre-tokenize │  Split by whitespace/punctuation             │
│  │              │  ["chatgpt", "is", "amazing", "!"]            │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  BPE/Sentence │  Apply learned merge rules                   │
│  │  Piece       │  ["chat", "gpt", "is", "amazing", "!"]        │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Vocabulary  │  Map to token IDs                             │
│  │  Lookup      │  [15496, 4231, 318, 1842, 0]                  │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Question: What are Token Embeddings vs Positional Embeddings?

#### Concept Breakdown

**Token Embeddings**: Semantic meaning of words
- Similar words have similar vectors
- "king" and "queen" are close in embedding space
- Learned during pre-training

**Positional Embeddings**: Position information
- Transformers process all tokens in parallel (no inherent order)
- Need to inject "this is the 1st word, 2nd word, etc."
- Can be learned or fixed (sinusoidal)

#### Token Embeddings: Step-by-Step

**Step 1: Vocabulary Mapping**
```python
vocab_size = 50000  # GPT-2
embedding_dim = 768

# Embedding matrix: (vocab_size, embedding_dim)
embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.02
```

**Step 2: Lookup**
```python
token_id = 15496  # "Hello"
embedding = embedding_matrix[token_id]
# Shape: (768,)
# Values: [-0.023, 0.156, -0.089, ..., 0.034]
```

**Step 3: Semantic Properties**
```python
# Famous example: King - Man + Woman ≈ Queen
king = embedding_matrix[tokenizer.encode("king")[0]]
man = embedding_matrix[tokenizer.encode("man")[0]]
woman = embedding_matrix[tokenizer.encode("woman")[0]]

result_vector = king - man + woman

# Find closest embedding to result_vector
# → Should be "queen"
```

#### Positional Embeddings: Step-by-Step

**Method 1: Learned Positional Embeddings**
```python
max_seq_length = 1024
embedding_dim = 768

# Learned during training
positional_embeddings = torch.randn(max_seq_length, embedding_dim)

# For position 0, 1, 2 in sequence:
pos_0_embedding = positional_embeddings[0]  # (768,)
pos_1_embedding = positional_embeddings[1]  # (768,)
pos_2_embedding = positional_embeddings[2]  # (768,)
```

**Method 2: Sinusoidal (Original Transformer)**
```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding from "Attention Is All You Need"
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * 
        -(np.log(10000.0) / d_model)
    )  # (d_model/2,)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices
    
    return pe

# Example: position 0, 1, 2 with d_model=4
pe = get_positional_encoding(3, 4)

# pe[0] = [0.0, 1.0, 0.0, 1.0]      ← Position 0
# pe[1] = [0.841, 0.540, 0.010, 0.999]  ← Position 1
# pe[2] = [0.909, -0.416, 0.020, 0.999] ← Position 2
```

**Why Sinusoidal?**
```
┌──────────────────────────────────────────────────────────────┐
│           SINUSOIDAL POSITIONAL ENCODING                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Dimension 0 (sin, low freq):  ╭────╮    ╭────╮             │
│                              ──╯    ╰────╯    ╰──            │
│  Position: 0   10   20   30   40   50   60   70              │
│                                                              │
│  Dimension 2 (sin, high freq):  ╭╮  ╭╮  ╭╮  ╭╮  ╭╮  ╭╮      │
│                              ───╯╰──╯╰──╯╰──╯╰──╯╰──╯       │
│  Position: 0   10   20   30   40   50   60   70              │
│                                                              │
│  Key Property:                                               │
│  PE(pos + k) can be represented as linear function of PE(pos)│
│  → Model can learn relative positions easily!                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Final Input Representation

```python
# Combine token and positional embeddings
sequence = ["The", "cat", "sat"]
token_ids = [1996, 4937, 2438]

# Look up embeddings
token_embeds = embedding_matrix[token_ids]  # (3, 768)
pos_embeds = positional_embeddings[:3]      # (3, 768)

# Add together
final_input = token_embeds + pos_embeds     # (3, 768)

# Also add segment embeddings (for BERT) if needed
```

#### Visualization: Complete Embedding Process

```
┌─────────────────────────────────────────────────────────────────┐
│                   INPUT EMBEDDING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Token IDs: [1996, 4937, 2438]  ← ["The", "cat", "sat"]         │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────────────────────────────────┐        │
│  │           TOKEN EMBEDDINGS                         │        │
│  │  (Learned semantic meanings)                       │        │
│  │                                                    │        │
│  │  "The" → [0.2, -0.5, 0.3, ..., 0.1] (768-dim)     │        │
│  │  "cat" → [0.1, 0.8, -0.2, ..., 0.4] (768-dim)     │        │
│  │  "sat" → [0.3, 0.2, 0.7, ..., -0.3] (768-dim)     │        │
│  └────────────────────────────────────────────────────┘        │
│       │                                                         │
│       │  ┌─────────────────────────────────────────────────┐   │
│       └──┤ POSITIONAL EMBEDDINGS                             │   │
│          │ (Position information)                            │   │
│          │                                                   │   │
│          │ Pos 0 → [0.0, 1.0, 0.0, ..., 0.5] (768-dim)      │   │
│          │ Pos 1 → [0.8, 0.6, 0.4, ..., 0.2] (768-dim)      │   │
│          │ Pos 2 → [0.9, 0.3, 0.8, ..., 0.1] (768-dim)      │   │
│          └─────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    ┌────────────┐                              │
│                    │   ADD (+)  │                              │
│                    └─────┬──────┘                              │
│                          ▼                                     │
│  Final Input:                                                    │
│  [0.2+0.0, -0.5+1.0, 0.3+0.0, ..., 0.1+0.5] = "The" at pos 0   │
│  [0.1+0.8, 0.8+0.6, -0.2+0.4, ..., 0.4+0.2] = "cat" at pos 1   │
│  [0.3+0.9, 0.2+0.3, 0.7+0.8, ..., -0.3+0.1] = "sat" at pos 2   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Generation & Decoding Strategies

### Question: Explain Different Decoding Strategies

#### Concept Breakdown

Decoding strategies determine HOW the model selects the next token from the probability distribution.

#### Strategy Comparison

| Strategy | Method | Output | Use Case |
|----------|--------|--------|----------|
| **Greedy** | Always pick highest probability | Deterministic, often repetitive | Factual tasks |
| **Beam Search** | Track top-K sequences | Higher probability, less diverse | Translation, summarization |
| **Top-K** | Sample from K most likely | Balanced diversity/quality | General generation |
| **Top-P (Nucleus)** | Sample from smallest set with cumulative prob > P | Adaptive diversity | Creative writing |
| **Temperature** | Scale logits before softmax | Control randomness | All strategies |

#### Step-by-Step Examples

**Setup: Model output logits for next token**
```python
vocabulary = ["the", "a", "cat", "dog", "runs", "sleeps", "quickly"]
logits = [2.0, 1.5, 3.0, 2.8, 1.0, 0.5, 0.3]

# Convert to probabilities
probs = softmax(logits)
# [0.15, 0.09, 0.30, 0.25, 0.08, 0.07, 0.06]
```

**Strategy 1: Greedy Decoding**
```python
def greedy_decode(logits):
    """Always pick the highest probability token"""
    return argmax(logits)

# logits = [2.0, 1.5, 3.0, 2.8, 1.0, 0.5, 0.3]
#              ↑
#          highest (index 2)

next_token = "cat"  # Always!

# Problem: Can get stuck in loops
# "The cat sat on the cat sat on the cat..."
```

**Strategy 2: Beam Search**
```python
def beam_search_decode(logits, beam_width=2):
    """
    Keep top-K most likely partial sequences
    """
    # At each step, expand each beam and keep top-K
    # Example with beam_width=2:
    
    # Step 1: Top 2 tokens
    beams = [("cat", 0.30), ("dog", 0.25)]
    
    # Step 2: Expand each, keep top 2 overall
    # "cat the" (0.30 × 0.40 = 0.12)
    # "cat a" (0.30 × 0.35 = 0.105)
    # "dog the" (0.25 × 0.45 = 0.1125)
    # "dog a" (0.25 × 0.30 = 0.075)
    
    beams = [("dog the", 0.1125), ("cat the", 0.12)]
    # Continue...

# Advantage: Better global sequence probability
# "The cat sat on the mat" > "The cat cat cat cat"
```

**Strategy 3: Top-K Sampling**
```python
def top_k_sampling(logits, k=3, temperature=1.0):
    """
    Sample from K most likely tokens
    """
    # Apply temperature
    logits = logits / temperature
    probs = softmax(logits)
    
    # [0.15, 0.09, 0.30, 0.25, 0.08, 0.07, 0.06]
    #                    ↑↑↑
    #                top 3
    
    # Keep only top K
    top_k_indices = [2, 3, 0]  # "cat", "dog", "the"
    top_k_probs = [0.30, 0.25, 0.15]
    
    # Renormalize
    top_k_probs = [p / sum(top_k_probs) for p in top_k_probs]
    # [0.43, 0.36, 0.21]
    
    # Sample
    next_token = np.random.choice(top_k_indices, p=top_k_probs)
    
    return next_token

# k=1: Same as greedy
# k=vocab_size: Pure sampling (too random)
# k=40-50: Good balance (GPT-2 default)
```

**Strategy 4: Top-P (Nucleus) Sampling**
```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Sample from smallest set of tokens with cumulative probability >= p
    """
    probs = softmax(logits / temperature)
    # Sorted: [("cat", 0.30), ("dog", 0.25), ("the", 0.15), 
    #          ("a", 0.09), ("runs", 0.08), ...]
    
    # Cumulative sum
    # "cat": 0.30 ( < 0.9, include)
    # "dog": 0.55 ( < 0.9, include)
    # "the": 0.70 ( < 0.9, include)
    # "a": 0.79 ( < 0.9, include)
    # "runs": 0.87 ( < 0.9, include)
    # "sleeps": 0.94 ( >= 0.9, stop!)
    
    nucleus = ["cat", "dog", "the", "a", "runs"]  # 5 tokens
    nucleus_probs = [0.30, 0.25, 0.15, 0.09, 0.08]
    
    # Renormalize to sum to 1.0
    nucleus_probs = [p / 0.87 for p in nucleus_probs]
    
    # Sample from nucleus
    next_token = np.random.choice(nucleus, p=nucleus_probs)
    
    return next_token

# Advantage: Dynamic vocabulary size
# High confidence → small nucleus (focused)
# Low confidence → large nucleus (exploratory)
```

**Strategy 5: Temperature Scaling**
```python
def apply_temperature(logits, temperature):
    """
    temperature < 1: More focused/peaked
    temperature > 1: More flat/random
    """
    return logits / temperature

# Original logits: [2.0, 1.5, 3.0, 2.8, 1.0, 0.5, 0.3]

# Temperature = 0.5 (focused)
scaled = [4.0, 3.0, 6.0, 5.6, 2.0, 1.0, 0.6]
probs = softmax(scaled)
# [0.05, 0.02, 0.71, 0.19, 0.01, 0.01, 0.01]
# Almost always picks "cat"

# Temperature = 1.0 (balanced)
probs = softmax([2.0, 1.5, 3.0, 2.8, 1.0, 0.5, 0.3])
# [0.15, 0.09, 0.30, 0.25, 0.08, 0.07, 0.06]

# Temperature = 2.0 (random)
scaled = [1.0, 0.75, 1.5, 1.4, 0.5, 0.25, 0.15]
probs = softmax(scaled)
# [0.18, 0.15, 0.25, 0.23, 0.11, 0.05, 0.03]
# Much more uniform
```

#### Visualization: Decoding Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                   DECODING STRATEGIES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GREEDY (Deterministic)                                         │
│  ├── Input: Probability distribution                            │
│  ├── Action: argmax(probs)                                      │
│  └── Output: "cat" (always)                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────────────────────┐                    │
│  │  Probability:  ████████████████░░░░░░  │  30%               │
│  │  "cat" chosen: ▲                      │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
│  TOP-K SAMPLING (k=3)                                           │
│  ├── Candidates: ["cat", "dog", "the"]                          │
│  ├── Renormalized: [43%, 36%, 21%]                              │
│  └── Sampled: "dog" (36% chance)                               │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────────────────────┐                    │
│  │  cat  ██████████████░░░░  43%         │                    │
│  │  dog  ███████████░░░░░░░  36%  ← chosen│                   │
│  │  the  ██████░░░░░░░░░░░░  21%         │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
│  TOP-P SAMPLING (p=0.9)                                         │
│  ├── Dynamic set: ["cat", "dog", "the", "a", "runs"]            │
│  ├── Cumulative: 87%                                            │
│  └── "sleeps" excluded (would push over 90%)                   │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────────────────────────┐                    │
│  │  Included (p ≤ 0.9):                  │                    │
│  │  cat ████████░░ 30%                   │                    │
│  │  dog ██████░░░░ 25%                   │                    │
│  │  the ████░░░░░░ 15%                   │                    │
│  │  a   ██░░░░░░░░  9%                   │                    │
│  │  runs ██░░░░░░░  8% ← cutoff at 87%   │                    │
│  │  ─────────────────────────             │                    │
│  │  sleeps█░░░░░░░  7% ← excluded        │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Temperature Effect Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│              TEMPERATURE EFFECT ON DISTRIBUTION                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Temp = 0.3  (Conservative/Focused)                             │
│  ┌─────────────────────────────────────┐                       │
│  │ cat ████████████████████████████████│ 85%                  │
│  │ dog ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 10%                  │
│  │ the █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  4%                  │
│  │ others ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  1%                  │
│  └─────────────────────────────────────┘                       │
│  → Very likely to pick "cat", almost deterministic             │
│                                                                 │
│  Temp = 1.0  (Balanced)                                         │
│  ┌─────────────────────────────────────┐                       │
│  │ cat ████████████░░░░░░░░░░░░░░░░░░░░│ 30%                  │
│  │ dog ██████████░░░░░░░░░░░░░░░░░░░░░░│ 25%                  │
│  │ the ██████░░░░░░░░░░░░░░░░░░░░░░░░░░│ 15%                  │
│  │ a   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  9%                  │
│  │ ... ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 21%                  │
│  └─────────────────────────────────────┘                       │
│  → Balanced exploration and exploitation                       │
│                                                                 │
│  Temp = 2.0  (Creative/Random)                                  │
│  ┌─────────────────────────────────────┐                       │
│  │ cat ████████░░░░░░░░░░░░░░░░░░░░░░░░│ 18%                  │
│  │ dog ██████░░░░░░░░░░░░░░░░░░░░░░░░░░│ 15%                  │
│  │ the █████░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 12%                  │
│  │ a   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 11%                  │
│  │ ... █████████████████░░░░░░░░░░░░░░░│ 44%                  │
│  └─────────────────────────────────────┘                       │
│  → Much more random, all tokens have similar probability       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary Tables

### LLM Architecture Comparison

| Model | Architecture | Parameters | Context | Special Features |
|-------|-------------|------------|---------|------------------|
| **GPT-2** | Decoder-only | 1.5B | 1024 | First popular GPT |
| **GPT-3** | Decoder-only | 175B | 2048 | Few-shot learning |
| **GPT-4** | Decoder-only | ~1.8T | 128K | Multimodal, RLHF |
| **BERT** | Encoder-only | 340M | 512 | Bidirectional |
| **T5** | Encoder-Decoder | 11B | 512 | Unified text-to-text |
| **LLaMA** | Decoder-only | 7B-65B | 4096 | Open weights |
| **Claude** | Decoder-only | Unknown | 200K | Constitutional AI |

### Tokenization Comparison

| Tokenizer | Vocab Size | Method | Special Features |
|-----------|------------|--------|------------------|
| **BPE (GPT-2)** | 50K | Byte Pair | Space as part of token |
| **WordPiece (BERT)** | 30K | Subword | ## prefix for continuations |
| **SentencePiece** | 32K | Unigram/BPE | Language agnostic |
| **tiktoken (GPT-4)** | 100K | BPE | Better compression |

### When to Use Each Decoding Strategy

| Strategy | Best For | Temperature |
|----------|----------|-------------|
| **Greedy** | Factual Q&A, code generation | 0.1-0.3 |
| **Beam Search** | Translation, summarization | 0.3-0.7 |
| **Top-K (40)** | General purpose | 0.7-1.0 |
| **Top-P (0.9)** | Creative writing, brainstorming | 0.8-1.2 |
| **High Temperature** | Poetry, fiction, exploration | 1.2-2.0 |

---

## LLM Capabilities & Limitations

### Overview

Understanding what LLMs can and cannot do is crucial for building effective AI systems. This section breaks down their capabilities, limitations, and the underlying reasons.

---

### Capabilities

#### 1. In-Context Learning (Few-Shot Learning)

**Concept Breakdown**

In-context learning is the ability of LLMs to learn new tasks from examples provided in the prompt, without updating model weights.

**How It Works**
```
Traditional ML:          In-Context Learning:
Training Data →         Prompt with examples →
Gradient Descent →      Forward Pass Only →
Updated Model           Immediate Task Performance
(weeks of training)     (zero parameter updates)
```

**Step-by-Step Example: Sentiment Classification**

```python
# Zero-shot (no examples)
prompt_zero_shot = """
Classify the sentiment: "This movie was terrible"
Sentiment:
"""
# Model might be confused about format

# Few-shot (with examples)
prompt_few_shot = """
Classify the sentiment of reviews:

Review: "This movie was amazing!"
Sentiment: Positive

Review: "I hated every minute of it."
Sentiment: Negative

Review: "The acting was superb but the plot was confusing."
Sentiment: Mixed

Review: "This movie was terrible"
Sentiment:
"""
# Model learns the pattern from examples
# Output: Negative
```

**Why It Works**
```
┌──────────────────────────────────────────────────────────────┐
│              IN-CONTEXT LEARNING MECHANISM                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Pre-training:                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Internet text contains millions of implicit tasks  │   │
│  │  - Q&A patterns: "Q: What is X? A: Y"              │   │
│  │  - Translation: "Hello → Bonjour"                 │   │
│  │  - Classification: "Review: X. Rating: 5/5"         │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  Pattern Recognition:                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  During inference, model recognizes task structure  │   │
│  │  - Examples create "task context"                   │   │
│  │  - Model retrieves similar patterns from training  │   │
│  │  - Attention heads specialize in task templates   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Key Finding:                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Emergent at scale!                                 │   │
│  │  • GPT-2 Small (125M): Poor few-shot performance  │   │
│  │  • GPT-3 (175B): Strong few-shot capability       │   │
│  │  • Pattern: Linear increase → Sudden capability   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Scaling Effect**
```
Model Size    Few-Shot Accuracy    Explanation
─────────────────────────────────────────────────
125M          25%                  Random-ish
1.3B          40%                  Some patterns
6.7B          55%                  Better recognition
175B (GPT-3)  75%                  Strong capability
```

---

#### 2. Chain-of-Thought Reasoning

**Concept Breakdown**

Chain-of-thought (CoT) prompting elicits step-by-step reasoning by showing the model how to break down problems.

**Standard vs Chain-of-Thought**

```python
# Standard Prompting (Direct answer)
prompt_standard = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11

Q: The cafeteria had 23 apples. If they used 20 to make lunch and 
   bought 6 more, how many apples do they have?
A:
"""
# Output might be: 9 (incorrect, 23-20=3, 3+6=9... wait that's correct)
# Or might be: 29 (23-20=3, but then 23+6=29 - wrong!)

# Chain-of-Thought Prompting (Step by step)
prompt_cot = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 
   6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and 
   bought 6 more, how many apples do they have?
A:
"""
# Output: The cafeteria started with 23 apples. They used 20, 
# so they had 23 - 20 = 3 apples left. Then they bought 6 more,
# so they have 3 + 6 = 9 apples. The answer is 9.
```

**Why Chain-of-Thought Works**

```
┌──────────────────────────────────────────────────────────────┐
│              CHAIN-OF-THOUGHT MECHANISM                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Without CoT:                                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input: Math problem                                   │  │
│  │       ↓                                              │  │
│  │  Model: Jumps to conclusion (prone to errors)         │  │
│  │       ↓                                              │  │
│  │  Output: Single number (often incorrect)             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  With CoT:                                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input: Math problem                                   │  │
│  │       ↓                                              │  │
│  │  Model: Generates intermediate reasoning steps       │  │
│  │       ↓                                              │  │
│  │  Each step is simpler, easier to compute correctly     │  │
│  │       ↓                                              │  │
│  │  Can self-correct if a step seems wrong              │  │
│  │       ↓                                              │  │
│  │  Output: Step-by-step → Final answer (more accurate)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Key Insight:                                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Language models are auto-regressive                   │  │
│  │  → Each token conditions on all previous tokens       │  │
│  │  → Writing reasoning commits to intermediate values   │  │
│  │  → Reduces cognitive load per step                    │  │
│  │  → Similar to how humans solve complex problems!      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Self-Consistency Improvement**
```python
# Generate multiple reasoning paths and take majority vote
results = []
for _ in range(10):  # Generate 10 different CoT solutions
    response = llm.generate(prompt_cot, temperature=0.7)
    # Extract final answer from each response
    answer = extract_final_number(response)
    results.append(answer)

# Take majority vote
final_answer = most_common(results)
# Significantly improves accuracy over single sample
```

---

#### 3. Tool Use and Function Calling

**Concept Breakdown**

Modern LLMs can use external tools (APIs, calculators, databases) by generating structured function calls.

**How Function Calling Works**

```
┌──────────────────────────────────────────────────────────────┐
│                    FUNCTION CALLING FLOW                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Define Available Functions                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  functions = [                                        │  │
│  │    {                                                  │  │
│  │      "name": "get_weather",                           │  │
│  │      "description": "Get current weather",            │  │
│  │      "parameters": {                                │  │
│  │        "location": {"type": "string"},               │  │
│  │        "unit": {"enum": ["celsius", "fahrenheit"]}   │  │
│  │      }                                                │  │
│  │    },                                                 │  │
│  │    {                                                  │  │
│  │      "name": "calculate",                             │  │
│  │      "description": "Perform math calculation"       │  │
│  │    }                                                  │  │
│  │  ]                                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  Step 2: User Query                                          │
│  "What's the weather in London?"                           │
│                           │                                  │
│                           ▼                                  │
│  Step 3: Model Generates Function Call                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  {                                                    │  │
│  │    "function": "get_weather",                         │  │
│  │    "arguments": {                                     │  │
│  │      "location": "London",                          │  │
│  │      "unit": "celsius"                                │  │
│  │    }                                                  │  │
│  │  }                                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  Step 4: Execute Function                                    │
│  result = get_weather(location="London", unit="celsius")   │
│  # Returns: {"temp": 15, "condition": "cloudy"}              │
│                           │                                  │
│                           ▼                                  │
│  Step 5: Generate Final Response                             │
│  "The current weather in London is 15°C and cloudy."        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**ReAct Pattern: Reasoning + Acting**
```python
# ReAct = Reasoning + Action
prompt = """
You are an AI assistant that can use tools. When you need to use a tool,
output:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [parameters]

Then wait for the Observation before continuing.

Example:
Question: What is the population of the capital of France?
Thought: I need to find the capital of France first, then its population.
Action: search
Action Input: {"query": "capital of France"}
Observation: The capital of France is Paris.
Thought: Now I need to find the population of Paris.
Action: search  
Action Input: {"query": "population of Paris 2024"}
Observation: The population of Paris is approximately 2.1 million.
Thought: I have the answer.
Final Answer: The population of Paris, the capital of France, 
is approximately 2.1 million.

Now answer this question:
Question: {user_question}
"""
```

---

#### 4. Code Generation and Understanding

**Capabilities Breakdown**

```
┌──────────────────────────────────────────────────────────────┐
│                 CODE CAPABILITIES                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Level 1: Code Completion                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  def calculate_average(numbers):                    │   │
│  │      total = sum(numbers)                         │   │
│  │      count = len(numbers)                         │   │
│  │      return total / count  # ← Completes this     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Level 2: Function from Description                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "Write a function to reverse a string"             │   │
│  │  →                                                  │   │
│  │  def reverse_string(s):                             │   │
│  │      return s[::-1]                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Level 3: Bug Detection and Fixing                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Code with bug:                                     │   │
│  │  for i in range(len(list)):                         │   │
│  │      if list[i] == target:                         │   │
│  │          return i                                   │   │
│  │  return -1                                          │   │
│  │                                                     │   │
│  │  Bug: Modifying list while iterating                │   │
│  │  Fix: Iterate over a copy or use index-based loop   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Level 4: Complex Algorithm Design                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "Implement Dijkstra's shortest path algorithm"       │   │
│  │  → Complete working implementation with explanation │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Level 5: Code Review and Optimization                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "Review this code for SQL injection vulnerabilities" │   │
│  │  → Identifies parameterized query issues             │   │
│  │  → Suggests secure alternatives                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Why Code is "Easier" for LLMs**
```
1. Structured Syntax: Strict grammar rules (unlike natural language)
2. Deterministic: Same input always produces same output
3. Training Data: Code is abundant, well-commented, has examples
4. Self-Contained: Less world knowledge required
5. Testable: Can verify correctness with execution
```

---

### Limitations

#### 1. Hallucinations

**Concept Breakdown**

Hallucination is when an LLM generates plausible-sounding but false or ungrounded information.

**Types of Hallucinations**

```
┌──────────────────────────────────────────────────────────────┐
│                  TYPES OF HALLUCINATIONS                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Factual Hallucination                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  User: "Who won the 2024 US Presidential Election?"  │  │
│  │  Model: "John Smith won the election." (Made up!)     │  │
│  │                                                        │  │
│  │  Why: Training cutoff date / No access to recent data │  │
│  │  Pattern: Confabulation - fills gaps with plausible   │  │
│  │          but false information                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  2. Source Hallucination                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  User: "Cite sources for climate change evidence"     │  │
│  │  Model: "According to a study by Dr. Jane Doe..."   │  │
│  │          (Study doesn't exist!)                       │  │
│  │                                                        │  │
│  │  Why: Pattern matching - generates citation format   │  │
│  │       without verifying existence                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  3. Logical Hallucination                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  User: "If A > B and B > C, is A > C?"               │  │
│  │  Model: "Not necessarily..." (Wrong! Transitivity)    │  │
│  │                                                        │  │
│  │  Why: Failure to apply logical rules consistently    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  4. Inconsistency Hallucination                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Turn 1: "What's 15 + 27?"                            │  │
│  │  Model: "42"                                          │  │
│  │                                                        │  │
│  │  Turn 2: "What did you say 15 + 27 was?"              │  │
│  │  Model: "I said it was 43." (Contradicts itself!)    │  │
│  │                                                        │  │
│  │  Why: Stateless nature - each response independent     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Why Hallucinations Occur**

```
┌──────────────────────────────────────────────────────────────┐
│              MECHANISM OF HALLUCINATIONS                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Training Objective: Next Token Prediction                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Model is trained to predict ANY likely next token  │  │
│  │  → Not trained to say "I don't know"                │  │
│  │  → Generates most probable continuation, not truth  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  2. Pattern Matching Over Understanding                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Model: "The capital of France is ______"            │  │
│  │  Pattern: "The capital of X is Y"                     │  │
│  │  For rare facts: Matches to nearest pattern         │  │
│  │  → May substitute similar but incorrect fact        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  3. Training Data Noise                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Internet contains:                                   │  │
│  │  • Misinformation                                     │  │
│  │  • Outdated facts                                     │  │
│  │  • Contradictory statements                           │  │
│  │  • Fiction presented as fact                          │  │
│  │  Model learns all of these patterns!                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  4. Context Window Limitations                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Long documents exceed context window                 │  │
│  │  → Can't verify against source text                   │  │
│  │  → Must rely on compressed knowledge                  │  │
│  │  → Details get fabricated                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Mitigation Strategies**

```python
# Strategy 1: Retrieval-Augmented Generation (RAG)
def rag_query(question, documents):
    # Retrieve relevant documents first
    relevant_docs = retrieve(question, documents)
    
    # Include in prompt
    prompt = f"""
    Answer based ONLY on these documents:
    {relevant_docs}
    
    Question: {question}
    If the answer isn't in the documents, say "I don't know".
    """
    return llm.generate(prompt)

# Strategy 2: Self-Verification
verification_prompt = """
Answer the question, then verify your answer:
1. State your answer
2. List evidence supporting it
3. Consider what would make it wrong
4. Rate confidence (1-10)
"""

# Strategy 3: Constrained Generation
# Force model to cite sources or say "unknown"
constrained_prompt = """
Rules:
- Only state facts from provided context
- Format: "According to [source]: [fact]"
- If unsure: "I don't have enough information"
"""
```

---

#### 2. Context Window Limitations

**Concept Breakdown**

The context window is the maximum number of tokens the model can process at once. This creates fundamental constraints.

**Context Window Sizes**

```
Model               Context Window    Approximate Pages
────────────────────────────────────────────────────
GPT-3               2,048 tokens    ~3 pages
GPT-3.5             4,096 tokens    ~6 pages
GPT-4               8,192 tokens    ~12 pages
GPT-4 Turbo         128K tokens     ~200 pages
Claude 2            100K tokens     ~150 pages
Claude 3            200K tokens     ~300 pages
```

**What Happens at the Limit**

```
┌──────────────────────────────────────────────────────────────┐
│              CONTEXT WINDOW OVERFLOW                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input (exceeds limit):                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ [System Prompt....................................]   │  │
│  │ [Document Part 1 (50K tokens)......................]   │  │
│  │ [Document Part 2 (50K tokens)......................]   │  │
│  │ [Document Part 3 (50K tokens)......................]   │  │
│  │ [User Question.....................................]   │  │
│  │                                                       │  │
│  │ Limit: 8K tokens ← TRUNCATION HAPPENS HERE          │  │
│  │                                                       │  │
│  │ [Content beyond limit is IGNORED...................]   │  │
│  │ [Model can't see this!.............................]   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Consequences:                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. Lost Information: Middle of long documents       │  │
│  │ 2. Poor Recall: Can't answer questions about         │  │
│  │    parts that were truncated                          │  │
│  │ 3. Incoherence: Missing context causes errors        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**The "Lost in the Middle" Problem**

```python
# Research finding: Models struggle with information in MIDDLE of context
# Performance by position:
position_performance = {
    "beginning": 90,    # High recall
    "middle": 40,       # Poor recall!
    "end": 85           # Good recall
}

# Reason: Attention weights dilute over long sequences
# → Early and late tokens get more attention
# → Middle tokens are "overlooked"
```

**Mitigation: Chunking and Retrieval**

```python
class ContextManager:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
    
    def process_long_document(self, document, question):
        # Step 1: Split into chunks
        chunks = self.chunk_document(document, chunk_size=1000)
        
        # Step 2: Embed each chunk
        chunk_embeddings = [embed(chunk) for chunk in chunks]
        
        # Step 3: Embed question
        question_embedding = embed(question)
        
        # Step 4: Find relevant chunks
        similarities = cosine_similarity(question_embedding, chunk_embeddings)
        top_chunks = chunks[top_k_indices(similarities, k=3)]
        
        # Step 5: Include only relevant chunks in context
        context = "\n\n".join(top_chunks)
        return context

# Result: Relevant info fits in context, irrelevant excluded
```

---

#### 3. Reasoning Limitations

**Concept Breakdown**

While LLMs show impressive reasoning, they have systematic limitations in:
- Multi-step logical deduction
- Mathematical precision
- Counterfactual reasoning
- Planning with constraints

**Systematic Failures**

```
┌──────────────────────────────────────────────────────────────┐
│              REASONING FAILURE MODES                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Arithmetic Errors                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  13 × 17 = ?                                          │  │
│  │  Model: "221" (Correct)                               │  │
│  │                                                       │  │
│  │  137 × 249 = ?                                        │  │
│  │  Model: "34,113" (Wrong! Actual: 34,113)            │  │
│  │  Wait... let me check... 137 × 249 = 34,113 ✓       │  │
│  │                                                       │  │
│  │  Larger: 12345 × 67890 = ?                          │  │
│  │  Model: "838,102,050" (May be wrong)                │  │
│  │  Actual: 838,102,050 (actually correct!)            │  │
│  │  But unreliable for very large numbers               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  2. Logical Consistency                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Premise: "All birds can fly."                        │  │
│  │  Premise: "Penguins are birds."                       │  │
│  │  Question: "Can penguins fly?"                        │  │
│  │                                                       │  │
│  │  Model might say "Yes" (following premise)            │  │
│  │  or "No" (using real-world knowledge)                 │  │
│  │  → Inconsistent depending on phrasing                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  3. Planning with Constraints                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  "Plan a 3-day trip to Paris visiting:               │  │
│  │   - Eiffel Tower (closes at 11pm)                   │  │
│  │   - Louvre (closed Tuesdays, open 9am-6pm)           │  │
│  │   - Versailles (30 min from Paris)                    │  │
│  │   All on a Tuesday-Wednesday-Thursday trip"           │  │
│  │                                                       │  │
│  │  Model might suggest Louvre on Tuesday (closed!)    │  │
│  │  → Fails to track multiple constraints                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  4. Counterfactual Reasoning                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  "If the Wright brothers had never been born,       │  │
│  │   would airplanes still have been invented?"        │  │
│  │                                                       │  │
│  │  Model struggles with:                                │  │
│  │  - Multiple causal chains                             │  │
│  │  - Historical contingency                             │  │
│  │  - Speculative scenarios                            │  │
│  │  → Often gives superficial answers                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Improving Reasoning**

```python
# Technique 1: Break into steps
step_prompt = """
Solve this step by step:
Step 1: Identify what we need to find
Step 2: List known values
Step 3: Set up equations
Step 4: Solve
Step 5: Verify answer
"""

# Technique 2: Use external tools
tool_prompt = """
For calculations, use the calculator tool.
For facts, use the search tool.
Only answer after verifying with tools.
"""

# Technique 3: Multiple verification paths
verify_prompt = """
Solve this problem, then:
1. Solve it a different way
2. Check if answers match
3. If different, identify the error
"""
```

---

#### 4. Bias and Fairness Issues

**Concept Breakdown**

LLMs can exhibit biases present in their training data, including:
- Gender stereotypes
- Racial biases
- Cultural biases
- Temporal biases (training cutoff)

**Types of Bias**

```
┌──────────────────────────────────────────────────────────────┐
│                    BIAS MANIFESTATIONS                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Gender Bias                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Prompt: "The doctor said to the nurse that ___       │  │
│  │          should listen carefully."                    │  │
│  │                                                       │  │
│  │  Model often completes with "she" (referring to       │  │
│  │  nurse), assuming doctor = male                       │  │
│  │                                                       │  │
│  │  Source: Training data reflects historical gender     │  │
│  │  roles in healthcare                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  2. Cultural Bias                                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  "What is a traditional family dinner?"               │  │
│  │  Model may assume Western cuisine/structure          │  │
│  │  → Over-representation of Western training data      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  3. Temporal Bias                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  "Who is the current US President?"                   │  │
│  │  (Asked in 2024, model trained in 2023)             │  │
│  │                                                       │  │
│  │  Model: "Joe Biden" (correct if training post-2020) │  │
│  │  But may not know about 2024 election results       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  4. Confirmation Bias                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  User: "I think X is true."                           │  │
│  │  Model often agrees and finds supporting arguments   │  │
│  │  → Tendency to align with user position              │  │
│  │  → Can reinforce misinformation                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Mitigation Approaches**

```python
# Approach 1: System Prompts
bias_mitigation_prompt = """
You are a helpful AI assistant. Important guidelines:
- Treat all people with equal respect regardless of gender, race, or background
- Avoid stereotypes in your responses
- Present diverse perspectives when appropriate
- Acknowledge uncertainty rather than assuming
"""

# Approach 2: RLHF (Reinforcement Learning from Human Feedback)
# Models are fine-tuned to prefer less biased responses
# Based on human preference rankings

# Approach 3: Careful Prompt Engineering
neutral_prompt = """
Complete the sentence without assumptions:
"The engineer discussed the project with ___ manager."
(Answer could be: "their", "the", "his", "her", etc.)
"""
```

---

### Summary Table: Capabilities vs Limitations

| Aspect | Capability | Limitation |
|--------|------------|------------|
| **Learning** | Few-shot in-context learning | No true understanding, pattern matching |
| **Reasoning** | Chain-of-thought helps | Multi-step logic, math errors |
| **Knowledge** | Broad factual knowledge | Hallucinations, cutoff dates |
| **Context** | Can process thousands of tokens | "Lost in middle", truncation |
| **Bias** | Can be prompted for neutrality | Inherent training data biases |
| **Code** | Generate, debug, explain | Complex architecture design |
| **Creativity** | Novel combinations, writing | Not truly original, trained patterns |
| **Tools** | Use APIs, calculators, search | Dependent on tool quality |

---

### Best Practices for Working with LLMs

```
┌──────────────────────────────────────────────────────────────┐
│                 BEST PRACTICES GUIDE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  DO:                                                         │
│  ✓ Use RAG for factual queries                             │
│  ✓ Request chain-of-thought for reasoning                  │
│  ✓ Verify critical information independently               │
│  ✓ Use temperature 0 for deterministic tasks               │
│  ✓ Break complex tasks into smaller steps                  │
│  ✓ Include examples in prompts (few-shot)                  │
│  ✓ Set clear constraints and formats                       │
│                                                              │
│  DON'T:                                                      │
│  ✗ Trust outputs without verification (hallucinations)     │
│  ✗ Expect perfect math on large numbers                    │
│  ✗ Assume model remembers earlier conversation             │
│  ✗ Use for high-stakes decisions without human review     │
│  ✗ Expect consistent answers to rephrased questions        │
│  ✗ Rely on long-document comprehension without chunking    │
│                                                              │
│  CONSIDER:                                                   │
│  ? Using multiple samples and voting for reliability       │
│  ? Providing explicit reasoning steps in prompts           │
│  ? Using smaller models for simple tasks (cost/speed)      │
│  ? Fine-tuning for domain-specific applications              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

*This guide provides comprehensive coverage of LLM fundamentals with detailed breakdowns, step-by-step calculations, and visualizations to aid understanding.*

**Complete Guide Sections:**
1. ✅ Core LLM Concepts
2. ✅ Transformer Architecture  
3. ✅ Attention Mechanism Deep Dive
4. ✅ Tokenization & Embeddings
5. ✅ Generation & Decoding Strategies
6. ✅ **LLM Capabilities & Limitations** (New!)
   - Capabilities: In-context learning, Chain-of-thought, Tool use, Code generation
   - Limitations: Hallucinations, Context window, Reasoning, Bias
   - Best practices and mitigation strategies

The file is saved at: [`llm_fundamentals_detailed_guide.md`](llm_fundamentals_detailed_guide.md)
}

---

## Prompt Engineering (25+ Questions)

### Overview

Prompt engineering is the practice of designing inputs (prompts) to get desired outputs from LLMs. It involves understanding how models interpret text and crafting prompts that guide them toward accurate, useful responses.

---

### Question 1: What is Zero-Shot vs Few-Shot vs Chain-of-Thought Prompting?

#### Concept Breakdown

**The Prompting Spectrum:**

| Technique | Examples Given | When to Use | Complexity |
|-----------|---------------|-------------|------------|
| **Zero-Shot** | 0 | Simple, well-known tasks | Low |
| **Few-Shot** | 2-5 | Pattern recognition needed | Medium |
| **Chain-of-Thought** | 0-3 + reasoning | Multi-step reasoning | High |

#### Step-by-Step Comparison

**Problem**: Sentiment Classification

**Step 1: Zero-Shot Prompting**
```python
prompt = """
Classify the sentiment of this review:
"This restaurant was amazing! The food was delicious and 
service was quick."

Sentiment:
"""

# Model Output: "Positive"
```

**Why it works:**
- Model has seen many sentiment classification patterns in training
- Task is straightforward and common
- No special formatting needed

**Result Simulation:**
```
Input: "This restaurant was amazing!"
↓
Model: Recognizes sentiment classification task
↓
Pattern Match: "amazing", "delicious", "quick" = positive words
↓
Output: "Positive"
```

---

**Step 2: Few-Shot Prompting**
```python
prompt = """
Classify the sentiment of reviews as Positive, Negative, or Mixed.

Review: "Best movie I've seen this year!"
Sentiment: Positive

Review: "Waste of time, terrible acting."
Sentiment: Negative

Review: "Good plot but slow pacing ruined it."
Sentiment: Mixed

Review: "The food was cold but the ambiance was nice."
Sentiment:
"""

# Model Output: "Mixed"
```

**Why it works:**
- Examples establish the expected format
- Model learns classification criteria from examples
- Pattern: "X but Y" structure often indicates mixed sentiment

**Result Simulation:**
```
Input Pattern Analysis:
"Best movie..." → Positive (strong positive words)
"Waste of time..." → Negative (strong negative words)
"Good plot but..." → Mixed (positive + negative)

Current Input:
"food was cold [negative] but ambiance was nice [positive]"
↓
Pattern Match: Matches "Mixed" example structure
↓
Output: "Mixed"
```

---

**Step 3: Chain-of-Thought Prompting**
```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 
   5 + 6 = 11. The answer is 11.

Q: A bakery has 24 cupcakes. They sell 15 in the morning and 
   bake 10 more in the afternoon. How many cupcakes do they have now?
A:
"""

# Model Output:
# "The bakery started with 24 cupcakes. They sold 15, so they had 
#  24 - 15 = 9 cupcakes. Then they baked 10 more, so they have 
#  9 + 10 = 19 cupcakes. The answer is 19."
```

**Why it works:**
- Model generates intermediate reasoning steps
- Each step is simpler than the whole problem
- Errors in early steps can be caught
- Final answer based on derived value, not guess

**Result Simulation:**
```
Problem: 24 cupcakes → sell 15 → bake 10

Without CoT:
Model might output: 29 (wrong: 24 + 10, forgot subtraction)
                 or 19 (correct, by luck)

With CoT:
Step 1: Identify starting amount = 24
Step 2: Calculate after sales: 24 - 15 = 9
Step 3: Calculate after baking: 9 + 10 = 19
Step 4: State final answer: 19

Each step verified before proceeding → Higher accuracy
```

---

### Summary Table: Prompt Engineering Techniques

| Technique | Use Case | Key Principle |
|-----------|----------|---------------|
| **Zero-Shot** | Simple tasks | Clear instructions |
| **Few-Shot** | Pattern learning | Examples establish format |
| **Chain-of-Thought** | Reasoning tasks | Break into steps |
| **Role Prompting** | Tone/style control | Persona influences output |
| **System Prompts** | Global behavior | Comprehensive constraints |
| **Prompt Chaining** | Complex workflows | Modular processing |

---

*This guide provides comprehensive coverage of LLM fundamentals with detailed breakdowns, step-by-step calculations, and visualizations to aid understanding.*

**Complete Guide Sections:**
1. ✅ Core LLM Concepts
2. ✅ Transformer Architecture
3. ✅ Attention Mechanism Deep Dive
4. ✅ Tokenization & Embeddings
5. ✅ Generation & Decoding Strategies
6. ✅ LLM Capabilities & Limitations
7. ✅ **Prompt Engineering** (New!)
   - Zero/Few/CoT prompting with comparisons
   - Role prompting with examples
   - System prompt architecture
   - Prompt chaining workflows

The file is saved at: [`llm_fundamentals_detailed_guide.md`](llm_fundamentals_detailed_guide.md)
</invoke>
