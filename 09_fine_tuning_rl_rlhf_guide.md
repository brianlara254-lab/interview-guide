# Fine-Tuning, Reinforcement Learning & RLHF - Comprehensive Guide

A detailed breakdown of LLM fine-tuning methods, reinforcement learning from human feedback (RLHF), and advanced training techniques with step-by-step explanations.

**Last Updated**: 2025 | **Verified with**: OpenAI, Anthropic, DeepSpeed, TRL documentation

---

## Table of Contents
1. [Fine-Tuning Fundamentals](#fine-tuning-fundamentals)
2. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
3. [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
4. [Reinforcement Learning Basics](#reinforcement-learning-basics)
5. [RLHF - Reinforcement Learning from Human Feedback](#rlhf---reinforcement-learning-from-human-feedback)
6. [Advanced Training Techniques](#advanced-training-techniques)
7. [Evaluation & Best Practices](#evaluation--best-practices)

---

## Fine-Tuning Fundamentals

### Question 1: What is Fine-Tuning and When Should You Use It?

#### Concept Breakdown

**Definition**: Fine-tuning is the process of taking a pre-trained model and continuing training on a specific dataset to adapt it for a particular task or domain.

**When to Use Fine-Tuning vs Other Methods:**

```
┌─────────────────────────────────────────────────────────────────┐
│              ADAPTATION METHODS COMPARISON                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prompt Engineering       Few-Shot Learning      Fine-Tuning    │
│  ─────────────────       ────────────────       ───────────     │
│  • No training           • No training          • Training      │
│  • Quick setup           • Example-based        • Data needed   │
│  • Limited context       • Context limited      • Full context  │
│  • Good for simple tasks • Medium complexity    • Best accuracy │
│                                                                 │
│  Use When:                 Use When:            Use When:       │
│  • Task is simple          • Examples available • High accuracy │
│  • Need quick results      • Task is specific   •   needed      │
│  • Budget limited          • Medium data        • Sufficient    │
│                                                   data          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Fine-Tuning vs RAG:**

| Aspect | Fine-Tuning | RAG |
|--------|-------------|-----|
| **Knowledge Update** | Permanent in weights | Dynamic retrieval |
| **Data Privacy** | Model learns private data | Data stays in DB |
| **Cost** | High (training) | Lower (inference) |
| **Flexibility** | Fixed after training | Update docs anytime |
| **Hallucinations** | Can still hallucinate | Grounded in sources |
| **Best For** | Style, format, domain tasks | Factual Q&A |

#### Step-by-Step: When to Choose Fine-Tuning

**Scenario Analysis:**

```python
decision_tree = {
    "Need to adapt model behavior?": {
        "Yes": {
            "Have 1000+ examples?": {
                "Yes": "Fine-tuning recommended",
                "No": "Use few-shot prompting"
            }
        },
        "No": {
            "Need current/private knowledge?": {
                "Yes": "Use RAG",
                "No": "Use base model with prompts"
            }
        }
    }
}
```

**Example Decision Process:**

```
Problem: Customer wants model to respond in Shakespearean English

Analysis:
├─ Adapt behavior? → YES (style change)
├─ Have examples? → YES (can collect Shakespeare texts)
└─ Decision → FINE-TUNING

Implementation:
1. Collect Shakespeare plays (~1M tokens)
2. Format as instruction-following pairs
3. Fine-tune base model
4. Model now responds in Shakespearean style permanently

Result: Model's weights now encode Shakespearean style
```

**vs RAG Example:**

```
Problem: Customer wants model to answer questions about their product manual

Analysis:
├─ Adapt behavior? → NO (don't want to change style)
├─ Need private knowledge? → YES (proprietary manual)
├─ Data changes frequently? → YES (product updates)
└─ Decision → RAG

Implementation:
1. Chunk product manual
2. Store in vector database
3. Retrieve relevant chunks at query time
4. Generate grounded responses

Result: Knowledge is dynamic, can update manual anytime
```

---

### Question 2: What are the Different Types of Fine-Tuning?

#### Concept Breakdown

**Fine-Tuning Taxonomy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING TYPES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FULL FINE-TUNING                                            │
│     └── Update all model parameters                             │
│         • Most flexible                                         │
│         • Highest memory cost                                   │
│         • Risk of catastrophic forgetting                       │
│                                                                 │
│  2. PARAMETER-EFFICIENT FINE-TUNING (PEFT)                      │
│     └── Update only small subset of parameters                  │
│         ├─ LoRA (Low-Rank Adaptation)                           │
│         ├─ QLoRA (Quantized LoRA)                               │
│         ├─ Prefix Tuning                                        │
│         └─ Prompt Tuning                                        │
│                                                                 │
│  3. LAYER-SPECIFIC FINE-TUNING                                  │
│     └── Freeze early layers, tune later layers                  │
│         • Preserve general knowledge                            │
│         • Adapt task-specific layers                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Fine-Tuning Type Selection

**Type 1: Full Fine-Tuning**
```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# By default, ALL parameters are trainable
# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable_params:,} / {total_params:,}")
# Output: Trainable: 6,771,970,048 / 6,771,970,048 (100%)

# Training configuration
training_args = TrainingArguments(
    output_dir="./full-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    # ... other args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**Result Simulation:**
```
Full Fine-Tuning Analysis:

Model: Llama-2-7B
Parameters: 6.7B
Memory Required: ~28GB (FP16) or ~56GB (FP32)
Training Time: ~10 hours (on 8x A100)
Cost: ~$200-300 (cloud GPUs)

Advantages:
✓ Maximum flexibility
✓ Best performance on target task
✓ Can fundamentally change model behavior

Disadvantages:
✗ High memory requirements
✗ Risk of catastrophic forgetting
✗ Expensive to train
✗ Large model checkpoints
```

**Type 2: LoRA (Low-Rank Adaptation)**
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=16,                    # Rank (typically 8-64)
    lora_alpha=32,           # Scaling factor (typically 2*r)
    target_modules=[         # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
lora_model = get_peft_model(model, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_model.parameters())

print(f"Trainable: {trainable_params:,} / {total_params:,}")
# Output: Trainable: 33,554,432 / 6,771,970,048 (~0.5%)
```

**Result Simulation:**
```
LoRA Fine-Tuning Analysis:

Model: Llama-2-7B with LoRA (r=16)
Trainable Parameters: 33.5M (0.5% of total)
Memory Required: ~14GB (can fit on single GPU)
Training Time: ~2 hours (on single A100)
Adapter Size: ~67MB (vs 13GB full model)
Cost: ~$20-30

How LoRA Works:
┌─────────────────────────────────────────────┐
│  Original Weight Matrix W (d × d)          │
│                                             │
│  Instead of updating W directly:            │
│  W' = W + ΔW                                │
│                                             │
│  LoRA approximates ΔW as:                   │
│  ΔW ≈ B × A  where:                         │
│    • B is d × r matrix                      │
│    • A is r × d matrix                      │
│    • r << d (rank, typically 16)            │
│                                             │
│  Parameters: r × d + r × d = 2 × r × d      │
│  vs d × d for full update                   │
│                                             │
│  Example: d=4096, r=16                      │
│  Full: 4096² = 16,777,216 params            │
│  LoRA: 2 × 16 × 4096 = 131,072 params       │
│  Reduction: 128x fewer parameters!          │
└─────────────────────────────────────────────┘

Advantages:
✓ 99.5% fewer parameters
✓ Can run on consumer GPUs
✓ Faster training
✓ Small adapter files (easy to share)
✓ Can combine multiple adapters

Disadvantages:
✗ Slightly lower performance than full tuning
✗ Need to choose rank hyperparameter
```

**Type 3: QLoRA (Quantized LoRA)**
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit precision
    bnb_4bit_quant_type="nf4",            # Normal float 4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,       # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto",                     # Auto layer placement
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # Fewer layers for speed
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
```

**Result Simulation:**
```
QLoRA Fine-Tuning Analysis:

Model: Llama-2-7B with QLoRA
Quantization: 4-bit
Memory Required: ~6GB (fits on consumer GPU!)
Training Time: ~4 hours (on RTX 4090)
Cost: ~$5 (consumer GPU)

How QLoRA Works:
┌─────────────────────────────────────────────┐
│  Step 1: Quantize Base Model to 4-bit       │
│    • 16-bit weights → 4-bit weights         │
│    • Memory reduction: 4x                   │
│    • Slight quality loss from quantization  │
│                                             │
│  Step 2: Add LoRA Adapters (16-bit)         │
│    • Train adapters in full precision       │
│    • Dequantize on-the-fly during forward   │
│                                             │
│  Step 3: Optimizer States in 8-bit          │
│    • paged optimizers for memory efficiency │
│                                             │
│  Memory Breakdown (7B model):               │
│    Base model (4-bit): ~4GB                 │
│    LoRA adapters (16-bit): ~0.5GB           │
│    Gradients: ~0.5GB                        │
│    Optimizer states: ~1GB                   │
│    Total: ~6GB                              │
└─────────────────────────────────────────────┘

Advantages:
✓ Fits on consumer GPUs (RTX 3090/4090)
✓ Very low cost
✓ Minimal quality loss
✓ Accessible to everyone

Disadvantages:
✗ Slower training (dequantization overhead)
✗ More hyperparameters to tune
```

---

## Supervised Fine-Tuning (SFT)

### Question 3: How to Prepare Data for SFT?

#### Concept Breakdown

**SFT Data Formats:**

```
┌─────────────────────────────────────────────────────────────────┐
│              INSTRUCTION-FOLLOWING DATA FORMATS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Format 1: Alpaca (Instruction-Input-Output)                    │
│  {                                                              │
│    "instruction": "Translate to French",                        │
│    "input": "Hello, how are you?",                              │
│    "output": "Bonjour, comment allez-vous?"                     │
│  }                                                              │
│                                                                 │
│  Format 2: ChatML (Conversation)                                │
│  [                                                              │
│    {"role": "system", "content": "You are a translator"},       │
│    {"role": "user", "content": "Hello"},                        │
│    {"role": "assistant", "content": "Bonjour"}                  │
│  ]                                                              │
│                                                                 │
│  Format 3: ShareGPT (Conversations)                             │
│  {                                                              │
│    "conversations": [                                           │
│      {"from": "human", "value": "Hello"},                       │
│      {"from": "gpt", "value": "Hi there!"}                      │
│    ]                                                            │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Data Preparation Pipeline

**Step 1: Raw Data Collection**
```python
# Example: Building a customer service dataset
raw_data = [
    {
        "customer_query": "My order hasn't arrived",
        "support_response": "I apologize for the delay. Let me check..."
    },
    {
        "customer_query": "How do I reset my password?",
        "support_response": "You can reset your password by..."
    },
    # ... more examples
]
```

**Step 2: Format Conversion**
```python
def convert_to_alpaca_format(raw_data):
    """Convert raw data to Alpaca format"""
    alpaca_data = []
    
    for item in raw_data:
        alpaca_item = {
            "instruction": "You are a helpful customer support agent. "
                          "Respond to the customer's question.",
            "input": item["customer_query"],
            "output": item["support_response"]
        }
        alpaca_data.append(alpaca_item)
    
    return alpaca_data

# Convert
formatted_data = convert_to_alpaca_format(raw_data)
```

**Step 3: Text Formatting for Training**
```python
def format_prompt(example):
    """Format example for causal LM training"""
    if example["input"]:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return prompt

# Create text field for training
dataset = dataset.map(lambda x: {"text": format_prompt(x)})
```

**Step 4: Tokenization**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

def tokenize_function(examples):
    """Tokenize text for training"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

**Result Simulation:**
```
Data Preparation Pipeline:

Raw Data: 1,000 customer service conversations
↓
Format Conversion → Alpaca format
↓
Text Formatting → Prompt templates
↓
Tokenization → Input IDs and attention masks
↓
Final Dataset: 1,000 examples, 512 tokens each

Memory Requirements:
- Raw text: ~2MB
- Tokenized: ~4MB
- During training: ~2GB (batch processing)
```

---

### Question 4: What are the Key Hyperparameters for SFT?

#### Concept Breakdown

**Critical Hyperparameters:**

| Hyperparameter | Typical Range | Effect | Tuning Strategy |
|----------------|---------------|--------|-----------------|
| **Learning Rate** | 1e-5 to 5e-5 | Step size for updates | Start small, increase if slow |
| **Batch Size** | 4-64 | Samples per update | Larger = faster, more memory |
| **Epochs** | 1-10 | Full dataset passes | Use early stopping |
| **Warmup Steps** | 100-1000 | Gradual LR increase | 5-10% of total steps |
| **Weight Decay** | 0.01-0.1 | L2 regularization | Prevent overfitting |
| **Gradient Clipping** | 0.5-1.0 | Max gradient norm | Prevent exploding gradients |

#### Step-by-Step: Hyperparameter Selection

**Learning Rate Selection:**
```python
# Learning rate scheduling
from transformers import get_linear_schedule_with_warmup

# Typical configuration
learning_rate = 2e-5  # Conservative starting point
num_epochs = 3
batch_size = 4

# Calculate total steps
total_steps = (len(train_dataset) // batch_size) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

# Learning rate schedule
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Result Simulation:**
```
Learning Rate Experiments:

Experiment 1: LR = 1e-4 (too high)
Epoch 1: Loss: 2.5 → 0.8 → NaN (diverged)
Result: ❌ Training unstable

Experiment 2: LR = 2e-5 (good)
Epoch 1: Loss: 2.5 → 1.2
Epoch 2: Loss: 1.2 → 0.8
Epoch 3: Loss: 0.8 → 0.6
Result: ✅ Stable convergence

Experiment 3: LR = 1e-6 (too low)
Epoch 1: Loss: 2.5 → 2.3
Epoch 2: Loss: 2.3 → 2.1
Epoch 3: Loss: 2.1 → 2.0
Result: ⚠️ Too slow, underfitting

Best Practice:
┌─────────────────────────────────────────┐
│  Start: 1e-5 to 2e-5                   │
│  If loss not decreasing: increase LR   │
│  If loss spikes/NaN: decrease LR       │
│  If plateau: increase LR or use decay  │
└─────────────────────────────────────────┘
```

---

## Reinforcement Learning Basics

### Question 5: What is Reinforcement Learning (RL) in the Context of LLMs?

#### Concept Breakdown

**RL Fundamentals:**

```
┌─────────────────────────────────────────────────────────────────┐
│           REINFORCEMENT LEARNING FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Environment (Task)          Agent (LLM)                       │
│   ┌─────────────┐             ┌─────────────┐                   │
│   │             │◄────────────│             │                   │
│   │  State      │   Action    │  Policy     │                   │
│   │  (Prompt)   │             │  π(a|s)     │                   │
│   │             ├────────────►│             │                   │
│   └──────┬──────┘             └─────────────┘                   │
│          │                                                      │
│          │ Reward                                               │
│          │ (Feedback)                                           │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │  Reward     │                                               │
│   │  Function   │                                               │
│   └─────────────┘                                               │
│                                                                 │
│   Objective: Maximize cumulative reward                         │
│   π* = argmax E[R|π]                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components for LLMs:**

| Component | In LLM Terms | Example |
|-----------|--------------|---------|
| **State** | Input prompt | "Translate: Hello" |
| **Action** | Generated tokens | "Bonjour" |
| **Policy** | Model distribution | P(token \| context) |
| **Reward** | Quality score | BLEU score, human rating |
| **Trajectory** | Full response | Entire generated text |

#### Step-by-Step: RL for Text Generation

**Example: Training LLM to generate helpful responses**

```python
# Step 1: Generate response (Action)
prompt = "Explain quantum computing"
response = model.generate(prompt)  # "Quantum computing uses qubits..."

# Step 2: Evaluate response (Reward)
helpfulness_score = evaluate_helpfulness(response)
# Score: 8.5/10 (based on clarity, accuracy, completeness)

# Step 3: Update policy
# Increase probability of generating similar helpful responses
# Decrease probability of generating unhelpful responses

# Step 4: Repeat
# Continue for many examples to improve policy
```

**Result Simulation:**
```
RL Training Progress:

Iteration 1:
Prompt: "Explain AI"
Response: "AI is artificial intelligence."
Reward: 5/10 (too brief)
Policy Update: Slightly decrease this pattern

Iteration 100:
Prompt: "Explain AI"
Response: "AI is technology that enables machines to 
          perform tasks requiring human intelligence, 
          such as visual perception, speech recognition,
          and decision-making."
Reward: 9/10 (comprehensive)
Policy Update: Increase this pattern

Iteration 1000:
Model consistently generates helpful, detailed responses
Average reward: 8.7/10
```

---

## RLHF - Reinforcement Learning from Human Feedback

### Question 6: What is RLHF and How Does It Work?

#### Concept Breakdown

**RLHF Pipeline (Used in ChatGPT, Claude):**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF THREE-STAGE PROCESS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: SFT (Supervised Fine-Tuning)                          │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  • Train on high-quality human demonstrations         │      │
│  │  • Learn to imitate human responses                   │      │
│  │  • Result: SFT Model                                  │      │
│  └───────────────────────────────────────────────────────┘      │
│                              ↓                                  │
│  STAGE 2: Reward Model Training                                 │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  • Humans rank multiple model outputs                 │      │
│  │  • Train reward model to predict preferences          │      │
│  │  • Result: Reward Model                               │      │
│  └───────────────────────────────────────────────────────┘      │
│                              ↓                                  │
│  STAGE 3: RL Fine-Tuning (PPO)                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  • Use reward model as feedback signal                │      │
│  │  • Optimize policy with PPO algorithm                 │      │
│  │  • Result: RLHF Model (Final)                         │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Complete RLHF Pipeline

**Stage 1: SFT (Supervised Fine-Tuning)**
```python
# Already covered in SFT section
# Result: sft_model
```

**Stage 2: Reward Model Training**
```python
from transformers import AutoModelForSequenceClassification

# Load base model for reward modeling
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "sft_model",
    num_labels=1  # Regression (output single score)
)

# Training data: ranked pairs
# For each prompt, humans rank multiple responses
preference_data = [
    {
        "prompt": "Explain Python",
        "chosen": "Python is a versatile programming language...",
        "rejected": "Python is a snake."  # Worse response
    },
    # ... more examples
]

# Loss function: Pairwise ranking loss
def compute_loss(chosen_reward, rejected_reward):
    """
    We want: chosen_reward > rejected_reward
    Loss = -log(sigmoid(chosen_reward - rejected_reward))
    """
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward))

# Train reward model
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    learning_rate=1e-5,
)

# Result: reward_model that predicts human preferences
```

**Result Simulation:**
```
Reward Model Training:

Input: "Explain Python"
Candidates:
├─ Response A: "Python is a programming language..." → Score: 8.5
├─ Response B: "Python is a snake" → Score: 2.1
└─ Response C: "Python was created by Guido..." → Score: 9.2

Model learns to predict that humans prefer A and C over B
Validation Accuracy: 78% (predicts preferences correctly)
```

**Stage 3: RL with PPO (Proximal Policy Optimization)**
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load SFT model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")

# PPO configuration
ppo_config = PPOConfig(
    model_name="sft_model",
    learning_rate=1.41e-5,
    batch_size=256,
    mini_batch_size=64,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,  # KL divergence constraint
    ppo_epochs=4,
    seed=0,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# Training loop
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        # 1. Generate responses
        queries = batch["query"]
        response_tensors = ppo_trainer.generate(queries)
        
        # 2. Get rewards from reward model
        texts = [q + r for q, r in zip(queries, response_tensors)]
        rewards = reward_model(texts)  # [batch_size]
        
        # 3. Update policy with PPO
        stats = ppo_trainer.step(queries, response_tensors, rewards)
        
        # Log stats
        print(f"Reward: {rewards.mean():.2f}")
        print(f"KL: {stats['objective/kl']: .4f}")
```

**Result Simulation:**
```
PPO Training Progress:

Epoch 1/3:
  Batch 100: Avg Reward: 2.3, KL: 0.05
  Batch 200: Avg Reward: 3.1, KL: 0.08
  Batch 300: Avg Reward: 3.8, KL: 0.10
  
Epoch 2/3:
  Batch 100: Avg Reward: 4.5, KL: 0.09
  Batch 200: Avg Reward: 5.2, KL: 0.11
  
Epoch 3/3:
  Batch 100: Avg Reward: 5.8, KL: 0.10
  
Final Model:
  • Higher reward (5.8 vs 2.3 initially)
  • KL divergence controlled (0.10)
  • More helpful responses
  • Less harmful outputs
```

---

### Question 7: What are the Challenges in RLHF?

#### Concept Breakdown

**Common Challenges:**

```
┌──────────────────────────────────────────────────────────────────┐
│                    RLHF CHALLENGES                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. REWARD HACKING                                               │
│     Model finds shortcuts to maximize reward                     │
│     Example: Always say "I cannot answer" (safe = high reward)   │
│     Solution: Reward model ensemble, diverse training data       │
│                                                                  │
│  2. DISTRIBUTION SHIFT                                           │
│     Training data ≠ Production data                              │
│     Model overfits to training prompts                           │
│     Solution: Diverse prompts, continuous training               │
│                                                                  │
│  3. KL DIVERGENCE TRADEOFF                                       │
│     Too strict: No improvement                                   │
│     Too loose: Model drifts, forgets base capabilities           │
│     Solution: Careful tuning of KL penalty                       │
│                                                                  │
│  4. ANNOTATION COST                                              │
│     Human labeling is expensive and slow                         │
│     Solution: Active learning, synthetic data, Constitutional AI │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Addressing Challenges

**Challenge 1: Reward Hacking**
```python
# Problem: Model learns to say "I cannot answer" for all queries
# (Gets reward for being safe, even for benign questions)

# Solution 1: Reward model ensemble
class EnsembleRewardModel:
    def __init__(self):
        self.models = [
            load_reward_model("model_1"),
            load_reward_model("model_2"),
            load_reward_model("model_3"),
        ]
    
    def __call__(self, text):
        scores = [model(text) for model in self.models]
        # Use median to reduce outlier impact
        return np.median(scores)

# Solution 2: Diverse prompts in training
# Ensure training data includes:
# - Safe questions (should answer)
# - Unsafe questions (should refuse)
# - Ambiguous questions (nuanced response)
```

**Challenge 2: KL Divergence Tuning**
```python
# KL divergence measures how far RL policy is from SFT policy
# Too high = model forgot base knowledge
# Too low = no improvement from RL

# Experiment with different KL targets
kl_targets = [0.05, 0.1, 0.2]
results = {}

for kl_target in kl_targets:
    model = train_ppo(kl_coef=kl_target)
    
    # Evaluate
    helpfulness = evaluate_helpfulness(model)
    safety = evaluate_safety(model)
    base_task_perf = evaluate_base_tasks(model)  # Check forgetting
    
    results[kl_target] = {
        "helpfulness": helpfulness,
        "safety": safety,
        "base_retention": base_task_perf
    }

# Choose KL target that balances all three
# Typical good value: 0.1
```

---

## Advanced Training Techniques

### Question 8: What is DPO (Direct Preference Optimization)?

#### Concept Breakdown

**DPO vs RLHF:**

```
┌─────────────────────────────────────────────────────────────────┐
│              DPO vs RLHF COMPARISON                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RLHF:                                                          │
│  SFT → Reward Model → PPO (3 stages)                            │
│  • Complex pipeline                                             │
│  • Requires reward model training                               │
│  • PPO can be unstable                                          │
│  • More hyperparameters                                         │
│                                                                 │
│  DPO:                                                           │
│  SFT → Direct Preference Optimization (2 stages)                │
│  • Simpler pipeline                                             │
│  • No reward model needed!                                      │
│  • More stable training                                         │
│  • Often better results                                         │
│                                                                 │
│  Key Insight:                                                   │
│  DPO derives optimal policy directly from preferences           │
│  without explicit reward modeling                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: DPO Implementation

```python
from trl import DPOTrainer, DPOConfig

# DPO only needs preference data (no reward model!)
preference_data = [
    {
        "prompt": "Explain AI",
        "chosen": "AI is technology enabling machines to perform...",
        "rejected": "AI is a company name."
    },
    # ... more examples
]

# DPO configuration
dpo_config = DPOConfig(
    beta=0.1,  # Temperature parameter for DPO
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,  # SFT model
    ref_model=ref_model,  # Reference model (frozen SFT)
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

# Train
dpo_trainer.train()

# DPO Loss Function:
# L_DPO = -E[log(sigmoid(beta * (log π(y_w|x) - log π(y_l|x))))]
# where:
#   y_w = preferred (chosen) response
#   y_l = dispreferred (rejected) response
#   beta = temperature parameter
```

**Result Simulation:**
```
DPO vs RLHF Results:

Model: Llama-2-7B
Dataset: 10k preference pairs

Metric              | RLHF    | DPO     | Improvement
────────────────────┼─────────┼─────────┼────────────
Helpfulness (GPT-4) | 7.2/10  | 7.8/10  | +8%
Safety              | 8.5/10  | 8.7/10  | +2%
Training Time       | 8 hrs   | 2 hrs   | 4x faster
Complexity          | High    | Low     | Simpler
Stability           | Medium  | High    | More stable

DPO Advantages:
✓ No reward model training
✓ Single-stage after SFT
✓ More stable
✓ Often better performance
✓ Faster training
```

---

## Evaluation & Best Practices

### Question 9: How to Evaluate Fine-Tuned Models?

#### Concept Breakdown

**Evaluation Dimensions:**

| Dimension | Metrics | Tools |
|-----------|---------|-------|
| **Task Performance** | Accuracy, F1, BLEU, ROUGE | Custom eval, HuggingFace Evaluate |
| **Safety** | Toxicity, bias, harmfulness | Perspective API, custom classifiers |
| **Capability Retention** | Perplexity on general corpus | lm-eval-harness |
| **Human Preference** | Win rate vs baseline | Human eval, GPT-4 as judge |

#### Step-by-Step: Comprehensive Evaluation

```python
from evaluate import load
import lm_eval

# 1. Automatic Metrics
bleu = load("bleu")
rouge = load("rouge")

def evaluate_generation(predictions, references):
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    return {
        "bleu": bleu_score["bleu"],
        "rougeL": rouge_score["rougeL"]
    }

# 2. LM Evaluation Harness (academic benchmarks)
results = lm_eval.simple_evaluate(
    model="finetuned_model",
    tasks=["hellaswag", "arc_challenge", "truthfulqa_mc"],
    batch_size=8,
)

# 3. GPT-4 as Judge
from openai import OpenAI

client = OpenAI()

def gpt4_judge(prompt, response_a, response_b):
    """Ask GPT-4 which response is better"""
    judge_prompt = f"""
    Prompt: {prompt}
    
    Response A: {response_a}
    Response B: {response_b}
    
    Which response is better? Answer only with 'A' or 'B'.
    """
    
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    return completion.choices[0].message.content

# Compute win rate
wins = 0
for example in test_set:
    winner = gpt4_judge(
        example["prompt"],
        finetuned_response,
        baseline_response
    )
    if winner == "A":
        wins += 1

win_rate = wins / len(test_set)
print(f"Win rate vs baseline: {win_rate:.2%}")
```

---

### Summary Table: Fine-Tuning Methods

| Method | Parameters | Memory | Speed | Quality | Use Case |
|--------|-----------|--------|-------|---------|----------|
| **Full FT** | 100% | Very High | Slow | Best | Maximum adaptation |
| **LoRA** | 0.5-2% | Medium | Fast | Very Good | Most use cases |
| **QLoRA** | 0.5-2% | Low | Medium | Good | Consumer GPUs |
| **Prefix Tuning** | 0.1% | Low | Fast | Good | Task adaptation |
| **Prompt Tuning** | 0.01% | Very Low | Fastest | Fair | Simple tasks |

### RL Methods Comparison

| Method | Stages | Reward Model | Stability | Performance |
|--------|--------|--------------|-----------|-------------|
| **RLHF (PPO)** | 3 | Yes | Medium | Good |
| **DPO** | 2 | No | High | Very Good |
| **KTO** | 2 | No | High | Good |
| **IPO** | 2 | No | High | Good |

### Best Practices Checklist

```
Data Preparation:
✓ Collect high-quality, diverse examples
✓ Balance different types of queries
✓ Include edge cases and safety examples
✓ Split into train/val/test (80/10/10)

Model Selection:
✓ Start with strong base model (Llama-2, Mistral)
✓ Use appropriate model size for task
✓ Consider quantization for resource constraints

Training:
✓ Start with conservative learning rate (1e-5 to 2e-5)
✓ Use learning rate warmup
✓ Monitor validation loss for overfitting
✓ Save checkpoints regularly
✓ Use gradient accumulation for larger effective batch size

Evaluation:
✓ Automatic metrics (BLEU, ROUGE)
✓ Benchmark tasks (lm-eval-harness)
✓ Human evaluation (win rates)
✓ Safety evaluations
✓ Capability retention tests

Deployment:
✓ A/B test against baseline
✓ Monitor production metrics
✓ Collect feedback for continuous improvement
```

---

## References & Further Reading

**Papers:**
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO)

**Libraries:**
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Unified fine-tuning framework

---

---

## Appendix: Mathematical Foundations & Deep Dive

### A.1 Fine-Tuning Mathematics

#### A.1.1 Cross-Entropy Loss Function

**Underlying Concept:**
The fundamental objective in language model training is to minimize the difference between predicted token probabilities and actual target tokens.

**Mathematical Formulation:**

For a sequence of tokens $x = (x_1, x_2, ..., x_T)$, the cross-entropy loss is:

$$
\mathcal{L}_{CE} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

Where:
- $P(x_t | x_{<t}; \theta)$ is the model's predicted probability of token $x_t$
- $\theta$ represents the model parameters
- The sum is over all positions $t$ in the sequence

**Why Cross-Entropy:**
```
┌─────────────────────────────────────────────────────────────┐
│  Cross-Entropy Properties:                                  │
│                                                             │
│  1. Minimized when prediction matches target exactly       │
│  2. Heavily penalizes confident wrong predictions          │
│  3. Provides stable gradients for optimization             │
│  4. Equivalent to maximizing log-likelihood                │
│                                                             │
│  Example:                                                   │
│  Target token: "cat" (vocab index 5234)                    │
│                                                             │
│  Good prediction: P("cat") = 0.9                           │
│  Loss = -log(0.9) = 0.105                                  │
│                                                             │
│  Bad prediction: P("cat") = 0.1                            │
│  Loss = -log(0.1) = 2.303 (22x higher!)                    │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    """
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    """
    # Flatten for computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Compute cross-entropy
    loss = F.cross_entropy(
        logits_flat, 
        targets_flat,
        ignore_index=-100  # Padding token
    )
    return loss

# Example usage
batch_size, seq_len, vocab_size = 4, 512, 50000
logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

loss = cross_entropy_loss(logits, targets)
print(f"Loss: {loss.item():.4f}")
```

---

#### A.1.2 Gradient Descent and Backpropagation

**Underlying Concept:**
Gradient descent iteratively updates parameters to minimize loss by moving in the direction of the negative gradient.

**Mathematical Formulation:**

**Parameter Update Rule:**
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Where:
- $\theta_t$: Parameters at step $t$
- $\eta$: Learning rate
- $\nabla_\theta \mathcal{L}$: Gradient of loss with respect to parameters

**Stochastic Gradient Descent (SGD):**
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}_i(\theta_t)
$$

Where $B$ is the batch size.

**Adam Optimizer (more sophisticated):**
```
┌─────────────────────────────────────────────────────────────┐
│  Adam Update Rule:                                          │
│                                                             │
│  m_t = β₁·m_{t-1} + (1-β₁)·g_t        (first moment)       │
│  v_t = β₂·v_{t-1} + (1-β₂)·g_t²       (second moment)      │
│                                                             │
│  m̂_t = m_t / (1-β₁^t)                  (bias correction)   │
│  v̂_t = v_t / (1-β₂^t)                  (bias correction)   │
│                                                             │
│  θ_t = θ_{t-1} - η·m̂_t / (√v̂_t + ε)                       │
│                                                             │
│  Typical values:                                            │
│  β₁ = 0.9, β₂ = 0.999, ε = 1e-8                            │
└─────────────────────────────────────────────────────────────┘
```

**Why Adam for Fine-Tuning:**
- Adaptive learning rates per parameter
- Handles sparse gradients well
- Works well with default hyperparameters
- Combines momentum and RMSprop benefits

**Implementation:**
```python
from torch.optim import AdamW

# Standard configuration for fine-tuning
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,              # Conservative learning rate
    betas=(0.9, 0.999),   # Standard Adam values
    eps=1e-8,
    weight_decay=0.01     # L2 regularization
)

# Learning rate scheduler
from transformers import get_linear_schedule_with_warmup

num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

---

#### A.1.3 Weight Initialization and Pre-training

**Why Pre-training Works:**

```
┌─────────────────────────────────────────────────────────────┐
│  Pre-training → Fine-tuning Transfer:                       │
│                                                             │
│  1. Pre-training on large corpus:                           │
│     • Learns general language patterns                      │
│     • Captures syntax, semantics, world knowledge           │
│     • Parameters in "good" region of loss landscape        │
│                                                             │
│  2. Fine-tuning on specific task:                          │
│     • Start from pre-trained weights (not random)          │
│     • Smaller dataset sufficient                           │
│     • Faster convergence                                    │
│     • Better generalization                                 │
│                                                             │
│  Mathematical intuition:                                    │
│  • Pre-training finds θ₀ close to optimal θ*               │
│  • Fine-tuning only needs small Δθ = θ* - θ₀               │
│  • Without pre-training: need large random → θ*            │
└─────────────────────────────────────────────────────────────┘
```

---

### A.2 LoRA Deep Mathematical Analysis

#### A.2.1 Low-Rank Matrix Approximation Theory

**Underlying Concept:**
Weight updates during fine-tuning often have low intrinsic rank, meaning they can be represented with fewer parameters.

**Singular Value Decomposition (SVD):**

Any matrix $W \in \mathbb{R}^{d \times d}$ can be decomposed as:

$$
W = U \Sigma V^T = \sum_{i=1}^{d} \sigma_i u_i v_i^T
$$

Where:
- $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_d \geq 0$ are singular values
- $u_i, v_i$ are left and right singular vectors
- Most information concentrated in top singular values

**Low-Rank Approximation:**

Keep only top $r$ singular values:

$$
W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^T = U_r \Sigma_r V_r^T
$$

**Connection to LoRA:**

```
┌─────────────────────────────────────────────────────────────┐
│  In LoRA, we approximate the update ΔW as:                  │
│                                                             │
│  ΔW ≈ B·A where:                                            │
│  • B ∈ ℝ^{d×r} (down-projection)                           │
│  • A ∈ ℝ^{r×d} (up-projection)                             │
│  • r << d (rank)                                           │
│                                                             │
│  This is equivalent to learning the top r components        │
│  of the SVD of the optimal weight update.                   │
│                                                             │
│  Why it works:                                              │
│  • Intrinsic dimension of fine-tuning is low               │
│  • Most directions in weight space don't matter            │
│  • LoRA learns the important directions                     │
└─────────────────────────────────────────────────────────────┘
```

**Rank Selection:**

```python
import numpy as np

def analyze_effective_rank(weight_matrix, threshold=0.9):
    """
    Analyze how many singular values needed to capture 
    threshold fraction of variance
    """
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    
    # Cumulative explained variance
    explained_variance = np.cumsum(S**2) / np.sum(S**2)
    
    # Find rank needed
    effective_rank = np.argmax(explained_variance >= threshold) + 1
    
    return {
        "singular_values": S[:20],  # Top 20
        "effective_rank_90": effective_rank,
        "explained_variance": explained_variance[:20]
    }

# Example for attention weight matrix
W_q = np.random.randn(4096, 4096)  # Simulated
analysis = analyze_effective_rank(W_q)
print(f"Effective rank (90% variance): {analysis['effective_rank_90']}")
# Often finds rank < 100 sufficient!
```

---

#### A.2.2 LoRA Gradient Computation

**Forward Pass:**

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

Where $W_0$ is frozen (pre-trained), $A$ and $B$ are trainable.

**Backward Pass (Gradients):**

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^T
$$

$$
\frac{\partial \mathcal{L}}{\partial A} = B^T \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^T
$$

**Implementation with Efficient Gradient Computation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=1):
        super().__init__()
        
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero (so ΔW = 0 initially)
        
    def forward(self, x, base_output):
        """
        x: input tensor
        base_output: output from frozen base layer
        """
        # LoRA path: x @ A^T @ B^T
        # Efficient computation: (x @ A^T) @ B^T
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        
        return base_output + lora_output

# Usage example
batch_size, seq_len, hidden_dim = 4, 512, 768
lora = LoRALayer(hidden_dim, hidden_dim, rank=8)

x = torch.randn(batch_size, seq_len, hidden_dim)
base_output = torch.randn(batch_size, seq_len, hidden_dim)  # From frozen layer

output = lora(x, base_output)
print(f"Output shape: {output.shape}")
print(f"Trainable params: {sum(p.numel() for p in lora.parameters())}")
# Only 8*768 + 768*8 = 12,288 params vs 768*768 = 589,824 (48x reduction!)
```

---

### A.3 Reinforcement Learning Mathematics

#### A.3.1 Policy Gradient Methods

**Objective:**

Maximize expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

Where:
- $\tau = (s_0, a_0, s_1, a_1, ...)$ is a trajectory
- $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ is total discounted reward
- $\pi_\theta$ is the policy (LLM) parameterized by $\theta$

**Policy Gradient Theorem:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t\right]
$$

**REINFORCE Algorithm (Basic):**

```python
def compute_reinforce_loss(log_probs, rewards):
    """
    log_probs: List of log π(a_t|s_t) for each step
    rewards: List of rewards r_t for each step
    """
    # Compute returns (cumulative discounted rewards)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
    
    # Policy gradient loss
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G  # Negative because we maximize
    
    return loss / len(log_probs)
```

**Why This Works:**
```
┌─────────────────────────────────────────────────────────────┐
│  Intuition:                                                 │
│                                                             │
│  • Increase probability of actions that led to high reward │
│  • Decrease probability of actions that led to low reward  │
│                                                             │
│  ∇_θ log π_θ(a|s) tells us how to change parameters       │
│  to increase/decrease probability of action a               │
│                                                             │
│  Multiplying by reward R scales the update:                │
│  • High R → big update in direction that increases P(a)    │
│  • Low R → big update in direction that decreases P(a)     │
└─────────────────────────────────────────────────────────────┘
```

---

#### A.3.2 PPO (Proximal Policy Optimization) Mathematics

**Problem with Vanilla Policy Gradient:**
- Large updates can destabilize training
- No constraint on how far policy moves

**PPO Solution:**
Add constraint to prevent large policy changes

**Clipped Surrogate Objective:**

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is probability ratio
- $\hat{A}_t$ is estimated advantage
- $\epsilon$ is hyperparameter (typically 0.1 or 0.2)

**Advantage Function:**

$$
\hat{A}_t = R_t - V(s_t)
$$

Where $V(s_t)$ is value function estimate (baseline).

**Why Clipping Helps:**
```
┌─────────────────────────────────────────────────────────────┐
│  Probability Ratio r(θ):                                    │
│                                                             │
│  • r(θ) = 1: New policy same as old                        │
│  • r(θ) > 1: New policy more likely to take action         │
│  • r(θ) < 1: New policy less likely to take action         │
│                                                             │
│  Without clipping:                                         │
│  • If advantage is positive, keeps increasing r(θ)         │
│  • Can lead to catastrophic policy collapse                │
│                                                             │
│  With clipping (ε=0.2):                                    │
│  • If r(θ) > 1.2, gradient becomes 0                       │
│  • Policy can only move so far in one update               │
│  • More stable training                                     │
└─────────────────────────────────────────────────────────────┘
```

**PPO Implementation:**

```python
def compute_ppo_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    """
    new_log_probs: Log probs from current policy π_θ
    old_log_probs: Log probs from old policy π_{θ_old} (detached)
    advantages: Advantage estimates Â_t
    """
    # Probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # Take minimum (pessimistic bound)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss

# Full PPO update
for epoch in range(K_epochs):  # Multiple epochs on same batch
    new_log_probs, state_values, entropy = policy.evaluate(states, actions)
    
    # PPO loss
    policy_loss = compute_ppo_loss(new_log_probs, old_log_probs, advantages)
    
    # Value function loss
    value_loss = F.mse_loss(state_values, returns)
    
    # Entropy bonus (encourage exploration)
    entropy_bonus = entropy.mean()
    
    # Total loss
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### A.4 RLHF Reward Modeling Mathematics

#### A.4.1 Bradley-Terry Model

**Underlying Concept:**
Model the probability that humans prefer one response over another.

**Mathematical Formulation:**

For two responses $y_1$ and $y_2$ to prompt $x$:

$$
P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2)) = \frac{1}{1 + e^{-(r(x, y_1) - r(x, y_2))}}
$$

Where:
- $r(x, y)$ is the reward model's scalar score
- $\sigma$ is the sigmoid function
- $y_1 \succ y_2$ means "$y_1$ preferred over $y_2$"

**Loss Function:**

For a dataset of comparisons $\mathcal{D} = \{(x, y_w, y_l)\}$:

$$
\mathcal{L}_R = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]
$$

**Intuition:**
```
┌─────────────────────────────────────────────────────────────┐
│  Reward Model Training:                                     │
│                                                             │
│  We want: r(x, y_w) >> r(x, y_l)                           │
│                                                             │
│  If r_w = 8.5 and r_l = 2.1:                               │
│  P(prefer y_w) = σ(8.5 - 2.1) = σ(6.4) ≈ 0.998            │
│                                                             │
│  Loss = -log(0.998) ≈ 0.002 (very low, good!)             │
│                                                             │
│  If r_w = 5.0 and r_l = 4.5:                               │
│  P(prefer y_w) = σ(5.0 - 4.5) = σ(0.5) ≈ 0.62             │
│                                                             │
│  Loss = -log(0.62) ≈ 0.48 (higher, needs improvement)     │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Add regression head
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        # Use last hidden state of last token
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

def compute_reward_model_loss(reward_model, batch):
    """
    batch contains:
    - prompt_input_ids
    - chosen_input_ids (preferred completion)
    - rejected_input_ids (dispreferred completion)
    """
    # Get rewards
    chosen_rewards = reward_model(
        batch['chosen_input_ids'],
        batch['chosen_attention_mask']
    )
    rejected_rewards = reward_model(
        batch['rejected_input_ids'],
        batch['rejected_attention_mask']
    )
    
    # Bradley-Terry loss
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    # Accuracy (for monitoring)
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, accuracy
```

---

#### A.4.2 DPO (Direct Preference Optimization) Derivation

**Key Insight:**
Instead of learning a reward model then doing RL, derive optimal policy directly from preferences.

**Mathematical Derivation:**

**Step 1: Optimal Policy Under KL Constraint**

Find policy $\pi$ that maximizes expected reward while staying close to reference $\pi_{ref}$:

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)] - \beta \mathbb{D}_{KL}(\pi(y|x) \| \pi_{ref}(y|x))
$$

**Step 2: Closed-Form Solution**

The optimal policy has form:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

Where $Z(x)$ is partition function.

**Step 3: Solve for Reward**

Rearrange to express reward in terms of policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

**Step 4: Plug into Bradley-Terry**

Substitute into preference model (partition function cancels!):

$$
P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

**Step 5: DPO Loss**

$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**Why DPO is Elegant:**
```
┌─────────────────────────────────────────────────────────────┐
│  DPO Advantages:                                            │
│                                                             │
│  1. No reward model needed!                                 │
│     • Directly optimizes policy from preferences           │
│     • Saves training time and computation                   │
│                                                             │
│  2. Single-stage training                                   │
│     • SFT → DPO (instead of SFT → RM → PPO)                │
│     • Simpler pipeline                                      │
│                                                             │
│  3. Theoretical guarantee                                   │
│     • Exact solution to constrained optimization           │
│     • Not an approximation                                  │
│                                                             │
│  4. Often works better in practice                          │
│     • More stable than PPO                                  │
│     • Better final performance                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
def compute_dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    policy_model: Current policy π_θ (being optimized)
    ref_model: Reference policy π_ref (frozen, usually SFT model)
    """
    # Get log probabilities from policy
    policy_chosen_logps = get_batch_logps(
        policy_model, 
        batch['chosen_input_ids'],
        batch['chosen_attention_mask']
    )
    policy_rejected_logps = get_batch_logps(
        policy_model,
        batch['rejected_input_ids'],
        batch['rejected_attention_mask']
    )
    
    # Get log probabilities from reference (no grad)
    with torch.no_grad():
        ref_chosen_logps = get_batch_logps(
            ref_model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask']
        )
        ref_rejected_logps = get_batch_logps(
            ref_model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask']
        )
    
    # Compute log-ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # DPO loss
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Metrics
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    return loss, chosen_rewards.mean(), rejected_rewards.mean()

def get_batch_logps(model, input_ids, attention_mask):
    """Compute log probabilities of sequences"""
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs of actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask padding and sum
    mask = (input_ids != -100).float()
    return (token_log_probs * mask).sum(-1)
```

---

### A.5 Advanced Topics

#### A.5.1 KL Divergence in RLHF

**Mathematical Definition:**

KL divergence measures difference between two probability distributions:

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

**In RLHF Context:**

$$
D_{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x)) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]
$$

**Why KL Penalty is Important:**
```
┌─────────────────────────────────────────────────────────────┐
│  Without KL Penalty:                                        │
│  • Policy can drift far from base model                    │
│  • May forget general capabilities                         │
│  • Can overfit to reward model                             │
│                                                             │
│  With KL Penalty:                                           │
│  • Stays close to reference policy                         │
│  • Preserves base model capabilities                       │
│  • More stable training                                     │
│                                                             │
│  Trade-off:                                                 │
│  • Too small β: No constraint, may overfit                 │
│  • Too large β: No learning, stuck at reference            │
│  • Typical β: 0.1 to 0.5                                   │
└─────────────────────────────────────────────────────────────┘
```

---

### Summary: Key Equations

| Topic | Key Equation | Purpose |
|-------|--------------|---------|
| **Cross-Entropy** | $\mathcal{L} = -\sum \log P(x_t \| x_{<t})$ | Language modeling objective |
| **LoRA** | $W' = W_0 + BA$ | Parameter-efficient updates |
| **Policy Gradient** | $\nabla J = \mathbb{E}[\nabla \log \pi \cdot R]$ | RL policy optimization |
| **PPO Clip** | $L^{CLIP} = \mathbb{E}[\min(rA, \text{clip}(r)A)]$ | Stable policy updates |
| **Bradley-Terry** | $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ | Preference modeling |
| **DPO** | $\mathcal{L} = -\log \sigma(\beta \log \frac{\pi}{\pi_{ref}})$ | Direct preference optimization |

---

*This guide provides comprehensive coverage of fine-tuning and RLHF with mathematical foundations, detailed breakdowns, and step-by-step implementations.*

**Complete Guide Series:**
- 01: Python & SQL Solutions
- 02: LLM Fundamentals
- 03: RAG Comprehensive
- 04: LangChain & Agents
- 05: Interview Questions
- 06: Study Guide
- 07: Python & ML Answers
- 08: RAG Answers
- 09: **Fine-Tuning & RLHF** (This Guide)

The file is saved at: `09_fine_tuning_rl_rlhf_guide.md`
