# LLM/GenAI Engineer Interview - Study Guide & Preparation Strategies

## Quick Reference for High-Priority Topics

Based on the job description emphasis and current interview trends (2025), focus your preparation in this order:

### 🔥 CRITICAL (Must-Know) - 40% of Interview
1. **RAG Systems** - Implementation, optimization, production deployment
2. **LangChain/LangGraph** - Chains, agents, tools, memory
3. **Prompt Engineering** - Few-shot, CoT, ReAct patterns
4. **Python/SQL** - Data manipulation, pipeline building
5. **Production Deployment** - Scaling, monitoring, debugging

### ⚡ HIGH PRIORITY (Frequently Asked) - 35% of Interview
6. **LLM Fundamentals** - Transformers, attention, tokenization
7. **Fine-tuning & RLHF** - LoRA, QLoRA, DPO
8. **Agent Design** - Multi-step reasoning, tool usage
9. **Vector Databases** - Embeddings, similarity search
10. **MLOps** - CI/CD, monitoring, versioning

### 📚 GOOD TO KNOW (Nice to Have) - 25% of Interview
11. **Advanced Architectures** - MoE, long-context models
12. **Multimodal LLMs** - Vision, audio integration
13. **Research Awareness** - Recent papers, trends
14. **Specialized Topics** - Constitutional AI, RLHF alternatives

---

## Common Interview Patterns & How to Answer

### Pattern 1: "Implement X from Scratch"
**Questions like:**
- "Implement a simple RAG system"
- "Build a basic agent with tool calling"
- "Create a prompt template manager"

**Answer Framework:**
```python
# 1. Start with requirements clarification
"Let me clarify a few things first:
- What's the expected data volume?
- Do we need real-time or batch processing?
- What's the accuracy/latency trade-off?"

# 2. Propose high-level architecture
"I'll break this into three components:
1. Data ingestion/indexing
2. Retrieval/processing
3. Generation/response"

# 3. Walk through implementation with code
# Show actual Python code

# 4. Discuss trade-offs and optimizations
"We could optimize this by:
- Caching frequent queries
- Batching embeddings
- Async processing"

# 5. Mention production considerations
"For production, we'd add:
- Error handling
- Logging/monitoring
- Rate limiting"
```

### Pattern 2: "System Design for LLM Application"
**Questions like:**
- "Design a customer support chatbot"
- "Build a document Q&A system"
- "Create a code generation assistant"

**Answer Framework:**
```
1. REQUIREMENTS (5 min)
   - Clarify scope, scale, constraints
   - Users: Who? How many?
   - Data: Volume, freshness, structure
   - Latency: Real-time? Batch?
   - Budget: API costs, infrastructure

2. HIGH-LEVEL ARCHITECTURE (10 min)
   - Draw component diagram
   - Data flow
   - API contracts
   - Storage solutions

3. DEEP DIVE (15 min)
   - Choose LLM approach (prompting/RAG/fine-tuning)
   - Vector store selection
   - Retrieval strategy
   - Caching strategy
   - Monitoring approach

4. TRADE-OFFS & SCALING (5 min)
   - Latency vs. accuracy
   - Cost optimization
   - Failure scenarios
   - Future extensibility
```

### Pattern 3: "Debugging & Optimization"
**Questions like:**
- "LLM is too slow, how do you fix it?"
- "RAG returns irrelevant results, debug this"
- "Agent gets stuck in loops, what's wrong?"

**Answer Framework:**
```
1. GATHER INFORMATION
   - What are the symptoms?
   - When did it start?
   - What changed recently?
   - What are the metrics showing?

2. HYPOTHESIZE ROOT CAUSES
   - Rank by probability
   - List quick tests for each

3. SYSTEMATIC DEBUGGING
   - Start with logs and metrics
   - Isolate components
   - Test incrementally
   - Measure impact

4. IMPLEMENT SOLUTION
   - Fix and verify
   - Add monitoring
   - Document learnings
```

---

## Code Examples for Common Interview Questions

### 1. Basic RAG Implementation (LangChain)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# Step 1: Load and split documents
loader = TextLoader("documents.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Step 2: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 3: Create retrieval chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce", "refine"
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Step 4: Query
query = "What is the main topic discussed?"
result = qa_chain({"query": query})
print(result["result"])
print(result["source_documents"])

# Optimization tips:
# - Use async for parallel processing
# - Cache embeddings
# - Implement re-ranking
# - Add metadata filtering
# - Monitor token usage
```

### 2. Custom Agent with Tools (LangChain)

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import requests

# Define custom tools
@tool
def search_database(query: str) -> str:
    """Search the internal database for information."""
    # Your database search logic
    return f"Database results for: {query}"

@tool
def calculate_metric(expression: str) -> str:
    """Calculate mathematical expressions or metrics."""
    try:
        result = eval(expression)  # In production, use safer evaluation
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # Your web search API call
    return f"Web results for: {query}"

# Create tools list
tools = [search_database, calculate_metric, web_search]

# Create agent
llm = OpenAI(temperature=0)

prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools])
    }
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# Execute
result = agent_executor.invoke({
    "input": "What is the revenue for Q4 and calculate the YoY growth?"
})
print(result["output"])
```

### 3. Prompt Template with Few-Shot Examples

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {
        "input": "How do I reset my password?",
        "output": "To reset your password:\n1. Go to login page\n2. Click 'Forgot Password'\n3. Enter your email\n4. Check your inbox for reset link\n5. Follow the link and create new password"
    },
    {
        "input": "I can't log in to my account",
        "output": "Let's troubleshoot your login issue:\n1. Verify your username/email\n2. Check if Caps Lock is on\n3. Try resetting your password\n4. Clear browser cache\n5. Try a different browser\n\nIf none work, contact support with your account details."
    },
    {
        "input": "Where is my order?",
        "output": "To track your order:\n1. Log into your account\n2. Go to 'Orders' section\n3. Find your order number\n4. Click 'Track Package'\n\nExpected delivery dates are shown there. If delayed, you'll see updated information."
    }
]

# Create example template
example_template = """
User: {input}
Assistant: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Create few-shot template
prefix = """You are a helpful customer support assistant. Answer questions clearly and concisely.

Here are some examples of good responses:
"""

suffix = """
User: {input}
Assistant: """

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input"],
    example_separator="\n"
)

# Use it
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)

query = "How do I update my billing information?"
response = llm(few_shot_prompt.format(input=query))
print(response)
```

### 4. Advanced RAG with Re-ranking

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import numpy as np

class AdvancedRAG:
    def __init__(self, documents):
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents, 
            self.embeddings,
            collection_name="advanced_rag"
        )
        
        # Initialize re-ranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def hybrid_search(self, query: str, k: int = 20) -> list:
        """Perform hybrid search combining dense and sparse retrieval"""
        # Dense retrieval (semantic)
        dense_results = self.vectorstore.similarity_search(query, k=k)
        
        # Sparse retrieval (keyword-based) - simplified
        # In production, use BM25 or similar
        keyword_scores = self._keyword_search(query, dense_results)
        
        # Combine scores (ensemble)
        combined_results = self._combine_retrieval_results(
            dense_results, 
            keyword_scores
        )
        
        return combined_results[:k]
    
    def rerank_results(self, query: str, documents: list, top_k: int = 3) -> list:
        """Re-rank retrieved documents using cross-encoder"""
        if not documents:
            return []
        
        # Prepare pairs for re-ranking
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get re-ranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        return [documents[i] for i in ranked_indices]
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Main retrieval pipeline"""
        # Step 1: Hybrid search
        candidates = self.hybrid_search(query, k=20)
        
        # Step 2: Re-rank
        final_results = self.rerank_results(query, candidates, top_k=top_k)
        
        return final_results
    
    def _keyword_search(self, query: str, documents: list) -> dict:
        """Simple keyword matching (BM25 would be better)"""
        query_terms = set(query.lower().split())
        scores = {}
        
        for i, doc in enumerate(documents):
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms & doc_terms)
            scores[i] = score
        
        return scores
    
    def _combine_retrieval_results(self, documents: list, keyword_scores: dict) -> list:
        """Combine dense and sparse retrieval results"""
        # Normalize scores and combine
        # This is simplified - production would use learned weights
        return sorted(
            documents,
            key=lambda x: keyword_scores.get(documents.index(x), 0),
            reverse=True
        )

# Usage
documents = [...]  # Your documents
rag = AdvancedRAG(documents)
results = rag.retrieve("What is the company's revenue?", top_k=3)
```

### 5. LLM Evaluation Framework

```python
from typing import List, Dict
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import json

class LLMEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def evaluate_generation(
        self, 
        predictions: List[str], 
        references: List[str],
        metrics: List[str] = ['rouge', 'bertscore', 'exact_match']
    ) -> Dict:
        """Comprehensive evaluation of LLM generations"""
        results = {}
        
        if 'rouge' in metrics:
            results['rouge'] = self._compute_rouge(predictions, references)
        
        if 'bertscore' in metrics:
            results['bertscore'] = self._compute_bertscore(predictions, references)
        
        if 'exact_match' in metrics:
            results['exact_match'] = self._compute_exact_match(predictions, references)
        
        return results
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores"""
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in scores.items()
        }
    
    def _compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BERTScore"""
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        
        return {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean())
        }
    
    def _compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy"""
        matches = sum(
            pred.strip().lower() == ref.strip().lower() 
            for pred, ref in zip(predictions, references)
        )
        return matches / len(predictions)
    
    def evaluate_rag_system(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        ground_truth_docs: List[List[str]],
        k: int = 3
    ) -> Dict:
        """Evaluate RAG retrieval quality"""
        precision_at_k = []
        recall_at_k = []
        mrr_scores = []
        
        for retrieved, ground_truth in zip(retrieved_docs, ground_truth_docs):
            # Precision@K
            relevant_retrieved = len(set(retrieved[:k]) & set(ground_truth))
            precision = relevant_retrieved / k if k > 0 else 0
            precision_at_k.append(precision)
            
            # Recall@K
            recall = relevant_retrieved / len(ground_truth) if ground_truth else 0
            recall_at_k.append(recall)
            
            # MRR
            for i, doc in enumerate(retrieved[:k], 1):
                if doc in ground_truth:
                    mrr_scores.append(1.0 / i)
                    break
            else:
                mrr_scores.append(0.0)
        
        return {
            'precision@k': np.mean(precision_at_k),
            'recall@k': np.mean(recall_at_k),
            'mrr': np.mean(mrr_scores)
        }

# Usage
evaluator = LLMEvaluator()

predictions = ["The capital of France is Paris.", "Python is a programming language."]
references = ["Paris is the capital of France.", "Python is a programming language."]

results = evaluator.evaluate_generation(predictions, references)
print(json.dumps(results, indent=2))
```

---

## SQL Questions - Common Patterns

### 1. Window Functions
```sql
-- Running total
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;

-- Top N per group
SELECT *
FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) as rn
    FROM products
) WHERE rn <= 3;

-- Moving average
SELECT 
    date,
    metric,
    AVG(metric) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7days
FROM metrics;
```

### 2. Complex Joins
```sql
-- Self-join for finding gaps
SELECT 
    t1.id,
    t1.date as start_date,
    MIN(t2.date) as end_date
FROM events t1
LEFT JOIN events t2 ON t2.date > t1.date
GROUP BY t1.id, t1.date
HAVING DATEDIFF(MIN(t2.date), t1.date) > 1;

-- Multiple condition joins
SELECT 
    u.user_id,
    u.name,
    COUNT(DISTINCT o.order_id) as orders,
    SUM(o.amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id 
    AND o.status = 'completed'
    AND o.date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
GROUP BY u.user_id, u.name;
```

### 3. CTEs and Subqueries
```sql
-- Recursive CTE for hierarchies
WITH RECURSIVE org_hierarchy AS (
    SELECT employee_id, manager_id, name, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    SELECT e.employee_id, e.manager_id, e.name, oh.level + 1
    FROM employees e
    JOIN org_hierarchy oh ON e.manager_id = oh.employee_id
)
SELECT * FROM org_hierarchy ORDER BY level, name;

-- Multiple CTEs for complex logic
WITH 
daily_metrics AS (
    SELECT date, SUM(revenue) as revenue
    FROM sales
    GROUP BY date
),
moving_avg AS (
    SELECT 
        date,
        revenue,
        AVG(revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as avg_7d
    FROM daily_metrics
)
SELECT * FROM moving_avg WHERE revenue > avg_7d * 1.2;
```

---

## System Design - Common Patterns

### 1. RAG System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        USER REQUEST                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      API GATEWAY                            │
│  - Rate limiting  - Auth  - Load balancing                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSOR                          │
│  - Query understanding  - Intent classification             │
│  - Query reformulation  - Cache check                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ VECTOR SEARCH   │ │ KEYWORD SEARCH  │
│  - FAISS/Chroma │ │  - BM25         │
│  - ANN search   │ │  - Elastic      │
└────────┬────────┘ └────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │   RE-RANKER     │
         │  - CrossEncoder │
         │  - Score fusion │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ CONTEXT BUILDER │
         │  - Chunk select │
         │  - Format prompt│
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │   LLM CALL      │
         │  - OpenAI/Claude│
         │  - Streaming    │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ RESPONSE PROC   │
         │  - Validation   │
         │  - Formatting   │
         │  - Citation add │
         └────────┬────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                        RESPONSE                             │
└─────────────────────────────────────────────────────────────┘

SUPPORTING SYSTEMS:
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Monitoring  │ │   Logging   │ │   Cache     │
│ - Metrics   │ │ - Trace IDs │ │ - Redis     │
│ - Alerts    │ │ - Errors    │ │ - Mem cache │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 2. Agent System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    USER INSTRUCTION                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
│  - Task decomposition                                       │
│  - Planning phase                                           │
│  - State management                                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ AGENT 1  │ │ AGENT 2  │ │ AGENT 3  │
│ Research │ │ Analysis │ │ Writing  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┴────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    TOOL REGISTRY                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Web      │ │Database │ │  API    │ │  Code   │
│Search   │ │ Query   │ │ Calls   │ │Executor │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## Study Resources by Priority

### Must-Read Papers
1. "Attention Is All You Need" (Vaswani et al.) - Transformers
2. "RAG: Retrieval-Augmented Generation" (Lewis et al.)
3. "ReAct: Synergizing Reasoning and Acting" (Yao et al.)
4. "Training language models to follow instructions" (InstructGPT)
5. "LoRA: Low-Rank Adaptation of Large Language Models"

### Essential Documentation
1. LangChain Documentation - Focus on: Agents, Chains, RAG
2. OpenAI API Documentation - GPT-4, Embeddings
3. Anthropic Claude Documentation - API, prompt engineering
4. HuggingFace Transformers Documentation
5. Vector Database Docs (Pinecone/Chroma/Weaviate)

### Practice Platforms
1. LeetCode (Medium) - Python, algorithms
2. StrataScratch - SQL, data problems
3. HuggingFace Hub - Model experimentation
4. GitHub - Study open-source LLM projects
5. Personal projects - Build mini RAG/agent systems

---

## Interview Day Tips

### Technical Interview (Coding/System Design)
1. **Clarify First**: Always ask questions before jumping to solution
2. **Think Aloud**: Explain your reasoning process
3. **Start Simple**: Basic solution first, then optimize
4. **Code Quality**: Clean, readable code with error handling
5. **Testing**: Mention edge cases and how you'd test

### Behavioral Interview
1. **STAR Method**: Situation, Task, Action, Result
2. **Be Specific**: Use concrete examples from your projects
3. **Show Impact**: Quantify results when possible
4. **Learn from Failures**: Discuss what you learned
5. **Ask Questions**: Show genuine interest in the role

### Common Red Flags to Avoid
- ❌ Not asking clarifying questions
- ❌ Jumping to code without planning
- ❌ Ignoring edge cases
- ❌ Not considering scale/production issues
- ❌ Being unable to explain trade-offs
- ❌ Not knowing basics of tools you claim to know

### Green Flags Interviewers Love
- ✅ Clear communication of thought process
- ✅ Asking about requirements and constraints
- ✅ Discussing multiple approaches and trade-offs
- ✅ Production mindset (monitoring, testing, scaling)
- ✅ Learning from feedback during interview
- ✅ Showing passion for the field

---

## Final Checklist Before Interview

### Technical Prep
- [ ] Can explain RAG pipeline from scratch
- [ ] Can implement basic agent with LangChain
- [ ] Comfortable with Python data structures
- [ ] Can write SQL for complex analytics
- [ ] Know prompt engineering patterns
- [ ] Understand fine-tuning vs. RAG trade-offs
- [ ] Can discuss production deployment strategies

### Projects to Reference
- [ ] Have 2-3 LLM projects ready to discuss in detail
- [ ] Can explain architecture decisions
- [ ] Know metrics and results
- [ ] Can discuss what you'd do differently

### Questions to Ask Interviewer
- What LLM use cases are you building?
- What's the tech stack for LLM applications?
- How do you evaluate LLM outputs?
- What are the biggest technical challenges?
- What does success look like in first 90 days?

---

**Good luck with your interview! Remember: It's not about knowing everything, but about showing how you think, learn, and solve problems.**

---

# 🎓 Deep Dive Explanations - Concepts, Math & Implementation

This section provides detailed explanations of the underlying concepts, mathematical foundations, benefits, and implementation details for all topics appearing in this study guide. Each topic is explained with:
1. **Concept**: What it is and how it works
2. **Math**: The mathematical foundations
3. **Why It's Good**: Benefits and trade-offs
4. **Implementation**: How to implement it

---

## 1. RAG Systems Deep Dive

### 1.1 Document Chunking

#### **Concept**
Document chunking splits large documents into smaller, semantically coherent segments. The goal is to create chunks that:
- Preserve complete semantic meaning
- Fit within embedding model context limits
- Enable precise retrieval of relevant information

#### **Math**
**Chunk Size Optimization Formula:**
The optimal chunk size balances information density with retrieval precision:

$$
\text{Chunk Score} = \alpha \cdot \text{Semantic Completeness} - \beta \cdot \text{Redundancy} - \gamma \cdot \text{Retrieval Noise}
$$

Where:
- $\alpha$ = importance of semantic coherence (typically 0.5)
- $\beta$ = penalty for overlap (typically 0.2)
- $\gamma$ = penalty for irrelevant content (typically 0.3)

**Overlap Calculation:**
For chunk size $S$ and overlap ratio $r$:
$$
\text{Overlap Size} = S \times r
$$
$$
\text{Effective Chunk Size} = S - (S \times r) = S(1-r)
$$

**Example:**
- Document: 10,000 tokens
- Chunk size: 512 tokens
- Overlap: 20%
- Number of chunks: $\lceil 10000 / (512 \times 0.8) \rceil = \lceil 10000 / 409.6 \rceil = 25$ chunks

#### **Why It's Good**
1. **Context Window Compliance**: Ensures chunks fit within embedding model limits (typically 512-2048 tokens)
2. **Retrieval Precision**: Smaller chunks reduce noise from irrelevant content
3. **Semantic Preservation**: Overlapping maintains context across chunk boundaries
4. **Storage Efficiency**: Optimal sizing reduces vector storage costs
5. **Latency**: Smaller chunks = faster embedding generation

#### **Implementation**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Mathematical approach to chunk sizing
class ChunkingStrategy:
    def __init__(self, chunk_size=512, overlap_ratio=0.2):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        
    def calculate_chunks(self, text_length: int) -> int:
        """Calculate number of chunks needed."""
        effective_size = self.chunk_size - self.overlap
        return (text_length + effective_size - 1) // effective_size
    
    def split(self, documents):
        """Split with semantic preservation."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)

# Usage
strategy = ChunkingStrategy(chunk_size=512, overlap_ratio=0.2)
num_chunks = strategy.calculate_chunks(10000)  # Returns 25
```

---

### 1.2 Embedding Models & Vector Representations

#### **Concept**
Embeddings map discrete tokens/text into continuous vector space where semantic similarity corresponds to geometric proximity. Modern embedding models (BERT, OpenAI, BGE) create dense vectors (768-3072 dimensions) that capture semantic meaning.

#### **Math**
**Vector Representation:**
$$
\text{Embed}: \text{Text} \rightarrow \mathbb{R}^d
$$
Where $d$ is embedding dimension (typically 768, 1024, or 1536)

**Cosine Similarity (Primary Metric):**
$$
\text{Cosine Similarity}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \cdot \sqrt{\sum_{i=1}^{d} b_i^2}}
$$
Range: $[-1, 1]$, where 1 = identical direction (semantically similar)

**Euclidean Distance (Alternative):**
$$
\text{Euclidean}(a, b) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}
$$

**Dot Product (For normalized vectors):**
$$
\text{Dot Product}(a, b) = \sum_{i=1}^{d} a_i b_i
$$

**Information Retrieval Metrics:**
- **Precision@k**: $\frac{\text{Relevant in top-k}}{k}$
- **Recall@k**: $\frac{\text{Relevant in top-k}}{\text{Total relevant}}$
- **MRR (Mean Reciprocal Rank)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$

#### **Why It's Good**
1. **Semantic Search**: Finds conceptually related content even with different keywords
2. **Dense Representation**: Compresses semantic meaning into fixed-size vectors
3. **Scalability**: Enables efficient similarity search via ANN algorithms
4. **Multilingual**: Cross-lingual embeddings enable search across languages
5. **Compositionality**: Can compose meanings (king - man + woman ≈ queen)

#### **Implementation**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name="text-embedding-3-large"):
        self.model = OpenAIEmbeddings(model=model_name)
        self.dimensions = 3072  # For text-embedding-3-large
    
    def embed(self, texts: list) -> np.ndarray:
        """Generate embeddings for texts."""
        return np.array(self.model.embed_documents(texts))
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def find_similar(self, query_emb: np.ndarray, 
                     doc_embs: np.ndarray, top_k: int = 5):
        """Find top-k most similar documents."""
        # Normalize for cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]

# Usage
engine = EmbeddingEngine()
documents = ["Machine learning is amazing", "Deep learning uses neural networks"]
query = "What is artificial intelligence?"

doc_embs = engine.embed(documents)
query_emb = engine.embed([query])[0]
indices, scores = engine.find_similar(query_emb, doc_embs)
```

---

### 1.3 Vector Similarity Search

#### **Concept**
Vector search retrieves the most similar vectors from a database using distance metrics. ANN (Approximate Nearest Neighbor) algorithms trade perfect accuracy for massive speedups (10x-100x faster than exact search).

#### **Math**
**HNSW (Hierarchical Navigable Small World) - Common ANN Algorithm:**

**Layer Probability:**
For layer $l$, insertion probability decreases exponentially:
$$P(l) = e^{-l/m} \cdot \text{max}(0, l - L_{max})$$

Where:
- $m$ = layer multiplier (typically 16)
- $L_{max}$ = maximum layer (calculated as $\lceil \log_M(N) \rceil$)
- $N$ = total number of vectors

**Search Complexity:**
- Exact search: $O(N \cdot d)$ (linear scan)
- HNSW search: $O(\log N)$ with high recall (>95%)
- IVF (Inverted File Index): $O(\sqrt{N})$ with tunable recall

**Recall vs Speed Trade-off:**
$$\text{Recall@10} \approx 1 - e^{-\frac{nprobe \cdot nlist}{N}}$$
Where:
- $nprobe$ = number of clusters to search
- $nlist$ = total number of clusters

#### **Why It's Good**
1. **Sub-linear Complexity**: Search millions of vectors in milliseconds
2. **Scalability**: Handles billions of vectors with partitioning
3. **Approximate Accuracy**: 95-99% recall with 100x speedup
4. **Hybrid Search**: Combines vector + metadata filtering
5. **Real-time**: Enables live RAG applications

#### **Implementation**
```python
import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dimension: int, index_type="HNSW"):
        self.dimension = dimension
        self.index_type = index_type
        self._build_index()
    
    def _build_index(self):
        """Build ANN index with mathematical optimizations."""
        if self.index_type == "HNSW":
            # HNSW: M=16, efConstruction=200
            self.index = faiss.IndexHNSWFlat(self.dimension, 16)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        elif self.index_type == "IVF":
            # IVF with nlist = 4*sqrt(N) heuristic
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = 100  # Adjust based on data size
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
    
    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to index."""
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
    
    def search(self, query: np.ndarray, k: int = 5, 
               nprobe: int = 10) -> tuple:
        """Search with recall tuning."""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe  # Trade speed for accuracy
        
        distances, indices = self.index.search(
            query.reshape(1, -1), k
        )
        return indices[0], distances[0]
    
    def benchmark_search(self, queries: np.ndarray, k: int = 5):
        """Benchmark search performance."""
        import time
        
        start = time.time()
        results = self.index.search(queries, k)
        elapsed = time.time() - start
        
        qps = len(queries) / elapsed
        latency = elapsed / len(queries) * 1000  # ms
        
        return {
            'queries_per_second': qps,
            'avg_latency_ms': latency,
            'total_time_s': elapsed
        }

# Usage
dimension = 768
index = VectorIndex(dimension, index_type="HNSW")

# Add 10000 random vectors
vectors = np.random.randn(10000, dimension).astype('float32')
index.add_vectors(vectors)

# Search
query = np.random.randn(dimension).astype('float32')
indices, distances = index.search(query, k=5)
```

---

## 2. LangChain & Agent Patterns Deep Dive

### 2.1 LangChain Expression Language (LCEL)

#### **Concept**
LCEL is a declarative way to compose LangChain components using the pipe operator (`|`), similar to Unix pipes or functional programming. It enables:
- Streaming support
- Async execution
- Parallel processing
- Automatic optimization

#### **Math**
**Chain Composition Algebra:**
$$
C = f_n \circ f_{n-1} \circ \ldots \circ f_2 \circ f_1
$$

Where each $f_i$ is a transformation function:
$$f_i: X_i \rightarrow X_{i+1}$$

**Parallel Processing Speedup:**
For $n$ parallel operations with execution time $t_i$:
$$
T_{\text{sequential}} = \sum_{i=1}^{n} t_i
$$
$$
T_{\text{parallel}} = \max(t_1, t_2, \ldots, t_n) + T_{\text{overhead}}
$$
$$
\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}
$$

#### **Why It's Good**
1. **Composability**: Build complex chains from simple components
2. **Streaming**: Automatic token-by-token streaming
3. **Async**: Native async/await support
4. **Type Safety**: Preserves type information through chains
5. **Debugging**: Easy to inspect intermediate results

#### **Implementation**
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LCEL Chain Composition
class LCELChainBuilder:
    def __init__(self, model_name="gpt-4"):
        self.llm = ChatOpenAI(model=model_name)
    
    def build_rag_chain(self, retriever):
        """Build RAG chain with LCEL composition."""
        
        # Define the chain algebraically:
        # chain = retrieve | format | generate
        
        template = """Answer based on context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Composition using pipe operator
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
        )
        
        return chain
    
    def _format_docs(self, docs):
        """Format documents for context."""
        return "\n\n".join([d.page_content for d in docs])
    
    def build_parallel_chain(self):
        """Execute operations in parallel."""
        
        # Parallel processing map
        parallel_chain = RunnableParallel(
            summary=self._summary_chain(),
            keywords=self._keyword_chain(),
            sentiment=self._sentiment_chain()
        )
        
        return parallel_chain
    
    def _summary_chain(self):
        return ChatPromptTemplate.from_template("Summarize: {text}") | self.llm
    
    def _keyword_chain(self):
        return ChatPromptTemplate.from_template("Extract keywords: {text}") | self.llm
    
    def _sentiment_chain(self):
        return ChatPromptTemplate.from_template("Sentiment of: {text}") | self.llm

# Usage
builder = LCELChainBuilder()
# Each component can be inspected, tested, and reused independently
```

---

### 2.2 ReAct Agent Pattern

#### **Concept**
ReAct (Reasoning + Acting) alternates between:
1. **Thought**: Internal reasoning about what to do
2. **Action**: Tool execution
3. **Observation**: Processing tool results

This loop continues until the task is complete.

#### **Math**
**ReAct as Markov Decision Process:**
$$
\text{State}_t = \{\text{Context}, \text{History}_t, \text{Observation}_t\}
$$
$$
\text{Action}_t = \pi_\theta(\text{State}_t)
$$
$$
\text{Reward} = \begin{cases} 
+1 & \text{if task completed correctly} \\
-0.1 & \text{per step (penalty for inefficiency)} \\
-1 & \text{if error or hallucination}
\end{cases}
$$

**Action Selection Probability:**
$$
P(a_t | s_t) = \text{softmax}(\text{LLM}(\text{prompt}_t))
$$

#### **Why It's Good**
1. **Transparency**: Explicit reasoning steps visible
2. **Correctability**: Can debug where reasoning failed
3. **Flexibility**: Adapts to task complexity dynamically
4. **Tool Integration**: Natural fit for external tool use
5. **Human Alignment**: Reasoning similar to human problem-solving

#### **Implementation**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

class ReActAgent:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _setup_tools(self):
        """Define available tools with mathematical descriptions."""
        
        @tool
        def calculate(expression: str) -> str:
            """Evaluate mathematical expressions safely."""
            try:
                # Safe evaluation with limited scope
                allowed_names = {
                    "abs": abs, "max": max, "min": min,
                    "sum": sum, "len": len
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return str(result)
            except:
                return "Error: Invalid expression"
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            # Implementation would connect to search API
            return f"Search results for: {query}"
        
        return [calculate, search]
    
    def _create_agent(self):
        """Create ReAct agent with explicit reasoning loop."""
        
        # ReAct prompt template
        template = """You are an AI assistant that helps users by thinking step by step.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: your reasoning about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
                "tool_names": ", ".join([t.name for t in self.tools])
            }
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def run(self, query: str) -> str:
        """Execute agent with query."""
        return self.agent.invoke({"input": query})

# Usage
agent = ReActAgent()
result = agent.run("What is 123 * 456, and what year was it calculated?")
```

---

### 2.3 Multi-Agent Systems (LangGraph)

#### **Concept**
Multi-agent systems coordinate multiple specialized agents through:
- **State sharing**: Common state all agents can access/modify
- **Routing**: Deciding which agent handles next step
- **Coordination**: Managing agent interactions and dependencies

#### **Math**
**State Graph Model:**
$$
G = (S, A, T, s_0, F)
$$
Where:
- $S$ = set of states
- $A$ = set of agents
- $T: S \times A \rightarrow S$ = transition function
- $s_0$ = initial state
- $F \subseteq S$ = final/terminal states

**Agent Selection Policy:**
$$
\pi(s_t) = \arg\max_{a \in A} Q(s_t, a)
$$
Where $Q(s, a)$ estimates the value of agent $a$ in state $s$.

**Convergence Criteria:**
$$
\text{Terminate when: } s_t \in F \text{ or } t > T_{max}
$$

#### **Why It's Good**
1. **Specialization**: Each agent masters one domain
2. **Scalability**: Add new capabilities by adding agents
3. **Robustness**: Single agent failure doesn't crash system
4. **Parallelism**: Independent agents run concurrently
5. **Maintainability**: Modular design enables isolated updates

#### **Implementation**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    research_results: dict
    draft_content: str
    final_output: str

class MultiAgentSystem:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._build_graph()
    
    def _build_graph(self):
        """Build multi-agent workflow graph."""
        
        # Add agent nodes
        self.workflow.add_node("researcher", self._research_agent)
        self.workflow.add_node("writer", self._writer_agent)
        self.workflow.add_node("editor", self._editor_agent)
        self.workflow.add_node("critic", self._critic_agent)
        
        # Define transitions (edges)
        self.workflow.add_conditional_edges(
            "researcher",
            self._should_continue_research,
            {"continue": "researcher", "write": "writer"}
        )
        
        self.workflow.add_edge("writer", "editor")
        
        self.workflow.add_conditional_edges(
            "editor",
            self._needs_revision,
            {"revise": "writer", "critique": "critic"}
        )
        
        self.workflow.add_conditional_edges(
            "critic",
            self._is_approved,
            {"approve": END, "revise": "writer"}
        )
        
        # Set entry point
        self.workflow.set_entry_point("researcher")
        
        self.app = self.workflow.compile()
    
    def _research_agent(self, state: AgentState):
        """Agent 1: Gather information."""
        # Research logic here
        return {
            "research_results": {"data": "research data"},
            "messages": ["Research completed"]
        }
    
    def _writer_agent(self, state: AgentState):
        """Agent 2: Draft content."""
        return {
            "draft_content": "Draft based on research",
            "messages": ["Draft created"]
        }
    
    def _editor_agent(self, state: AgentState):
        """Agent 3: Edit and refine."""
        return {"messages": ["Editing complete"]}
    
    def _critic_agent(self, state: AgentState):
        """Agent 4: Final review."""
        return {
            "final_output": state["draft_content"],
            "messages": ["Content approved"]
        }
    
    def _should_continue_research(self, state):
        """Decision function: continue research or move to writing."""
        if len(state["research_results"]) < 3:
            return "continue"
        return "write"
    
    def _needs_revision(self, state):
        """Decision function: needs more editing or move to critique."""
        # Logic to determine quality
        return "critique"
    
    def _is_approved(self, state):
        """Decision function: approved or needs revision."""
        return "approve"
    
    def run(self, initial_input: str):
        """Execute multi-agent workflow."""
        initial_state = {
            "messages": [initial_input],
            "research_results": {},
            "draft_content": "",
            "final_output": ""
        }
        return self.app.invoke(initial_state)

# Usage
system = MultiAgentSystem()
result = system.run("Write an article about AI advancements")
```

---

## 3. Prompt Engineering Deep Dive

### 3.1 Few-Shot Prompting

#### **Concept**
Few-shot prompting provides examples of input-output pairs to guide the LLM's behavior. The LLM learns the pattern from examples rather than explicit instructions.

#### **Math**
**In-Context Learning as Bayesian Inference:**
$$
P(Y|X, C) \propto P(Y|X) \cdot P(C|X, Y)
$$
Where:
- $X$ = input query
- $Y$ = desired output
- $C$ = context (examples)

**Optimal Number of Shots:**
Empirical studies show:
- 1-3 shots: Good for simple tasks
- 4-8 shots: Optimal for most tasks
- >8 shots: Diminishing returns, context limit issues

**Example Selection Impact:**
$$
\text{Performance} \propto \text{Similarity}(\text{example}, \text{query}) + \text{Diversity}(\text{examples})
$$

#### **Why It's Good**
1. **No Training Required**: Works with any pre-trained model
2. **Flexible**: Change behavior by changing examples
3. **Effective**: Often matches fine-tuned performance
4. **Interpretable**: Examples show expected behavior
5. **Universal**: Works across all LLM providers

#### **Implementation**
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

class FewShotEngine:
    def __init__(self, task_description: str):
        self.task = task_description
        self.examples = []
    
    def add_example(self, input_text: str, output_text: str):
        """Add training example with mathematical weighting."""
        self.examples.append({
            "input": input_text,
            "output": output_text,
            "weight": 1.0  # Can be adjusted for importance
        })
    
    def build_prompt(self, query: str) -> str:
        """Build few-shot prompt with optimal formatting."""
        
        # Example template
        example_template = """
Input: {input}
Output: {output}
"""
        
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template=example_template
        )
        
        # Select top-k most relevant examples
        selected_examples = self._select_examples(query, k=3)
        
        # Build few-shot template
        few_shot_prompt = FewShotPromptTemplate(
            examples=selected_examples,
            example_prompt=example_prompt,
            prefix=self.task,
            suffix="\nInput: {input}\nOutput:",
            input_variables=["input"]
        )
        
        return few_shot_prompt.format(input=query)
    
    def _select_examples(self, query: str, k: int):
        """Select k most similar examples using embedding similarity."""
        if len(self.examples) <= k:
            return self.examples
        
        # Simple selection: could use embeddings for similarity
        # Return most recent/high-weight examples
        sorted_examples = sorted(
            self.examples,
            key=lambda x: x.get("weight", 1.0),
            reverse=True
        )
        return sorted_examples[:k]

# Usage - Sentiment Classification
engine = FewShotEngine(
    "Classify the sentiment of the text as positive, negative, or neutral."
)

engine.add_example(
    "This product exceeded my expectations!",
    "positive"
)
engine.add_example(
    "Terrible quality, broke after one day.",
    "negative"
)
engine.add_example(
    "It's an okay product, nothing special.",
    "neutral"
)

prompt = engine.build_prompt("Absolutely love this, best purchase ever!")
```

---

### 3.2 Chain-of-Thought (CoT) Prompting

#### **Concept**
CoT prompting explicitly asks the model to show its reasoning step-by-step before giving the final answer. This significantly improves performance on complex reasoning tasks.

#### **Math**
**Decomposition Strategy:**
For problem $P$ with solution $S$:
$$P \rightarrow \{S_1, S_2, \ldots, S_n\} \rightarrow S$$

**Reasoning Path Probability:**
$$
P(S|P) = \prod_{i=1}^{n} P(S_i|S_{i-1}, \ldots, S_1, P)
$$

**Self-Consistency (CoT-SC):**
Generate $k$ reasoning paths, take majority vote:
$$\hat{S} = \arg\max_{S} \sum_{i=1}^{k} \mathbb{1}(S_i = S)$$

**Accuracy Improvement:**
Studies show 10-40% accuracy improvement on reasoning tasks with CoT.

#### **Why It's Good**
1. **Error Localization**: Can identify where reasoning failed
2. **Better Accuracy**: Forces systematic thinking
3. **Explainability**: Shows how answer was derived
4. **Debugging**: Can prompt to fix specific step
5. **Versatile**: Works across math, logic, planning tasks

#### **Implementation**
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class ChainOfThought:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatOpenAI(model=llm_model)
    
    def solve(self, problem: str, steps: int = None) -> dict:
        """Solve problem with explicit step-by-step reasoning."""
        
        # Mathematical CoT template
        cot_template = """Solve the following problem step by step.
        Show your reasoning clearly before giving the final answer.
        
        Problem: {problem}
        
        Let's work through this:
        Step 1:"""
        
        prompt = PromptTemplate(
            template=cot_template,
            input_variables=["problem"]
        )
        
        # Generate reasoning
        response = self.llm.invoke(prompt.format(problem=problem))
        
        # Parse reasoning and answer
        parsed = self._parse_cot_response(response.content)
        
        return {
            "reasoning": parsed["reasoning"],
            "answer": parsed["answer"],
            "confidence": self._calculate_confidence(parsed)
        }
    
    def solve_with_self_consistency(self, problem: str, k: int = 5) -> dict:
        """Self-consistency: multiple paths, majority vote."""
        
        answers = []
        reasoning_paths = []
        
        for i in range(k):
            result = self.solve(problem)
            answers.append(result["answer"])
            reasoning_paths.append(result["reasoning"])
        
        # Majority vote
        from collections import Counter
        vote_counts = Counter(answers)
        final_answer = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[final_answer] / k
        
        return {
            "answer": final_answer,
            "confidence": confidence,
            "reasoning_paths": reasoning_paths,
            "vote_distribution": dict(vote_counts)
        }
    
    def _parse_cot_response(self, response: str) -> dict:
        """Parse reasoning steps and final answer."""
        lines = response.strip().split("\n")
        
        reasoning = []
        answer = ""
        
        for line in lines:
            if line.startswith("Step"):
                reasoning.append(line)
            elif "Final Answer" in line or "Answer:" in line:
                answer = line.split(":")[-1].strip()
        
        if not answer:
            answer = lines[-1]  # Assume last line is answer
        
        return {"reasoning": "\n".join(reasoning), "answer": answer}
    
    def _calculate_confidence(self, parsed: dict) -> float:
        """Estimate confidence based on reasoning quality."""
        # Simple heuristic: more steps = more careful reasoning
        steps = len(parsed["reasoning"].split("\n"))
        base_conf = min(0.5 + (steps * 0.05), 0.95)
        return base_conf

# Usage
cot = ChainOfThought()

# Math problem
result = cot.solve("""
If a train travels 120 km in 2 hours, and then continues at the same speed 
for another 3 hours, how far does it travel in total?
""")

print(f"Reasoning: {result['reasoning']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")

# Self-consistency for critical decisions
consistent_result = cot.solve_with_self_consistency(
    "Calculate 15% of $84.50", 
    k=3
)
print(f"Final answer (self-consistency): {consistent_result['answer']}")
print(f"Vote confidence: {consistent_result['confidence']:.2f}")
```

---

## 4. SQL Window Functions Deep Dive

### 4.1 Window Function Fundamentals

#### **Concept**
Window functions perform calculations across a set of rows ("window") related to the current row, without collapsing groups like aggregate functions do.

#### **Math**
**Window Definition:**
$$
\text{Window} = \text{PARTITION BY} \times \text{ORDER BY} \times \text{FRAME}
$$

**Frame Specification:**
$$
\text{Frame} = [\text{start}, \text{end}] \text{ relative to current row}
$$

Common frames:
- `ROWS UNBOUNDED PRECEDING`: $[-\infty, 0]$
- `ROWS 2 PRECEDING`: $[-2, 0]$
- `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: $[-\infty, \text{current value}]$

**Running Total:**
$$
\text{RunningSum}_i = \sum_{j=1}^{i} x_j
$$

**Moving Average (k-period):**
$$
\text{MA}_i = \frac{1}{k} \sum_{j=i-k+1}^{i} x_j
$$

**Row Number:**
$$
\text{RowNumber} = \text{RANK of row in ordered partition}
$$

**Percent Rank:**
$$
\text{PercentRank} = \frac{\text{Rank} - 1}{\text{Total Rows} - 1}
$$

#### **Why It's Good**
1. **Efficiency**: Single pass through data
2. **Simplicity**: No self-joins or subqueries needed
3. **Power**: Complex analytics in one query
4. **Standards**: ANSI SQL standard, portable across databases
5. **Performance**: Optimized by query planners

#### **Implementation**
```python
import sqlite3

class SQLWindowFunctions:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create sample data for window function demonstrations."""
        cursor = self.conn.cursor()
        
        # Sales data with date, region, amount
        cursor.execute("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                date TEXT,
                region TEXT,
                salesperson TEXT,
                amount REAL
            )
        """)
        
        # Insert sample data
        data = [
            ("2024-01-01", "North", "Alice", 1000),
            ("2024-01-01", "South", "Bob", 1500),
            ("2024-01-02", "North", "Alice", 1200),
            ("2024-01-02", "South", "Bob", 800),
            ("2024-01-03", "North", "Alice", 900),
            ("2024-01-03", "South", "Bob", 2000),
            ("2024-01-03", "East", "Carol", 1300),
        ]
        
        cursor.executemany(
            "INSERT INTO sales (date, region, salesperson, amount) VALUES (?, ?, ?, ?)",
            data
        )
        self.conn.commit()
    
    def running_total(self):
        """Calculate running total (cumulative sum)."""
        query = """
            SELECT 
                date,
                region,
                salesperson,
                amount,
                SUM(amount) OVER (
                    ORDER BY date, id
                    ROWS UNBOUNDED PRECEDING
                ) as running_total,
                SUM(amount) OVER (
                    PARTITION BY region
                    ORDER BY date
                    ROWS UNBOUNDED PRECEDING
                ) as regional_running_total
            FROM sales
            ORDER BY date, id
        """
        return self._execute_and_format(query)
    
    def ranking_functions(self):
        """Demonstrate ROW_NUMBER, RANK, DENSE_RANK."""
        query = """
            SELECT 
                salesperson,
                SUM(amount) as total_sales,
                ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) as row_num,
                RANK() OVER (ORDER BY SUM(amount) DESC) as rank_num,
                DENSE_RANK() OVER (ORDER BY SUM(amount) DESC) as dense_rank,
                NTILE(4) OVER (ORDER BY SUM(amount) DESC) as quartile,
                PERCENT_RANK() OVER (ORDER BY SUM(amount) DESC) as pct_rank
            FROM sales
            GROUP BY salesperson
            ORDER BY total_sales DESC
        """
        return self._execute_and_format(query)
    
    def moving_calculations(self, window_size: int = 2):
        """Calculate moving averages and sums."""
        query = f"""
            SELECT 
                date,
                amount,
                AVG(amount) OVER (
                    ORDER BY date
                    ROWS {window_size} PRECEDING
                ) as moving_avg_{window_size}day,
                LAG(amount, 1) OVER (ORDER BY date) as prev_day,
                LEAD(amount, 1) OVER (ORDER BY date) as next_day,
                amount - LAG(amount, 1) OVER (ORDER BY date) as day_over_day_change,
                FIRST_VALUE(amount) OVER (ORDER BY date) as first_amount,
                LAST_VALUE(amount) OVER (
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as last_amount
            FROM sales
            ORDER BY date
        """
        return self._execute_and_format(query)
    
    def top_n_per_group(self, n: int = 2):
        """Select top N per group using window functions."""
        query = f"""
            SELECT *
            FROM (
                SELECT 
                    region,
                    salesperson,
                    amount,
                    ROW_NUMBER() OVER (
                        PARTITION BY region 
                        ORDER BY amount DESC
                    ) as rn
                FROM sales
            )
            WHERE rn <= {n}
            ORDER BY region, rn
        """
        return self._execute_and_format(query)
    
    def _execute_and_format(self, query: str) -> dict:
        """Execute query and return formatted results."""
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "query": query
        }

# Usage
demo = SQLWindowFunctions()

# Running totals
print("=== Running Totals ===")
result = demo.running_total()
print(f"Columns: {result['columns']}")
for row in result['rows'][:3]:
    print(row)

# Rankings
print("\n=== Rankings ===")
result = demo.ranking_functions()
print(f"Columns: {result['columns']}")
for row in result['rows']:
    print(row)

# Moving calculations
print("\n=== Moving Calculations ===")
result = demo.moving_calculations(window_size=2)
for row in result['rows']:
    print(row)
```

---

## 5. System Design Deep Dive

### 5.1 RAG System Architecture

#### **Concept**
RAG combines retrieval (finding relevant documents) with generation (LLM producing answers). The architecture involves multiple components working together.

#### **Math**
**End-to-End Latency:**
$$
T_{total} = T_{query} + T_{embed} + T_{search} + T_{rerank} + T_{generate}
$$

Typical breakdown:
- Query parsing: 10-50ms
- Embedding generation: 100-500ms
- Vector search: 10-100ms
- Re-ranking: 50-200ms
- LLM generation: 500-5000ms

**Throughput Calculation:**
$$
\text{Throughput} = \frac{1}{T_{total}} \times \text{Parallel Workers}
$$

**Caching Hit Rate:**
$$
\text{Effective Latency} = H \times T_{cache} + (1-H) \times T_{compute}
$$
Where $H$ = cache hit rate, typically 0.6-0.9

#### **Why It's Good**
1. **Accuracy**: Reduces hallucinations with factual grounding
2. **Freshness**: Update knowledge base without retraining
3. **Attribution**: Can cite sources for transparency
4. **Cost**: Smaller context = lower inference costs
5. **Privacy**: Keep sensitive data on-premise

#### **Implementation**
```python
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class RAGMetrics:
    query_time_ms: float
    embed_time_ms: float
    search_time_ms: float
    generate_time_ms: float
    total_time_ms: float
    tokens_used: int

class ProductionRAGSystem:
    def __init__(self):
        self.cache = {}  # Query -> Result cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def query(self, question: str, use_cache: bool = True) -> dict:
        """Execute RAG query with full observability."""
        
        metrics = {}
        start_total = time.time()
        
        # 1. Check cache
        if use_cache and question in self.cache:
            self.cache_hits += 1
            return {
                "answer": self.cache[question],
                "source": "cache",
                "metrics": {"cache_hit": True}
            }
        self.cache_misses += 1
        
        # 2. Query understanding
        t0 = time.time()
        parsed_query = self._parse_query(question)
        metrics['query_time_ms'] = (time.time() - t0) * 1000
        
        # 3. Generate embedding
        t0 = time.time()
        query_embedding = await self._embed(parsed_query)
        metrics['embed_time_ms'] = (time.time() - t0) * 1000
        
        # 4. Vector search (k=10)
        t0 = time.time()
        candidates = await self._vector_search(query_embedding, k=10)
        metrics['search_time_ms'] = (time.time() - t0) * 1000
        
        # 5. Re-rank with cross-encoder
        t0 = time.time()
        reranked = self._rerank(candidates, parsed_query)
        top_k = reranked[:5]  # Use top 5
        metrics['rerank_time_ms'] = (time.time() - t0) * 1000
        
        # 6. Generate answer
        t0 = time.time()
        context = self._build_context(top_k)
        answer, tokens = await self._generate(question, context)
        metrics['generate_time_ms'] = (time.time() - t0) * 1000
        
        # 7. Update cache
        if use_cache:
            self.cache[question] = answer
        
        metrics['total_time_ms'] = (time.time() - start_total) * 1000
        metrics['tokens_used'] = tokens
        
        return {
            "answer": answer,
            "sources": top_k,
            "metrics": metrics,
            "source": "computed"
        }
    
    def get_cache_stats(self) -> dict:
        """Calculate cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return {"hit_rate": 0}
        
        hit_rate = self.cache_hits / total
        return {
            "hit_rate": hit_rate,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.cache),
            "estimated_latency_reduction_ms": 
                hit_rate * 1000  # Assuming 1s avg compute time
        }
    
    def _parse_query(self, question: str) -> str:
        """Parse and expand query."""
        return question  # Simplified
    
    async def _embed(self, text: str):
        """Generate embedding."""
        # Embedding model call
        pass
    
    async def _vector_search(self, embedding, k: int):
        """Search vector database."""
        pass
    
    def _rerank(self, candidates, query):
        """Re-rank with cross-encoder."""
        return candidates  # Simplified
    
    def _build_context(self, documents):
        """Build context from documents."""
        return "\n\n".join([d.content for d in documents])
    
    async def _generate(self, question: str, context: str):
        """Generate answer with LLM."""
        # LLM generation
        return "Answer", 100
```

---

## 6. LLM Evaluation Metrics Deep Dive

### 6.1 RAGAS Metrics

#### **Concept**
RAGAS (Retrieval-Augmented Generation Assessment) provides metrics to evaluate RAG systems without human annotations.

#### **Math**
**Faithfulness:**
Measures if answer is grounded in context:
$$
\text{Faithfulness} = \frac{|\text{Claims in Answer} \cap \text{Supported by Context}|}{|\text{Claims in Answer}|}
$$

**Answer Relevance:**
$$
\text{Relevance} = \frac{1}{n} \sum_{i=1}^{n} \cos(\text{Emb}(q), \text{Emb}(a_i))
$$
Where $q$ = question, $a_i$ = sentences in answer

**Context Precision:**
$$
\text{Precision@k} = \frac{\text{Relevant Retrieved}}{k}
$$

**Context Recall:**
$$
\text{Recall} = \frac{|\text{Ground Truth} \cap \text{Retrieved}|}{|\text{Ground Truth}|}
$$

**Context Entity Recall:**
$$
\text{Entity Recall} = \frac{|\text{Entities in GT} \cap \text{Entities in Retrieved}|}{|\text{Entities in GT}|}
$$

#### **Why It's Good**
1. **Automated**: No human labels needed
2. **Comprehensive**: Evaluates all RAG components
3. **Actionable**: Pinpoints which component needs improvement
4. **Standardized**: Industry-standard metric suite
5. **Fast**: LLM-based evaluation, no training required

#### **Implementation**
```python
from typing import List, Dict
import numpy as np

class RAGASEvaluator:
    """RAGAS metrics implementation."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.embedding_model = None  # Initialize embedding model
    
    def evaluate(self, question: str, answer: str, 
                 contexts: List[str], ground_truth: str = None) -> Dict:
        """Calculate all RAGAS metrics."""
        
        metrics = {}
        
        # 1. Faithfulness
        metrics['faithfulness'] = self._calculate_faithfulness(
            answer, contexts
        )
        
        # 2. Answer Relevance
        metrics['answer_relevancy'] = self._calculate_answer_relevance(
            question, answer
        )
        
        # 3. Context Relevance
        metrics['context_relevancy'] = self._calculate_context_relevance(
            question, contexts
        )
        
        # 4. Context Precision
        if ground_truth:
            metrics['context_precision'] = self._calculate_context_precision(
                contexts, ground_truth
            )
            
            # 5. Context Recall
            metrics['context_recall'] = self._calculate_context_recall(
                contexts, ground_truth
            )
        
        # Overall RAGAS score
        metrics['ragas_score'] = np.mean([
            metrics.get('faithfulness', 0),
            metrics.get('answer_relevancy', 0),
            metrics.get('context_relevancy', 0)
        ])
        
        return metrics
    
    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Check if answer claims are supported by context."""
        
        # Step 1: Extract claims from answer
        claims_prompt = f"""
        Extract all factual claims from this answer as a list.
        Answer: {answer}
        Claims:"""
        
        claims_response = self.llm.generate(claims_prompt)
        claims = [c.strip() for c in claims_response.split("\n") if c.strip()]
        
        if not claims:
            return 1.0  # No claims = vacuously faithful
        
        # Step 2: Verify each claim
        verified_claims = 0
        context_combined = "\n".join(contexts)
        
        for claim in claims:
            verify_prompt = f"""
            Context: {context_combined}
            Claim: {claim}
            Is this claim supported by the context? Answer Yes or No.
            """
            
            verification = self.llm.generate(verify_prompt).strip().lower()
            if "yes" in verification:
                verified_claims += 1
        
        return verified_claims / len(claims)
    
    def _calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Calculate semantic relevance of answer to question."""
        
        # Break answer into sentences
        sentences = answer.split(". ")
        
        # Get embeddings
        q_emb = self._embed(question)
        
        similarities = []
        for sent in sentences:
            if sent.strip():
                s_emb = self._embed(sent)
                sim = self._cosine_similarity(q_emb, s_emb)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_context_relevance(self, question: str, 
                                     contexts: List[str]) -> float:
        """Calculate how relevant retrieved contexts are to question."""
        
        q_emb = self._embed(question)
        
        # Extract sentences from all contexts
        all_sentences = []
        for ctx in contexts:
            all_sentences.extend(ctx.split(". "))
        
        # Calculate relevance for each sentence
        relevant_count = 0
        for sent in all_sentences:
            if sent.strip():
                s_emb = self._embed(sent)
                sim = self._cosine_similarity(q_emb, s_emb)
                if sim > 0.7:  # Threshold for relevance
                    relevant_count += 1
        
        return relevant_count / len(all_sentences) if all_sentences else 0
    
    def _calculate_context_precision(self, contexts: List[str], 
                                    ground_truth: str) -> float:
        """Precision: relevant chunks / total chunks retrieved."""
        
        gt_emb = self._embed(ground_truth)
        
        relevant = 0
        for ctx in contexts:
            ctx_emb = self._embed(ctx)
            sim = self._cosine_similarity(gt_emb, ctx_emb)
            if sim > 0.8:  # High similarity threshold
                relevant += 1
        
        return relevant / len(contexts) if contexts else 0
    
    def _calculate_context_recall(self, contexts: List[str], 
                                 ground_truth: str) -> float:
        """Recall: relevant information covered / total relevant info."""
        
        # Extract key information from ground truth
        info_extraction_prompt = f"""
        Extract key facts/information from:
        {ground_truth}
        
        List each key fact on a new line.
        """
        
        key_facts = self.llm.generate(info_extraction_prompt).split("\n")
        key_facts = [f.strip() for f in key_facts if f.strip()]
        
        if not key_facts:
            return 1.0
        
        # Check which facts are covered in contexts
        context_combined = "\n".join(contexts)
        
        covered_facts = 0
        for fact in key_facts:
            check_prompt = f"""
            Context: {context_combined}
            Fact: {fact}
            Is this fact present or implied in the context? Yes/No
            """
            result = self.llm.generate(check_prompt).strip().lower()
            if "yes" in result:
                covered_facts += 1
        
        return covered_facts / len(key_facts)
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Implementation using embedding model
        return np.random.randn(768)  # Placeholder
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Usage
# evaluator = RAGASEvaluator(llm_client)
# metrics = evaluator.evaluate(
#     question="What is RAG?",
#     answer="RAG is retrieval-augmented generation...",
#     contexts=["RAG combines retrieval with generation..."],
#     ground_truth="Retrieval-Augmented Generation (RAG) is..."
# )
```

---

## 7. Summary: Key Mathematical Formulas

### RAG & Embeddings
- Cosine Similarity: $\frac{a \cdot b}{||a|| \cdot ||b||}$
- Euclidean Distance: $\sqrt{\sum (a_i - b_i)^2}$
- Optimal Chunks: $\lceil N / (S \times (1-r)) \rceil$

### Prompt Engineering
- CoT Self-Consistency: $\arg\max_{S} \sum_{i=1}^{k} \mathbb{1}(S_i = S)$
- In-Context Learning: $P(Y|X, C) \propto P(Y|X) \cdot P(C|X, Y)$

### Window Functions
- Moving Average: $\frac{1}{k} \sum_{j=i-k+1}^{i} x_j$
- Percent Rank: $\frac{\text{Rank} - 1}{\text{Total Rows} - 1}$

### System Design
- Total Latency: $T_{query} + T_{embed} + T_{search} + T_{generate}$
- Cache Hit Rate: $H \times T_{cache} + (1-H) \times T_{compute}$

### Evaluation
- Faithfulness: $\frac{|\text{Claims} \cap \text{Supported}|}{|\text{Claims}|}$
- Context Recall: $\frac{|\text{GT} \cap \text{Retrieved}|}{|\text{GT}|}$
