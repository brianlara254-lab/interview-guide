# Retrieval-Augmented Generation (RAG) - Comprehensive Guide

A detailed breakdown of RAG systems, implementation strategies, optimization techniques, and advanced patterns with up-to-date library methods and step-by-step explanations.

**Last Updated**: 2025 | **LangChain Version**: 0.3.x | **LangChain Core**: 1.2.16

---

## Table of Contents
1. [RAG Fundamentals](#rag-fundamentals)
2. [Document Ingestion & Processing](#document-ingestion--processing)
3. [Embedding Models & Vector Stores](#embedding-models--vector-stores)
4. [Retrieval Strategies](#retrieval-strategies)
5. [Advanced RAG Patterns](#advanced-rag-patterns)
6. [Evaluation & Optimization](#evaluation--optimization)

---

## RAG Fundamentals

### Question 1: What is RAG and Why Use It?

#### Concept Breakdown

**Definition**: Retrieval-Augmented Generation (RAG) enhances LLM outputs by retrieving relevant information from external knowledge sources and incorporating it into the generation context.

**The Problem RAG Solves:**

```
┌──────────────────────────────────────────────────────────────┐
│              WITHOUT RAG vs WITH RAG                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  WITHOUT RAG                                                 │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  User: "What are Q4 2024 sales?"                      │   │
│  │  Model: "I don't have access to current data..."      │   │
│  │         OR hallucinates numbers!                      │   │
│  │  Problems:                                            │   │
│  │  • Knowledge cutoff date                              │   │
│  │  • No access to private data                          │   │
│  │  • Hallucination risk                                 │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  WITH RAG                                                    │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  User: "What are Q4 2024 sales?"                      │   │
│  │       ↓                                               │   │
│  │  System: Retrieves Q4 2024 report from vector DB      │   │
│  │       ↓                                               │   │
│  │  Model: "According to the Q4 2024 report, sales       │   │
│  │         were $45.2M, up 15% from Q3..."               │   │
│  │  Benefits:                                            │   │
│  │  • Current information                                │   │
│  │  • Grounded in sources                                │   │
│  │  • Verifiable answers                                 │   │
│  └───────────────────────────────────────────────────────┘   │   
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: How RAG Works

**Step 1: Ingestion Phase (Pre-processing)**
```python
# Documents → Chunks → Embeddings → Vector Store

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. Load documents
loader = PyPDFLoader("annual_report_2024.pdf")
documents = loader.load()  # List of Document objects

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap for context continuity
    separators=["\n\n", "\n", " ", ""]  # Split hierarchy
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Latest embedding model
    dimensions=3072                    # High-quality embeddings
)

# 4. Store in vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Save to disk
    collection_name="annual_reports"
)
```

**Step 2: Retrieval Phase (Runtime)**
```python
# Query → Embedding → Similarity Search → Relevant Chunks

user_query = "What were the Q4 sales figures?"

# Embed the query
query_embedding = embeddings.embed_query(user_query)

# Retrieve similar documents
retriever = vectorstore.as_retriever(
    search_type="similarity",      # "mmr" or "similarity_score_threshold"
    search_kwargs={"k": 5}         # Return top 5 chunks
)

relevant_docs = retriever.invoke(user_query)
# Returns: List[Document] with page_content and metadata
```

**Step 3: Augmentation & Generation**
```python
# Chunks + Query → Augmented Prompt → LLM → Response

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1  # Low for factual tasks
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
    retriever=retriever,
    return_source_documents=True,  # Include citations
    verbose=True
)

# Execute query
result = qa_chain.invoke({"query": user_query})

# Result structure:
# {
#     "result": "Q4 2024 sales were $45.2M...",
#     "source_documents": [Document1, Document2, ...]
# }
```

**Result Simulation:**

```
User Query: "What were Q4 2024 sales?"

Step 1: Query Embedding
- "What were Q4 2024 sales?" → 3072-dimensional vector
- Similarity to stored chunks calculated

Step 2: Retrieved Chunks (Top 3)
┌─────────────────────────────────────────────────────────────┐
│ Chunk 1 (Score: 0.94)                                       │
│ "Q4 2024 Financial Highlights: Total revenue reached $45.2M │
│  representing 15% YoY growth..."                            │
├─────────────────────────────────────────────────────────────┤
│ Chunk 2 (Score: 0.89)                                       │
│ "Quarterly Breakdown: Q4 showed strongest performance..."   │
├─────────────────────────────────────────────────────────────┤
│ Chunk 3 (Score: 0.85)                                       │
│ "Sales by Region: North America contributed $28M..."        │
└─────────────────────────────────────────────────────────────┘

Step 3: Augmented Prompt
"""
Context information:
[Q4 2024 Financial Highlights...]
[Quarterly Breakdown...]
[Sales by Region...]

Question: What were Q4 2024 sales?
Answer based only on the context provided.
"""

Step 4: Generated Response
"According to the Q4 2024 Financial Highlights, total revenue 
reached $45.2 million, representing 15% year-over-year growth."

Sources: [Citations to specific chunks]
```

---

### Question 2: What are the Different RAG Architectures?

#### Concept Breakdown

**RAG Architecture Patterns:**

| Architecture | Description | Use Case | Complexity |
|-------------|-------------|----------|------------|
| **Naive RAG** | Basic retrieve-then-generate | Simple Q&A | Low |
| **Advanced RAG** | Pre/post retrieval optimization | Production systems | Medium |
| **Modular RAG** | Flexible, interchangeable components | Complex workflows | High |
| **Agentic RAG** | Multi-step reasoning with tools | Research, analysis | Very High |

#### Step-by-Step: Architecture Comparison

**Pattern 1: Naive RAG**
```python
from langchain.chains import RetrievalQA

# Simplest form - direct retrieval and generation
naive_rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Flow:
# Query → Retrieve → Stuff into prompt → Generate
```

**Pattern 2: Advanced RAG with Query Rewriting**
```python
from langchain.retrievers import MultiQueryRetriever

# Problem: User query may not match document wording
# Solution: Generate multiple query variations

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Example transformation:
# User: "tell me about cloud costs"
# Generated queries:
#   - "What are cloud computing expenses?"
#   - "How much does AWS cost?"
#   - "Cloud infrastructure pricing"
# Retrieve for all → Deduplicate → Rerank
```

**Pattern 3: Modular RAG with Reranking**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Step 1: Initial broad retrieval (recall-focused)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Step 2: Rerank for precision
cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

# Combine into retrieval pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Flow:
# Query → Retrieve 20 chunks → Rerank → Top 5 → Generate
```

**Pattern 4: Agentic RAG**
```python
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define retrieval tool
retrieval_tool = Tool(
    name="document_search",
    func=retriever.invoke,
    description="Search internal documents for information"
)

# Define web search tool
web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the internet for current information"
)

# Create agent with multiple tools
tools = [retrieval_tool, web_search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use tools as needed."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Agent decides:
# - Use document_search for company-specific queries
# - Use web_search for current events
# - Combine both for comprehensive answers
```

---

## Document Ingestion & Processing

### Question 3: How to Choose Chunk Size and Overlap?

#### Concept Breakdown

**Chunking Trade-offs:**

```
Small Chunks (256-512 tokens)     Large Chunks (1024-2048 tokens)
├─ More precise retrieval         ├─ More context per chunk
├─ Lower semantic coherence       ├─ Better context preservation
├─ More chunks = higher cost      ├─ May include irrelevant info
└─ Better for specific facts      └─ Better for summaries

Overlap helps maintain continuity between chunks
```

#### Step-by-Step: Chunking Strategy Selection

**Step 1: Analyze Document Type**
```python
document_analysis = {
    "code": {
        "chunk_size": 500,      # Functions/methods
        "overlap": 50,
        "splitter": Language.APYTHON,  # Syntax-aware
        "strategy": "Keep functions together"
    },
    "legal_contracts": {
        "chunk_size": 1000,     # Paragraphs/clauses
        "overlap": 200,
        "splitter": RecursiveCharacterTextSplitter,
        "strategy": "Preserve clause boundaries"
    },
    "research_papers": {
        "chunk_size": 1500,     # Sections
        "overlap": 300,
        "splitter": RecursiveCharacterTextSplitter,
        "strategy": "Keep sections together"
    },
    "conversation": {
        "chunk_size": 1000,
        "overlap": 200,
        "splitter": RecursiveCharacterTextSplitter,
        "strategy": "Keep speaker turns"
    }
}
```

**Step 2: Implement Hierarchical Chunking**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent-Child Chunking (Latest LangChain pattern)
# Large parent chunks for context, small child chunks for retrieval

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ". ", " ", ""]
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

# Store relationship between parent and child chunks
# Retrieve small chunks but provide parent context
```

**Step 3: Test and Measure**
```python
def evaluate_chunking(docs, chunk_sizes, overlap_sizes):
    """
    Evaluate different chunking strategies
    """
    results = []
    
    for chunk_size in chunk_sizes:
        for overlap in overlap_sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            chunks = splitter.split_documents(docs)
            
            # Metrics
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
            num_chunks = len(chunks)
            
            # Retrieval test
            test_queries = ["test query 1", "test query 2"]
            avg_relevance = test_retrieval_quality(chunks, test_queries)
            
            results.append({
                "chunk_size": chunk_size,
                "overlap": overlap,
                "num_chunks": num_chunks,
                "avg_size": avg_chunk_size,
                "relevance": avg_relevance
            })
    
    return results

# Run evaluation
results = evaluate_chunking(
    docs=documents,
    chunk_sizes=[256, 512, 1024],
    overlap_sizes=[0, 50, 100, 200]
)

# Choose based on relevance score and cost constraints
```

---

## Embedding Models & Vector Stores

### Question 4: Which Embedding Model Should I Use?

#### Concept Breakdown

**Embedding Model Comparison (2024):**

| Model | Dimensions | Context | Strengths | Cost |
|-------|-----------|---------|-----------|------|
| **text-embedding-3-large** | 3072 | 8192 | Best quality | $$ |
| **text-embedding-3-small** | 1536 | 8192 | Fast, cheap | $ |
| **text-embedding-ada-002** | 1536 | 8192 | Legacy, reliable | $ |
| **BAAI/bge-large-en** | 1024 | 512 | Open source | Free |
| **sentence-transformers/all-mpnet** | 768 | 512 | General purpose | Free |
| **Cohere embed-english-v3** | 1024 | 512 | Multilingual | $$ |
| **Voyage AI** | 1024 | 4096 | Domain-specific | $$$ |

#### Step-by-Step: Embedding Selection

**Step 1: Benchmark on Your Data**
```python
from langchain.evaluation import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

# Load evaluation dataset
dataset = load_dataset("your_domain_qa_pairs")

# Test different models
models = {
    "openai_large": OpenAIEmbeddings(model="text-embedding-3-large"),
    "openai_small": OpenAIEmbeddings(model="text-embedding-3-small"),
    "bge": HuggingFaceEmbeddings(model_name="BAAI/bge-large-en"),
    "mpnet": HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
}

results = {}

for name, embedding_model in models.items():
    # Encode questions and answers
    question_embeddings = embedding_model.embed_documents(dataset["questions"])
    answer_embeddings = embedding_model.embed_documents(dataset["answers"])
    
    # Calculate recall@k
    recall_scores = []
    for i, q_emb in enumerate(question_embeddings):
        similarities = cosine_similarity([q_emb], answer_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-5:]
        recall_scores.append(i in top_k_indices)
    
    results[name] = {
        "recall@5": np.mean(recall_scores),
        "avg_latency": measure_latency(embedding_model),
        "cost_per_1m": get_cost(embedding_model)
    }

# Select based on accuracy/cost tradeoff
```

**Step 2: Configure with LangChain**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Option 1: OpenAI (Best quality, higher cost)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # Optional: reduce dimensions for cost savings
    dimensions=1536,  # Down from 3072, still high quality
    api_key=os.getenv("OPENAI_API_KEY")
)

# Option 2: Local HuggingFace (Free, private)
local_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cuda"},  # Use GPU
    encode_kwargs={"normalize_embeddings": True}
)

# Option 3: Voyage AI (Domain-specific, best for enterprise)
voyage_embeddings = VoyageEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-2"  # or "voyage-law-2" for legal
)
```

---

### Question 5: Which Vector Store Should I Use?

#### Concept Breakdown

**Vector Store Comparison (2024):**

| Database | Type | Best For | Scaling | Features |
|----------|------|----------|---------|----------|
| **Pinecone** | Managed Cloud | Production | Auto | Metadata filtering, hybrid |
| **Weaviate** | Open/Managed | Enterprise | Horizontal | GraphQL, modular AI |
| **Chroma** | Embedded/Local | Development | Single node | Easy setup, Python-native |
| **Qdrant** | Open/Managed | Performance | Horizontal | Fast, Rust-based |
| **Milvus/Zilliz** | Open/Managed | Large scale | Distributed | Billion-scale |
| **pgvector** | PostgreSQL | Existing PG users | Vertical | SQL integration |
| **Redis** | In-memory | Real-time | Vertical | Low latency |

#### Step-by-Step: Vector Store Selection

**Step 1: Development with Chroma**
```python
from langchain_community.vectorstores import Chroma

# Local development - easiest setup
chroma_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Persistent storage
    collection_name="my_documents"
)

# Query
results = chroma_store.similarity_search(
    query="What are the sales figures?",
    k=5,
    filter={"source": "annual_report.pdf"}  # Metadata filtering
)
```

**Step 2: Production with Pinecone**
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if not exists
if "my-index" not in pc.list_indexes().names():
    pc.create_index(
        name="my-index",
        dimension=1536,  # Match embedding dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect and store
pinecone_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index",
    namespace="production"  # Logical separation
)

# Query with metadata filtering
results = pinecone_store.similarity_search(
    query="sales figures",
    k=5,
    filter={
        "year": {"$eq": 2024},
        "department": {"$in": ["sales", "finance"]}
    }
)
```

**Step 3: Enterprise with Weaviate**
```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

# Connect to Weaviate
client = weaviate.Client(
    url="https://my-cluster.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(
        api_key=os.getenv("WEAVIATE_API_KEY")
    )
)

# Store with rich metadata
weaviate_store = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    index_name="Documents",
    text_key="content",
    attributes=["source", "page", "author", "date"]
)

# Hybrid search (vector + BM25)
results = weaviate_store.hybrid_search(
    query="Q4 revenue",
    k=5,
    alpha=0.75  # Balance between semantic (1.0) and keyword (0.0)
)
```

---

## Retrieval Strategies

### Question 6: What are the Different Retrieval Methods?

#### Concept Breakdown

**Retrieval Methods Comparison:**

| Method | Description | Best For | Implementation |
|--------|-------------|----------|----------------|
| **Similarity** | Cosine/dot product similarity | General semantic search | `search_type="similarity"` |
| **MMR** | Maximal Marginal Relevance | Diverse results | `search_type="mmr"` |
| **Threshold** | Score cutoff | High precision | `search_type="similarity_score_threshold"` |
| **Hybrid** | Vector + BM25 | Keyword + semantic | Custom implementation |

#### Step-by-Step: Retrieval Implementation

**Method 1: Similarity Search (Default)**
```python
# Simple semantic similarity
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# How it works:
# 1. Embed query: "cloud costs" → vector q
# 2. For each chunk vector c, calculate: score = cos_sim(q, c)
# 3. Return top-k chunks with highest scores
```

**Method 2: MMR (Diversity-Aware)**
```python
from langchain.retrievers import MaxMarginalRelevanceRetriever

# Problem: Similarity search may return redundant results
# Solution: MMR balances relevance with diversity

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # Number to return
        "fetch_k": 20,    # Consider 20 candidates first
        "lambda_mult": 0.5  # Balance: 0=diverse, 1=relevant
    }
)

# Algorithm:
# 1. Retrieve fetch_k most similar chunks
# 2. Iteratively select chunks that maximize:
#    MMR = λ * Similarity(query, doc) - (1-λ) * max_sim(selected_docs, doc)
# 3. Continue until k chunks selected
```

**Method 3: Similarity with Threshold**
```python
# Only return results above confidence threshold
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.8  # Minimum similarity score
    }
)

# Use case: When you prefer "no results" over "bad results"
# If no chunks score above 0.8, return empty list
```

**Method 4: Hybrid Search (Vector + Keyword)**
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 for keyword matching
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Vector for semantic matching
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Equal weighting
)

# Results are reranked by weighted combination
# keyword_score * 0.5 + semantic_score * 0.5
```

---

## Advanced RAG Patterns

### Question 7: How to Implement Self-Querying RAG?

#### Concept Breakdown

**Self-Querying**: LLM converts natural language queries into structured queries (with filters) before retrieval.

**Use Case**: "What are the sales figures for Q4 2023 in the APAC region?"
- Extract: year=2023, quarter=Q4, region=APAC
- Apply filters before vector search

#### Step-by-Step: Self-Query Implementation

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Step 1: Define document metadata schema
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year the document refers to",
        type="integer"
    ),
    AttributeInfo(
        name="quarter",
        description="The fiscal quarter (Q1, Q2, Q3, Q4)",
        type="string"
    ),
    AttributeInfo(
        name="region",
        description="Geographic region (APAC, EMEA, Americas)",
        type="string"
    ),
    AttributeInfo(
        name="department",
        description="Company department",
        type="string"
    ),
    AttributeInfo(
        name="revenue",
        description="Whether the document mentions revenue",
        type="boolean"
    )
]

# Step 2: Create self-querying retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Financial reports and sales data",
    metadata_field_info=metadata_field_info,
    verbose=True  # See the generated query
)

# Step 3: Query
user_query = "What were APAC sales in Q4 2023?"
results = self_query_retriever.invoke(user_query)

# What happens internally:
# 1. LLM generates structured query:
#    {
#        "query": "APAC sales",
#        "filter": {
#            "and": [
#                {"eq": {"region": "APAC"}},
#                {"eq": {"quarter": "Q4"}},
#                {"eq": {"year": 2023}}
#            ]
#        }
#    }
#
# 2. Apply metadata filter in vector DB
# 3. Perform semantic search on filtered subset
# 4. Return results
```

---

### Question 8: How to Implement RAG Fusion?

#### Concept Breakdown

**RAG Fusion**: Generate multiple query variations, retrieve for each, then fuse (rerank) results.

**Why**: Different phrasings may match different relevant chunks.

#### Step-by-Step: RAG Fusion Implementation

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Step 1: Generate query variations
generate_queries_template = """
You are a helpful assistant that generates multiple search queries 
based on a single input query.

Generate {num_queries} different versions of the question that 
capture different aspects or phrasings.

Original question: {question}

Provide each query on a new line:
"""

generate_queries_prompt = ChatPromptTemplate.from_template(
    generate_queries_template
)

query_generator = (
    generate_queries_prompt 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.strip().split("\n"))
)

# Step 2: Retrieve for each query
def retrieve_multiple(queries):
    all_results = []
    for query in queries:
        results = retriever.invoke(query)
        all_results.extend(results)
    return all_results

# Step 3: Rerank with Reciprocal Rank Fusion
def reciprocal_rank_fusion(results_lists, k=60):
    """
    Fuse multiple ranked lists using RRF formula:
    score = Σ(1 / (k + rank))
    """
    scores = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id") or hash(doc.page_content)
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank)
    
    # Sort by score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked]

# Step 4: Full RAG Fusion Chain
rag_fusion_chain = (
    {
        "question": RunnablePassthrough(),
        "num_queries": lambda _: 5
    }
    | query_generator
    | retrieve_multiple
    | reciprocal_rank_fusion
    | (lambda docs: "\n\n".join(d.page_content for d in docs[:5]))
    | ChatPromptTemplate.from_template(
        "Answer based on context:\\n{context}\\n\\nQuestion: {question}"
    )
    | llm
    | StrOutputParser()
)

# Usage
answer = rag_fusion_chain.invoke("What are the benefits of RAG?")
```

---

## Evaluation & Optimization

### Question 9: How to Evaluate RAG Systems?

#### Concept Breakdown

**RAG Evaluation Metrics:**

| Metric | What it Measures | Target |
|--------|-----------------|--------|
| **Context Precision** | Retrieved chunks contain answer | > 80% |
| **Context Recall** | All relevant chunks retrieved | > 80% |
| **Answer Relevance** | Answer matches question | > 85% |
| **Faithfulness** | Answer grounded in context | > 90% |
| **Latency** | Response time | < 2s |

#### Step-by-Step: RAG Evaluation

**Step 1: Create Evaluation Dataset**
```python
eval_dataset = [
    {
        "question": "What were Q4 2024 sales?",
        "answer": "Q4 2024 sales were $45.2M",
        "contexts": [
            "Q4 2024 Financial Highlights: Total revenue reached $45.2M..."
        ],
        "ground_truth": "$45.2 million in Q4 2024"
    },
    # ... more examples
]
```

**Step 2: Use RAGAS Framework**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Prepare dataset
dataset = Dataset.from_list(eval_dataset)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=evaluation_llm,  # Separate LLM for evaluation
)

print(results)
# {
#     "context_precision": 0.85,
#     "context_recall": 0.78,
#     "faithfulness": 0.92,
#     "answer_relevancy": 0.88
# }
```

**Step 3: A/B Testing Different Configurations**
```python
def test_configuration(config):
    """Test a RAG configuration"""
    retriever = create_retriever(config)
    chain = create_rag_chain(retriever, config)
    
    scores = []
    for item in eval_dataset:
        result = chain.invoke(item["question"])
        score = evaluate_answer(result, item["ground_truth"])
        scores.append(score)
    
    return {
        "config": config,
        "avg_score": np.mean(scores),
        "latency": measure_latency(chain)
    }

# Test different configurations
configs = [
    {"chunk_size": 500, "embedding": "small", "top_k": 3},
    {"chunk_size": 1000, "embedding": "large", "top_k": 5},
    {"chunk_size": 1500, "embedding": "large", "top_k": 7},
]

results = [test_configuration(c) for c in configs]
best_config = max(results, key=lambda x: x["avg_score"])
```

---

### Summary Table: RAG Components

| Component | Options | Recommendation |
|-----------|---------|----------------|
| **Embeddings** | OpenAI, Cohere, Local | OpenAI large for quality, BGE for cost |
| **Vector Store** | Pinecone, Weaviate, Chroma | Pinecone for prod, Chroma for dev |
| **Chunking** | Size: 256-2048 | Start with 1000, test on your data |
| **Retrieval** | Similarity, MMR, Hybrid | MMR for diversity, Hybrid for best results |
| **LLM** | GPT-4, Claude, Local | GPT-4 for accuracy, Claude for context |

### Quick Start Template

```python
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Load and process documents
documents = load_your_documents()
splits = split_documents(documents)

# 2. Create vector store
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# 3. Create RAG chain
prompt = hub.pull("rlm/rag-prompt")  # Pre-built RAG prompt

llm = ChatOpenAI(model="gpt-4-turbo-preview")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Query
result = rag_chain.invoke("What is RAG?")
```

---

*This guide provides comprehensive coverage of RAG systems with up-to-date libraries (LangChain 0.3.x), detailed breakdowns, and step-by-step implementations.*
