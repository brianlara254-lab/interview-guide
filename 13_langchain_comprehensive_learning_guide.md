# The Complete Guide to LangChain (v0.3.x): From Concepts to Production

> A comprehensive deep-dive into LangChain's architecture, components, and best practices for building production-grade LLM applications.

**Reading Time**: 30-35 minutes  
**Prerequisites**: Basic Python, familiarity with LLMs  
**LangChain Version**: 0.3.x (Latest as of 2024-2025)

---

## Table of Contents

1. [Introduction: Why LangChain?](#1-introduction-why-langchain)
2. [Core Architecture & Philosophy](#2-core-architecture--philosophy)
3. [LCEL: The LangChain Expression Language](#3-lcel-the-langchain-expression-language)
4. [Components Deep Dive](#4-components-deep-dive)
5. [Building RAG Applications](#5-building-rag-applications)
6. [Agents and Tool Use](#6-agents-and-tool-use)
7. [Memory and State Management](#7-memory-and-state-management)
8. [Production Patterns](#8-production-patterns)
9. [Common Pitfalls and Best Practices](#9-common-pitfalls-and-best-practices)
10. [Complete Project: Research Assistant](#10-complete-project-research-assistant)

---

## 1. Introduction: Why LangChain?

### The Problem LangChain Solves

Building LLM applications involves more than just calling an API. You need to:

- **Manage context windows** - LLMs have limited context; you need strategies for handling long documents
- **Chain operations** - Complex tasks require multiple LLM calls with dependencies
- **Integrate external data** - Connect to databases, APIs, and document stores
- **Handle failures** - Retry logic, fallbacks, and error recovery
- **Maintain state** - Conversational memory across interactions
- **Deploy to production** - Observability, streaming, and async support

LangChain provides a **unified framework** for all of this.

### Key Differentiators

| Feature | Raw API | LangChain |
|---------|---------|-----------|
| Prompt management | Hardcoded strings | Templates with validation |
| Chaining | Manual orchestration | Composable pipelines |
| External data | Custom integration | 100+ pre-built integrations |
| Observability | Manual logging | Built-in tracing |
| Streaming | Complex implementation | One parameter change |

### Installation

```bash
# Core installation
pip install langchain

# With specific provider
pip install langchain langchain-openai

# Full installation with all integrations
pip install langchain[all]

# For production (v0.3.x recommended)
pip install langchain==0.3.0 langchain-core==0.3.0
```

---

## 2. Core Architecture & Philosophy

### The Runnable Interface

At the heart of LangChain is the **`Runnable`** interface. Everything in LangChain is a Runnable:

```python
from langchain_core.runnables import Runnable

# All these are Runnables:
# - Chat models
# - Prompt templates  
# - Output parsers
# - Chains
# - Agents
```

**Key insight**: Once you understand Runnable, you understand LangChain.

### The Runnable Contract

Every Runnable implements:

```python
class Runnable(ABC):
    def invoke(self, input: Input) -> Output:
        """Synchronous execution"""
        pass
    
    async def ainvoke(self, input: Input) -> Output:
        """Asynchronous execution"""
        pass
    
    def stream(self, input: Input) -> Iterator[Output]:
        """Stream tokens as they're generated"""
        pass
    
    def batch(self, inputs: List[Input]) -> List[Output]:
        """Execute on multiple inputs efficiently"""
        pass
```

This uniform interface means you can:
- Swap components without changing code
- Move from sync to async with one method change
- Enable streaming with a single parameter
- Batch process efficiently

---

## 3. LCEL: The LangChain Expression Language

LCEL is the most important concept in modern LangChain. It's a declarative way to compose chains.

### The Pipe Operator (`|`)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Define components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI(model="gpt-4")

# Compose with LCEL
chain = prompt | model

# Execute
result = chain.invoke({"topic": "programming"})
print(result.content)
```

**What's happening?**
1. `invoke` receives `{"topic": "programming"}`
2. `prompt` formats it into messages → `SystemMessage`, `HumanMessage`
3. `model` receives messages → generates response
4. Result flows through automatically

### Why LCEL Matters

**Before LCEL (v0.0.x - v0.1.x)**:
```python
from langchain import LLMChain, PromptTemplate

prompt = PromptTemplate(template="Tell me about {topic}")
chain = LLMChain(llm=model, prompt=prompt)
result = chain.predict(topic="AI")  # Different API!
```

**With LCEL (v0.2.x - v0.3.x)**:
```python
chain = prompt | model
result = chain.invoke({"topic": "AI"})  # Standard interface
```

Benefits:
- ✅ Consistent API across all components
- ✅ Type safety and validation
- ✅ Automatic async support
- ✅ Built-in streaming
- ✅ Better debugging

### LCEL Composition Patterns

#### 1. Sequential Chains

```python
from langchain_core.output_parsers import StrOutputParser

# Multiple steps
chain = (
    prompt 
    | model 
    | StrOutputParser()  # Extract text from AIMessage
    | (lambda x: x.upper())  # Custom Python function
)

result = chain.invoke({"topic": "python"})
```

#### 2. Parallel Branches (RunnableParallel)

```python
from langchain_core.runnables import RunnableParallel

# Execute multiple paths in parallel
chain = RunnableParallel(
    joke=prompt1 | model,
    fact=prompt2 | model,
    poem=prompt3 | model
)

result = chain.invoke({"topic": "AI"})
# Result: {"joke": AIMessage(...), "fact": AIMessage(...), "poem": AIMessage(...)}
```

**Use case**: Generate multiple content types simultaneously.

#### 3. Conditional Routing (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch

# Route based on condition
branch = RunnableBranch(
    (lambda x: "urgent" in x["query"], urgent_chain),
    (lambda x: "question" in x["query"], qa_chain),
    default_chain  # Fallback
)

result = branch.invoke({"query": "urgent: system is down"})
```

#### 4. Passing Data Through (RunnablePassthrough)

```python
from langchain_core.runnables import RunnablePassthrough

# Maintain context through the chain
chain = (
    {"original": RunnablePassthrough(), "summary": summarization_chain}
    | formatting_prompt
    | model
)
```

### LCEL Helper Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `RunnableLambda` | Wrap Python function | `RunnableLambda(lambda x: x * 2)` |
| `RunnablePassthrough` | Pass input through | `{"key": RunnablePassthrough()}` |
| `RunnableParallel` | Execute in parallel | `RunnableParallel(a=chain1, b=chain2)` |
| `RunnableBranch` | Conditional routing | `RunnableBranch((condition, chain))` |
| `itemgetter` | Extract dict keys | `itemgetter("messages")` |

---

## 4. Components Deep Dive

### 4.1 Chat Models

#### The Base Interface

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# All implement same interface
openai = ChatOpenAI(model="gpt-4", temperature=0.7)
claude = ChatAnthropic(model="claude-3-opus-20240229")
gemini = ChatGoogleGenerativeAI(model="gemini-pro")

# Same invocation pattern for all
messages = [
    ("system", "You are a helpful assistant"),
    ("human", "Explain quantum computing")
]

response = openai.invoke(messages)
```

#### Key Parameters

```python
model = ChatOpenAI(
    model="gpt-4",              # Model identifier
    temperature=0.7,            # Creativity (0-2)
    max_tokens=1000,            # Max response length
    top_p=1.0,                  # Nucleus sampling
    frequency_penalty=0.0,      # Reduce repetition
    presence_penalty=0.0,       # Encourage new topics
    n=1,                        # Number of completions
    stop=["END"],               # Stop sequences
    streaming=True              # Enable streaming
)
```

#### Message Types

```python
from langchain_core.messages import (
    SystemMessage,      # Instructions to model
    HumanMessage,       # User input
    AIMessage,          # Model response
    ToolMessage,        # Tool execution result
    FunctionMessage     # Legacy function result
)

messages = [
    SystemMessage(content="You are a coding assistant"),
    HumanMessage(content="Write a Python function to sort a list"),
    AIMessage(content="Here's the function..."),
    HumanMessage(content="Can you optimize it?")
]
```

#### Streaming Responses

```python
# Stream tokens as they're generated
for chunk in model.stream("Tell me a long story"):
    print(chunk.content, end="", flush=True)

# Async streaming
async for chunk in model.astream("Tell me a story"):
    print(chunk.content, end="")
```

### 4.2 Prompt Templates

#### ChatPromptTemplate

The most important template class:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Basic template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be {tone}."),
    ("human", "{question}")
])

# With message history
history_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),  # Dynamic messages
    ("human", "{input}")
])

# Usage
prompt = template.invoke({
    "role": "coding expert",
    "tone": "concise",
    "question": "How do I use decorators?"
})
```

#### Few-Shot Prompting

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "10*5", "output": "50"}
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor"),
    few_shot_prompt,
    ("human", "{input}")
])
```

#### Output Parsing with Templates

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    rating: float = Field(description="IMDB rating")

parser = JsonOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_template("""
Extract movie information from the query.
{format_instructions}

Query: {query}
""").partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
movie = chain.invoke({"query": "Inception from 2010 rated 8.8"})
# Result: Movie(title="Inception", year=2010, rating=8.8)
```

### 4.3 Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,        # Extract text content
    JsonOutputParser,       # Parse JSON responses
    PydanticOutputParser,   # Parse into Pydantic models
    CommaSeparatedListOutputParser,  # Parse lists
    StructuredOutputParser  # Parse complex structures
)

# String parser (most common)
str_parser = StrOutputParser()
chain = prompt | model | str_parser

# JSON parser
json_parser = JsonOutputParser()
chain = prompt | model | json_parser
result = chain.invoke({"topic": "AI"})
# Returns Python dict

# Pydantic parser for type safety
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

pydantic_parser = PydanticOutputParser(pydantic_object=Person)
```

### 4.4 Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader
)

# PDF loader
pdf_loader = PyPDFLoader("document.pdf")
pages = pdf_loader.load()  # Returns List[Document]

# Web loader
web_loader = WebBaseLoader("https://example.com/article")
web_docs = web_loader.load()

# Directory loader (batch processing)
dir_loader = DirectoryLoader(
    "./data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs = dir_loader.load()
```

### 4.5 Text Splitters

Critical for RAG - chunking strategy affects retrieval quality:

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

# Most recommended: Recursive splitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Priority order
)

chunks = recursive_splitter.split_documents(docs)

# Token-based (for token-limited models)
token_splitter = TokenTextSplitter(
    chunk_size=500,  # Tokens, not characters
    chunk_overlap=50
)
```

**Chunking Best Practices:**
- ✅ Use `RecursiveCharacterTextSplitter` for most text
- ✅ Set `chunk_overlap` to 10-20% of chunk size
- ✅ Keep chunks under model's context window
- ✅ Consider semantic splitting for code/docs

### 4.6 Vector Stores and Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone

# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Similarity search
docs = vectorstore.similarity_search(
    query="What is machine learning?",
    k=4  # Top 4 results
)

# Similarity with score
docs_and_scores = vectorstore.similarity_search_with_score(
    query="AI applications",
    k=5
)
```

#### Advanced Retrieval

```python
# MMR (Maximal Marginal Relevance) - diverse results
docs = vectorstore.max_marginal_relevance_search(
    query="machine learning",
    k=4,
    fetch_k=20,  # Fetch more, then diversify
    lambda_mult=0.5  # Balance relevance vs diversity
)

# Metadata filtering
docs = vectorstore.similarity_search(
    query="revenue",
    filter={"source": "financial_report.pdf"}
)
```

---

## 5. Building RAG Applications

### The Complete RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Load and process documents
loader = PyPDFLoader("annual_report.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Create RAG prompt
template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

If you don't know the answer, say "I don't have enough information."
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Build RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)

# 6. Use
answer = rag_chain.invoke("What was the revenue in Q3?")
```

### Understanding the RAG Chain

```
User Query
    ↓
RunnablePassthrough() → "question": query
    ↓
Retriever → Relevant documents
    ↓
format_docs() → "context": formatted text
    ↓
Prompt Template (combines context + question)
    ↓
LLM generates answer
    ↓
Output parser extracts text
    ↓
Final Answer
```

### Advanced RAG Patterns

#### Multi-Query Retrieval

Generate multiple query variations for better retrieval:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Generate 3 query variations
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(temperature=0)
)

docs = multi_query_retriever.get_relevant_documents("AI impact on jobs")
# Returns docs for: "AI impact on jobs", "artificial intelligence employment", 
# "automation job market"
```

#### Contextual Compression

Filter irrelevant content from retrieved chunks:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(ChatOpenAI())

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Only relevant sentences from chunks are kept
docs = compression_retriever.get_relevant_documents("revenue growth")
```

#### Ensemble Retrieval (RRF)

Combine multiple retrievers with Reciprocal Rank Fusion:

```python
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore_retriever],
    weights=[0.5, 0.5]
)
```

### RAG Evaluation

```python
from langchain.evaluation import (
    QAEvalChain,
    CotQAEvalChain
)

# Create evaluation dataset
eval_questions = [
    {"question": "Q1", "answer": "A1", "contexts": [...]},
    {"question": "Q2", "answer": "A2", "contexts": [...]}
]

# Evaluate
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    eval_questions,
    metrics=[faithfulness, answer_relevancy]
)
```

---

## 6. Agents and Tool Use

### What Are Agents?

Agents are systems that:
1. Receive a user input
2. Decide which action to take (using LLM reasoning)
3. Execute the action (call tools)
4. Observe the result
5. Repeat until task is complete

### Creating Tools

```python
from langchain.tools import tool
from langchain_core.tools import Tool

# Method 1: Decorator (recommended)
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

# Method 2: Explicit creation
calculator = Tool(
    name="Calculator",
    func=lambda x: eval(x),  # Warning: eval is unsafe
    description="Useful for math calculations"
)

# Method 3: Structured tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results")

@tool(args_schema=SearchInput)
def search(query: str, num_results: int = 5) -> str:
    """Search for information."""
    return f"Results for {query}"
```

### Agent Types

#### 1. ReAct Agent (Reasoning + Acting)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

tools = [search_web, calculator]

# ReAct prompt template
react_template = """Answer the following questions as best you can...

You have access to the following tools:
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

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(
    llm=ChatOpenAI(temperature=0),
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

result = agent_executor.invoke({"input": "What is the weather in NY?"})
```

#### 2. OpenAI Functions Agent

Most reliable for OpenAI models:

```python
from langchain.agents import create_openai_functions_agent

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
```

#### 3. Structured Chat Agent

Best for multi-input tools:

```python
from langchain.agents import create_structured_chat_agent

agent = create_structured_chat_agent(
    llm=ChatOpenAI(),
    tools=tools,
    prompt=prompt
)
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Now agent remembers previous interactions
```

---

## 7. Memory and State Management

### Types of Memory

```python
from langchain.memory import (
    ConversationBufferMemory,      # Stores all messages
    ConversationBufferWindowMemory, # Last k exchanges
    ConversationSummaryMemory,      # Summarized history
    ConversationEntityMemory,       # Track entities
    VectorStoreRetrieverMemory      # Semantic retrieval
)
```

### Conversation Buffer Memory

```python
memory = ConversationBufferMemory(
    memory_key="history",      # Variable name in prompt
    input_key="input",         # User input key
    return_messages=True       # Return as Message objects
)

# Add messages
memory.save_context(
    {"input": "Hi there!"},
    {"output": "Hello! How can I help?"}
)

# Load memory
memory.load_memory_variables({})
# Returns: {"history": [HumanMessage(...), AIMessage(...)]}
```

### Conversation Summary Memory

For long conversations - summarize instead of storing raw messages:

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),  # LLM to create summaries
    max_token_limit=1000
)

# Automatically summarizes when limit reached
```

### Using Memory in Chains

```python
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=ConversationBufferMemory(),
    verbose=True
)

response = conversation.predict(input="Hi!")
response = conversation.predict(input="What's my name?")  # Remembers context
```

### LCEL with Memory

```python
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Store for multiple sessions
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# Each session maintains separate history
result = chain_with_history.invoke(
    {"question": "Hello!"},
    config={"configurable": {"session_id": "user_123"}}
)
```

---

## 8. Production Patterns

### Streaming in Production

```python
from fastapi import FastAPI, StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
model = ChatOpenAI(streaming=True)

@app.post("/chat")
async def chat(message: str):
    async def generate():
        async for chunk in model.astream(message):
            yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")
```

### Async Operations

```python
# Batch processing
inputs = [{"topic": "AI"}, {"topic": "ML"}, {"topic": "DL"}]
results = await chain.abatch(inputs)

# Parallel execution
import asyncio

tasks = [chain.ainvoke({"topic": t}) for t in topics]
results = await asyncio.gather(*tasks)
```

### Error Handling and Retries

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_concurrency=5,
    recursion_limit=10
)

# With fallback
from langchain_core.runnables import RunnableWithFallbacks

chain_with_fallback = chain.with_fallbacks(
    fallbacks=[alternative_chain],
    exceptions_to_handle=(RateLimitError,)
)
```

### Observability with LangSmith

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "production-chatbot"

# All runs now traced automatically
trace = chain.invoke({"question": "Hello"})
```

### Caching

```python
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache, SQLiteCache

# In-memory cache
set_llm_cache(InMemoryCache())

# Persistent SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Same prompt → cached response (faster, cheaper)
```

---

## 9. Common Pitfalls and Best Practices

### ❌ Pitfall 1: Not Using LCEL

```python
# Old way (deprecated)
from langchain import LLMChain
chain = LLMChain(llm=model, prompt=prompt)

# ✅ Modern way
chain = prompt | model | parser
```

### ❌ Pitfall 2: Hardcoded Prompts

```python
# Bad
response = model.invoke("Tell me about {topic}".format(topic=user_input))

# ✅ Good
template = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = template | model
```

### ❌ Pitfall 3: No Error Handling

```python
# Bad
result = chain.invoke(input)

# ✅ Good
try:
    result = chain.invoke(input)
except OutputParserException as e:
    # Handle parsing failure
    result = fallback_chain.invoke(input)
```

### ❌ Pitfall 4: Wrong Chunk Size

```python
# Bad - chunks too large
splitter = CharacterTextSplitter(chunk_size=4000)

# ✅ Good - consider model context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### ✅ Best Practices Summary

1. **Always use LCEL** for new projects
2. **Type your inputs/outputs** with Pydantic
3. **Use streaming** for better UX
4. **Implement caching** to reduce costs
5. **Add observability** (LangSmith) from day one
6. **Test with evals** before deploying
7. **Handle failures** gracefully
8. **Monitor token usage** and costs

---

## 10. Complete Project: Research Assistant

Putting it all together:

```python
"""
Research Assistant - Complete LangChain Application
Features:
- Web search
- Document analysis
- Source citations
- Conversation memory
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

# Tools
tools = [
    DuckDuckGoSearchRun(),
    # Add more tools as needed
]

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# RAG Setup
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Agent Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Use tools to gather information,
    then provide comprehensive answers with citations."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5
)

# Use the agent
response = agent_executor.invoke({
    "input": "Research the latest developments in LLMs and provide a summary"
})

print(response["output"])
```

---

## Conclusion

LangChain v0.3.x represents a mature, production-ready framework for building LLM applications. The key takeaways:

1. **LCEL is everything** - Learn the pipe operator and Runnable interface
2. **Start simple** - Build basic chains before complex agents
3. **Think composition** - Small, reusable components beat monolithic chains
4. **Production matters** - Observability, caching, and error handling from day one
5. **Stay updated** - LangChain evolves rapidly; follow the changelog

### Resources for Continued Learning

- **Documentation**: https://python.langchain.com
- **GitHub**: https://github.com/langchain-ai/langchain
- **Cookbook**: https://github.com/langchain-ai/langchain/tree/master/cookbook
- **Discord**: Active community for questions
- **LangSmith**: https://smith.langchain.com for tracing

---

**About the Author**: *Building AI systems that actually work in production. Follow for more deep dives into LangChain, LLMs, and agentic architectures.*

**Tags**: `#LangChain` `#LLM` `#AI` `#Python` `#MachineLearning` `#RAG` `#OpenAI` `#GenerativeAI`
