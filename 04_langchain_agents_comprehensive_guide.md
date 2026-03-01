# LangChain & Agent Frameworks - Comprehensive Guide

A detailed breakdown of LangChain components, agent architectures, tool use, and advanced patterns with up-to-date library methods and step-by-step implementations.

**Last Updated**: 2025 | **LangChain Version**: 0.3.x | **LangChain Core**: 1.2.10

---

## Table of Contents
1. [LangChain Fundamentals](#langchain-fundamentals)
2. [Chains and Runnables](#chains-and-runnables)
3. [Agent Architectures](#agent-architectures)
4. [Tool Use and Function Calling](#tool-use-and-function-calling)
5. [Memory and Persistence](#memory-and-persistence)
6. [Advanced Patterns](#advanced-patterns)
7. [Production Multi-Agent Orchestration](#production-multi-agent-orchestration)

---

## LangChain Fundamentals

### Question 1: What is LangChain and Why Use It?

#### Concept Breakdown

**Definition**: LangChain is a Python/JS framework for building applications with LLMs through composable components.

**Core Philosophy**: Modular, composable building blocks that can be chained together.

```
┌─────────────────────────────────────────────────────────────┐
│              LANGCHAIN COMPONENT HIERARCHY                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: Components (Building Blocks)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Chat Models (GPT-4, Claude)                        │  │
│  │  • Prompt Templates                                   │  │
│  │  • Document Loaders                                   │  │
│  │  • Text Splitters                                     │  │
│  │  • Embeddings                                         │  │
│  │  • Vector Stores                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  Level 2: Chains (Combinations)                             │
│  ├───────────────────────────────────────────────────────┐  │
│  │  • LLMChain (Prompt → Model → Output)                 │  │
│  │  • RetrievalQA (RAG pattern)                          │  │
│  │  • ConversationalRetrievalChain                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  Level 3: Agents (Dynamic Execution)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Tools + LLM + Strategy = Agent                     │  │
│  │  • ReAct, Plan-and-Execute, etc.                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  Level 4: Applications                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Chatbots, RAG Systems, Code Assistants             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: First LangChain Application

**Step 1: Installation**
```bash
# Core packages
pip install langchain langchain-openai

# Optional integrations
pip install langchain-community langchain-huggingface
```

**Step 2: Basic Chain**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Create model
model = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)

# 2. Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# 3. Create output parser
output_parser = StrOutputParser()

# 4. Compose chain using LCEL (LangChain Expression Language)
chain = prompt | model | output_parser

# 5. Invoke
result = chain.invoke({"input": "What is LangChain?"})
print(result)
```

**Step 3: Understanding the Pipe Operator (`|`)**
```python
# The pipe operator creates a processing pipeline
# Each component's output becomes the next component's input

# Step by step:
# 1. {"input": "What is LangChain?"}
#    ↓
# 2. prompt.invoke() → ChatPromptValue(messages=[...])
#    ↓
# 3. model.invoke() → AIMessage(content="...")
#    ↓
# 4. output_parser.invoke() → "LangChain is..."

# This is equivalent to:
result = output_parser.invoke(
    model.invoke(
        prompt.invoke({"input": "What is LangChain?"})
    )
)
```

---

### Question 2: What is the Difference Between Legacy Chains and LCEL?

#### Concept Breakdown

**Legacy Chains** (Pre-LangChain 0.1):
- Class-based, inheritance-heavy
- Less flexible, harder to customize
- Hidden execution logic

**LCEL (LangChain Expression Language)**:
- Functional, composable approach
- Pipe operator (`|`) for composition
- Transparent, streaming-native
- Async support built-in

#### Step-by-Step: Comparison

**Legacy Approach (Old)**
```python
from langchain import LLMChain, PromptTemplate
from langchain_openai import OpenAI

# Define components separately
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

llm = OpenAI(temperature=0.9)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Run
result = chain.predict(product="colorful socks")
# → "Rainbow Toes"
```

**LCEL Approach (New)**
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser

# More explicit, composable
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

llm = OpenAI(temperature=0.9)
output_parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | llm | output_parser

# Invoke (explicit input/output)
result = chain.invoke({"product": "colorful socks"})
# → "Rainbow Toes"
```

**Key Differences:**

| Aspect | Legacy (LLMChain) | LCEL |
|--------|-------------------|------|
| **Composition** | Class inheritance | Pipe operator `\|` |
| **Streaming** | Not built-in | Native support |
| **Async** | Separate methods | Same interface |
| **Debugging** | Harder to trace | Step-by-step transparent |
| **Batch** | Separate method | `.batch()` method |

---

## Chains and Runnables

### Question 3: What are Runnables and How Do They Work?

#### Concept Breakdown

**Runnable**: The core abstraction in LangChain - anything that can be invoked with `.invoke()`, `.batch()`, or `.stream()`.

**Runnable Interface:**
```python
class Runnable:
    def invoke(self, input, config=None) → Output
    def batch(self, inputs, config=None) → List[Output]
    def stream(self, input, config=None) → Iterator[Output]
    def ainvoke(self, input, config=None) → Awaitable[Output]  # Async
```

#### Step-by-Step: Runnable Types

**Type 1: RunnableLambda (Custom Functions)**
```python
from langchain_core.runnables import RunnableLambda

# Wrap any function as a Runnable
def multiply_by_two(x: int) -> int:
    return x * 2

runnable = RunnableLambda(multiply_by_two)

# Can now use with .invoke(), .batch(), .stream()
result = runnable.invoke(5)  # → 10
results = runnable.batch([1, 2, 3])  # → [2, 4, 6]
```

**Type 2: RunnablePassthrough (Pass Through)**
```python
from langchain_core.runnables import RunnablePassthrough

# Passes input through unchanged
passthrough = RunnablePassthrough()

result = passthrough.invoke({"key": "value"})  # → {"key": "value"}

# Common use case: Assign new keys
chain = (
    RunnablePassthrough.assign(
        double=lambda x: x["number"] * 2
    )
    | (lambda x: f"Original: {x['number']}, Double: {x['double']}")
)

result = chain.invoke({"number": 5})
# → "Original: 5, Double: 10"
```

**Type 3: RunnableParallel (Branching)**
```python
from langchain_core.runnables import RunnableParallel

# Execute multiple branches in parallel
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

result = parallel.invoke("Long article text...")
# → {
#     "summary": "Brief summary...",
#     "keywords": ["AI", "machine learning"],
#     "sentiment": "positive"
# }
```

**Type 4: RunnableSequence (Chaining)**
```python
from langchain_core.runnables import RunnableSequence

# The pipe operator creates a RunnableSequence
sequence = step1 | step2 | step3

# Equivalent to:
sequence = RunnableSequence(first=step1, middle=[step2], last=step3)
```

---

### Question 4: How to Create a Complex Chain with Branching?

#### Concept Breakdown

**Scenario**: Process a document through multiple parallel analyses, then combine results.

#### Step-by-Step: Complex Chain Implementation

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Setup
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Define parallel processing branches
branches = RunnableParallel(
    # Branch 1: Summary
    summary=ChatPromptTemplate.from_template(
        "Summarize this in one sentence:\\n{document}"
    ) | llm | StrOutputParser(),
    
    # Branch 2: Key Points  
    key_points=ChatPromptTemplate.from_template(
        "List 3 key points from:\\n{document}"
    ) | llm | StrOutputParser(),
    
    # Branch 3: Sentiment
    sentiment=ChatPromptTemplate.from_template(
        "Classify sentiment (positive/neutral/negative):\\n{document}"
    ) | llm | StrOutputParser(),
    
    # Branch 4: Keep original
    original=RunnablePassthrough()
)

# Combine results
combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document analyst. Combine the analyses."),
    ("user", """
    Document: {original}
    
    Summary: {summary}
    Key Points: {key_points}
    Sentiment: {sentiment}
    
    Provide a comprehensive analysis:
    """)
])

# Full chain
full_chain = branches | combine_prompt | llm | StrOutputParser()

# Execute
document = "Your long document here..."
result = full_chain.invoke({"document": document})

# What happens:
# 1. Input: {"document": "..."}
# 2. Parallel execution:
#    - summary: "Summary..."
#    - key_points: "1. Point A..."
#    - sentiment: "positive"
#    - original: {"document": "..."}
# 3. Combine: All outputs merged into final prompt
# 4. LLM generates comprehensive analysis
```

**Execution Flow Visualization:**

```
Input: {"document": "long text..."}
    │
    ▼
┌─────────────────────────────────────────────┐
│         RUNNABLE PARALLEL                   │
│  (Executes all branches concurrently)       │
├─────────────┬──────────────┬────────────────┤
│             │              │                │
▼             ▼              ▼                ▼
summary    key_points    sentiment      original
│             │              │                │
│             │              │                │
└─────────────┴──────────────┴────────────────┘
    │
    ▼
Merged Dict:
{
    "summary": "...",
    "key_points": "...",
    "sentiment": "...",
    "original": "..."
}
    │
    ▼
Combine Prompt
    │
    ▼
LLM
    │
    ▼
Final Analysis
```

---

## Agent Architectures

### Question 5: What are Agents and How Do They Work?

#### Concept Breakdown

**Agent**: An LLM-powered system that can make decisions, use tools, and take actions to accomplish goals.

**Agent Components:**
```
┌──────────────────────────────────────────────────────────────┐
│                    AGENT ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                            │
│  │    LLM      │ ← Core reasoning engine                    │
│  │  (Brain)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │   Planning  │────▶│  Tool Use   │────▶│  Observation│    │
│  │  (Decide)   │     │  (Act)      │     │  (Result)   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│         ▲                                          │         │
│         └──────────────────────────────────────────┘         │
│                        (Loop until done)                     │
│                                                              │
│  Available Tools:                                            │
│  • Search (web, internal docs)                               │
│  • Calculator (math operations)                              │
│  • APIs (external services)                                  │
│  • Code Execution                                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: ReAct Agent (Reason + Act)

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Step 1: Define Tools
tools = [
    Tool(
        name="search",
        func=lambda q: f"Search results for: {q}",
        description="Useful for finding current information"
    ),
    Tool(
        name="calculator",
        func=lambda expr: str(eval(expr)),
        description="Useful for math calculations"
    )
]

# Step 2: Create LLM
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

# Step 3: Create ReAct Agent
# ReAct = Reasoning + Acting
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)

# Step 4: Create Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See reasoning process
    max_iterations=5  # Prevent infinite loops
)

# Step 5: Run
result = agent_executor.invoke({
    "input": "What is the population of Tokyo divided by 1000?"
})
```

**Execution Trace:**

```
Question: "What is the population of Tokyo divided by 1000?"

Thought: I need to find Tokyo's population, then divide by 1000.
Action: search
Action Input: "Tokyo population 2024"

Observation: "Tokyo has approximately 14 million residents"

Thought: Now I need to calculate 14,000,000 / 1000
Action: calculator
Action Input: "14000000 / 1000"

Observation: "14000"

Thought: I now know the final answer
Final Answer: 14,000
```

---

### Question 6: What is the Difference Between Different Agent Types?

#### Concept Breakdown

**Agent Type Comparison:**

| Agent Type | Strategy | Best For | Complexity |
|------------|----------|----------|------------|
| **Zero-Shot ReAct** | Reason step-by-step | Simple tasks | Low |
| **Structured Chat** | JSON-based tool calls | Multiple tools | Medium |
| **OpenAI Tools** | Native function calling | OpenAI models | Low |
| **Plan-and-Execute** | Plan first, then execute | Complex multi-step | High |
| **Self-Ask** | Ask follow-up questions | Research tasks | Medium |

#### Step-by-Step: Agent Type Selection

**Type 1: Structured Chat Agent**
```python
from langchain.agents import create_structured_chat_agent

# Uses JSON format for tool calls (more reliable)
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Tool call format:
# {
#     "action": "search",
#     "action_input": "Tokyo population"
# }
```

**Type 2: OpenAI Tools Agent**
```python
from langchain.agents import create_openai_tools_agent

# Uses OpenAI's native function calling
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Most efficient for OpenAI models
# Models specifically trained for function calling
```

**Type 3: Plan-and-Execute Agent**
```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# Two-phase approach:
# 1. Planning phase: Create step-by-step plan
# 2. Execution phase: Execute each step

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor)

# Best for complex tasks requiring multiple steps
# Example: "Research Tesla stock, analyze competitors, write report"
```

---

## Tool Use and Function Calling

### Question 7: How to Create and Use Custom Tools?

#### Concept Breakdown

**Tool**: A function the agent can call, with:
- Name: How to reference it
- Description: When to use it
- Function: What it does
- Parameters: Input schema

#### Step-by-Step: Creating Custom Tools

**Method 1: @tool Decorator (Simple)**
```python
from langchain.tools import tool

@tool
def search_stock_price(ticker: str) -> str:
    """
    Search for current stock price of a company.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA)
    
    Returns:
        Current stock price and daily change
    """
    # Implementation
    import yfinance as yf
    stock = yf.Ticker(ticker)
    price = stock.info['currentPrice']
    return f"${price}"

# Tool is automatically registered with:
# - Name: search_stock_price
# - Description: From docstring
# - Schema: From type hints
```

**Method 2: StructuredTool (Complex)**
```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(description="Number of results", default=5)

def web_search(query: str, num_results: int = 5) -> str:
    """Perform web search."""
    # Implementation
    return f"Results for {query}"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the internet for current information",
    args_schema=SearchInput,
    return_direct=False  # Let LLM process result
)
```

**Method 3: Tool Class (Full Control)**
```python
from langchain.tools import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def _run(self, expression: str) -> str:
        """Execute calculation."""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async execution."""
        return self._run(expression)
```

---

## Memory and Persistence

### Question 8: How to Add Memory to Chains?

#### Concept Breakdown

**Memory Types:**

| Memory Type | What It Stores | Use Case |
|-------------|---------------|----------|
| **Buffer** | All messages | Short conversations |
| **Buffer Window** | Last N messages | Long conversations |
| **Summary** | Summarized history | Very long conversations |
| **Entity** | Key facts about entities | Relationship tracking |
| **Vector Store** | Similar past messages | Contextual recall |

#### Step-by-Step: Memory Implementation

**Simple Conversation Buffer**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# First message
conversation.predict(input="Hi, I'm Bob")
# Memory: [Human: "Hi, I'm Bob", AI: "Hello Bob!"]

# Second message (has context)
conversation.predict(input="What's my name?")
# AI can answer: "Your name is Bob"
```

**Window Memory (Sliding Window)**
```python
from langchain.memory import ConversationBufferWindowMemory

# Only remember last 2 exchanges (k=2)
memory = ConversationBufferWindowMemory(k=2)

# Older messages are forgotten
# Useful for managing token limits
```

**Summary Memory (For Long Conversations)**
```python
from langchain.memory import ConversationSummaryMemory

# Summarize conversation instead of storing raw messages
memory = ConversationSummaryMemory(
    llm=llm,  # LLM to use for summarization
    max_token_limit=100
)

# After several exchanges:
# Memory: "The human introduced himself as Bob. 
#          They discussed AI and LangChain."
```

**LCEL with Memory (Modern Approach)**
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

# Create chain
chain = prompt | llm | output_parser

# Add memory wrapper
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),  # Get/store history
    input_messages_key="input",
    history_messages_key="history"
)

# Invoke with session ID
result = chain_with_history.invoke(
    {"input": "Hi, I'm Bob"},
    config={"configurable": {"session_id": "user_123"}}
)
```

---

## Advanced Patterns

### Question 9: How to Build a Multi-Agent System?

#### Concept Breakdown

**Multi-Agent Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│                  MULTI-AGENT SYSTEM                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────────┐                          │
│                    │  Supervisor │                          │
│                    │   Agent     │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│          ┌────────────────┼────────────────┐                │
│          ▼                ▼                ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Research   │  │   Writer    │  │   Critic    │         │
│  │    Agent    │  │    Agent    │  │    Agent    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│        │               │               │                   │
│        └───────────────┴───────────────┘                   │
│                        │                                    │
│                        ▼                                    │
│                 Final Output                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Multi-Agent Implementation

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_agent: str
    task_complete: bool

# Create specialized agents
def create_research_agent():
    system_msg = """You are a research agent. Find relevant information.
    Return your findings as bullet points."""
    
    def research(state):
        messages = state["messages"]
        response = llm.invoke([
            SystemMessage(content=system_msg),
            *messages
        ])
        return {
            "messages": [response],
            "next_agent": "writer"
        }
    
    return research

def create_writer_agent():
    system_msg = """You are a writer. Create content based on research."""
    
    def write(state):
        messages = state["messages"]
        response = llm.invoke([
            SystemMessage(content=system_msg),
            *messages
        ])
        return {
            "messages": [response],
            "next_agent": "critic"
        }
    
    return write

def create_critic_agent():
    system_msg = """You are a critic. Review and suggest improvements."""
    
    def critique(state):
        messages = state["messages"]
        response = llm.invoke([
            SystemMessage(content=system_msg),
            *messages
        ])
        
        # Determine if complete
        if "APPROVED" in response.content:
            return {
                "messages": [response],
                "next_agent": END,
                "task_complete": True
            }
        else:
            return {
                "messages": [response],
                "next_agent": "writer"
            }
    
    return critique

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", create_research_agent())
workflow.add_node("writer", create_writer_agent())
workflow.add_node("critic", create_critic_agent())

# Add edges
workflow.set_entry_point("research")
workflow.add_conditional_edges(
    "research",
    lambda x: x["next_agent"],
    {"writer": "writer"}
)
workflow.add_conditional_edges(
    "writer",
    lambda x: x["next_agent"],
    {"critic": "critic"}
)
workflow.add_conditional_edges(
    "critic",
    lambda x: x["next_agent"],
    {"writer": "writer", END: END}
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="Write about AI safety")],
    "next_agent": "research",
    "task_complete": False
})
```

---

## Production Multi-Agent Orchestration

This section covers real-world patterns for deploying multi-agent systems in production, including communication strategies, failure handling, observability, and conflict resolution.

---

### 1. Multi-Agent Communication Patterns

In production systems, agents need robust communication mechanisms. Here are the three primary patterns:

#### **Pattern 1: Shared State (Blackboard Architecture)**
All agents read from and write to a central state store.

**How it works:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent A   │◄───►│ Shared State│◄───►│   Agent B   │
│  (Research) │     │  (Redis/DB) │     │  (Writer)   │
└─────────────┘     └─────────────┘     └─────────────┘
         ▲                    ▲                  ▲
         └────────────────────┴──────────────────┘
                   All agents sync via state
```

**Pros:**
- Simple to implement
- Full observability of all decisions
- Easy to replay/debug

**Cons:**
- State conflicts when agents write simultaneously
- Requires locking mechanisms
- Can become bottleneck at scale

**Implementation with Redis:**
```python
import redis
import json
from typing import Dict, Any
import threading

class SharedStateManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.lock = threading.Lock()
    
    def get_state(self, session_id: str) -> Dict[str, Any]:
        """Retrieve shared state for a session."""
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else {}
    
    def update_state(self, session_id: str, agent_id: str, 
                     updates: Dict[str, Any]) -> bool:
        """Update state with optimistic locking."""
        with self.lock:
            # Get current state
            current = self.get_state(session_id)
            
            # Add metadata
            updates["_last_updated_by"] = agent_id
            updates["_timestamp"] = time.time()
            
            # Merge updates
            current.update(updates)
            
            # Save back
            self.redis.set(
                f"session:{session_id}", 
                json.dumps(current),
                ex=3600  # 1 hour expiry
            )
            return True
    
    def acquire_agent_lock(self, session_id: str, agent_id: str) -> bool:
        """Ensure only one agent acts at a time."""
        lock_key = f"lock:{session_id}"
        return self.redis.set(
            lock_key, agent_id, nx=True, ex=30  # 30s timeout
        )
    
    def release_agent_lock(self, session_id: str):
        """Release agent lock."""
        self.redis.delete(f"lock:{session_id}")
```

---

#### **Pattern 2: Message Passing (Actor Model)**
Agents communicate by sending messages to each other via a message broker.

**How it works:**
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Agent A   │─────►│   Message    │─────►│   Agent B   │
│  (Research) │      │   Broker     │      │  (Writer)   │
└─────────────┘      │  (RabbitMQ/  │      └─────────────┘
                     │   Kafka)     │
                     └──────────────┘
```

**Pros:**
- Decoupled agents can scale independently
- Agents can be written in different languages
- Natural retry and dead-letter queue support
- Better fault isolation

**Cons:**
- More complex to debug
- Message ordering challenges
- Eventual consistency

**Implementation with RabbitMQ:**
```python
import pika
import json
from typing import Callable
import threading

class MessagePassingAgent:
    def __init__(self, agent_id: str, rabbitmq_url: str):
        self.agent_id = agent_id
        self.connection = pika.BlockingConnection(
            pika.URLParameters(rabbitmq_url)
        )
        self.channel = self.connection.channel()
        
        # Declare exchange
        self.channel.exchange_declare(
            exchange='agents', 
            exchange_type='topic'
        )
        
        # Create queue for this agent
        self.queue = self.channel.queue_declare(
            queue=f"agent_{agent_id}", 
            durable=True
        ).method.queue
        
        # Bind to relevant routing keys
        self.channel.queue_bind(
            exchange='agents',
            queue=self.queue,
            routing_key=f"to.{agent_id}"
        )
        self.channel.queue_bind(
            exchange='agents',
            queue=self.queue,
            routing_key="broadcast"
        )
    
    def send_message(self, to_agent: str, message_type: str, 
                     payload: dict):
        """Send message to another agent."""
        message = {
            "from": self.agent_id,
            "to": to_agent,
            "type": message_type,
            "payload": payload,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4())
        }
        
        routing_key = f"to.{to_agent}" if to_agent != "broadcast" else "broadcast"
        
        self.channel.basic_publish(
            exchange='agents',
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                message_id=message["message_id"]
            )
        )
    
    def register_handler(self, message_type: str, 
                         handler: Callable[[dict], dict]):
        """Register message handler."""
        def callback(ch, method, properties, body):
            try:
                message = json.loads(body)
                if message.get("type") == message_type:
                    result = handler(message["payload"])
                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                    # Send response if needed
                    if result and message.get("from"):
                        self.send_message(
                            message["from"], 
                            f"{message_type}_response", 
                            result
                        )
            except Exception as e:
                # Reject message for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.log_error(message, e)
        
        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=callback
        )
    
    def start(self):
        """Start consuming messages."""
        self.channel.start_consuming()
    
    def log_error(self, message: dict, error: Exception):
        """Log failed message for analysis."""
        error_log = {
            "message": message,
            "error": str(error),
            "agent": self.agent_id,
            "timestamp": time.time()
        }
        # Send to dead letter queue or error log
        self.channel.basic_publish(
            exchange='agents',
            routing_key='errors',
            body=json.dumps(error_log)
        )
```

---

#### **Pattern 3: Direct RPC (Request-Reply)**
Agents call each other directly via HTTP/gRPC.

**How it works:**
```
┌─────────────┐     HTTP/GRPC      ┌─────────────┐
│   Agent A   │◄────Request───────►│   Agent B   │
│  (Research) │     Response       │  (Writer)   │
└─────────────┘                    └─────────────┘
```

**Pros:**
- Synchronous, easier to reason about
- Type-safe with gRPC
- Fast for same-datacenter communication

**Cons:**
- Tight coupling between agents
- Cascading failures possible
- Harder to scale independently

**Implementation with FastAPI:**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

class AgentRequest(BaseModel):
    session_id: str
    task: str
    context: dict
    timeout: int = 30

class AgentResponse(BaseModel):
    status: str
    result: dict
    agent_id: str
    processing_time_ms: float

# Agent registry
AGENT_REGISTRY = {
    "research": "http://research-agent:8000",
    "writer": "http://writer-agent:8000",
    "critic": "http://critic-agent:8000"
}

@app.post("/delegate/{agent_type}")
async def delegate_task(
    agent_type: str, 
    request: AgentRequest,
    background_tasks: BackgroundTasks
) -> AgentResponse:
    """Delegate task to another agent via RPC."""
    
    if agent_type not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_url = AGENT_REGISTRY[agent_type]
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            response = await client.post(
                f"{agent_url}/process",
                json=request.dict()
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            # Track in background
            background_tasks.add_task(
                log_interaction,
                from_agent="orchestrator",
                to_agent=agent_type,
                duration_ms=processing_time,
                success=True
            )
            
            return AgentResponse(
                status="success",
                result=result,
                agent_id=agent_type,
                processing_time_ms=processing_time
            )
    
    except httpx.TimeoutException:
        background_tasks.add_task(
            log_interaction,
            from_agent="orchestrator",
            to_agent=agent_type,
            duration_ms=(time.time() - start_time) * 1000,
            success=False,
            error="timeout"
        )
        raise HTTPException(status_code=504, detail="Agent timeout")
    
    except Exception as e:
        background_tasks.add_task(
            log_interaction,
            from_agent="orchestrator",
            to_agent=agent_type,
            duration_ms=(time.time() - start_time) * 1000,
            success=False,
            error=str(e)
        )
        raise HTTPException(status_code=502, detail=f"Agent error: {str(e)}")

async def log_interaction(**kwargs):
    """Log for monitoring."""
    # Send to monitoring system
    pass
```

---

### 2. Failure Handling and Recovery Strategies

#### **Strategy 1: Retry with Exponential Backoff**

When an agent fails, retry with increasing delays:

```python
import asyncio
import random
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar('T')

class AgentRetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions
    
    async def execute(self, operation: Callable[[], T], 
                      context: dict = None) -> T:
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay with exponential backoff + jitter
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                # Add jitter (±25%) to prevent thundering herd
                jitter = delay * 0.25 * (2 * random.random() - 1)
                actual_delay = delay + jitter
                
                print(f"Attempt {attempt + 1} failed: {e}. "
                      f"Retrying in {actual_delay:.2f}s...")
                
                await asyncio.sleep(actual_delay)
        
        raise last_exception

# Usage
retry_policy = AgentRetryPolicy(
    max_retries=3,
    base_delay=1.0,
    retryable_exceptions=(AgentTimeout, AgentError)
)

async def call_research_agent(query: str):
    return await retry_policy.execute(
        lambda: research_agent.process(query),
        context={"query": query}
    )
```

**Backoff Formula:**
$$\text{Delay}_n = \min(\text{base} \times \text{factor}^n, \text{max_delay}) \times (1 + \text{jitter})$$

---

#### **Strategy 2: Circuit Breaker Pattern**

Prevent cascading failures by stopping calls to failing agents:

```python
import time
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject fast
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.lock = Lock()
    
    def can_execute(self) -> bool:
        """Check if call should proceed."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    print("Circuit CLOSED - service recovered")
            else:
                self.failure_count = 0
    
    def record_failure(self):
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print("Circuit OPEN - recovery failed")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"Circuit OPEN after {self.failure_count} failures")
    
    async def call(self, operation, fallback=None):
        """Execute with circuit breaker protection."""
        if not self.can_execute():
            if fallback:
                return await fallback()
            raise CircuitBreakerOpen("Circuit is OPEN - service unavailable")
        
        try:
            result = await operation()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

# Usage per agent
agent_breakers = {
    "research": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
    "writer": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "critic": CircuitBreaker(failure_threshold=3, recovery_timeout=30)
}

async def safe_call_research(query: str):
    breaker = agent_breakers["research"]
    
    async def fallback():
        # Use cached result or simplified agent
        return {"result": "Using cached research", "source": "fallback"}
    
    return await breaker.call(
        lambda: research_agent.process(query),
        fallback=fallback
    )
```

**State Transitions:**
```
CLOSED ──►[failures ≥ threshold]──► OPEN ──►[timeout]──► HALF_OPEN
   ▲                                               │
   │                                               │
   └──[successes ≥ threshold]──────────────────────┘
```

---

#### **Strategy 3: Dead Letter Queue (DLQ)**

Failed tasks go to DLQ for later analysis and replay:

```python
import json
from datetime import datetime

class DeadLetterQueue:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.dlq_key = "agent:dlq"
    
    def send_to_dlq(self, task: dict, error: Exception, 
                    agent_id: str, retry_count: int):
        """Send failed task to DLQ."""
        dlq_message = {
            "original_task": task,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            },
            "agent_id": agent_id,
            "failed_at": datetime.utcnow().isoformat(),
            "retry_count": retry_count,
            "dlq_entry_id": str(uuid.uuid4())
        }
        
        # Add to DLQ
        self.redis.lpush(self.dlq_key, json.dumps(dlq_message))
        
        # Alert if DLQ grows too fast
        dlq_size = self.redis.llen(self.dlq_key)
        if dlq_size > 100:
            alert_ops(f"DLQ size critical: {dlq_size} messages")
    
    def replay_task(self, dlq_entry_id: str, 
                    agent_executor: callable) -> bool:
        """Manually replay a task from DLQ."""
        # Find and remove from DLQ
        dlq_messages = self.redis.lrange(self.dlq_key, 0, -1)
        
        for msg in dlq_messages:
            data = json.loads(msg)
            if data["dlq_entry_id"] == dlq_entry_id:
                try:
                    # Attempt replay
                    result = agent_executor(data["original_task"])
                    # Remove from DLQ on success
                    self.redis.lrem(self.dlq_key, 0, msg)
                    return True
                except Exception as e:
                    # Update retry count
                    data["retry_count"] += 1
                    data["last_replay_error"] = str(e)
                    self.redis.lrem(self.dlq_key, 0, msg)
                    self.redis.lpush(self.dlq_key, json.dumps(data))
                    return False
        
        return False
    
    def get_dlq_stats(self) -> dict:
        """Get DLQ statistics."""
        messages = self.redis.lrange(self.dlq_key, 0, -1)
        
        stats = {
            "total_messages": len(messages),
            "by_agent": {},
            "by_error_type": {},
            "oldest_failure": None,
            "avg_retry_count": 0
        }
        
        total_retries = 0
        for msg in messages:
            data = json.loads(msg)
            agent = data["agent_id"]
            error_type = data["error"]["type"]
            
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
            stats["by_error_type"][error_type] = \
                stats["by_error_type"].get(error_type, 0) + 1
            total_retries += data.get("retry_count", 0)
        
        if messages:
            stats["avg_retry_count"] = total_retries / len(messages)
        
        return stats
```

---

#### **Strategy 4: Fallback and Degradation**

When primary agent fails, use fallback:

```python
from typing import List, Optional

class AgentWithFallback:
    def __init__(
        self,
        primary_agent,
        fallback_agents: List,  # Ordered by preference
        use_cache_on_failure: bool = True
    ):
        self.primary = primary_agent
        self.fallbacks = fallback_agents
        self.cache = {}
        self.use_cache = use_cache_on_failure
    
    async def process(self, task: dict) -> dict:
        """Process with fallback chain."""
        cache_key = self._generate_cache_key(task)
        
        agents = [self.primary] + self.fallbacks
        last_error = None
        
        for i, agent in enumerate(agents):
            try:
                result = await agent.process(task)
                
                # Cache successful result
                self.cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time(),
                    "agent_used": agent.name
                }
                
                # Add metadata
                result["_meta"] = {
                    "agent_tier": "primary" if i == 0 else f"fallback_{i}",
                    "attempt": i + 1
                }
                
                return result
                
            except Exception as e:
                last_error = e
                print(f"Agent {agent.name} failed: {e}")
                continue
        
        # All agents failed - try cache
        if self.use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            age_hours = (time.time() - cached["timestamp"]) / 3600
            
            return {
                "result": cached["result"],
                "_meta": {
                    "agent_tier": "cached",
                    "cache_age_hours": age_hours,
                    "warning": "Using cached result due to agent failures"
                }
            }
        
        # Complete failure
        raise AgentCascadeFailure(
            f"All agents failed. Last error: {last_error}"
        )
    
    def _generate_cache_key(self, task: dict) -> str:
        """Generate deterministic cache key."""
        import hashlib
        task_str = json.dumps(task, sort_keys=True)
        return hashlib.md5(task_str.encode()).hexdigest()

# Usage
research_with_fallback = AgentWithFallback(
    primary_agent=OpenAIResearchAgent(model="gpt-4"),
    fallback_agents=[
        OpenAIResearchAgent(model="gpt-3.5-turbo"),
        LocalResearchAgent(model="llama-2-70b"),
        SimpleRuleBasedAgent()
    ],
    use_cache_on_failure=True
)
```

---

### 3. Observability and Monitoring

Production multi-agent systems need comprehensive observability:

#### **Metrics to Track**

```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class AgentMetrics:
    agent_id: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    circuit_breaker_opens: int = 0
    retry_attempts: int = 0
    last_error: str = None

class MultiAgentMonitor:
    def __init__(self):
        self.metrics: Dict[str, AgentMetrics] = {}
        self.latency_history: Dict[str, List[float]] = {}
        self.message_flow: List[dict] = []
    
    def record_call(self, agent_id: str, success: bool, 
                    latency_ms: float, error: str = None):
        """Record agent call metrics."""
        if agent_id not in self.metrics:
            self.metrics[agent_id] = AgentMetrics(agent_id=agent_id)
            self.latency_history[agent_id] = []
        
        metric = self.metrics[agent_id]
        metric.total_calls += 1
        
        if success:
            metric.successful_calls += 1
        else:
            metric.failed_calls += 1
            metric.last_error = error
        
        # Update latency
        self.latency_history[agent_id].append(latency_ms)
        if len(self.latency_history[agent_id]) > 1000:
            self.latency_history[agent_id] = self.latency_history[agent_id][-1000:]
        
        # Calculate statistics
        latencies = self.latency_history[agent_id]
        metric.avg_latency_ms = sum(latencies) / len(latencies)
        metric.p99_latency_ms = sorted(latencies)[int(len(latencies) * 0.99)]
    
    def record_message(self, from_agent: str, to_agent: str, 
                       message_type: str, size_bytes: int):
        """Record inter-agent communication."""
        self.message_flow.append({
            "timestamp": time.time(),
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "size_bytes": size_bytes
        })
    
    def get_health_status(self) -> dict:
        """Get system health overview."""
        status = {
            "healthy_agents": [],
            "degraded_agents": [],
            "unhealthy_agents": [],
            "overall_health": "healthy"
        }
        
        for agent_id, metric in self.metrics.items():
            if metric.total_calls == 0:
                continue
            
            error_rate = metric.failed_calls / metric.total_calls
            
            if error_rate < 0.01 and metric.avg_latency_ms < 1000:
                status["healthy_agents"].append(agent_id)
            elif error_rate < 0.05 and metric.avg_latency_ms < 5000:
                status["degraded_agents"].append({
                    "agent": agent_id,
                    "error_rate": error_rate,
                    "avg_latency": metric.avg_latency_ms
                })
            else:
                status["unhealthy_agents"].append({
                    "agent": agent_id,
                    "error_rate": error_rate,
                    "avg_latency": metric.avg_latency_ms,
                    "last_error": metric.last_error
                })
        
        # Determine overall health
        if status["unhealthy_agents"]:
            status["overall_health"] = "critical"
        elif status["degraded_agents"]:
            status["overall_health"] = "degraded"
        
        return status
    
    def export_metrics(self) -> dict:
        """Export for Prometheus/Grafana."""
        return {
            "agent_metrics": {
                agent_id: {
                    "success_rate": m.successful_calls / max(m.total_calls, 1),
                    "error_rate": m.failed_calls / max(m.total_calls, 1),
                    "avg_latency_ms": m.avg_latency_ms,
                    "p99_latency_ms": m.p99_latency_ms,
                    "circuit_breaker_opens": m.circuit_breaker_opens
                }
                for agent_id, m in self.metrics.items()
            },
            "message_flow_rate": len(self.message_flow) / 60,  # per minute
            "timestamp": time.time()
        }

# Prometheus integration
from prometheus_client import Counter, Histogram, Gauge

agent_calls_total = Counter(
    'agent_calls_total', 
    'Total calls per agent', 
    ['agent_id', 'status']
)
agent_latency = Histogram(
    'agent_latency_seconds', 
    'Call latency', 
    ['agent_id']
)
agent_circuit_breaker = Gauge(
    'agent_circuit_breaker_state', 
    'Circuit breaker state (0=closed, 1=open, 2=half_open)', 
    ['agent_id']
)
```

---

### 4. Conflict Resolution in Multi-Agent Systems

When multiple agents work on shared state, conflicts can occur:

#### **Conflict Types**

1. **Write-Write Conflicts**: Two agents modify same data
2. **Read-Write Conflicts**: Agent reads stale data
3. **Decision Conflicts**: Agents disagree on next action

#### **Resolution Strategies**

```python
from enum import Enum
from typing import Optional
import hashlib

class ConflictStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    VERSION_VECTOR = "version_vector"
    OPERATIONAL_TRANSFORM = "operational_transform"
    AGENT_HIERARCHY = "agent_hierarchy"
    CONSENSUS_VOTING = "consensus_voting"

class ConflictResolver:
    def __init__(self, strategy: ConflictStrategy):
        self.strategy = strategy
        self.agent_priority = {
            "orchestrator": 100,
            "critic": 90,
            "writer": 80,
            "research": 70
        }
    
    def resolve_write_conflict(
        self, 
        existing_value: dict,
        new_value: dict,
        agent_id: str
    ) -> dict:
        """Resolve write-write conflict."""
        
        if self.strategy == ConflictStrategy.LAST_WRITE_WINS:
            return {
                **new_value,
                "_conflict_resolved": True,
                "_resolution_strategy": "last_write_wins",
                "_winner": agent_id
            }
        
        elif self.strategy == ConflictStrategy.AGENT_HIERARCHY:
            existing_agent = existing_value.get("_written_by", "unknown")
            
            existing_priority = self.agent_priority.get(existing_agent, 0)
            new_priority = self.agent_priority.get(agent_id, 0)
            
            if new_priority >= existing_priority:
                winner = agent_id
                result = new_value
            else:
                winner = existing_agent
                result = existing_value
            
            return {
                **result,
                "_conflict_resolved": True,
                "_resolution_strategy": "agent_hierarchy",
                "_winner": winner,
                "_loser": agent_id if winner != agent_id else existing_agent
            }
        
        elif self.strategy == ConflictStrategy.CONSENSUS_VOTING:
            # For decision conflicts, agents vote
            return self._resolve_by_consensus(existing_value, new_value, agent_id)
        
        return new_value
    
    def _resolve_by_consensus(self, option_a: dict, option_b: dict, 
                             proposing_agent: str) -> dict:
        """Multiple agents vote on best option."""
        # Simplified - in practice, gather votes from all agents
        votes = {proposing_agent: option_b}
        
        # Count votes
        option_a_votes = sum(1 for v in votes.values() if v == option_a)
        option_b_votes = sum(1 for v in votes.values() if v == option_b)
        
        winner = option_b if option_b_votes > option_a_votes else option_a
        
        return {
            **winner,
            "_conflict_resolved": True,
            "_resolution_strategy": "consensus",
            "_votes": {"option_a": option_a_votes, "option_b": option_b_votes}
        }
    
    def detect_conflict(self, state: dict, proposed_changes: dict) -> bool:
        """Detect if proposed changes conflict with current state."""
        # Check if any keys overlap
        state_keys = set(state.keys())
        change_keys = set(proposed_changes.keys())
        
        # Remove metadata keys
        metadata_keys = {"_timestamp", "_last_updated_by", "_version"}
        overlapping = (state_keys & change_keys) - metadata_keys
        
        if not overlapping:
            return False
        
        # Check if values actually differ
        for key in overlapping:
            if state.get(key) != proposed_changes.get(key):
                return True
        
        return False

# Usage in agent system
class ConflictAwareAgentSystem:
    def __init__(self):
        self.resolver = ConflictResolver(
            strategy=ConflictStrategy.AGENT_HIERARCHY
        )
        self.state_store = {}
    
    async def update_state(self, agent_id: str, changes: dict) -> dict:
        """Update state with conflict detection and resolution."""
        current_state = self.state_store.get("shared", {})
        
        if self.resolver.detect_conflict(current_state, changes):
            print(f"Conflict detected! Resolving using {self.resolver.strategy}")
            resolved = self.resolver.resolve_write_conflict(
                current_state, changes, agent_id
            )
        else:
            resolved = {**current_state, **changes}
        
        # Add metadata
        resolved["_last_updated_by"] = agent_id
        resolved["_timestamp"] = time.time()
        resolved["_version"] = current_state.get("_version", 0) + 1
        
        self.state_store["shared"] = resolved
        return resolved
```

---

### 5. Production Multi-Agent Architecture Example

Here's a complete production-ready multi-agent system:

```python
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class AgentConfig:
    agent_id: str
    agent_type: str
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker_threshold: int = 5
    priority: int = 50

class ProductionMultiAgentSystem:
    """
    Production-grade multi-agent system with:
    - Message passing communication
    - Circuit breaker protection
    - Retry with exponential backoff
    - Dead letter queue
    - Comprehensive monitoring
    - Conflict resolution
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitor = MultiAgentMonitor()
        self.dlq = DeadLetterQueue(redis_client)
        self.resolver = ConflictResolver(ConflictStrategy.AGENT_HIERARCHY)
        self.state_manager = SharedStateManager()
        self.message_broker = MessageBroker()
        
        self.logger = logging.getLogger("multi_agent_system")
    
    def register_agent(self, config: AgentConfig, agent_instance):
        """Register an agent with production safeguards."""
        self.agents[config.agent_id] = {
            "instance": agent_instance,
            "config": config
        }
        
        self.circuit_breakers[config.agent_id] = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=60.0
        )
        
        self.logger.info(f"Registered agent: {config.agent_id}")
    
    async def orchestrate(self, task: dict, 
                         agent_sequence: List[str]) -> dict:
        """Execute task through agent sequence with full resilience."""
        context = {"original_task": task, "results": {}}
        
        for agent_id in agent_sequence:
            if agent_id not in self.agents:
                raise ValueError(f"Unknown agent: {agent_id}")
            
            agent = self.agents[agent_id]
            breaker = self.circuit_breakers[agent_id]
            
            start_time = time.time()
            
            try:
                # Check circuit breaker
                if not breaker.can_execute():
                    self.logger.warning(
                        f"Circuit OPEN for {agent_id}, using fallback"
                    )
                    result = await self._fallback(agent_id, context)
                else:
                    # Execute with retry policy
                    result = await self._execute_with_retry(
                        agent, context, breaker
                    )
                    breaker.record_success()
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_call(agent_id, True, latency_ms)
                
                # Update shared state
                context["results"][agent_id] = result
                await self.state_manager.update_state(
                    task["session_id"], agent_id, result
                )
                
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_call(agent_id, False, latency_ms, str(e))
                breaker.record_failure()
                
                # Send to DLQ after retries exhausted
                self.dlq.send_to_dlq(
                    {"agent_id": agent_id, "context": context},
                    e, agent_id, agent["config"].retries
                )
                
                # Decide: fail or continue?
                if self._is_critical_agent(agent_id):
                    raise AgentCascadeFailure(f"Critical agent {agent_id} failed")
                
                # Continue with partial results
                context["results"][agent_id] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Return final result
        return {
            "status": "completed",
            "context": context,
            "health": self.monitor.get_health_status()
        }
    
    async def _execute_with_retry(self, agent, context, breaker):
        """Execute agent with retry logic."""
        retry_policy = AgentRetryPolicy(
            max_retries=agent["config"].retries,
            base_delay=1.0
        )
        
        return await retry_policy.execute(
            lambda: agent["instance"].process(context),
            context={"agent_id": agent["config"].agent_id}
        )
    
    async def _fallback(self, agent_id: str, context: dict) -> dict:
        """Provide fallback result when agent is unavailable."""
        # Try cache first
        cached = await self.state_manager.get_cached_result(
            context.get("session_id"), agent_id
        )
        
        if cached:
            return {**cached, "source": "cache", "fallback": True}
        
        # Return default response
        return {
            "status": "unavailable",
            "message": f"Agent {agent_id} temporarily unavailable",
            "fallback": True
        }
    
    def _is_critical_agent(self, agent_id: str) -> bool:
        """Determine if agent failure should stop workflow."""
        critical_agents = {"orchestrator", "validator", "critic"}
        return agent_id in critical_agents

# Example usage
async def main():
    system = ProductionMultiAgentSystem()
    
    # Register agents
    system.register_agent(
        AgentConfig("research", "researcher", priority=70),
        ResearchAgent()
    )
    system.register_agent(
        AgentConfig("writer", "writer", priority=80),
        WriterAgent()
    )
    system.register_agent(
        AgentConfig("critic", "critic", priority=90),
        CriticAgent()
    )
    
    # Execute workflow
    result = await system.orchestrate(
        task={"session_id": "abc123", "query": "Write about AI safety"},
        agent_sequence=["research", "writer", "critic"]
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Summary: Production Multi-Agent Best Practices

| Aspect | Best Practice | Implementation |
|--------|---------------|----------------|
| **Communication** | Use message passing for decoupling | RabbitMQ/Kafka with DLQ |
| **Failure Handling** | Circuit breaker + retry with backoff | Exponential backoff, 3-5 retries |
| **State Management** | Optimistic locking with conflict resolution | Version vectors or agent hierarchy |
| **Observability** | Track latency, errors, message flow | Prometheus + Grafana dashboards |
| **Recovery** | Dead letter queue for failed tasks | Redis-based DLQ with replay capability |
| **Degradation** | Graceful fallbacks | Cached results, simplified agents |

---

### Quick Reference: Multi-Agent Failure Resolution

```python
# Circuit breaker for failing agents
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

# Retry with exponential backoff
retry_policy = AgentRetryPolicy(max_retries=3, base_delay=1.0)

# Dead letter queue for analysis
dlq.send_to_dlq(task, error, agent_id, retry_count)

# Conflict resolution
resolver = ConflictResolver(strategy=ConflictStrategy.AGENT_HIERARCHY)

# Monitoring
monitor.record_call(agent_id, success, latency_ms)
```

---

### Summary Table: LangChain Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Models** | LLM interface | `ChatOpenAI`, `ChatAnthropic` |
| **Prompts** | Input formatting | `ChatPromptTemplate` |
| **Parsers** | Output processing | `StrOutputParser`, `JsonOutputParser` |
| **Chains** | Component composition | `RunnableSequence`, `RunnableParallel` |
| **Agents** | Dynamic execution | `AgentExecutor`, various agent types |
| **Tools** | External capabilities | `Tool`, `StructuredTool` |
| **Memory** | Conversation state | `ConversationBufferMemory` |
| **Retrievers** | Document search | `VectorStoreRetriever` |

### Quick Reference: LCEL Patterns

```python
# Basic chain
chain = prompt | model | parser

# With input transformation
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Parallel processing
chain = RunnableParallel(
    branch1=chain1,
    branch2=chain2
) | combine_prompt | model | parser

# With memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

---

*This guide provides comprehensive coverage of LangChain and Agent frameworks with up-to-date libraries (LangChain 0.3.x), detailed breakdowns, and step-by-step implementations.*
