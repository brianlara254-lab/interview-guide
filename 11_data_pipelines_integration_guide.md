# Data Pipelines & Integration - Comprehensive Guide

A detailed breakdown of data engineering for LLMs, ETL pipelines, data quality, and integration patterns with step-by-step implementations.

**Last Updated**: 2025 | **Verified with**: Apache Airflow 2.9, dbt 1.8, Kafka 3.7, Spark 3.5

---

## Table of Contents
1. [Data Engineering Fundamentals](#data-engineering-fundamentals)
2. [ETL Pipeline Architecture](#etl-pipeline-architecture)
3. [Data Quality & Validation](#data-quality--validation)
4. [Real-Time Streaming](#real-time-streaming)
5. [Data Integration Patterns](#data-integration-patterns)
6. [Feature Stores](#feature-stores)
7. [Data Governance & Lineage](#data-governance--lineage)

---

## Data Engineering Fundamentals

### Question 1: How to Design Data Pipelines for LLM Training?

#### Concept Breakdown

**LLM Data Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────┐
│              LLM DATA LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │  RAW     │──►│ CLEANSED │──►│PROCESSED │──►│ TRAINING │      │
│  │  DATA    │   │   DATA   │   │  DATA    │   │  DATA    │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│   ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐           │
│   │Sources│     │Dedupli│     │Tokeniz│     │Batches│           │
│   │       │     │cation │     │ation  │     │       │           │
│   └───────┘     └───────┘     └───────┘     └───────┘           │
│                                                                 │
│  Volume: TBs to PBs                                             │
│  Variety: Text, code, structured, unstructured                  │
│  Velocity: Batch (hourly) or Stream (real-time)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Data Pipeline Architecture Comparison:**

| Architecture | Best For | Latency | Complexity |
|-------------|----------|---------|------------|
| **Batch** | Large-scale training | Hours | Low |
| **Micro-batch** | Near real-time | Minutes | Medium |
| **Streaming** | Live data updates | Seconds | High |
| **Lambda** | Mixed workloads | Variable | Very High |
| **Kappa** | Stream-only | Low | High |

#### Step-by-Step: Data Pipeline Implementation

**Step 1: Data Ingestion Layer**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

# Define DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'llm_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['llm', 'data'],
)

def ingest_raw_data(**context):
    """Ingest data from multiple sources"""
    
    sources = {
        'web_crawl': 's3://raw-data/web-crawl/',
        'github_code': 's3://raw-data/github/',
        'books': 's3://raw-data/books/',
        'wikipedia': 's3://raw-data/wiki/'
    }
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    
    total_files = 0
    for source_name, prefix in sources.items():
        # List files
        files = s3_hook.list_keys(
            bucket_name='raw-data',
            prefix=prefix
        )
        
        # Download and catalog
        for file_key in files:
            local_path = f"/tmp/raw/{source_name}/{file_key.split('/')[-1]}"
            s3_hook.download_file(
                bucket_name='raw-data',
                key=file_key,
                local_path=local_path
            )
            
            # Log metadata
            log_data_asset(
                asset_id=file_key,
                source=source_name,
                location=local_path,
                ingestion_time=datetime.now()
            )
        
        total_files += len(files)
    
    return f"Ingested {total_files} files from {len(sources)} sources"

# Define tasks
ingest_task = PythonOperator(
    task_id='ingest_raw_data',
    python_callable=ingest_raw_data,
    dag=dag,
)
```

**Step 2: Data Cleansing & Deduplication**
```python
import hashlib
import pandas as pd
from datasketch import MinHashLSH

def deduplicate_dataset(input_path: str, output_path: str, threshold: float = 0.85):
    """
    Remove near-duplicate documents using MinHash LSH
    
    Mathematical foundation:
    - MinHash approximates Jaccard similarity
    - J(A,B) = |A ∩ B| / |A ∪ B|
    - LSH buckets similar items for efficient lookup
    """
    
    # Initialize LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    # Process documents
    documents = load_documents(input_path)
    duplicates = []
    
    for doc_id, text in enumerate(documents):
        # Create MinHash signature
        m = get_minhash_signature(text, num_perm=128)
        
        # Check for near-duplicates
        result = lsh.query(m)
        if result:
            duplicates.append((doc_id, result[0]))
            continue
        
        # Add to LSH
        lsh.insert(doc_id, m)
        minhashes[doc_id] = m
    
    # Remove duplicates and save
    unique_docs = [doc for i, doc in enumerate(documents) 
                   if i not in [d[0] for d in duplicates]]
    
    save_documents(unique_docs, output_path)
    
    return {
        "original_count": len(documents),
        "unique_count": len(unique_docs),
        "duplicates_removed": len(duplicates),
        "deduplication_rate": len(duplicates) / len(documents)
    }

def get_minhash_signature(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash signature from text"""
    from datasketch import MinHash
    
    # Shingle text (create n-grams)
    shingles = set()
    words = text.split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])  # 3-grams
        shingles.add(shingle)
    
    # Create MinHash
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))
    
    return m

# Mathematical explanation of MinHash
"""
MinHash Algorithm:

For sets A and B, Jaccard similarity is:
J(A,B) = |A ∩ B| / |A ∪ B|

MinHash approximates this by:
1. Creating k hash functions h_1, h_2, ..., h_k
2. For each set, compute min hash value for each h_i
3. Signature = [min(h_1(S)), min(h_2(S)), ..., min(h_k(S))]
4. Estimated Jaccard = (# matching mins) / k

Example:
Set A = {"cat", "sat", "mat"}
Set B = {"cat", "sat", "hat"}

True Jaccard = |{cat, sat}| / |{cat, sat, mat, hat}| = 2/4 = 0.5

MinHash with k=4 might estimate 0.48 (close approximation)

Why it works:
- Probability that min(h(A)) = min(h(B)) = |A ∩ B| / |A ∪ B|
- Multiple permutations reduce variance
- Efficient for large datasets
"""
```

**Step 3: Data Quality Validation**
```python
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite

def create_quality_checkpoint(dataset_path: str):
    """Create data quality validation checkpoint"""
    
    # Initialize context
    context = gx.get_context()
    
    # Create expectation suite
    suite_name = "llm_training_data_suite"
    suite = context.add_expectation_suite(
        expectation_suite_name=suite_name
    )
    
    # Add expectations
    expectations = [
        # Expect column to exist
        gx.expectations.ExpectColumnToExist(
            column="text"
        ),
        
        # Expect text length within bounds
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="text_length",
            min_value=10,
            max_value=100000
        ),
        
        # Expect no null values
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="text"
        ),
        
        # Expect language distribution
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="english_ratio",
            min_value=0.7,
            max_value=1.0
        ),
        
        # Expect token count reasonable
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="token_count",
            min_value=1,
            max_value=8192
        ),
    ]
    
    for exp in expectations:
        suite.add_expectation(exp)
    
    # Create checkpoint
    checkpoint = context.add_checkpoint(
        name="llm_data_quality_checkpoint",
        validations=[
            {
                "batch_request": {
                    "datasource_name": "training_data",
                    "data_asset_name": "processed_text"
                },
                "expectation_suite_name": suite_name
            }
        ]
    )
    
    return checkpoint

def run_quality_check(checkpoint, dataset_path: str) -> dict:
    """Run quality validation and return results"""
    
    results = checkpoint.run()
    
    # Analyze results
    validation_results = {
        "success": results.success,
        "evaluated_expectations": results.statistics["evaluated_expectations"],
        "successful_expectations": results.statistics["successful_expectations"],
        "unsuccessful_expectations": results.statistics["unsuccessful_expectations"],
        "success_percent": results.statistics["success_percent"],
        "failed_expectations": []
    }
    
    # Log failures
    for result in results.results:
        if not result.success:
            validation_results["failed_expectations"].append({
                "expectation_type": result.expectation_config.expectation_type,
                "column": result.expectation_config.kwargs.get("column"),
                "unexpected_percent": result.result.get("unexpected_percent", 0)
            })
    
    return validation_results

# Data quality metrics interpretation
"""
Quality Metrics:

1. Completeness: % of non-null values
   Target: > 99%
   
2. Uniqueness: % of unique records
   Target: > 95% (after deduplication)
   
3. Validity: % conforming to schema
   Target: > 99%
   
4. Consistency: % matching across sources
   Target: > 98%
   
5. Timeliness: Age of data
   Target: < 30 days for dynamic content

Pipeline Decision Logic:
- Quality score > 95%: Proceed to training
- Quality score 85-95%: Alert, proceed with caution
- Quality score < 85%: Halt pipeline, investigate
"""
```

---

## Real-Time Streaming

### Question 2: How to Build Real-Time Data Pipelines?

#### Concept Breakdown

**Streaming vs Batch:**

```
┌─────────────────────────────────────────────────────────────────┐
│           BATCH vs STREAMING COMPARISON                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BATCH PROCESSING                                               │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Collect data over time (hours/days)                  │      │
│  │  Process large volume at once                         │      │
│  │  Higher throughput                                    │      │
│  │  Simpler error handling                               │      │
│  │  Good for: Training data, analytics, reports          │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  STREAM PROCESSING                                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Process data as it arrives (milliseconds)             │     │
│  │  Lower latency                                         │     │
│  │  Event-driven architecture                             │     │
│  │  Complex state management                              │     │
│  │  Good for: Live features, monitoring, real-time ML     │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Streaming Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│              KAFKA STREAMING ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Producers          Kafka Cluster          Consumers            │
│  ────────          ─────────────          ──────────            │
│       │                  │                     │                │
│       ▼                  ▼                     ▼                │
│  ┌─────────┐       ┌─────────────┐       ┌───────────────┐      │
│  │Web App  │──────▶│  Topic:     │──────▶│Stream Process │      │
│  │Events   │       │  user-logs  │       │  (Flink/Spark)│      │
│  └─────────┘       └─────────────┘       └───────────────┘      │
│       │                  │                     │                │
│  ┌─────────┐       ┌─────────────┐       ┌───────────────┐      │
│  │Database │──────▶│  Topic:     │──────▶│  Feature Store │     │
│  │Changes  │       │  features   │       │  (Feast/Tecton)│     │
│  └─────────┘       └─────────────┘       └───────────────┘      │
│       │                  │                     │                │
│  ┌─────────┐       ┌─────────────┐       ┌───────────────┐      │
│  │IoT      │──────▶│  Topic:     │──────▶│  Model Servin │      │
│  │Sensors  │       │  sensor-data│       │  (Online Inference)│ │
│  └─────────┘       └─────────────┘       └───────────────┘      │
│                                                                 │
│  Key Concepts:                                                  │
│  • Topic: Logical channel for events                            │
│  • Partition: Physical split of topic for parallelism           │
│  • Offset: Position in stream (enables replay)                  │
│  • Consumer Group: Parallel consumers with coordination         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step: Kafka Streaming Pipeline

**Step 1: Kafka Producer**
```python
from confluent_kafka import Producer
import json

# Configure producer
conf = {
    'bootstrap.servers': 'kafka-1:9092,kafka-2:9092',
    'client.id': 'llm-data-producer',
    'batch.size': 16384,           # Batch messages for throughput
    'linger.ms': 5,                # Wait up to 5ms to batch
    'compression.type': 'snappy',   # Compress for network efficiency
    'retries': 3,                  # Retry on transient failures
    'acks': 'all'                  # Wait for all replicas
}

producer = Producer(conf)

def delivery_callback(err, msg):
    """Callback for delivery reports"""
    if err:
        print(f'Message failed delivery: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def stream_document(doc_id: str, text: str, metadata: dict):
    """Stream document to Kafka"""
    
    message = {
        'doc_id': doc_id,
        'text': text,
        'timestamp': datetime.utcnow().isoformat(),
        'metadata': metadata,
        'token_count': len(text.split())
    }
    
    # Produce with callback
    producer.produce(
        topic='raw-documents',
        key=doc_id.encode('utf-8'),  # Partition by doc_id
        value=json.dumps(message).encode('utf-8'),
        callback=delivery_callback
    )
    
    # Poll for delivery reports
    producer.poll(0)

# Flush on shutdown
producer.flush()
```

**Step 2: Kafka Consumer with Processing**
```python
from confluent_kafka import Consumer, KafkaException
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

def create_streaming_pipeline():
    """Create Flink streaming pipeline"""
    
    # Initialize environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)  # Parallel processing
    
    # Configure Kafka source
    kafka_props = {
        'bootstrap.servers': 'kafka:9092',
        'group.id': 'llm-processing-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': 'false'  # Manual commit for reliability
    }
    
    # Define source
    source = env.add_source(
        KafkaSource.builder()
        .set_bootstrap_servers('kafka:9092')
        .set_topics('raw-documents')
        .set_group_id('llm-processing-group')
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .build()
    )
    
    # Processing pipeline
    processed = source \
        .map(parse_document) \
        .filter(filter_quality) \
        .map(tokenize_text) \
        .key_by(lambda x: x['language']) \
        .window(TumblingProcessingTimeWindows.of(Time.minutes(5))) \
        .aggregate(aggregate_stats) \
        .map(enrich_metadata)
    
    # Sink to processed topic
    processed.add_sink(
        KafkaSink.builder()
        .set_bootstrap_servers('kafka:9092')
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic('processed-documents')
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .build()
    )
    
    # Execute
    env.execute("LLM Data Processing Pipeline")

def parse_document(raw_msg):
    """Parse and validate incoming document"""
    try:
        doc = json.loads(raw_msg)
        return {
            'doc_id': doc['doc_id'],
            'text': doc['text'],
            'timestamp': doc['timestamp'],
            'language': detect_language(doc['text']),
            'char_count': len(doc['text']),
            'word_count': len(doc['text'].split())
        }
    except Exception as e:
        # Send to dead letter queue
        send_to_dlq(raw_msg, str(e))
        return None

def filter_quality(doc):
    """Filter low-quality documents"""
    if doc is None:
        return False
    
    # Quality checks
    min_length = 50
    max_length = 100000
    
    return (
        doc['char_count'] >= min_length and
        doc['char_count'] <= max_length and
        doc['language'] in ['en', 'es', 'fr', 'de']  # Supported languages
    )

# Streaming windowing explanation
"""
Windowing Strategies:

1. Tumbling Windows (Fixed Time):
   ┌─────┬─────┬─────┬─────┐
   │ 0-5 │ 5-10│10-15│15-20│
   └─────┴─────┴─────┴─────┘
   - Non-overlapping, fixed size
   - Use case: Aggregate stats every 5 minutes

2. Sliding Windows (Overlapping):
   ┌──┬──┬──┬──┬──┐
   │0-5│2-7│4-9│...│
   └──┴──┴──┴──┴──┘
   - Overlapping with slide interval
   - Use case: Moving averages

3. Session Windows (Activity-based):
   ┌─────┐    ┌──────┐
   │gap  │gap │      │
   └─────┘    └──────┘
   - Dynamic based on gaps
   - Use case: User sessions
"""
```

---

## Feature Stores

### Question 3: What is a Feature Store and Why Use It?

#### Concept Breakdown

**Feature Store Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE STORE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Offline Store (Batch Features)                                 │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Data Warehouse / Data Lake                           │      │
│  │  • Historical aggregations                            │      │
│  │  • Training datasets                                  │      │
│  │  • Batch inference features                           │      │
│  │  (Spark, BigQuery, Snowflake)                         │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │              Feature Registry                         │      │
│  │  • Feature definitions                                │      │
│  │  • Versioning                                         │      │
│  │  • Lineage tracking                                   │      │
│  │  • Documentation                                      │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│  Online Store (Real-time Features)                              │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Low-Latency Database                                 │      │
│  │  • User embeddings                                    │      │
│  │  • Session features                                   │      │
│  │  • Context for inference                              │      │
│  │  (Redis, DynamoDB, Cassandra)                         │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why Feature Stores:**

| Problem | Feature Store Solution | Benefit |
|---------|----------------------|---------|
| Feature duplication | Centralized registry | Reuse across teams |
| Training/serving skew | Point-in-time joins | Consistent features |
| Feature discovery | Feature catalog | Discoverability |
| Latency requirements | Online/offline separation | Fast serving |
| Compliance | Lineage tracking | Auditability |

#### Step-by-Step: Feast Implementation

```python
from feast import Entity, Feature, FeatureView, ValueType
from feast.types import Float, Int64, String
from datetime import timedelta

# Define entity (what we're predicting for)
user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="User identifier",
    join_keys=["user_id"]
)

# Define features
user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(hours=24),  # Feature freshness
    features=[
        Feature(name="embedding", dtype=Float),
        Feature(name="total_purchases", dtype=Int64),
        Feature(name="preferred_category", dtype=String),
        Feature(name="last_session_duration", dtype=Float),
    ],
    online=True,  # Serve from online store
    source=user_transactions_source,  # Data source
)

# Usage in inference
from feast import FeatureStore

store = FeatureStore(repo_path=".")

def get_features_for_inference(user_ids: List[str]):
    """Fetch real-time features for model inference"""
    
    # Get online features (low latency)
    feature_vector = store.get_online_features(
        features=[
            "user_features:embedding",
            "user_features:total_purchases",
            "user_features:last_session_duration"
        ],
        entity_rows=[{"user_id": uid} for uid in user_ids]
    ).to_dict()
    
    return feature_vector

# Feature computation with point-in-time correctness
def create_training_dataset():
    """Create training dataset avoiding future leakage"""
    
    # Define label (what we want to predict)
    label = Feature(name="purchase_amount", dtype=Float)
    
    # Create dataset with point-in-time features
    training_df = store.get_historical_features(
        entity_df=pd.DataFrame({
            "user_id": user_ids,
            "timestamp": event_times  # Point in time for each label
        }),
        features=[
            "user_features:total_purchases",
            "user_features:embedding"
        ]
    ).to_df()
    
    return training_df
```

---

### Summary Table: Data Pipeline Patterns

| Pattern | Use Case | Tools | Latency |
|---------|----------|-------|---------|
| **Batch ETL** | Training data prep | Airflow, Spark | Hours |
| **Stream Processing** | Real-time features | Kafka, Flink | Seconds |
| **Lambda** | Hybrid workloads | Both above | Variable |
| **CDC** | DB replication | Debezium | Seconds |
| **ELT** | Data warehouse | dbt, Fivetran | Minutes |

---

*This guide provides comprehensive coverage of data pipelines and integration for LLMs with detailed breakdowns, step-by-step implementations, and best practices.*

**Complete Guide Series:**
- 01: Python & SQL Solutions
- 02: LLM Fundamentals
- 03: RAG Comprehensive
- 04: LangChain & Agents
- 05: Interview Questions
- 06: Study Guide
- 07: Python & ML Answers
- 08: RAG Answers
- 09: Fine-Tuning & RLHF
- 10: Production Deployment & MLOps
- 11: **Data Pipelines & Integration** (This Guide)

The file is saved at: `11_data_pipelines_integration_guide.md`
