# Python & SQL Interview Solutions with Concept Explanations

A comprehensive guide covering Python programming and SQL concepts with detailed explanations, example schemas, and practical solutions.

---

## Table of Contents
1. [Example Database Schema](#example-database-schema)
2. [Python Fundamentals](#python-fundamentals)
3. [SQL Fundamentals](#sql-fundamentals)
4. [Advanced SQL](#advanced-sql)
5. [Python + SQL Integration](#python--sql-integration)

---



## Python Fundamentals

### 1. Lists vs Tuples vs Sets vs Dictionaries

#### Concept Explanation

| Data Structure | Ordered | Mutable | Duplicates | Lookup Speed | Use Case |
|---------------|---------|---------|------------|--------------|----------|
| **List** | Yes | Yes | Allowed | O(n) | Sequential data, stack/queue operations |
| **Tuple** | Yes | No | Allowed | O(n) | Fixed data, dictionary keys, return values |
| **Set** | No | Yes | Not Allowed | O(1) | Membership testing, deduplication |
| **Dictionary** | No* | Yes | Keys: No, Values: Yes | O(1) | Key-value mapping, caching |

*Python 3.7+ maintains insertion order

```python
# ============ LISTS ============
# Dynamic array implementation
# Underlying concept: Contiguous memory allocation with over-allocation

# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = list(range(10))  # [0, 1, 2, ..., 9]

# List operations - Time Complexities:
# - Append: O(1) amortized (occasional resize O(n))
# - Insert at index: O(n) - requires shifting elements
# - Delete: O(n) - requires shifting elements
# - Access by index: O(1)
# - Search: O(n)

# Memory layout visualization:
# Index:    [0]     [1]     [2]     [3]
# Memory:  |apple| |banana| |cherry| |None|  <- None = available space
# Pointer:   ↑
#           base

# Practical example: Processing order history
orders = [
    {"order_id": 1, "amount": 100.50, "status": "completed"},
    {"order_id": 2, "amount": 250.00, "status": "pending"},
    {"order_id": 3, "amount": 75.25, "status": "completed"}
]

# Add new order
orders.append({"order_id": 4, "amount": 300.00, "status": "processing"})

# Filter completed orders
completed = [o for o in orders if o["status"] == "completed"]


# ============ TUPLES ============
# Immutable sequence - similar to lists but cannot be modified after creation
# Underlying concept: Fixed-size array with reference counting

# Why use tuples?
# 1. Data integrity (hashable, can be dict keys)
# 2. Memory efficiency (smaller than lists)
# 3. Performance (slightly faster iteration)
# 4. Semantic meaning (indicates "record" structure)

# Creating tuples
coordinates = (40.7128, -74.0060)  # NYC coordinates
rgb_color = (255, 128, 0)

# Tuple packing/unpacking
user_record = ("john_doe", "john@example.com", 25)
username, email, age = user_record  # Unpacking

# Named tuples for better readability
from collections import namedtuple
User = namedtuple('User', ['username', 'email', 'age'])
user = User("john_doe", "john@example.com", 25)
print(user.username)  # More readable than user[0]

# Tuple as dictionary key (requires immutability)
location_sales = {
    ("NYC", "2024-01"): 50000,
    ("LA", "2024-01"): 45000
}


# ============ SETS ============
# Hash table implementation (like dict keys only)
# Underlying concept: Hash table with open addressing

# Time Complexities:
# - Add: O(1) average
# - Remove: O(1) average
# - Membership test (x in set): O(1) average
# - Union/Intersection: O(min(len(s), len(t)))

# Hash table visualization:
# Index:   0     1      2      3      4      5      6
#        |apple| None |banana| None | None |cherry| None|
#               ↑
#            hash("banana") % 7 = 2

# Creating sets
unique_categories = {"electronics", "clothing", "food"}
from_list = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Set operations - useful for data analysis
customers_2023 = {"alice", "bob", "charlie"}
customers_2024 = {"bob", "charlie", "diana"}

# Set operations
retained = customers_2023 & customers_2024      # Intersection: {"bob", "charlie"}
churned = customers_2023 - customers_2024       # Difference: {"alice"}
new_customers = customers_2024 - customers_2023 # Difference: {"diana"}
all_customers = customers_2023 | customers_2024 # Union: {"alice", "bob", "charlie", "diana"}

# Real-world: Find duplicate orders
def find_duplicate_orders(order_ids):
    seen = set()
    duplicates = set()
    for oid in order_ids:
        if oid in seen:
            duplicates.add(oid)
        else:
            seen.add(oid)
    return duplicates


# ============ DICTIONARIES ============
# Hash map implementation - key-value pairs
# Underlying concept: Hash table with dynamic resizing

# Time Complexities:
# - Get/Set/Delete: O(1) average
# - Keys/Values/Items: O(n)
# - Space: O(n)

# Creating dictionaries
user_profile = {
    "user_id": 101,
    "username": "jane_smith",
    "email": "jane@example.com",
    "preferences": {
        "theme": "dark",
        "notifications": True
    }
}

# Dictionary methods
user_profile.get("age", 25)  # Default value if key doesn't exist
user_profile.setdefault("country", "US")  # Set if not exists

# Dictionary comprehensions
prices = {"apple": 1.00, "banana": 0.50, "cherry": 2.00}
discounted = {k: v * 0.9 for k, v in prices.items()}  # 10% off

# Merging dictionaries (Python 3.9+)
defaults = {"theme": "light", "language": "en"}
user_prefs = {"theme": "dark"}
combined = defaults | user_prefs  # {"theme": "dark", "language": "en"}

# Real-world: Product catalog lookup
product_catalog = {
    1001: {"name": "Laptop", "price": 999.99, "stock": 50},
    1002: {"name": "Mouse", "price": 29.99, "stock": 200},
    1003: {"name": "Keyboard", "price": 79.99, "stock": 150}
}

# Fast lookup
product = product_catalog.get(1001)  # O(1) vs O(n) for list search
```

---

### 2. List Comprehensions vs Generator Expressions

#### Concept Explanation

**List Comprehensions**: Create entire list in memory
**Generator Expressions**: Create iterator, yield one item at a time

```python
# ============ LIST COMPREHENSIONS ============
# Syntax: [expression for item in iterable if condition]
# Memory: Stores all results immediately

# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]

# With condition (filtering)
evens = [x for x in numbers if x % 2 == 0]  # [2, 4]

# Multiple iterables
pairs = [(x, y) for x in [1, 2] for y in ['a', 'b']]
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Practical: Transform SQL results
users_data = [
    {"user_id": 1, "name": "Alice", "age": 25},
    {"user_id": 2, "name": "Bob", "age": 30},
]
names_upper = [u["name"].upper() for u in users_data]


# ============ GENERATOR EXPRESSIONS ============
# Syntax: (expression for item in iterable if condition)
# Memory: Lazy evaluation, one item at a time

# Generator expression
numbers = range(1, 1000001)
squares_gen = (x**2 for x in numbers)  # No memory allocated yet!

# Memory comparison:
import sys

# List comprehension - all values in memory
squares_list = [x**2 for x in range(10000)]
print(f"List size: {sys.getsizeof(squares_list)} bytes")  # ~80000 bytes

# Generator expression - iterator object only
squares_gen = (x**2 for x in range(10000))
print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")  # ~112 bytes

# Generator usage patterns
def process_large_file(filepath):
    """Process file line by line without loading entire file"""
    with open(filepath, 'r') as f:
        # Generator expression - processes one line at a time
        return (line.strip().upper() for line in f if line.strip())

# Chaining generators
def get_active_users(user_generator):
    """Filter active users lazily"""
    return (u for u in user_generator if u.get('is_active'))

def extract_emails(user_generator):
    """Extract emails lazily"""
    return (u.get('email') for u in user_generator)

# Chain: users -> active_users -> emails (all lazy!)
all_users = ({"name": f"User{i}", "is_active": i % 2 == 0} for i in range(1000000))
active_emails = extract_emails(get_active_users(all_users))

# Consume only what you need
for email in active_emails:
    if email:
        send_notification(email)
    if some_condition:
        break  # Stop early without processing remaining items


# ============ WHEN TO USE EACH ============
"""
Use List Comprehensions when:
- You need to access elements multiple times
- You need random access (indexing)
- The dataset is small enough to fit in memory
- You need the length

Use Generator Expressions when:
- Processing large datasets
- Streaming data processing
- Chaining multiple operations
- Only need to iterate once
- Memory is constrained
"""
```

---

### 3. Decorators in Python

#### Concept Explanation

A decorator is a design pattern that allows you to modify or enhance functions without changing their source code. It's implemented using higher-order functions (functions that take/return functions).

```python
# ============ BASIC DECORATOR STRUCTURE ============
"""
Decorator execution flow:
1. Function defined
2. Decorator called with function as argument
3. Decorator returns wrapper function
4. Original function name now points to wrapper

Before decoration:
    my_func -> function object

After decoration:
    my_func -> wrapper -> calls original function
"""

def my_decorator(func):
    """
    The decorator function receives the original function.
    Returns a wrapper that adds behavior before/after.
    """
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    """Original function"""
    return f"Hello, {name}!"

# Equivalent to: say_hello = my_decorator(say_hello)


# ============ PRACTICAL DECORATORS ============

# 1. Timing Decorator - Profile function execution
def timer(func):
    """Measure and log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def expensive_query():
    """Simulate database query"""
    import time
    time.sleep(0.1)
    return ["result1", "result2"]


# 2. Retry Decorator - Handle transient failures
def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """
    Retry a function on failure with exponential backoff.
    Common for network calls, database operations.
    """
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    wait = delay * (2 ** (attempt - 1))  # Exponential backoff
                    print(f"Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1, exceptions=(ConnectionError,))
def fetch_data_from_api(url):
    """API call that might fail intermittently"""
    import random
    if random.random() < 0.7:  # 70% failure rate for demo
        raise ConnectionError("Network error")
    return {"data": "success"}


# 3. Cache/Memoization Decorator
def memoize(func):
    """
    Cache function results to avoid recomputation.
    Useful for expensive, deterministic functions.
    """
    import functools
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    wrapper.cache = cache  # Expose cache for inspection
    return wrapper

@memoize
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Without memoization: O(2^n) time
# With memoization: O(n) time


# 4. Validation Decorator - Input sanitization
def validate_types(**expected_types):
    """Validate function argument types at runtime"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get argument names
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each argument
            for arg_name, expected_type in expected_types.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{arg_name} must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(user_id=int, name=str)
def create_user(user_id, name, email=None):
    """Create a new user with validated inputs"""
    return {"id": user_id, "name": name, "email": email}


# 5. Rate Limiter Decorator - API throttling
def rate_limit(calls_per_minute=60):
    """Limit function call frequency"""
    import functools
    import time
    from collections import deque
    
    def decorator(func):
        call_times = deque()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove calls older than 1 minute
            while call_times and call_times[0] < now - 60:
                call_times.popleft()
            
            if len(call_times) >= calls_per_minute:
                raise RuntimeError("Rate limit exceeded")
            
            call_times.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

@rate_limit(calls_per_minute=10)
def call_external_api(endpoint):
    """API call with rate limiting"""
    return f"Data from {endpoint}"


# ============ CLASS-BASED DECORATORS ============
class CountCalls:
    """Decorator as a class - maintains state"""
    
    def __init__(self, func):
        import functools
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count} of {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def process_order(order_id):
    return f"Processing order {order_id}"


# ============ DECORATORS IN ML/DATA PIPELINES ============
def log_data_pipeline(stage_name):
    """Log data transformations in ML pipeline"""
    import functools
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            input_shape = len(data) if hasattr(data, '__len__') else 'unknown'
            logger.info(f"[{stage_name}] Input shape: {input_shape}")
            
            result = func(data, *args, **kwargs)
            
            output_shape = len(result) if hasattr(result, '__len__') else 'unknown'
            logger.info(f"[{stage_name}] Output shape: {output_shape}")
            
            return result
        return wrapper
    return decorator

@log_data_pipeline("feature_engineering")
def extract_features(raw_data):
    """Extract features from raw data"""
    return [{"feature": item * 2} for item in raw_data]

@log_data_pipeline("normalization")
def normalize_features(features):
    """Normalize feature values"""
    max_val = max(f["feature"] for f in features)
    return [{"feature": f["feature"] / max_val} for f in features]
```

---
## Example Database Schema

### E-Commerce Database Schema

```sql
-- Users Table
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    country VARCHAR(50),
    age INT CHECK (age >= 13)
);

-- Products Table
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(200),
    category_id INT,
    price DECIMAL(10, 2),
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Categories Table
CREATE TABLE categories (
    category_id INT PRIMARY KEY,
    category_name VARCHAR(100),
    parent_category_id INT,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
);

-- Orders Table
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20) CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Order Items Table
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Reviews Table
CREATE TABLE reviews (
    review_id INT PRIMARY KEY,
    product_id INT,
    user_id INT,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

---

## SQL Fundamentals

### 1. SELECT, WHERE, ORDER BY, LIMIT

#### Concept Explanation

The basic SQL query execution order (logical processing):
1. **FROM** - Identify source tables
2. **WHERE** - Filter rows
3. **GROUP BY** - Group rows
4. **HAVING** - Filter groups
5. **SELECT** - Select columns
6. **ORDER BY** - Sort results
7. **LIMIT/OFFSET** - Limit rows

```sql
-- ============ BASIC SELECT ============
-- Retrieve all columns
SELECT * FROM users;

-- Retrieve specific columns (more efficient)
SELECT user_id, username, email FROM users;

-- Alias columns for readability
SELECT 
    user_id AS id,
    username AS name,
    created_at AS registration_date
FROM users;

-- DISTINCT - Remove duplicates
SELECT DISTINCT country FROM users;

-- COUNT DISTINCT
SELECT COUNT(DISTINCT country) AS unique_countries FROM users;


-- ============ WHERE CLAUSE ============
-- Filter rows based on conditions

-- Comparison operators: =, <>, !=, <, >, <=, >=
SELECT * FROM products WHERE price > 100;

-- Logical operators: AND, OR, NOT
SELECT * FROM products 
WHERE price > 50 AND stock_quantity > 0;

-- IN operator (better than multiple OR)
SELECT * FROM users 
WHERE country IN ('USA', 'UK', 'Canada');

-- BETWEEN (inclusive)
SELECT * FROM products 
WHERE price BETWEEN 50 AND 100;

-- LIKE - Pattern matching
-- % = zero or more characters
-- _ = single character
SELECT * FROM users WHERE username LIKE 'john%';  -- starts with 'john'
SELECT * FROM users WHERE email LIKE '%@gmail.com';  -- ends with '@gmail.com'
SELECT * FROM users WHERE username LIKE '_ohn';  -- 4 chars, ends with 'ohn'

-- NULL handling (use IS, not =)
SELECT * FROM users WHERE email IS NULL;
SELECT * FROM users WHERE email IS NOT NULL;


-- ============ ORDER BY ============
-- Sort results

-- Ascending (default)
SELECT * FROM products ORDER BY price;
SELECT * FROM products ORDER BY price ASC;

-- Descending
SELECT * FROM products ORDER BY price DESC;

-- Multiple columns
SELECT * FROM products 
ORDER BY category_id ASC, price DESC;
-- First by category (A-Z), then by price (high to low)

-- ORDER BY with expressions
SELECT 
    product_name,
    price,
    stock_quantity,
    price * stock_quantity AS inventory_value
FROM products
ORDER BY inventory_value DESC;


-- ============ LIMIT/OFFSET ============
-- Pagination

-- Top N results
SELECT * FROM products 
ORDER BY price DESC 
LIMIT 10;  -- Top 10 most expensive

-- Pagination
SELECT * FROM users 
ORDER BY user_id 
LIMIT 10 OFFSET 20;  -- Page 3 (20 records per page)

-- Alternative syntax (MySQL/PostgreSQL)
SELECT * FROM users 
ORDER BY user_id 
LIMIT 20, 10;  -- Same as above

-- Alternative syntax (SQL Server)
SELECT TOP 10 * FROM products ORDER BY price DESC;

-- Alternative syntax (Oracle)
SELECT * FROM products 
WHERE ROWNUM <= 10 
ORDER BY price DESC;
```

---

### 2. JOIN Operations

#### Concept Explanation

JOINs combine rows from two or more tables based on a related column.

**Types of JOINs:**
- **INNER JOIN**: Returns only matching rows from both tables
- **LEFT JOIN**: Returns all rows from left table, matching rows from right (NULL if no match)
- **RIGHT JOIN**: Returns all rows from right table, matching rows from left
- **FULL OUTER JOIN**: Returns all rows when there's a match in either table

```sql
-- ============ INNER JOIN ============
-- Returns only rows where there's a match in BOTH tables

-- Visual representation:
-- Table A        Table B        Result (INNER JOIN)
-- ┌────┬─────┐   ┌────┬─────┐   ┌────┬─────┬─────┐
-- │ id │ name│   │ id │ val │   │ id │ name│ val │
-- ├────┼─────┤   ├────┼─────┤   ├────┼─────┼─────┤
-- │ 1  │ A   │   │ 2  │ X   │   │ 2  │ B   │ X   │
-- │ 2  │ B   │   │ 3  │ Y   │   │ 3  │ C   │ Y   │
-- │ 3  │ C   │   │ 4  │ Z   │   └────┴─────┴─────┘
-- │ 4  │ D   │   └────┴─────┘
-- └────┴─────┘
-- (id=4 has no match in B, id=4 in B has no match in A)

-- Get orders with user information
SELECT 
    o.order_id,
    o.order_date,
    u.username,
    u.email
FROM orders o
INNER JOIN users u ON o.user_id = u.user_id;

-- Get order details with product info
SELECT 
    o.order_id,
    u.username,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    (oi.quantity * oi.unit_price) AS line_total
FROM orders o
INNER JOIN users u ON o.user_id = u.user_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id;


-- ============ LEFT JOIN ============
-- Returns ALL rows from left table, matching rows from right (NULL if no match)

-- Visual representation:
-- Table A        Table B        Result (LEFT JOIN A -> B)
-- ┌────┬─────┐   ┌────┬─────┐   ┌────┬─────┬──────┐
-- │ id │ name│   │ a_id│val │   │ id │ name│ val  │
-- ├────┼─────┤   ├─────┼─────┤   ├────┼─────┼──────┤
-- │ 1  │ A   │   │ 2   │ X   │   │ 1  │ A   │ NULL │
-- │ 2  │ B   │   │ 3   │ Y   │   │ 2  │ B   │ X    │
-- │ 3  │ C   │   └─────┴─────┘   │ 3  │ C   │ Y    │
-- └────┴─────┘                   └────┴─────┴──────┘

-- Find users who haven't placed any orders
SELECT 
    u.user_id,
    u.username,
    u.email,
    o.order_id  -- Will be NULL for users without orders
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_id IS NULL;

-- Get all products with their category (even uncategorized)
SELECT 
    p.product_id,
    p.product_name,
    c.category_name  -- NULL if product has no category
FROM products p
LEFT JOIN categories c ON p.category_id = c.category_id;


-- ============ RIGHT JOIN ============
-- Returns ALL rows from right table, matching rows from left
-- (Less common, usually rewritten as LEFT JOIN for readability)

-- Equivalent to LEFT JOIN with tables swapped
SELECT 
    c.category_name,
    p.product_name
FROM products p
RIGHT JOIN categories c ON p.category_id = c.category_id;

-- Better: Use LEFT JOIN
SELECT 
    c.category_name,
    p.product_name
FROM categories c
LEFT JOIN products p ON c.category_id = p.category_id;


-- ============ FULL OUTER JOIN ============
-- Returns ALL rows from both tables, matching where possible

-- Visual representation:
-- Table A        Table B        Result (FULL OUTER JOIN)
-- ┌────┬─────┐   ┌────┬─────┐   ┌────┬─────┬──────┐
-- │ id │ name│   │ id │ val │   │ id │ name│ val  │
-- ├────┼─────┤   ├────┼─────┤   ├────┼─────┼──────┤
-- │ 1  │ A   │   │ 2  │ X   │   │ 1  │ A   │ NULL │
-- │ 2  │ B   │   │ 3  │ Y   │   │ 2  │ B   │ X    │
-- │ 3  │ C   │   │ 4  │ Z   │   │ 3  │ C   │ Y    │
-- └────┴─────┘   └────┴─────┘   │ 4  │ NULL│ Z    │
--                               └────┴─────┴──────┘

-- Find all customers and orders, even unmatched
SELECT 
    u.username,
    o.order_id,
    o.order_date
FROM users u
FULL OUTER JOIN orders o ON u.user_id = o.user_id;

-- Note: MySQL doesn't support FULL OUTER JOIN, use UNION:
SELECT u.username, o.order_id FROM users u LEFT JOIN orders o ON u.user_id = o.user_id
UNION
SELECT u.username, o.order_id FROM users u RIGHT JOIN orders o ON u.user_id = o.user_id;


-- ============ SELF JOIN ============
-- Join a table to itself

-- Find employees and their managers
-- Table: employees(employee_id, name, manager_id)
SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;

-- Find categories and their parent categories
SELECT 
    c.category_name AS category,
    p.category_name AS parent_category
FROM categories c
LEFT JOIN categories p ON c.parent_category_id = p.category_id;


-- ============ CROSS JOIN ============
-- Cartesian product - every row from A with every row from B
-- Use case: Generate all combinations

-- Generate all user-product pairs for a survey
SELECT 
    u.user_id,
    p.product_id
FROM users u
CROSS JOIN products p;
-- Result: |users| × |products| rows
```

---

### 3. Aggregate Functions and GROUP BY

#### Concept Explanation

Aggregate functions perform calculations on a set of values and return a single value.

**Common Aggregate Functions:**
- `COUNT()` - Count rows
- `SUM()` - Sum of values
- `AVG()` - Average of values
- `MAX()` - Maximum value
- `MIN()` - Minimum value

```sql
-- ============ BASIC AGGREGATES ============

-- COUNT
SELECT COUNT(*) AS total_users FROM users;
SELECT COUNT(email) AS users_with_email FROM users;  -- Excludes NULL
SELECT COUNT(DISTINCT country) AS unique_countries FROM users;

-- SUM
SELECT SUM(total_amount) AS total_revenue FROM orders;
SELECT SUM(quantity) AS total_items_sold FROM order_items;

-- AVG
SELECT AVG(price) AS avg_product_price FROM products;
SELECT AVG(total_amount) AS avg_order_value FROM orders;

-- MAX/MIN
SELECT MAX(price) AS highest_price FROM products;
SELECT MIN(created_at) AS first_order_date FROM orders;

-- Multiple aggregates
SELECT 
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS avg_order_value,
    MIN(total_amount) AS min_order,
    MAX(total_amount) AS max_order
FROM orders;


-- ============ GROUP BY ============
-- Group rows with same values into summary rows

-- Revenue by country
SELECT 
    u.country,
    COUNT(DISTINCT o.order_id) AS num_orders,
    SUM(o.total_amount) AS total_revenue,
    AVG(o.total_amount) AS avg_order_value
FROM users u
JOIN orders o ON u.user_id = o.user_id
GROUP BY u.country;

-- Sales by category
SELECT 
    c.category_name,
    COUNT(DISTINCT p.product_id) AS num_products,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.unit_price) AS category_revenue
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY c.category_id, c.category_name;

-- Monthly sales trend
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;


-- ============ HAVING CLAUSE ============
-- Filter groups (like WHERE but for aggregated results)
-- WHERE filters rows BEFORE grouping
-- HAVING filters groups AFTER aggregation

-- Find categories with revenue > $10,000
SELECT 
    c.category_name,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY c.category_id, c.category_name
HAVING revenue > 10000;

-- Find users who placed 5+ orders
SELECT 
    u.user_id,
    u.username,
    COUNT(o.order_id) AS order_count
FROM users u
JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.username
HAVING order_count >= 5;

-- Find products with avg rating < 3
SELECT 
    p.product_name,
    AVG(r.rating) AS avg_rating,
    COUNT(r.review_id) AS review_count
FROM products p
JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.product_name
HAVING avg_rating < 3 AND review_count >= 5;


-- ============ GROUP BY WITH ROLLUP ============
-- Create subtotals and grand total

SELECT 
    COALESCE(c.category_name, 'TOTAL') AS category,
    COALESCE(p.product_name, 'Subtotal') AS product,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY c.category_name, p.product_name WITH ROLLUP;

-- Result includes:
-- - Each product with its category
-- - Subtotal for each category
-- - Grand total (all NULLs in group columns)
```

---

### 4. Window Functions

#### Concept Explanation

Window functions perform calculations across a set of rows related to the current row without collapsing them into a single row (unlike GROUP BY).

**Key Concepts:**
- `OVER()` - Defines the window (set of rows)
- `PARTITION BY` - Divides rows into groups
- `ORDER BY` - Orders rows within the window
- `FRAME` - Specifies which rows to include (ROWS/RANGE)

```sql
-- ============ ROW_NUMBER, RANK, DENSE_RANK ============

-- ROW_NUMBER(): Sequential number, unique per row
-- RANK(): Same values get same rank, skips next ranks
-- DENSE_RANK(): Same values get same rank, no gaps

-- Example data and results:
-- Score | ROW_NUMBER | RANK | DENSE_RANK
-- ──────┼────────────┼──────┼────────────
-- 100   │ 1          │ 1    │ 1
-- 95    │ 2          │ 2    │ 2
-- 95    │ 3          │ 2    │ 2
-- 90    │ 4          │ 4    │ 3
-- 85    │ 5          │ 5    │ 4

-- Top 3 products by revenue in each category
SELECT *
FROM (
    SELECT 
        c.category_name,
        p.product_name,
        SUM(oi.quantity * oi.unit_price) AS revenue,
        ROW_NUMBER() OVER (
            PARTITION BY c.category_id 
            ORDER BY SUM(oi.quantity * oi.unit_price) DESC
        ) AS rank_in_category
    FROM categories c
    JOIN products p ON c.category_id = p.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY c.category_id, c.category_name, p.product_id, p.product_name
) ranked
WHERE rank_in_category <= 3;


-- ============ LEAD and LAG ============
-- Access data from previous/next rows

-- Month-over-month sales comparison
SELECT 
    month,
    monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY month) AS prev_month_revenue,
    monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month) AS revenue_change,
    ROUND(
        (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month)) 
        / LAG(monthly_revenue) OVER (ORDER BY month) * 100, 
        2
    ) AS pct_change
FROM (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') AS month,
        SUM(total_amount) AS monthly_revenue
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
) monthly_sales;


-- ============ Running Totals and Moving Averages ============

-- Running total of sales (cumulative sum)
SELECT 
    order_date,
    daily_sales,
    SUM(daily_sales) OVER (
        ORDER BY order_date 
        ROWS UNBOUNDED PRECEDING
    ) AS running_total
FROM (
    SELECT 
        DATE(order_date) AS order_date,
        SUM(total_amount) AS daily_sales
    FROM orders
    GROUP BY DATE(order_date)
) daily;

-- 7-day moving average
SELECT 
    order_date,
    daily_sales,
    AVG(daily_sales) OVER (
        ORDER BY order_date 
        ROWS 6 PRECEDING
    ) AS moving_avg_7day
FROM (
    SELECT 
        DATE(order_date) AS order_date,
        SUM(total_amount) AS daily_sales
    FROM orders
    GROUP BY DATE(order_date)
) daily;


-- ============ FIRST_VALUE, LAST_VALUE, NTH_VALUE ============

-- First and last order for each user
SELECT DISTINCT
    u.user_id,
    u.username,
    FIRST_VALUE(o.order_id) OVER (
        PARTITION BY u.user_id 
        ORDER BY o.order_date
    ) AS first_order_id,
    LAST_VALUE(o.order_id) OVER (
        PARTITION BY u.user_id 
        ORDER BY o.order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_order_id
FROM users u
JOIN orders o ON u.user_id = o.user_id;


-- ============ PERCENT_RANK and CUME_DIST ============

-- Calculate percentile of each product's price
SELECT 
    product_name,
    price,
    PERCENT_RANK() OVER (ORDER BY price) AS price_percentile,
    CUME_DIST() OVER (ORDER BY price) AS cumulative_distribution,
    NTILE(4) OVER (ORDER BY price) AS price_quartile
FROM products;


-- ============ Advanced Window Functions Example ============
-- Customer segmentation: Identify new vs returning customers by month

WITH customer_monthly_activity AS (
    SELECT DISTINCT
        u.user_id,
        DATE_FORMAT(o.order_date, '%Y-%m') AS order_month
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
),
customer_first_month AS (
    SELECT 
        user_id,
        MIN(order_month) AS first_month
    FROM customer_monthly_activity
    GROUP BY user_id
)
SELECT 
    cma.order_month,
    COUNT(DISTINCT CASE 
        WHEN cma.order_month = cfm.first_month THEN cma.user_id 
    END) AS new_customers,
    COUNT(DISTINCT CASE 
        WHEN cma.order_month != cfm.first_month THEN cma.user_id 
    END) AS returning_customers
FROM customer_monthly_activity cma
JOIN customer_first_month cfm ON cma.user_id = cfm.user_id
GROUP BY cma.order_month
ORDER BY cma.order_month;
```

---

## Advanced SQL

### 1. Common Table Expressions (CTEs)

#### Concept Explanation

CTEs are temporary named result sets that exist only during query execution. They improve readability and allow recursive queries.

```sql
-- ============ BASIC CTE ============
WITH monthly_revenue AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') AS month,
        SUM(total_amount) AS revenue
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT 
    month,
    revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) AS revenue_change
FROM monthly_revenue;


-- ============ MULTIPLE CTES ============
WITH 
order_stats AS (
    SELECT 
        user_id,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_spent,
        AVG(total_amount) AS avg_order_value
    FROM orders
    GROUP BY user_id
),
customer_segments AS (
    SELECT 
        user_id,
        order_count,
        total_spent,
        avg_order_value,
        CASE 
            WHEN order_count >= 10 AND total_spent > 1000 THEN 'VIP'
            WHEN order_count >= 5 THEN 'Loyal'
            WHEN order_count >= 2 THEN 'Regular'
            ELSE 'New'
        END AS segment
    FROM order_stats
)
SELECT 
    cs.segment,
    COUNT(*) AS customer_count,
    AVG(cs.total_spent) AS avg_customer_value
FROM customer_segments cs
GROUP BY cs.segment;


-- ============ RECURSIVE CTE ============
-- Traverse hierarchical data (e.g., category tree)

-- Get all categories with their full path
WITH RECURSIVE category_tree AS (
    -- Anchor: Root categories (no parent)
    SELECT 
        category_id,
        category_name,
        parent_category_id,
        category_name AS full_path,
        0 AS level
    FROM categories
    WHERE parent_category_id IS NULL
    
    UNION ALL
    
    -- Recursive: Child categories
    SELECT 
        c.category_id,
        c.category_name,
        c.parent_category_id,
        CONCAT(ct.full_path, ' > ', c.category_name) AS full_path,
        ct.level + 1 AS level
    FROM categories c
    JOIN category_tree ct ON c.parent_category_id = ct.category_id
)
SELECT 
    REPEAT('  ', level) || category_name AS indented_name,
    full_path,
    level
FROM category_tree
ORDER BY full_path;


-- ============ CTE vs SUBQUERY ============
-- CTEs are often more readable

-- Subquery approach (less readable)
SELECT 
    high_value_users.country,
    COUNT(*) AS user_count
FROM (
    SELECT 
        u.user_id,
        u.country,
        SUM(o.total_amount) AS total_spent
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.country
    HAVING total_spent > 1000
) high_value_users
GROUP BY high_value_users.country;

-- CTE approach (more readable)
WITH high_value_users AS (
    SELECT 
        u.user_id,
        u.country,
        SUM(o.total_amount) AS total_spent
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.country
    HAVING total_spent > 1000
)
SELECT 
    country,
    COUNT(*) AS user_count
FROM high_value_users
GROUP BY country;
```

---

### 2. Subqueries

#### Concept Explanation

Subqueries are queries nested inside another query. They can return:
- Single value (scalar subquery)
- Single column (single-column subquery)
- Multiple columns (multi-column subquery)
- Multiple rows (table subquery)

```sql
-- ============ SCALAR SUBQUERY ============
-- Returns single value

-- Products priced above average
SELECT 
    product_name,
    price,
    (SELECT AVG(price) FROM products) AS avg_price,
    price - (SELECT AVG(price) FROM products) AS diff_from_avg
FROM products
WHERE price > (SELECT AVG(price) FROM products);


-- ============ CORRELATED SUBQUERY ============
-- References outer query (executed once per outer row)

-- Users who placed more orders than average
SELECT 
    u.user_id,
    u.username,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.user_id) AS order_count
FROM users u
WHERE (
    SELECT COUNT(*) 
    FROM orders o 
    WHERE o.user_id = u.user_id
) > (
    SELECT AVG(order_count) 
    FROM (
        SELECT COUNT(*) AS order_count 
        FROM orders 
        GROUP BY user_id
    ) avg_calc
);


-- ============ IN / NOT IN ============
-- Check membership in subquery results

-- Products that have been ordered
SELECT product_name, price
FROM products
WHERE product_id IN (
    SELECT DISTINCT product_id 
    FROM order_items
);

-- Products that have never been ordered
SELECT product_name, price
FROM products
WHERE product_id NOT IN (
    SELECT DISTINCT product_id 
    FROM order_items
);
-- WARNING: NOT IN returns empty result if subquery has NULLs!
-- Better to use NOT EXISTS:
SELECT product_name, price
FROM products p
WHERE NOT EXISTS (
    SELECT 1 FROM order_items oi WHERE oi.product_id = p.product_id
);


-- ============ EXISTS / NOT EXISTS ============
-- Check if subquery returns any rows

-- Categories with products in stock
SELECT category_name
FROM categories c
WHERE EXISTS (
    SELECT 1 
    FROM products p 
    WHERE p.category_id = c.category_id 
    AND p.stock_quantity > 0
);


-- ============ ALL / ANY ============
-- Compare with all/any values from subquery

-- Products more expensive than ALL electronics
SELECT product_name, price
FROM products
WHERE price > ALL (
    SELECT price 
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
    WHERE c.category_name = 'Electronics'
);


-- ============ DERIVED TABLE ============
-- Subquery in FROM clause

SELECT segment, COUNT(*) AS user_count, AVG(total_spent) AS avg_spent
FROM (
    SELECT 
        u.user_id,
        SUM(o.total_amount) AS total_spent,
        CASE 
            WHEN SUM(o.total_amount) > 1000 THEN 'High Value'
            WHEN SUM(o.total_amount) > 500 THEN 'Medium Value'
            ELSE 'Low Value'
        END AS segment
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id
) user_segments
GROUP BY segment;


-- ============ SUBQUERY IN SELECT ============
-- Multiple subqueries for different calculations

SELECT 
    c.category_name,
    (SELECT COUNT(*) FROM products p WHERE p.category_id = c.category_id) AS product_count,
    (SELECT COALESCE(SUM(oi.quantity), 0) 
     FROM order_items oi 
     JOIN products p ON oi.product_id = p.product_id 
     WHERE p.category_id = c.category_id) AS units_sold,
    (SELECT COALESCE(SUM(oi.quantity * oi.unit_price), 0)
     FROM order_items oi 
     JOIN products p ON oi.product_id = p.product_id 
     WHERE p.category_id = c.category_id) AS total_revenue
FROM categories c;
```

---

### 3. Indexing and Query Optimization

#### Concept Explanation

Indexes are data structures that improve query speed at the cost of storage and write performance.

```sql
-- ============ CREATING INDEXES ============

-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index (multiple columns)
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Unique index
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Partial index (filtered)
CREATE INDEX idx_active_users ON users(created_at) 
WHERE is_active = 1;

-- Covering index (includes additional columns)
CREATE INDEX idx_orders_covering ON orders(user_id, order_date) 
INCLUDE (total_amount, status);


-- ============ VIEWING EXECUTION PLANS ============

-- MySQL
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- PostgreSQL
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM users WHERE email = 'test@example.com';

-- Key metrics to look for:
-- - type: ALL (full table scan) vs index/ref/range
-- - rows: Number of rows examined
-- - Extra: Using filesort, Using temporary (bad), Using index (good)


-- ============ QUERY OPTIMIZATION TECHNIQUES ============

-- 1. SELECT only needed columns (not SELECT *)
-- BAD:
SELECT * FROM orders WHERE user_id = 123;

-- GOOD:
SELECT order_id, order_date, total_amount FROM orders WHERE user_id = 123;


-- 2. Use appropriate WHERE conditions
-- BAD (function on column prevents index use):
SELECT * FROM orders WHERE DATE(order_date) = '2024-01-01';

-- GOOD (index-friendly):
SELECT * FROM orders 
WHERE order_date >= '2024-01-01' 
AND order_date < '2024-01-02';


-- 3. Avoid implicit conversions
-- BAD:
SELECT * FROM products WHERE product_id = '123';  -- String vs INT

-- GOOD:
SELECT * FROM products WHERE product_id = 123;


-- 4. Use LIMIT with ORDER BY carefully
-- With proper index on (user_id, order_date DESC):
SELECT * FROM orders 
WHERE user_id = 123 
ORDER BY order_date DESC 
LIMIT 10;


-- 5. Optimize pagination
-- BAD (OFFSET gets slower as page increases):
SELECT * FROM orders 
ORDER BY order_id 
LIMIT 10 OFFSET 10000;

-- GOOD (Keyset pagination):
SELECT * FROM orders 
WHERE order_id > 10000 
ORDER BY order_id 
LIMIT 10;


-- 6. Batch operations
-- BAD (N+1 queries):
-- For each user: SELECT * FROM orders WHERE user_id = ?

-- GOOD (single query with JOIN):
SELECT u.user_id, u.username, o.order_id, o.total_amount
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE u.user_id IN (1, 2, 3, 4, 5);


-- 7. Proper JOIN order
-- Join smallest tables first, filter early
SELECT 
    oi.product_id,
    SUM(oi.quantity) AS total_sold
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id  -- Filter here
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date > '2024-01-01'  -- Filter as early as possible
GROUP BY oi.product_id;
```

---

### 4. Handling NULL Values

#### Concept Explanation

NULL represents missing or unknown data. Special handling is required because NULL != NULL.

```sql
-- ============ NULL COMPARISONS ============

-- NULL comparison always returns UNKNOWN (treated as FALSE)
SELECT 1 WHERE NULL = NULL;     -- Returns nothing
SELECT 1 WHERE NULL != NULL;    -- Returns nothing
SELECT 1 WHERE NULL IS NULL;    -- Returns 1


-- ============ COALESCE ============
-- Returns first non-NULL value

-- Provide default for NULL
SELECT 
    user_id,
    username,
    COALESCE(email, 'No email provided') AS email_display
FROM users;

-- Chain multiple fallbacks
SELECT 
    user_id,
    COALESCE(preferred_name, legal_name, username) AS display_name
FROM users;

-- Calculate with NULL handling
SELECT 
    order_id,
    COALESCE(discount_amount, 0) AS discount,
    total_amount - COALESCE(discount_amount, 0) AS final_amount
FROM orders;


-- ============ NULLIF ============
-- Returns NULL if two values are equal

-- Avoid division by zero
SELECT 
    product_id,
    total_sales,
    total_quantity,
    total_sales / NULLIF(total_quantity, 0) AS avg_price
FROM sales_summary;


-- ============ ISNULL / IFNULL ============
-- Database-specific NULL handling

-- MySQL/PostgreSQL
SELECT IFNULL(email, 'N/A') FROM users;

-- SQL Server
SELECT ISNULL(email, 'N/A') FROM users;

-- PostgreSQL
SELECT COALESCE(email, 'N/A') FROM users;  -- Standard


-- ============ NULL IN AGGREGATIONS ============

-- COUNT(*) counts NULLs
-- COUNT(column) ignores NULLs

SELECT 
    COUNT(*) AS total_rows,           -- Includes NULLs
    COUNT(email) AS emails_present,   -- Excludes NULLs
    COUNT(*) - COUNT(email) AS null_emails
FROM users;

-- SUM, AVG, MAX, MIN ignore NULLs
SELECT 
    AVG(price) AS avg_price_with_nulls,  -- Excludes NULL prices
    AVG(COALESCE(price, 0)) AS avg_price_with_zeros  -- Treats NULL as 0
FROM products;


-- ============ JOIN NULL HANDLING ============

-- NULLs don't match in JOINs
-- Use COALESCE or IS NULL checks

SELECT 
    u.user_id,
    u.username,
    COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.username;
```

---

## Python + SQL Integration

### 1. Using SQLAlchemy (ORM Approach)

```python
"""
SQLAlchemy: Python SQL Toolkit and ORM
- Object-Relational Mapping: Map Python classes to database tables
- Provides abstraction over different database systems
- Supports both ORM (high-level) and Core (low-level) approaches
"""

from sqlalchemy import create_engine, Column, Integer, String, \
    Float, DateTime, ForeignKey, func, desc
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import text
from datetime import datetime

# ============ SETUP ============
# Create engine (connection to database)
# Format: dialect+driver://username:password@host:port/database
engine = create_engine('postgresql://user:pass@localhost/ecommerce')
# For SQLite: 'sqlite:///ecommerce.db'
# For MySQL: 'mysql+pymysql://user:pass@localhost/ecommerce'

# Base class for declarative models
Base = declarative_base()

# Session factory
Session = sessionmaker(bind=engine)


# ============ MODEL DEFINITION ============
class User(Base):
    """
    ORM class maps to 'users' table
    Each instance represents a row
    """
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    country = Column(String(50))
    age = Column(Integer)
    
    # Relationship: One user has many orders
    orders = relationship("Order", back_populates="user")
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


class Category(Base):
    __tablename__ = 'categories'
    
    category_id = Column(Integer, primary_key=True)
    category_name = Column(String(100))
    parent_category_id = Column(Integer, ForeignKey('categories.category_id'))
    
    # Self-referential relationship for hierarchy
    parent = relationship("Category", remote_side=[category_id])
    products = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = 'products'
    
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(200))
    category_id = Column(Integer, ForeignKey('categories.category_id'))
    price = Column(Float)
    stock_quantity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")


class Order(Base):
    __tablename__ = 'orders'
    
    order_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    order_date = Column(DateTime, default=datetime.utcnow)
    total_amount = Column(Float)
    status = Column(String(20))
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    __tablename__ = 'order_items'
    
    order_item_id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'))
    product_id = Column(Integer, ForeignKey('products.product_id'))
    quantity = Column(Integer)
    unit_price = Column(Float)
    
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


# Create tables
Base.metadata.create_all(engine)


# ============ CRUD OPERATIONS ============

def create_user(session, username, email, country=None, age=None):
    """CREATE operation"""
    new_user = User(
        username=username,
        email=email,
        country=country,
        age=age
    )
    session.add(new_user)
    session.commit()
    return new_user


def get_user_by_id(session, user_id):
    """READ single record"""
    return session.query(User).filter_by(user_id=user_id).first()


def get_all_users(session, limit=100):
    """READ multiple records"""
    return session.query(User).limit(limit).all()


def update_user_email(session, user_id, new_email):
    """UPDATE operation"""
    user = session.query(User).filter_by(user_id=user_id).first()
    if user:
        user.email = new_email
        session.commit()
    return user


def delete_user(session, user_id):
    """DELETE operation"""
    user = session.query(User).filter_by(user_id=user_id).first()
    if user:
        session.delete(user)
        session.commit()
    return user


# ============ QUERY EXAMPLES ============

def get_users_with_orders(session):
    """
    Equivalent SQL:
    SELECT u.*, COUNT(o.order_id) as order_count
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id
    """
    from sqlalchemy import func
    
    results = session.query(
        User,
        func.count(Order.order_id).label('order_count')
    ).join(Order).group_by(User.user_id).all()
    
    return results


def get_high_value_users(session, min_spent=1000):
    """
    Equivalent SQL:
    SELECT u.user_id, u.username, SUM(o.total_amount) as total_spent
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.username
    HAVING SUM(o.total_amount) > 1000
    """
    from sqlalchemy import func
    
    return session.query(
        User.user_id,
        User.username,
        func.sum(Order.total_amount).label('total_spent')
    ).join(Order).group_by(User.user_id, User.username) \
     .having(func.sum(Order.total_amount) > min_spent).all()


def get_products_by_category(session):
    """
    Equivalent SQL:
    SELECT c.category_name, p.product_name, p.price
    FROM categories c
    JOIN products p ON c.category_id = p.category_id
    ORDER BY c.category_name, p.price DESC
    """
    return session.query(
        Category.category_name,
        Product.product_name,
        Product.price
    ).join(Product).order_by(
        Category.category_name,
        desc(Product.price)
    ).all()


def get_monthly_revenue(session):
    """
    Equivalent SQL:
    SELECT DATE_FORMAT(order_date, '%Y-%m') as month, 
           SUM(total_amount) as revenue
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
    ORDER BY month
    """
    from sqlalchemy import func, extract
    
    return session.query(
        func.date_trunc('month', Order.order_date).label('month'),
        func.sum(Order.total_amount).label('revenue')
    ).group_by('month').order_by('month').all()


# ============ RAW SQL WITH SQLALCHEMY ============

def execute_raw_query(session, query, params=None):
    """Execute raw SQL when ORM is not sufficient"""
    result = session.execute(text(query), params or {})
    return result.fetchall()


def complex_analytics_query(session):
    """
    Use raw SQL for complex queries that are hard to express in ORM
    """
    query = """
    WITH monthly_stats AS (
        SELECT 
            DATE_FORMAT(order_date, '%%Y-%%m') as month,
            COUNT(*) as order_count,
            SUM(total_amount) as revenue
        FROM orders
        GROUP BY DATE_FORMAT(order_date, '%%Y-%%m')
    )
    SELECT 
        month,
        order_count,
        revenue,
        revenue - LAG(revenue) OVER (ORDER BY month) as revenue_change,
        ROUND(revenue / LAG(revenue) OVER (ORDER BY month) * 100, 2) as pct_change
    FROM monthly_stats
    ORDER BY month
    """
    return session.execute(text(query)).fetchall()


# ============ CONNECTION POOLING ============

def create_optimized_engine():
    """
    Configure connection pool for production use
    """
    return create_engine(
        'postgresql://user:pass@localhost/ecommerce',
        pool_size=10,           # Keep 10 connections ready
        max_overflow=20,        # Allow up to 20 additional connections
        pool_timeout=30,        # Wait up to 30s for available connection
        pool_recycle=3600,      # Recycle connections after 1 hour
        echo=False              # Set True to see all SQL
    )


# ============ CONTEXT MANAGER USAGE ============

from contextlib import contextmanager

@contextmanager
def get_session():
    """
    Context manager for safe session handling
    Automatically commits on success, rolls back on exception
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# Usage
with get_session() as session:
    user = create_user(session, "newuser", "new@example.com")
    print(f"Created user: {user}")
    # Auto-committed on exit, or rolled back on exception
```

---

### 2. Using psycopg2 / pymysql (Direct Driver Approach)

```python
"""
Direct database driver approach
- Lower level, more control
- Better performance for bulk operations
- Useful for simple queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager


# ============ CONNECTION MANAGEMENT ============

@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    Ensures connections are properly closed
    """
    conn = psycopg2.connect(
        host="localhost",
        database="ecommerce",
        user="user",
        password="password"
    )
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(conn, cursor_factory=None):
    """
    Context manager for cursors
    Automatically commits or rolls back transactions
    """
    cursor = conn.cursor(cursor_factory=cursor_factory)
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()


# ============ BASIC CRUD ============

def create_user(username, email, country=None, age=None):
    """Insert single record"""
    with get_db_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                INSERT INTO users (username, email, country, age)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id
            """, (username, email, country, age))
            
            user_id = cur.fetchone()[0]
            return user_id


def get_user_by_id(user_id):
    """Fetch single record"""
    with get_db_connection() as conn:
        # RealDictCursor returns dict-like rows
        with get_cursor(conn, RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username, email, country, age, created_at
                FROM users
                WHERE user_id = %s
            """, (user_id,))
            
            return cur.fetchone()


def update_user(user_id, **kwargs):
    """Dynamic update with any fields"""
    allowed_fields = ['username', 'email', 'country', 'age']
    
    # Filter to only allowed fields
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not updates:
        return None
    
    # Build dynamic query
    set_clause = ', '.join(f"{k} = %s" for k in updates.keys())
    values = list(updates.values()) + [user_id]
    
    with get_db_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                UPDATE users
                SET {set_clause}
                WHERE user_id = %s
            """, values)
            
            return cur.rowcount


def delete_user(user_id):
    """Delete record"""
    with get_db_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                DELETE FROM users
                WHERE user_id = %s
            """, (user_id,))
            
            return cur.rowcount


# ============ BULK OPERATIONS ============

def bulk_insert_users(users_data):
    """
    Efficient bulk insert using execute_values
    Much faster than individual inserts
    
    users_data: list of tuples [(username, email, country, age), ...]
    """
    with get_db_connection() as conn:
        with get_cursor(conn) as cur:
            execute_values(
                cur,
                """
                INSERT INTO users (username, email, country, age)
                VALUES %s
                RETURNING user_id
                """,
                users_data,
                template=None,
                page_size=1000
            )
            
            return cur.fetchall()


def bulk_update_prices(category_id, percentage_increase):
    """
    Bulk update using single query
    """
    with get_db_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("""
                UPDATE products
                SET price = price * (1 + %s / 100.0)
                WHERE category_id = %s
            """, (percentage_increase, category_id))
            
            return cur.rowcount


# ============ PARAMETERIZED QUERIES ============

def get_users_by_country(countries):
    """
    Dynamic IN clause with proper parameterization
    Prevents SQL injection
    """
    with get_db_connection() as conn:
        with get_cursor(conn, RealDictCursor) as cur:
            # psycopg2 handles list parameters automatically
            cur.execute("""
                SELECT user_id, username, email, country
                FROM users
                WHERE country = ANY(%s)
                ORDER BY country, username
            """, (countries,))
            
            return cur.fetchall()


def search_products(name_pattern, min_price=None, max_price=None, category_id=None):
    """
    Dynamic query building with safe parameterization
    """
    conditions = ["product_name ILIKE %s"]
    params = [f"%{name_pattern}%"]
    
    if min_price is not None:
        conditions.append("price >= %s")
        params.append(min_price)
    
    if max_price is not None:
        conditions.append("price <= %s")
        params.append(max_price)
    
    if category_id is not None:
        conditions.append("category_id = %s")
        params.append(category_id)
    
    where_clause = " AND ".join(conditions)
    
    with get_db_connection() as conn:
        with get_cursor(conn, RealDictCursor) as cur:
            cur.execute(f"""
                SELECT product_id, product_name, price, stock_quantity
                FROM products
                WHERE {where_clause}
                ORDER BY product_name
            """, params)
            
            return cur.fetchall()


# ============ TRANSACTIONS ============

def transfer_order_to_user(order_id, from_user_id, to_user_id):
    """
    Multi-step transaction with rollback on failure
    """
    with get_db_connection() as conn:
        try:
            with conn.cursor() as cur:
                # Step 1: Verify order belongs to from_user
                cur.execute("""
                    SELECT order_id FROM orders
                    WHERE order_id = %s AND user_id = %s
                    FOR UPDATE  -- Lock the row
                """, (order_id, from_user_id))
                
                if not cur.fetchone():
                    raise ValueError("Order not found or not owned by user")
                
                # Step 2: Update order ownership
                cur.execute("""
                    UPDATE orders
                    SET user_id = %s
                    WHERE order_id = %s
                """, (to_user_id, order_id))
                
                # Step 3: Log the transfer
                cur.execute("""
                    INSERT INTO order_transfers (order_id, from_user_id, to_user_id, transferred_at)
                    VALUES (%s, %s, %s, NOW())
                """, (order_id, from_user_id, to_user_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()


# ============ ANALYTICS QUERIES ============

def get_customer_lifetime_value():
    """
    Complex analytics query with window functions
    """
    with get_db_connection() as conn:
        with get_cursor(conn, RealDictCursor) as cur:
            cur.execute("""
                WITH user_metrics AS (
                    SELECT 
                        u.user_id,
                        u.username,
                        COUNT(DISTINCT o.order_id) AS order_count,
                        SUM(o.total_amount) AS total_spent,
                        MIN(o.order_date) AS first_order,
                        MAX(o.order_date) AS last_order,
                        AVG(o.total_amount) AS avg_order_value
                    FROM users u
                    LEFT JOIN orders o ON u.user_id = o.user_id
                    GROUP BY u.user_id, u.username
                )
                SELECT 
                    user_id,
                    username,
                    order_count,
                    total_spent,
                    avg_order_value,
                    DATE_PART('day', last_order - first_order) AS customer_tenure_days,
                    CASE 
                        WHEN total_spent > 1000 THEN 'VIP'
                        WHEN total_spent > 500 THEN 'High Value'
                        WHEN total_spent > 0 THEN 'Regular'
                        ELSE 'No Purchase'
                    END AS segment,
                    NTILE(4) OVER (ORDER BY total_spent DESC) AS spending_quartile
                FROM user_metrics
                ORDER BY total_spent DESC
            """)
            
            return cur.fetchall()


def get_product_recommendations(user_id, limit=5):
    """
    Collaborative filtering-style recommendation
    """
    with get_db_connection() as conn:
        with get_cursor(conn, RealDictCursor) as cur:
            cur.execute("""
                WITH user_purchases AS (
                    SELECT DISTINCT product_id
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.order_id
                    WHERE o.user_id = %s
                ),
                similar_users AS (
                    SELECT DISTINCT o.user_id
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.order_id
                    WHERE oi.product_id IN (SELECT product_id FROM user_purchases)
                    AND o.user_id != %s
                ),
                recommendations AS (
                    SELECT 
                        p.product_id,
                        p.product_name,
                        p.price,
                        COUNT(*) AS purchase_frequency
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.order_id
                    JOIN products p ON oi.product_id = p.product_id
                    WHERE o.user_id IN (SELECT user_id FROM similar_users)
                    AND oi.product_id NOT IN (SELECT product_id FROM user_purchases)
                    GROUP BY p.product_id, p.product_name, p.price
                )
                SELECT * FROM recommendations
                ORDER BY purchase_frequency DESC
                LIMIT %s
            """, (user_id, user_id, limit))
            
            return cur.fetchall()


# ============ CONNECTION POOLING ============

from psycopg2 import pool

# Create global connection pool
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    database="ecommerce",
    user="user",
    password="password"
)


@contextmanager
def get_pooled_connection():
    """Get connection from pool"""
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)


# Usage with pool
with get_pooled_connection() as conn:
    with get_cursor(conn, RealDictCursor) as cur:
        cur.execute("SELECT * FROM users LIMIT 10")
        users = cur.fetchall()
```

---

### 3. Pandas Integration

```python
"""
Using pandas for data analysis with SQL
- Read SQL results directly into DataFrames
- Perform analysis in Python
- Write results back to database
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Create engine
engine = create_engine('postgresql://user:pass@localhost/ecommerce')


# ============ READING SQL INTO PANDAS ============

def load_orders_dataframe():
    """Load entire table into DataFrame"""
    df = pd.read_sql_table('orders', engine)
    return df


def load_custom_query():
    """Load query results into DataFrame"""
    query = """
    SELECT 
        o.order_id,
        o.order_date,
        o.total_amount,
        u.username,
        u.country
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    WHERE o.order_date >= '2024-01-01'
    """
    df = pd.read_sql_query(query, engine)
    return df


def load_in_chunks():
    """Load large datasets in chunks to manage memory"""
    chunks = []
    chunk_size = 10000
    
    for chunk in pd.read_sql_query(
        "SELECT * FROM order_items", 
        engine, 
        chunksize=chunk_size
    ):
        # Process each chunk
        processed = chunk.groupby('product_id')['quantity'].sum()
        chunks.append(processed)
    
    # Combine results
    return pd.concat(chunks).groupby(level=0).sum()


# ============ DATA ANALYSIS ============

def analyze_customer_segments():
    """Customer segmentation analysis"""
    query = """
    SELECT 
        u.user_id,
        u.country,
        COUNT(o.order_id) as order_count,
        SUM(o.total_amount) as total_spent,
        MIN(o.order_date) as first_order,
        MAX(o.order_date) as last_order
    FROM users u
    LEFT JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.country
    """
    
    df = pd.read_sql_query(query, engine)
    
    # Calculate metrics
    df['avg_order_value'] = df['total_spent'] / df['order_count'].replace(0, np.nan)
    df['customer_tenure_days'] = (
        pd.to_datetime(df['last_order']) - pd.to_datetime(df['first_order'])
    ).dt.days
    
    # Create segments
    def segment_customer(row):
        if row['total_spent'] > 1000:
            return 'VIP'
        elif row['total_spent'] > 500:
            return 'High Value'
        elif row['total_spent'] > 0:
            return 'Regular'
        else:
            return 'No Purchase'
    
    df['segment'] = df.apply(segment_customer, axis=1)
    
    # Aggregate by segment
    segment_summary = df.groupby('segment').agg({
        'user_id': 'count',
        'total_spent': ['sum', 'mean'],
        'order_count': 'mean',
        'avg_order_value': 'mean'
    }).round(2)
    
    return df, segment_summary


def time_series_analysis():
    """Time series analysis of sales"""
    query = """
    SELECT 
        DATE(order_date) as date,
        COUNT(*) as orders,
        SUM(total_amount) as revenue
    FROM orders
    GROUP BY DATE(order_date)
    ORDER BY date
    """
    
    df = pd.read_sql_query(query, engine, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # Calculate rolling statistics
    df['revenue_7d_avg'] = df['revenue'].rolling(window=7).mean()
    df['revenue_30d_avg'] = df['revenue'].rolling(window=30).mean()
    
    # Calculate day-over-day change
    df['revenue_change'] = df['revenue'].pct_change() * 100
    
    # Resample to weekly
    weekly = df.resample('W').agg({
        'orders': 'sum',
        'revenue': 'sum'
    })
    
    return df, weekly


# ============ WRITING BACK TO DATABASE ============

def save_analysis_results(df, table_name):
    """Save DataFrame to database table"""
    df.to_sql(
        table_name,
        engine,
        if_exists='replace',  # Options: 'fail', 'replace', 'append'
        index=False,
        chunksize=1000,       # Insert in batches
        method='multi'        # Optimize bulk insert
    )


def update_from_dataframe(df, table_name, key_column):
    """
    Update database table from DataFrame
    """
    with engine.connect() as conn:
        for _, row in df.iterrows():
            # Build update statement
            set_clause = ', '.join([
                f"{col} = :{col}" 
                for col in df.columns 
                if col != key_column
            ])
            
            query = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE {key_column} = :{key_column}
            """
            
            conn.execute(query, row.to_dict())
        
        conn.commit()


# ============ ADVANCED: SQLALCHEMY + PANDAS PIPELINE ============

def ml_feature_engineering():
    """
    Extract features from database for ML model
    """
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Customer features
        customer_features = pd.read_sql(text("""
            SELECT 
                u.user_id,
                u.country,
                u.age,
                COUNT(DISTINCT o.order_id) as total_orders,
                SUM(o.total_amount) as total_spent,
                AVG(o.total_amount) as avg_order_value,
                MAX(o.order_date) as last_order_date,
                MIN(o.order_date) as first_order_date,
                COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')) as active_months
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id
            GROUP BY u.user_id, u.country, u.age
        """), conn)
        
        # Product features
        product_features = pd.read_sql(text("""
            SELECT 
                p.product_id,
                c.category_name,
                p.price,
                COUNT(DISTINCT oi.order_id) as times_ordered,
                SUM(oi.quantity) as total_units_sold,
                AVG(r.rating) as avg_rating
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            LEFT JOIN reviews r ON p.product_id = r.product_id
            GROUP BY p.product_id, c.category_name, p.price
        """), conn)
    
    # Feature engineering in pandas
    customer_features['days_since_last_order'] = (
        pd.Timestamp.now() - pd.to_datetime(customer_features['last_order_date'])
    ).dt.days
    
    customer_features['customer_lifetime_days'] = (
        pd.to_datetime(customer_features['last_order_date']) - 
        pd.to_datetime(customer_features['first_order_date'])
    ).dt.days
    
    return customer_features, product_features
```

---

## Summary: Key Concepts

### Python Concepts

| Concept | Key Point | Common Use Case |
|---------|-----------|-----------------|
| Lists vs Tuples | Mutability | Lists for collections, tuples for records |
| Dicts vs Sets | Key-value vs unique values | Dicts for lookups, sets for deduplication |
| List Comprehensions | Memory vs readability | Transform/filter collections |
| Generator Expressions | Lazy evaluation | Large/streaming data processing |
| Decorators | Function modification | Logging, caching, validation |

### SQL Concepts

| Concept | Key Point | Common Use Case |
|---------|-----------|-----------------|
| JOINs | Combine tables | Related data from multiple tables |
| GROUP BY | Aggregate rows | Summarization, reporting |
| Window Functions | Row-level calculations | Running totals, rankings |
| CTEs | Named subqueries | Complex query organization |
| Indexes | Query performance | Speed up WHERE, JOIN, ORDER BY |

### Integration Concepts

| Approach | Best For | Performance |
|----------|----------|-------------|
| SQLAlchemy ORM | CRUD, relationships | Good for small/medium data |
| SQLAlchemy Core | Complex queries | Better for analytical queries |
| Raw SQL (psycopg2) | Bulk operations | Best for large datasets |
| Pandas | Analysis, visualization | Good for in-memory processing |

---

## Practice Problems

### Problem 1: Find Top Customers by Category
**Question**: For each product category, find the top 3 customers by total spending in that category.

**Solution Approach**:
```sql
WITH category_customer_spending AS (
    SELECT 
        c.category_name,
        u.user_id,
        u.username,
        SUM(oi.quantity * oi.unit_price) AS category_spent
    FROM categories c
    JOIN products p ON c.category_id = p.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    JOIN users u ON o.user_id = u.user_id
    GROUP BY c.category_name, u.user_id, u.username
),
ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY category_name 
            ORDER BY category_spent DESC
        ) AS rank_in_category
    FROM category_customer_spending
)
SELECT category_name, username, category_spent
FROM ranked
WHERE rank_in_category <= 3
ORDER BY category_name, rank_in_category;
```

### Problem 2: Detect Customer Churn
**Question**: Identify customers who haven't placed an order in the last 90 days but had placed orders before.

**Solution Approach**:
```python
import pandas as pd
from datetime import datetime, timedelta

def detect_churn_risk(engine, days_inactive=90):
    query = """
    SELECT 
        u.user_id,
        u.username,
        u.email,
        COUNT(o.order_id) AS total_orders,
        MAX(o.order_date) AS last_order_date,
        SUM(o.total_amount) AS total_spent
    FROM users u
    LEFT JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.username, u.email
    HAVING MAX(o.order_date) < NOW() - INTERVAL '%s days'
       AND COUNT(o.order_id) > 0
    """ % days_inactive
    
    df = pd.read_sql_query(query, engine)
    df['days_since_last_order'] = (
        datetime.now() - pd.to_datetime(df['last_order_date'])
    ).dt.days
    
    # Categorize churn risk
    def churn_risk(days):
        if days > 180:
            return 'High Risk'
        elif days > 120:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    df['churn_risk'] = df['days_since_last_order'].apply(churn_risk)
    
    return df
```

### Problem 3: Product Affinity Analysis
**Question**: Find pairs of products frequently bought together.

**Solution Approach**:
```sql
SELECT 
    p1.product_name AS product_a,
    p2.product_name AS product_b,
    COUNT(*) AS times_bought_together
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
    AND oi1.product_id < oi2.product_id  -- Avoid duplicates
JOIN products p1 ON oi1.product_id = p1.product_id
JOIN products p2 ON oi2.product_id = p2.product_id
GROUP BY p1.product_name, p2.product_name
HAVING COUNT(*) >= 5  -- Minimum threshold
ORDER BY times_bought_together DESC
LIMIT 20;
```

---

## Detailed Problem Solving with Concept Breakdown

This section walks through common interview problems with detailed concept explanations, step-by-step solutions, and result simulations.

---

## PROBLEM 1: Find Users Who Never Placed an Order

### Problem Statement
Given the `users` and `orders` tables, find all users who have never placed an order.

### Sample Data

```sql
-- users table
| user_id | username  | email              | country |
|---------|-----------|--------------------|---------|
| 1       | alice     | alice@email.com    | USA     |
| 2       | bob       | bob@email.com      | UK      |
| 3       | charlie   | charlie@email.com  | USA     |
| 4       | diana     | diana@email.com    | Canada  |
| 5       | eve       | eve@email.com      | UK      |

-- orders table
| order_id | user_id | order_date | total_amount |
|----------|---------|------------|--------------|
| 101      | 1       | 2024-01-15 | 150.00       |
| 102      | 1       | 2024-02-20 | 230.50       |
| 103      | 3       | 2024-01-25 | 89.99        |
| 104      | 3       | 2024-03-10 | 445.00       |
```

### Concept Breakdown

**1. Understanding JOIN Types**
- **INNER JOIN**: Returns only matching rows from both tables
- **LEFT JOIN**: Returns ALL rows from left table, matching rows from right (NULL if no match)
- This problem requires finding rows in `users` with NO match in `orders`

**2. The Key Insight**
After a LEFT JOIN, unmatched rows from the left table will have NULL values in columns from the right table. We can filter for these NULLs.

### Step-by-Step Solution

#### Step 1: Start with the LEFT JOIN
```sql
SELECT 
    u.user_id,
    u.username,
    o.order_id  -- Will be NULL for users without orders
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id;
```

**Intermediate Result:**
| user_id | username | order_id |
|---------|----------|----------|
| 1       | alice    | 101      |
| 1       | alice    | 102      |
| 2       | bob      | NULL     |  ← No orders!
| 3       | charlie  | 103      |
| 3       | charlie  | 104      |
| 4       | diana    | NULL     |  ← No orders!
| 5       | eve      | NULL     |  ← No orders!

#### Step 2: Filter for NULL order_id
```sql
SELECT 
    u.user_id,
    u.username,
    u.email
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_id IS NULL;
```

**Final Result:**
| user_id | username | email             |
|---------|----------|-------------------|
| 2       | bob      | bob@email.com     |
| 4       | diana    | diana@email.com   |
| 5       | eve      | eve@email.com     |

### Alternative: Using NOT EXISTS
```sql
SELECT u.user_id, u.username, u.email
FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.user_id
);
```
**Why this works:** For each user, the subquery checks if any order exists. NOT EXISTS returns TRUE only when the subquery returns no rows.

### Alternative: Using NOT IN
```sql
SELECT user_id, username, email
FROM users
WHERE user_id NOT IN (
    SELECT DISTINCT user_id FROM orders WHERE user_id IS NOT NULL
);
```
**⚠️ Warning:** If `orders.user_id` can be NULL, NOT IN returns empty result! Always ensure no NULLs or use NOT EXISTS.

---

## PROBLEM 2: Second Highest Salary

### Problem Statement
Find the second highest distinct salary from the `employees` table.

### Sample Data
```sql
| employee_id | name    | salary |
|-------------|---------|--------|
| 1           | Alice   | 100000 |
| 2           | Bob     | 80000  |
| 3           | Charlie | 120000 |
| 4           | Diana   | 120000 |
| 5           | Eve     | 90000  |
```

### Concept Breakdown

**1. Understanding DISTINCT vs ALL**
- DISTINCT removes duplicates (120000 appears once)
- ALL keeps duplicates

**2. Ordering Concepts**
- DESC: Highest to lowest
- LIMIT n: Returns first n rows
- OFFSET n: Skips first n rows

**3. The Approach**
To get 2nd highest: 
1. Sort salaries in descending order
2. Remove duplicates
3. Skip the first row (highest)
4. Take the next row (2nd highest)

### Step-by-Step Solution

#### Method 1: Using LIMIT and OFFSET
```sql
SELECT DISTINCT salary AS SecondHighestSalary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

**Execution Flow:**

Step 1: `SELECT DISTINCT salary`
| salary  |
|---------|
| 100000  |
| 80000   |
| 120000  |  ← Duplicate removed
| 90000   |

Step 2: `ORDER BY salary DESC`
| salary  |
|---------|
| 120000  |
| 100000  |
| 90000   |
| 80000   |

Step 3: `OFFSET 1` (Skip first row)
| salary  |
|---------|
| 100000  |  ← Now at top
| 90000   |
| 80000   |

Step 4: `LIMIT 1`
| salary  |
|---------|
| 100000  |

#### Method 2: Using Subquery
```sql
SELECT MAX(salary) AS SecondHighestSalary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

**Execution Flow:**

Step 1: Inner query `(SELECT MAX(salary) FROM employees)`
- Result: 120000 (highest)

Step 2: Outer query filters
```sql
WHERE salary < 120000
```
Remaining salaries: 100000, 80000, 90000

Step 3: `MAX(salary)` from filtered set
- Result: 100000

**Final Result:**
| SecondHighestSalary |
|---------------------|
| 100000              |

#### Method 3: Using Window Function (DENSE_RANK)
```sql
SELECT salary AS SecondHighestSalary
FROM (
    SELECT 
        salary,
        DENSE_RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees
) ranked
WHERE salary_rank = 2;
```

**Execution Flow:**

Step 1: Window function assigns ranks
| salary  | salary_rank |
|---------|-------------|
| 120000  | 1           |
| 120000  | 1           |
| 100000  | 2           |
| 90000   | 3           |
| 80000   | 4           |

Step 2: Filter WHERE salary_rank = 2
| salary  |
|---------|
| 100000  |

**Why DENSE_RANK not ROW_NUMBER?**
- DENSE_RANK: 1, 1, 2, 3, 4 (no gaps)
- ROW_NUMBER: 1, 2, 3, 4, 5 (with gaps)
- If two people tie for 1st, DENSE_RANK correctly identifies next as 2nd

---

## PROBLEM 3: Department Top Three Salaries

### Problem Statement
Find the top 3 highest-paid employees in each department.

### Sample Data
```sql
-- Employee table
| id | name  | salary | department_id |
|----|-------|--------|---------------|
| 1  | Joe   | 85000  | 1             |
| 2  | Henry | 80000  | 2             |
| 3  | Sam   | 60000  | 2             |
| 4  | Max   | 90000  | 1             |
| 5  | Janet | 69000  | 1             |
| 6  | Randy | 85000  | 1             |
| 7  | Will  | 70000  | 1             |

-- Department table
| id | name  |
|----|-------|
| 1  | IT    |
| 2  | Sales |
```

### Concept Breakdown

**1. PARTITION BY**
- Divides result set into partitions (groups)
- Window function operates within each partition independently
- Similar to GROUP BY but doesn't collapse rows

**2. Ranking Functions**
- `ROW_NUMBER()`: Unique rank, even for ties (1, 2, 3, 4)
- `RANK()`: Ties get same rank, skips next (1, 2, 2, 4)
- `DENSE_RANK()`: Ties get same rank, no skip (1, 2, 2, 3)

**3. The Challenge**
We need to rank within each department separately, not globally.

### Step-by-Step Solution

#### Step 1: Understand the Data
IT Department salaries: 90000, 85000, 85000, 70000, 69000
Sales Department salaries: 80000, 60000

#### Step 2: Apply Window Function with PARTITION
```sql
SELECT 
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary,
    DENSE_RANK() OVER (
        PARTITION BY e.department_id 
        ORDER BY e.salary DESC
    ) AS salary_rank
FROM Employee e
JOIN Department d ON e.department_id = d.id;
```

**Intermediate Result:**
| Department | Employee | Salary | salary_rank |
|------------|----------|--------|-------------|
| IT         | Max      | 90000  | 1           |
| IT         | Joe      | 85000  | 2           |
| IT         | Randy    | 85000  | 2           |
| IT         | Will     | 70000  | 3           |
| IT         | Janet    | 69000  | 4           |
| Sales      | Henry    | 80000  | 1           |
| Sales      | Sam      | 60000  | 2           |

**Concept Explanation:**
- PARTITION BY creates separate ranking sequences for each department
- IT department: ranks 1, 2, 2, 3, 4
- Sales department: ranks 1, 2

#### Step 3: Filter for Top 3
```sql
SELECT Department, Employee, Salary
FROM (
    SELECT 
        d.name AS Department,
        e.name AS Employee,
        e.salary AS Salary,
        DENSE_RANK() OVER (
            PARTITION BY e.department_id 
            ORDER BY e.salary DESC
        ) AS salary_rank
    FROM Employee e
    JOIN Department d ON e.department_id = d.id
) ranked
WHERE salary_rank <= 3;
```

**Final Result:**
| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Max      | 90000  |
| IT         | Joe      | 85000  |
| IT         | Randy    | 85000  |
| IT         | Will     | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |

**Why include Will (rank 3)?** Because DENSE_RANK gives 1, 2, 2, 3... so rank 3 is still in top 3 distinct salary levels.

---

## PROBLEM 4: Nth Highest Salary (Function)

### Problem Statement
Write a SQL function to get the Nth highest salary from the Employee table.

### Sample Data
```sql
| id | name  | salary |
|----|-------|--------|
| 1  | Joe   | 100000 |
| 2  | Bob   | 80000  |
| 3  | Alice | 120000 |
| 4  | Diana | 90000  |
| 5  | Eve   | 75000  |
```

### Concept Breakdown

**1. Stored Functions**
- Reusable SQL code that accepts parameters
- Returns a value or table
- Encapsulates complex logic

**2. Variable Handling**
- DECLARE: Define variables
- SET: Assign values
- RETURN: Output result

**3. Edge Cases**
- N = 0 (invalid)
- N > number of distinct salaries (return NULL)
- N = 1 (highest salary)

### Step-by-Step Solution

#### Solution 1: Using LIMIT (MySQL/PostgreSQL)
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    DECLARE result INT;
    
    -- Handle invalid input
    IF N <= 0 THEN
        RETURN NULL;
    END IF;
    
    -- Offset is N-1 because LIMIT starts at 0
    SET N = N - 1;
    
    SELECT DISTINCT salary INTO result
    FROM Employee
    ORDER BY salary DESC
    LIMIT 1 OFFSET N;
    
    RETURN result;
END;
```

**Execution Trace for N=2:**

Step 1: `SET N = N - 1` → N becomes 1

Step 2: Query execution
```sql
SELECT DISTINCT salary 
FROM Employee 
ORDER BY salary DESC 
LIMIT 1 OFFSET 1;
```

Step 3: DISTINCT salaries ordered
| salary  |
|---------|
| 120000  |  ← OFFSET 0
| 100000  |  ← OFFSET 1 (this one!)
| 90000   |  ← OFFSET 2
| 80000   |
| 75000   |

Step 4: LIMIT 1 OFFSET 1 returns 100000

#### Solution 2: Using Window Function
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    RETURN (
        SELECT salary
        FROM (
            SELECT 
                salary,
                DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
            FROM Employee
        ) ranked
        WHERE rnk = N
        LIMIT 1
    );
END;
```

**Test Cases:**

| Function Call              | Result  | Explanation              |
|----------------------------|---------|--------------------------|
| getNthHighestSalary(1)     | 120000  | Highest salary           |
| getNthHighestSalary(2)     | 100000  | Second highest           |
| getNthHighestSalary(3)     | 90000   | Third highest            |
| getNthHighestSalary(6)     | NULL    | No 6th distinct salary   |
| getNthHighestSalary(0)     | NULL    | Invalid input handled    |
| getNthHighestSalary(-1)    | NULL    | Invalid input handled    |

---

## PROBLEM 5: Consecutive Numbers

### Problem Statement
Find all numbers that appear at least three times consecutively in the `Logs` table.

### Sample Data
```sql
| id | num |
|----|-----|
| 1  | 1   |
| 2  | 1   |
| 3  | 1   |
| 4  | 2   |
| 5  | 1   |
| 6  | 2   |
| 7  | 2   |
```

### Concept Breakdown

**1. Self JOIN**
- Joining a table to itself
- Used to compare rows with other rows

**2. Consecutive Condition**
- Row N: id = x, num = y
- Row N+1: id = x+1, num = y
- Row N+2: id = x+2, num = y

**3. The Pattern**
We need to find where three consecutive rows have the same `num` value.

### Step-by-Step Solution

#### Method 1: Self JOIN (Most Intuitive)
```sql
SELECT DISTINCT
    l1.num AS ConsecutiveNums
FROM Logs l1
JOIN Logs l2 ON l1.id = l2.id - 1 AND l1.num = l2.num
JOIN Logs l3 ON l1.id = l3.id - 2 AND l1.num = l3.num;
```

**Execution Trace:**

Step 1: l1 JOIN l2 (consecutive rows with same num)
Matching pairs (id, id+1 with same num):
- (1, 2): num=1, num=1 ✓
- (2, 3): num=1, num=1 ✓

Step 2: Join result JOIN l3 (two-step consecutive)
Checking if l1.id = l3.id - 2 AND l1.num = l3.num:
- For l1.id=1, l2.id=2: check l3.id=3, num=1 ✓ MATCH!

**Result:**
| ConsecutiveNums |
|-----------------|
| 1               |

#### Method 2: Using Window Function (LEAD)
```sql
SELECT DISTINCT num AS ConsecutiveNums
FROM (
    SELECT 
        num,
        LEAD(num, 1) OVER (ORDER BY id) AS next_num,
        LEAD(num, 2) OVER (ORDER BY id) AS next_next_num
    FROM Logs
) sub
WHERE num = next_num AND num = next_next_num;
```

**Step-by-Step Execution:**

Step 1: Apply LEAD functions
| id | num | next_num | next_next_num |
|----|-----|----------|---------------|
| 1  | 1   | 1        | 1             |
| 2  | 1   | 1        | 2             |
| 3  | 1   | 2        | 1             |
| 4  | 2   | 1        | 2             |
| 5  | 1   | 2        | 2             |
| 6  | 2   | 2        | NULL          |
| 7  | 2   | NULL     | NULL          |

**Concept: LEAD Function**
- `LEAD(num, 1)`: Gets num from next row
- `LEAD(num, 2)`: Gets num from row after next
- NULL when no next row exists

Step 2: Filter WHERE num = next_num AND num = next_next_num
Only row with id=1 matches: 1 = 1 AND 1 = 1 ✓

**Final Result:**
| ConsecutiveNums |
|-----------------|
| 1               |

---

## PROBLEM 6: Department Highest Salary

### Problem Statement
Find employees who earn the highest salary in each department.

### Sample Data
```sql
-- Employee
| id | name  | salary | department_id |
|----|-------|--------|---------------|
| 1  | Joe   | 70000  | 1             |
| 2  | Jim   | 90000  | 1             |
| 3  | Henry | 80000  | 2             |
| 4  | Sam   | 60000  | 2             |
| 5  | Max   | 90000  | 1             |

-- Department
| id | name  |
|----|-------|
| 1  | IT    |
| 2  | Sales |
```

### Concept Breakdown

**1. IN Clause with Subquery**
- Check if a value exists in a set returned by subquery
- Format: `WHERE (col1, col2) IN (SELECT col1, col2 FROM ...)`

**2. Correlated vs Non-Correlated Subqueries**
- Non-correlated: Runs once, independent of outer query
- Correlated: Runs for each row of outer query

**3. Composite Key Matching**
- Match multiple columns simultaneously: `(dept_id, salary)`

### Step-by-Step Solution

#### Step 1: Find Max Salary Per Department
```sql
SELECT department_id, MAX(salary) as max_salary
FROM Employee
GROUP BY department_id;
```

**Result:**
| department_id | max_salary |
|---------------|------------|
| 1             | 90000      |
| 2             | 80000      |

#### Step 2: Match Employees to These Maximums
```sql
SELECT 
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM Employee e
JOIN Department d ON e.department_id = d.id
WHERE (e.department_id, e.salary) IN (
    SELECT department_id, MAX(salary)
    FROM Employee
    GROUP BY department_id
);
```

**Execution Flow:**

For each employee, check if (department_id, salary) matches any row in subquery:
- Joe (1, 70000): Does (1, 70000) match? No (max is 90000) ❌
- Jim (1, 90000): Does (1, 90000) match? Yes! ✓
- Henry (2, 80000): Does (2, 80000) match? Yes! ✓
- Sam (2, 60000): Does (2, 60000) match? No (max is 80000) ❌
- Max (1, 90000): Does (1, 90000) match? Yes! ✓

**Final Result:**
| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
| IT         | Max      | 90000  |

**Note:** Both Jim and Max from IT department appear because they tie for highest salary.

#### Alternative: Using Window Function
```sql
SELECT Department, Employee, Salary
FROM (
    SELECT 
        d.name AS Department,
        e.name AS Employee,
        e.salary AS Salary,
        RANK() OVER (PARTITION BY e.department_id ORDER BY e.salary DESC) AS rnk
    FROM Employee e
    JOIN Department d ON e.department_id = d.id
) ranked
WHERE rnk = 1;
```

---

## PROBLEM 7: Rank Scores

### Problem Statement
Rank scores in descending order. The ranking should be dense (no gaps between ranks).

### Sample Data
```sql
| id | score |
|----|-------|
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |
```

### Concept Breakdown

**1. DENSE_RANK vs RANK vs ROW_NUMBER**

| score | ROW_NUMBER | RANK | DENSE_RANK |
|-------|------------|------|------------|
| 4.00  | 1          | 1    | 1          |
| 4.00  | 2          | 1    | 1          |
| 3.85  | 3          | 3    | 2          |
| 3.65  | 4          | 4    | 3          |
| 3.65  | 5          | 4    | 3          |
| 3.50  | 6          | 6    | 4          |

**Differences:**
- ROW_NUMBER: Always unique, even for ties
- RANK: Same values get same rank, skips next ranks (1, 1, 3...)
- DENSE_RANK: Same values get same rank, no gaps (1, 1, 2...)

### Step-by-Step Solution

```sql
SELECT 
    score,
    DENSE_RANK() OVER (ORDER BY score DESC) AS rank_position
FROM Scores
ORDER BY score DESC;
```

**Execution Trace:**

Step 1: Sort by score DESC
| id | score |
|----|-------|
| 3  | 4.00  |
| 5  | 4.00  |
| 4  | 3.85  |
| 2  | 3.65  |
| 6  | 3.65  |
| 1  | 3.50  |

Step 2: Apply DENSE_RANK
| score | rank_position |
|-------|---------------|
| 4.00  | 1             |
| 4.00  | 1             |  (same score, same rank)
| 3.85  | 2             |  (next rank is 2, not 3)
| 3.65  | 3             |
| 3.65  | 3             |  (same score, same rank)
| 3.50  | 4             |

**Final Result:**
| score | rank |
|-------|------|
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |

---

## PROBLEM 8: Exchange Seats (Odd/Even Swap)

### Problem Statement
Swap the seat id of every two consecutive students. If odd number of students, last one stays.

### Sample Data
```sql
| id | student |
|----|---------|
| 1  | Alice   |
| 2  | Bob     |
| 3  | Charlie |
| 4  | Diana   |
| 5  | Eve     |
```

### Concept Breakdown

**1. CASE Expression**
- Conditional logic in SQL
- WHEN condition THEN result
- ELSE default_result

**2. Odd/Even Logic**
- Odd: id % 2 = 1
- Even: id % 2 = 0
- Pattern: (1,2), (3,4), (5,6)... swap pairs

**3. The Swap Pattern**
- Odd id → id + 1 (swap with next)
- Even id → id - 1 (swap with previous)
- Last odd (if count is odd) → stays same

### Step-by-Step Solution

```sql
SELECT 
    CASE 
        WHEN id = (SELECT MAX(id) FROM Seat) AND id % 2 = 1 
            THEN id      -- Last student if odd count, stays put
        WHEN id % 2 = 1 
            THEN id + 1  -- Odd: swap with next
        ELSE 
            id - 1       -- Even: swap with previous
    END AS id,
    student
FROM Seat
ORDER BY id;
```

**Execution Trace:**

Step 1: Check max id = 5 (odd)

Step 2: Process each row:

| Original id | student | Condition Check              | New id |
|-------------|---------|------------------------------|--------|
| 1           | Alice   | 1 is odd, not max → 1 + 1    | 2      |
| 2           | Bob     | 2 is even → 2 - 1            | 1      |
| 3           | Charlie | 3 is odd, not max → 3 + 1    | 4      |
| 4           | Diana   | 4 is even → 4 - 1            | 3      |
| 5           | Eve     | 5 is odd AND is max → 5      | 5      |

Step 3: ORDER BY new id

**Final Result:**
| id | student |
|----|---------|
| 1  | Bob     |
| 2  | Alice   |
| 3  | Diana   |
| 4  | Charlie |
| 5  | Eve     |

---

## PROBLEM 9: Tree Node Classification

### Problem Statement
Given a tree table with `id`, `p_id` (parent id), classify each node as Root, Inner, or Leaf.

### Sample Data
```sql
| id | p_id |
|----|------|
| 1  | null |
| 2  | 1    |
| 3  | 1    |
| 4  | 2    |
| 5  | 2    |
```

### Concept Breakdown

**1. Tree Structure Types**
- **Root**: No parent (p_id IS NULL)
- **Leaf**: No children (id not in any p_id)
- **Inner**: Has both parent and children

**2. EXISTS Subquery**
- Returns TRUE if subquery returns any row
- Efficient for checking existence

**3. IN vs EXISTS**
- IN: Good for small lists
- EXISTS: Better for correlated subqueries

### Step-by-Step Solution

```sql
SELECT 
    id,
    CASE 
        WHEN p_id IS NULL THEN 'Root'
        WHEN NOT EXISTS (SELECT 1 FROM Tree t2 WHERE t2.p_id = t.id) THEN 'Leaf'
        ELSE 'Inner'
    END AS node_type
FROM Tree t;
```

**Execution Trace:**

For each row, evaluate CASE:

| id | p_id | Check 1: IS NULL? | Check 2: EXISTS child? | Result  |
|----|------|-------------------|------------------------|---------|
| 1  | null | YES               | SKIP                   | Root    |
| 2  | 1    | NO                | EXISTS? (3,4 have p_id=2) YES | Inner   |
| 3  | 1    | NO                | EXISTS? (none have p_id=3) NO | Leaf    |
| 4  | 2    | NO                | EXISTS? (none have p_id=4) NO | Leaf    |
| 5  | 2    | NO                | EXISTS? (none have p_id=5) NO | Leaf    |

**Final Result:**
| id | Type  |
|----|-------|
| 1  | Root  |
| 2  | Inner |
| 3  | Leaf  |
| 4  | Leaf  |
| 5  | Leaf  |

**Tree Visualization:**
```
      1 (Root)
     / \
    2   3 (Leaf)
   / \
  4   5 (Leaves)
  (Inner)
```

---

## PROBLEM 10: Trips and Users (Cancellation Rate)

### Problem Statement
Find cancellation rate of requests by date, excluding banned users and drivers.

### Sample Data
```sql
-- Trips
| id | client_id | driver_id | city_id | status      | request_date |
|----|-----------|-----------|---------|-------------|--------------|
| 1  | 1         | 10        | 1       | completed   | 2024-01-01   |
| 2  | 2         | 11        | 1       | cancelled   | 2024-01-01   |
| 3  | 3         | 12        | 6       | completed   | 2024-01-01   |
| 4  | 4         | 13        | 6       | cancelled   | 2024-01-02   |
| 5  | 1         | 10        | 1       | completed   | 2024-01-02   |

-- Users
| users_id | banned | role   |
|----------|--------|--------|
| 1        | No     | client |
| 2        | Yes    | client |
| 3        | No     | client |
| 4        | No     | client |
| 10       | No     | driver |
| 11       | No     | driver |
| 12       | No     | driver |
| 13       | Yes    | driver |
```

### Concept Breakdown

**1. Filtering with Subqueries**
- Exclude rows where client_id or driver_id is in banned list
- Use NOT IN with filtered subquery

**2. Aggregation with Conditions**
- SUM(CASE WHEN condition THEN 1 ELSE 0 END): Conditional counting
- ROUND(value, 2): Format to 2 decimal places

**3. CAST for Division**
- Integer division truncates: 1/2 = 0
- CAST to DECIMAL/REAL: 1.0/2 = 0.5

### Step-by-Step Solution

```sql
SELECT 
    request_date AS Day,
    ROUND(
        CAST(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS DECIMAL) /
        COUNT(*),
        2
    ) AS CancellationRate
FROM Trips
WHERE client_id NOT IN (
    SELECT users_id FROM Users WHERE banned = 'Yes'
)
AND driver_id NOT IN (
    SELECT users_id FROM Users WHERE banned = 'Yes'
)
GROUP BY request_date
ORDER BY request_date;
```

**Step-by-Step Execution:**

Step 1: Identify banned users
- Banned clients: users_id = 2
- Banned drivers: users_id = 13

Step 2: Filter trips
| id | client_id | driver_id | status      | request_date | Valid?              |
|----|-----------|-----------|-------------|--------------|---------------------|
| 1  | 1         | 10        | completed   | 2024-01-01   | ✓ (neither banned)  |
| 2  | 2         | 11        | cancelled   | 2024-01-01   | ✗ (client 2 banned) |
| 3  | 3         | 12        | completed   | 2024-01-01   | ✓                   |
| 4  | 4         | 13        | cancelled   | 2024-01-02   | ✗ (driver 13 banned)|
| 5  | 1         | 10        | completed   | 2024-01-02   | ✓                   |

Valid trips: 1, 3, 5

Step 3: Group by date and calculate

**2024-01-01:**
- Total: 2 trips (id 1, 3)
- Cancelled: 0
- Rate: 0/2 = 0.00

**2024-01-02:**
- Total: 1 trip (id 5)
- Cancelled: 0
- Rate: 0/1 = 0.00

**Final Result:**
| Day        | CancellationRate |
|------------|------------------|
| 2024-01-01 | 0.00             |
| 2024-01-02 | 0.00             |

---

## Python Problem Solving

### PROBLEM 11: Two Sum

### Problem Statement
Given an array of integers `nums` and an integer `target`, return indices of two numbers that add up to target.

### Sample Data
```python
nums = [2, 7, 11, 15]
target = 9

# Expected output: [0, 1] because nums[0] + nums[1] = 2 + 7 = 9
```

### Concept Breakdown

**1. Hash Map (Dictionary)**
- Key: number from array
- Value: index of that number
- Lookup time: O(1)

**2. Complement Concept**
- For each number `num`, we need `target - num`
- If complement exists in hash map, we found our pair

**3. Time Complexity Trade-offs**
- Brute force: O(n²) - check all pairs
- Hash map: O(n) - single pass with O(n) space

### Step-by-Step Solution

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Find two indices where elements sum to target.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Hash map to store number -> index
    num_to_index = {}
    
    for i, num in enumerate(nums):
        # Calculate what number we need to reach target
        complement = target - num
        
        # Check if complement exists in hash map
        if complement in num_to_index:
            # Found! Return indices
            return [num_to_index[complement], i]
        
        # Store current number and its index
        num_to_index[num] = i
    
    # No solution found
    return []
```

**Execution Trace:**

```python
nums = [2, 7, 11, 15]
target = 9

# Initial: num_to_index = {}

i=0, num=2:
  complement = 9 - 2 = 7
  Is 7 in num_to_index? No
  Store: num_to_index = {2: 0}

i=1, num=7:
  complement = 9 - 7 = 2
  Is 2 in num_to_index? YES! at index 0
  Return [0, 1] ✓
```

### Alternative: Brute Force (for comparison)
```python
def two_sum_brute_force(nums: list[int], target: int) -> list[int]:
    """
    Check all pairs. Time: O(n²), Space: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

---

### PROBLEM 12: Best Time to Buy and Sell Stock

### Problem Statement
Given stock prices array, find maximum profit from one buy and one sell. Must buy before sell.

### Sample Data
```python
prices = [7, 1, 5, 3, 6, 4]

# Expected: 5
# Buy at 1 (index 1), sell at 6 (index 4)
# Profit: 6 - 1 = 5
```

### Concept Breakdown

**1. Greedy Algorithm**
- At each step, track minimum price seen so far
- Calculate potential profit if sold today
- Keep track of maximum profit

**2. Single Pass Optimization**
- Track `min_price` as we iterate
- Update `max_profit` at each step
- No need to check all pairs

**3. The Key Insight**
The maximum profit ending at day `i` is: `prices[i] - min(prices[0:i])`

### Step-by-Step Solution

```python
def max_profit(prices: list[int]) -> int:
    """
    Find maximum profit from single buy and sell.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not prices or len(prices) < 2:
        return 0
    
    min_price = float('inf')  # Track lowest price seen
    max_profit = 0            # Track best profit
    
    for price in prices:
        # Update minimum price if current is lower
        if price < min_price:
            min_price = price
        
        # Calculate profit if sold at current price
        current_profit = price - min_price
        
        # Update maximum profit
        if current_profit > max_profit:
            max_profit = current_profit
    
    return max_profit
```

**Execution Trace:**

```python
prices = [7, 1, 5, 3, 6, 4]

Day 0, price=7:
  min_price = min(inf, 7) = 7
  profit = 7 - 7 = 0
  max_profit = 0

Day 1, price=1:
  min_price = min(7, 1) = 1  ← New low!
  profit = 1 - 1 = 0
  max_profit = 0

Day 2, price=5:
  min_price stays 1
  profit = 5 - 1 = 4  ← New best!
  max_profit = 4

Day 3, price=3:
  min_price stays 1
  profit = 3 - 1 = 2
  max_profit = 4

Day 4, price=6:
  min_price stays 1
  profit = 6 - 1 = 5  ← New best!
  max_profit = 5

Day 5, price=4:
  min_price stays 1
  profit = 4 - 1 = 3
  max_profit = 5

Return: 5
```

---

### PROBLEM 13: Group Anagrams

### Problem Statement
Given an array of strings, group anagrams together.

### Sample Data
```python
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

# Expected: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

### Concept Breakdown

**1. Anagram Property**
Two strings are anagrams if they have the same character counts.
"eat" and "tea" both have: {'e':1, 'a':1, 't':1}

**2. Sorting as Key**
Sorting characters of anagrams produces the same string:
sorted("eat") = sorted("tea") = ['a', 'e', 't'] → "aet"

**3. Hash Map for Grouping**
- Key: sorted string (signature)
- Value: list of anagrams with that signature

### Step-by-Step Solution

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.
    
    Time Complexity: O(n * k log k) where k = max string length
    Space Complexity: O(n * k)
    """
    from collections import defaultdict
    
    # Map from sorted string to list of anagrams
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Create signature by sorting characters
        # Anagrams will have identical signatures
        signature = ''.join(sorted(s))
        
        # Add to appropriate group
        anagram_groups[signature].append(s)
    
    # Return grouped anagrams
    return list(anagram_groups.values())
```

**Execution Trace:**

```python
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

Initialize: anagram_groups = {}

s = "eat":
  signature = sorted("eat") = ['a', 'e', 't'] → "aet"
  anagram_groups = {"aet": ["eat"]}

s = "tea":
  signature = sorted("tea") = ['a', 'e', 't'] → "aet"
  anagram_groups = {"aet": ["eat", "tea"]}

s = "tan":
  signature = sorted("tan") = ['a', 'n', 't'] → "ant"
  anagram_groups = {"aet": ["eat", "tea"], "ant": ["tan"]}

s = "ate":
  signature = "aet"
  anagram_groups = {"aet": ["eat", "tea", "ate"], "ant": ["tan"]}

s = "nat":
  signature = "ant"
  anagram_groups = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}

s = "bat":
  signature = "abt"
  anagram_groups = {"aet": ["eat", "tea", "ate"], 
                    "ant": ["tan", "nat"], 
                    "abt": ["bat"]}

Return: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

### Alternative: Character Count as Key
```python
def group_anagrams_count(strs: list[str]) -> list[list[str]]:
    """
    Use character count tuple as key.
    Time: O(n * k), Space: O(n * k)
    Faster for long strings (no sorting needed)
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Count each character (26 lowercase letters)
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        # Use tuple (hashable) as key
        groups[tuple(count)].append(s)
    
    return list(groups.values())
```

---

### PROBLEM 14: Top K Frequent Elements

### Problem Statement
Given an integer array and integer k, return the k most frequent elements.

### Sample Data
```python
nums = [1, 1, 1, 2, 2, 3]
k = 2

# Expected: [1, 2] (1 appears 3 times, 2 appears 2 times)
```

### Concept Breakdown

**1. Frequency Counting**
Use Counter to get frequency of each element.

**2. Heap Data Structure**
- Min-heap of size k keeps track of top k elements
- Heap operations: O(log k)
- Total time: O(n log k)

**3. Alternative: Bucket Sort**
- Index = frequency
- Value = list of elements with that frequency
- O(n) time, O(n) space

### Step-by-Step Solution (Heap)

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements using heap.
    
    Time Complexity: O(n log k)
    Space Complexity: O(n)
    """
    from collections import Counter
    import heapq
    
    # Step 1: Count frequencies
    count = Counter(nums)
    # count = {1: 3, 2: 2, 3: 1}
    
    # Step 2: Use min-heap to track top k
    # Store (frequency, num) pairs
    heap = []
    
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        else:
            # If current more frequent than least in heap
            if freq > heap[0][0]:
                heapq.heapreplace(heap, (freq, num))
    
    # Extract numbers from heap
    return [num for freq, num in heap]
```

**Execution Trace:**

```python
nums = [1, 1, 1, 2, 2, 3], k = 2

Step 1: Counter = {1: 3, 2: 2, 3: 1}

Step 2: Build heap of size k=2

num=1, freq=3:
  heap size 0 < 2, push (3, 1)
  heap = [(3, 1)]

num=2, freq=2:
  heap size 1 < 2, push (2, 2)
  heap = [(2, 2), (3, 1)]  (min-heap: 2 is root)

num=3, freq=1:
  heap size 2 == k
  freq=1 > heap[0][0]=2? No, skip
  heap = [(2, 2), (3, 1)]

Result: [2, 1] or [1, 2] (order may vary)
```

### Alternative: Bucket Sort (O(n))

```python
def top_k_frequent_bucket(nums: list[int], k: int) -> list[int]:
    """
    O(n) solution using bucket sort concept.
    """
    from collections import Counter, defaultdict
    
    count = Counter(nums)
    
    # Bucket: frequency -> list of numbers
    # Index represents frequency
    freq_buckets = defaultdict(list)
    
    for num, freq in count.items():
        freq_buckets[freq].append(num)
    
    # Collect from highest frequency down
    result = []
    for freq in range(len(nums), 0, -1):
        if freq in freq_buckets:
            result.extend(freq_buckets[freq])
            if len(result) >= k:
                return result[:k]
    
    return result
```

**Bucket Sort Execution:**

```python
nums = [1, 1, 1, 2, 2, 3], k = 2

Step 1: count = {1: 3, 2: 2, 3: 1}

Step 2: Build buckets
  freq 3: [1]
  freq 2: [2]
  freq 1: [3]
  
  freq_buckets = {3: [1], 2: [2], 1: [3]}

Step 3: Collect from high to low frequency
  freq=6 to freq=4: empty
  freq=3: add [1], result = [1]
  freq=2: add [2], result = [1, 2] ✓ (have k=2)

Return: [1, 2]
```

---

## Summary: Problem-Solving Patterns

### SQL Patterns

| Pattern | Use Case | Key Concepts |
|---------|----------|--------------|
| **LEFT JOIN + IS NULL** | Find missing relationships | Join types, NULL handling |
| **LIMIT + OFFSET** | Get Nth item | Sorting, pagination |
| **DENSE_RANK()** | Rank with ties | Window functions, PARTITION BY |
| **Self JOIN** | Compare rows within table | Row relationships |
| **CASE Expression** | Conditional logic | Control flow in SQL |
| **NOT EXISTS/IN** | Existence checks | Subqueries |

### Python Patterns

| Pattern | Use Case | Key Concepts |
|---------|----------|--------------|
| **Hash Map** | Fast lookup, counting | Dictionary, set |
| **Two Pointers** | Search in sorted array | Array manipulation |
| **Sliding Window** | Subarray problems | Range queries |
| **Greedy** | Optimization problems | Local optimal choices |
| **Heap** | Top K problems | Priority queue |
| **Bucket Sort** | Frequency-based sorting | Index as frequency |

---

## Quick Reference: Time & Space Complexity

### SQL Operations

| Operation | Time | Notes |
|-----------|------|-------|
| SELECT | O(n) | Full table scan |
| WHERE (indexed) | O(log n) | B-tree index lookup |
| JOIN | O(n×m) | Nested loop worst case |
| JOIN (indexed) | O(n log m) | Index-assisted |
| GROUP BY | O(n log n) | Requires sorting |
| ORDER BY | O(n log n) | Comparison sort |
| Window Function | O(n log n) | With ORDER BY |

### Python Operations

| Data Structure | Access | Search | Insert | Delete |
|----------------|--------|--------|--------|--------|
| List | O(1) | O(n) | O(n) | O(n) |
| Dict | O(1) | O(1) | O(1) | O(1) |
| Set | - | O(1) | O(1) | O(1) |
| Heap | - | - | O(log n) | O(log n) |

---

*This guide covers fundamental and advanced Python and SQL concepts with practical examples. Practice these concepts with real datasets to solidify your understanding.*
