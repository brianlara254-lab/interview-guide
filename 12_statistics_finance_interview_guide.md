# Statistics Interview Guide for Data Science & Data Analyst (Finance Domain)

A comprehensive statistics interview guide tailored for Finance domain roles, with real-world examples from payment networks, credit card issuers, and financial services industry scenarios.

---

## Table of Contents
1. [Descriptive Statistics for Transaction Data](#1-descriptive-statistics-for-transaction-data)
2. [Probability & Distributions in Finance](#2-probability--distributions-in-finance)
3. [Hypothesis Testing & A/B Testing](#3-hypothesis-testing--ab-testing)
4. [Sampling Methods & Statistical Inference](#4-sampling-methods--statistical-inference)
5. [Regression Analysis for Risk Modeling](#5-regression-analysis-for-risk-modeling)
6. [Time Series Analysis for Financial Data](#6-time-series-analysis-for-financial-data)
7. [Fraud Detection Statistics](#7-fraud-detection-statistics)
8. [Credit Risk Statistics](#8-credit-risk-statistics)
9. [Customer Retention & Churn Analytics](#9-customer-retention--churn-analytics)
10. [Bayesian Statistics in Finance](#10-bayesian-statistics-in-finance)
11. [Statistical Metrics for Model Evaluation](#11-statistical-metrics-for-model-evaluation)
12. [Interview Questions & Solutions](#12-interview-questions--solutions)

---

## 1. Descriptive Statistics for Transaction Data

### 1.1 Central Tendency Measures in Credit Card Analytics

| Measure | Formula | Payment Network Real-World Use Case | Interpretation |
|---------|---------|--------------------------|----------------|
| **Mean** | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$ | Average transaction amount per customer | Sensitive to outliers (large purchases) |
| **Median** | Middle value of sorted data | Median spend to understand typical customer behavior | Robust to outliers, represents "typical" transaction |
| **Mode** | Most frequent value | Most common transaction amount (e.g., $25, $50) | Identifies standard purchase patterns |
| **Geometric Mean** | $(\prod_{i=1}^{n}x_i)^{1/n}$ | Average growth rate of customer spending | Better for multiplicative growth rates |
| **Trimmed Mean** | Mean after removing top/bottom x% | Average spend excluding luxury purchases | Reduces impact of high-value outliers |

**Real Payment Network Example:**
```
Scenario: Analyzing Premium cardholder monthly spending

Dataset (monthly spend in USD):
[2500, 3200, 2800, 15000, 2900, 3100, 2600, 45000, 2750, 3000]

Mean: $8,185 (skewed by $45K luxury purchase)
Median: $2,950 (better representation of typical spend)
Mode: No clear mode (all unique)
5% Trimmed Mean: $2,987.5 (excludes $15K and $45K outliers)

Payment Network Insight: For marketing campaigns, use MEDIAN for segment targeting.
```

### 1.2 Dispersion & Variability Measures

| Measure | Formula | Finance Application |
|---------|---------|---------------------|
| **Variance** | $\sigma^2 = \frac{\sum(x_i - \bar{x})^2}{n}$ | Risk assessment in portfolios |
| **Standard Deviation** | $\sigma = \sqrt{\sigma^2}$ | Volatility of transaction amounts |
| **Coefficient of Variation** | $CV = \frac{\sigma}{\mu} \times 100\%$ | Comparing variability across card types |
| **Interquartile Range (IQR)** | $Q_3 - Q_1$ | Identifying outlier transactions |
| **Range** | $max - min$ | Daily transaction spread |

**Payment Network Example - Card Portfolio Analysis:**
```
Comparing spending variability across card tiers:

Green Card: Mean = $2,500, σ = $400, CV = 16%
Gold Card: Mean = $5,000, σ = $1,200, CV = 24%
Platinum Card: Mean = $12,000, σ = $4,800, CV = 40%

Interpretation:
- Higher tier cards have HIGHER relative variability (CV)
- Premium cardholders show diverse spending (business + luxury)
- Marketing strategy: Gold card users more predictable for offers
```

### 1.3 Percentiles & Quartiles in Fraud Detection

**Transaction Amount Distribution Analysis:**

```python
import numpy as np

# Payment Network transaction data (simplified)
transactions = np.array([
    12.50, 15.00, 23.99, 45.00, 52.30, 78.50,  # P0-P50 (everyday)
    89.99, 120.00, 156.40, 199.00, 250.00,      # P50-P80 (moderate)
    345.00, 499.00, 675.00, 850.00, 1200.00,    # P80-P95 (high)
    2500.00, 4500.00, 8900.00, 15000.00         # P95-P99 (luxury)
])

percentiles = {
    'P50 (Median)': np.percentile(transactions, 50),    # $78.50
    'P75 (Q3)': np.percentile(transactions, 75),        # $345.00
    'P90': np.percentile(transactions, 90),             # $1,200.00
    'P95': np.percentile(transactions, 95),             # $2,500.00
    'P99': np.percentile(transactions, 99),             # $8,900.00
    'P99.9': np.percentile(transactions, 99.9)          # $15,000.00
}

# Payment Network Fraud Detection Rule:
# Flag transactions > P99 of user's historical spending
# This catches 99% of potential fraud while minimizing false positives
```

### 1.4 Skewness & Kurtosis in Financial Data

| Metric | Formula | Interpretation for Payment Network |
|--------|---------|-------------------------|
| **Skewness** | $E[(\frac{X-\mu}{\sigma})^3]$ | Transaction amounts typically RIGHT-SKEWED (few large purchases) |
| **Kurtosis** | $E[(\frac{X-\mu}{\sigma})^4] - 3$ | High kurtosis indicates heavy tails (occasional extreme purchases) |

**Real-World Data:**
```
Payment Network Transaction Amount Distribution Characteristics:

Skewness: +2.5 (highly right-skewed)
- Most transactions are small (< $100)
- Long tail of high-value transactions

Kurtosis: +8.2 (leptokurtic - fat tails)
- More extreme values than normal distribution
- Important for risk modeling

Implications:
1. Use median instead of mean for central tendency
2. Log-transformation before parametric modeling
3. Heavy-tailed distributions for risk assessment
```

---

## 2. Probability & Distributions in Finance

### 2.1 Discrete Distributions

#### 2.1.1 Binomial Distribution - Fraud Detection

**Scenario:** Payment Network monitors 10,000 transactions. Historical fraud rate is 0.1%.

```
Parameters:
n = 10,000 (trials/transactions)
p = 0.001 (fraud probability)
q = 0.999 (legitimate probability)

Expected frauds: E[X] = np = 10
Variance: Var(X) = npq = 9.99

P(exactly 15 frauds) = C(10000,15) × (0.001)^15 × (0.999)^9985
P(>20 frauds) = 1 - P(X ≤ 20) = 0.0032 (0.32%)

Payment Network Application:
If we observe >20 frauds in 10K transactions, investigate immediately
(Only 0.32% chance if fraud rate is truly 0.1%)
```

#### 2.1.2 Poisson Distribution - Transaction Velocity

**Use Case:** Modeling number of transactions per minute during Black Friday

```
λ = average transactions per minute = 500

P(k transactions) = (e^(-λ) × λ^k) / k!

P(exactly 550 transactions) = (e^(-500) × 500^550) / 550!
P(>600 transactions) = 1 - Σ(k=0 to 600) P(k)

Payment Network Capacity Planning:
- P(X > 700) < 0.01 means 99% confidence 700/min capacity sufficient
- Need to provision infrastructure for λ + 3√λ = 500 + 67 = 567 peak
```

### 2.2 Continuous Distributions

#### 2.2.1 Normal Distribution - Credit Scores

**Payment Network Credit Score Distribution:**
```
X ~ N(μ=720, σ=80)  # FICO scores of Payment Network applicants

P(X > 750) = P(Z > (750-720)/80) = P(Z > 0.375) = 0.354 (35.4%)
P(650 < X < 800) = P(-0.875 < Z < 1.0) = 0.650 (65.0%)

Z-score interpretation for applicants:
Z = (Score - 720) / 80

Z > 2.0 (Score > 880): Premium offers (top 2.3%)
Z < -2.0 (Score < 560): Decline or secured card (bottom 2.3%)
```

#### 2.2.2 Log-Normal Distribution - Transaction Amounts

**Why Log-Normal for Financial Data:**
```
Transaction amounts are bounded below by 0 and have long right tail.
If log(X) ~ Normal, then X ~ Log-Normal

Payment Network Transaction Amount Model:
log(Transaction) ~ N(4.2, 1.5)

Properties:
- Mean: exp(4.2 + 1.5²/2) = exp(5.325) = $205.54
- Median: exp(4.2) = $66.69
- Mode: exp(4.2 - 1.5²) = $14.88

Right-skewed: Mean >> Median >> Mode
```

#### 2.2.3 Exponential Distribution - Time Between Transactions

**Use Case:** Modeling time between customer purchases
```
λ = average transactions per day = 0.5

PDF: f(t) = λe^(-λt) = 0.5e^(-0.5t)
CDF: F(t) = 1 - e^(-λt) = 1 - e^(-0.5t)

P(no transaction in 7 days) = e^(-0.5×7) = e^(-3.5) = 0.030 (3.0%)
Expected time between transactions: 1/λ = 2 days

Payment Network Churn Prediction:
If no transaction in 14 days: P = e^(-7) = 0.0009 (0.09%)
This is very unlikely → Potential churn signal
```

### 2.3 Heavy-Tailed Distributions

#### 2.3.1 Pareto Distribution - High-Value Customers

**80/20 Rule in Payment Network Portfolio:**
```
20% of customers generate 80% of transaction volume

PDF: f(x) = α × xₘ^α / x^(α+1)  for x ≥ xₘ

Parameters:
α (shape) ≈ log(4)/log(5) ≈ 1.16 for 80/20 rule
xₘ = minimum transaction amount

Payment Network VIP Customer Identification:
- Top 1% of customers: Spending > $100K/year
- Pareto analysis identifies these "whales"
- Dedicated relationship managers assigned
```

#### 2.3.2 Student's t-Distribution - Risk Metrics

**Use Case:** Calculating Value at Risk (VaR) with small samples
```
For portfolio returns with n=30 observations:
- Normal distribution underestimates tail risk
- t-distribution (df=29) accounts for uncertainty in variance

VaR at 95% confidence:
Normal: μ - 1.645σ
t-dist: μ - 1.699σ (more conservative, 3.3% higher)

Payment Network Risk Management: Use t-distribution for < 100 observations
```

---

## 3. Hypothesis Testing & A/B Testing

### 3.1 One-Sample Tests

#### 3.1.1 Z-Test for Proportions - Fraud Rate Change

**Scenario:** Payment Network wants to test if fraud rate has increased from historical 0.1%

```
H₀: p = 0.001 (fraud rate unchanged)
H₁: p > 0.001 (fraud rate increased)

Sample: n = 50,000 transactions, x = 65 frauds
Sample proportion: p̂ = 65/50000 = 0.0013

Test Statistic:
z = (p̂ - p₀) / √(p₀(1-p₀)/n)
z = (0.0013 - 0.001) / √(0.001×0.999/50000)
z = 0.0003 / 0.000141 = 2.12

Critical value (α=0.05, one-tailed): 1.645
p-value: P(Z > 2.12) = 0.017

Decision: Reject H₀ (p < 0.05)
Conclusion: Significant evidence fraud rate increased
Payment Network Action: Activate enhanced monitoring
```

#### 3.1.2 t-Test for Means - Average Transaction Value

**Scenario:** Testing if Gold Card members spend more than $5,000/month

```
H₀: μ = $5,000
H₁: μ > $5,000

Sample: n = 100 members
x̄ = $5,450, s = $1,200

t = (x̄ - μ₀) / (s/√n)
t = (5450 - 5000) / (1200/10) = 450 / 120 = 3.75

df = 99, critical value (α=0.05): 1.66
p-value: P(t₉₉ > 3.75) < 0.001

Decision: Reject H₀
Conclusion: Gold members spend significantly more than $5,000
```

### 3.2 Two-Sample Tests

#### 3.2.1 Comparing Two Proportions - Conversion Rates

**A/B Test: New vs. Old Application Landing Page**

```
Variant A (Old): n₁ = 10,000, conversions = 450, p̂₁ = 4.5%
Variant B (New): n₂ = 10,000, conversions = 520, p̂₂ = 5.2%

H₀: p₁ = p₂ (no difference)
H₁: p₁ ≠ p₂ (two-tailed)

Pooled proportion: p̄ = (450+520)/(10000+10000) = 0.0485

SE = √[p̄(1-p̄)(1/n₁ + 1/n₂)]
SE = √[0.0485×0.9515×0.0002] = 0.00303

z = (p̂₂ - p̂₁) / SE = (0.052 - 0.045) / 0.00303 = 2.31

Critical value (α=0.05, two-tailed): ±1.96
p-value: 2 × P(Z > 2.31) = 0.021

Decision: Reject H₀
Conclusion: New page has significantly higher conversion
Payment Network Action: Roll out new page to 100% traffic
Lift: (5.2-4.5)/4.5 = 15.6% improvement
```

#### 3.2.2 Two-Sample t-Test - Spend Comparison

**Comparing Millennial vs. Gen Z Spending:**

```
Millennial (n₁=500): x̄₁ = $3,200, s₁ = $800
Gen Z (n₂=500): x̄₂ = $2,850, s₂ = $650

Assuming unequal variances (Welch's t-test):

t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
t = (3200 - 2850) / √(640000/500 + 422500/500)
t = 350 / √(1280 + 845) = 350 / 46.1 = 7.59

df ≈ (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
df ≈ 947

Critical value (α=0.01): ±2.58
p-value < 0.0001

Conclusion: Highly significant difference in spending
Payment Network Strategy: Different product offerings for each segment
```

### 3.3 Chi-Square Tests

#### 3.3.1 Chi-Square Goodness of Fit - Transaction Category Distribution

**Testing if spending matches expected category distribution:**

```
Expected distribution (historical):
Dining: 35%, Travel: 25%, Shopping: 20%, Other: 20%

Observed (current month, n=1000 transactions):
Dining: 380, Travel: 220, Shopping: 180, Other: 220

H₀: Observed matches expected distribution
H₁: Distribution has changed

Expected counts:
Dining: 350, Travel: 250, Shopping: 200, Other: 200

χ² = Σ((O-E)²/E)
χ² = (380-350)²/350 + (220-250)²/250 + (180-200)²/200 + (220-200)²/200
χ² = 2.57 + 3.60 + 2.00 + 2.00 = 8.17

df = 4 - 1 = 3
Critical value (α=0.05): 7.815

Decision: Reject H₀ (8.17 > 7.815)
Conclusion: Spending pattern changed significantly
Payment Network Action: Adjust rewards program emphasis
```

#### 3.3.2 Chi-Square Test of Independence - Fraud by Card Type

```
                    Fraud    No Fraud    Total
Platinum            45       9,955       10,000
Gold                120      23,880      24,000
Green               85       16,915      17,000
Total               250      50,750      51,000

H₀: Fraud rate independent of card type
H₁: Fraud rate depends on card type

Expected frequencies (row total × column total / grand total):
Platinum/Fraud: 10000×250/51000 = 49.02
Platinum/No Fraud: 10000×50750/51000 = 9950.98
...

χ² = Σ((O-E)²/E) across all cells
χ² = (45-49.02)²/49.02 + (9955-9950.98)²/9950.98 + ...
χ² = 0.33 + 0.0016 + ... (sum all 6 cells) = 8.92

df = (3-1) × (2-1) = 2
Critical value (α=0.05): 5.991

Decision: Reject H₀
Conclusion: Fraud rate varies significantly by card type
Fraud rates: Platinum 0.45%, Gold 0.50%, Green 0.50%
(Platinum slightly lower - better security features)
```

### 3.4 ANOVA - Comparing Multiple Groups

#### 3.4.1 One-Way ANOVA - Spend by Region

**Comparing average spend across 4 US regions:**

```
Groups: Northeast, Southeast, Midwest, West
Sample size per region: n = 200

Hypotheses:
H₀: μ_NE = μ_SE = μ_MW = μ_W (all regions equal)
H₁: At least one region differs

Data:
Region      Mean    Std Dev
Northeast   $3,400  $850
Southeast   $3,100  $720
Midwest     $2,900  $680
West        $3,800  $920

Calculations:
SSB (Between) = Σnᵢ(x̄ᵢ - x̄)²
SSW (Within) = Σ(nᵢ-1)sᵢ²
SST = SSB + SSW

MSB = SSB / (k-1)
MSW = SSW / (N-k)
F = MSB / MSW

Results:
F = 12.45
df₁ = 3, df₂ = 796
Critical value (α=0.05): 2.62
p-value < 0.001

Decision: Reject H₀
Conclusion: Significant regional differences
Post-hoc (Tukey HSD): West > Northeast > Southeast > Midwest
```

### 3.5 A/B Testing Framework for Payment Network

#### 3.5.1 Sample Size Calculation

**Determining required sample for offer test:**

```
Parameters:
Baseline conversion rate (p₁): 5%
Minimum detectable effect: 1% (absolute)
Target conversion (p₂): 6%
Power: 80%
Significance level: 5%

Effect size (Cohen's h):
h = 2 × (arcsin(√p₂) - arcsin(√p₁))
h = 2 × (0.253 - 0.230) = 0.046

Sample size per variant:
n = 2 × (Z₁₋α/₂ + Z₁₋β)² / h²
n = 2 × (1.96 + 0.84)² / (0.046)²
n = 2 × 7.84 / 0.0021
n ≈ 7,467 per variant

Total: ~15,000 customers
Duration: If 1,000 eligible per day → 15 days
```

#### 3.5.2 Sequential Testing

**Early stopping in Payment Network fraud model testing:**

```
Traditional: Fixed sample size, analyze once
Sequential: Analyze as data arrives, stop early if significant

Group Sequential Design:
- Maximum 4 interim analyses
- O'Brien-Fleming boundary (conservative early, lenient late)
- Overall α maintained at 5%

Benefits:
- Average 30% reduction in sample size
- Faster decision making for fraud model deployment
- Ethical: Stop harmful experiments early
```

---

## 4. Sampling Methods & Statistical Inference

### 4.1 Sampling Techniques in Credit Card Data

| Method | Description | Payment Network Application |
|--------|-------------|------------------|
| **Simple Random** | Equal probability for all | Selecting audit sample |
| **Stratified** | Sample from subgroups | Proportionate by card tier |
| **Cluster** | Sample groups (clusters) | Geographic cluster sampling |
| **Systematic** | Every k-th element | Transaction log analysis |
| **Oversampling** | Increase minority class | Fraud case collection |

### 4.2 Stratified Sampling Example

**Payment Network Customer Satisfaction Survey:**

```
Population (1,000,000 cardholders):
- Platinum: 50,000 (5%)
- Gold: 200,000 (20%)
- Green: 350,000 (35%)
- Blue: 400,000 (40%)

Sample size: 10,000

Proportionate allocation:
Platinum: 10,000 × 0.05 = 500
Gold: 10,000 × 0.20 = 2,000
Green: 10,000 × 0.35 = 3,500
Blue: 10,000 × 0.40 = 4,000

Advantage: Ensures representation of premium customers
Precision gain: Up to 40% reduction in variance vs. simple random
```

### 4.3 Bootstrapping

**Estimating confidence interval for median transaction:**

```python
import numpy as np

# Original sample
transactions = np.array([45, 52, 48, 150, 55, 49, 2000, 51, 47, 53])
n = len(transactions)
B = 10000  # bootstrap samples

# Bootstrap procedure
bootstrap_medians = []
for _ in range(B):
    sample = np.random.choice(transactions, size=n, replace=True)
    bootstrap_medians.append(np.median(sample))

# 95% Confidence Interval
ci_lower = np.percentile(bootstrap_medians, 2.5)
ci_upper = np.percentile(bootstrap_medians, 97.5)

# Result: Median spend CI [49.0, 53.5]
# Payment Network can say with 95% confidence that true median is in this range
```

### 4.4 Confidence Intervals

#### 4.4.1 CI for Proportion - Approval Rate

```
Sample: 1,000 applications, 720 approved
Sample proportion: p̂ = 0.72

95% CI formula: p̂ ± Z × √(p̂(1-p̂)/n)

CI = 0.72 ± 1.96 × √(0.72×0.28/1000)
CI = 0.72 ± 1.96 × 0.0142
CI = 0.72 ± 0.0278
CI = (0.692, 0.748)

Interpretation: 95% confident true approval rate is 69.2% to 74.8%
Payment Network can report: "Approval rate is 72% ± 2.8%"
```

#### 4.4.2 CI for Mean - CLV Estimation

```
Sample of 500 customers:
Mean CLV: $8,500
Standard deviation: $2,400

95% CI: x̄ ± t₀.₀₂₅,₄₉₉ × (s/√n)
95% CI: 8500 ± 1.965 × (2400/√500)
95% CI: 8500 ± 1.965 × 107.3
95% CI: 8500 ± 210.9
95% CI: ($8,289, $8,711)

Margin of error: ±$211 (2.5% of mean)
```

---

## 5. Regression Analysis for Risk Modeling

### 5.1 Linear Regression

#### 5.1.1 Simple Linear Regression - Credit Limit Prediction

```
Predicting Credit Limit based on Income:

CreditLimit = β₀ + β₁ × Income + ε

Coefficients from Payment Network data:
β₀ = $500 (base limit)
β₁ = 0.15 (15% of annual income)

Interpretation:
- For every $10,000 increase in income, limit increases by $1,500
- If income = $80,000, predicted limit = $500 + 0.15×80000 = $12,500

Model Evaluation:
R² = 0.72 (72% of limit variation explained by income)
RMSE = $2,400 (typical prediction error)
```

#### 5.1.2 Multiple Regression - Default Probability

```
Log-odds(Default) = β₀ + β₁×CreditScore + β₂×DTI + β₃×Income + β₄×EmployYears

Estimated coefficients:
β₀ = 5.2
β₁ = -0.008 (per point)
β₂ = 0.12 (per 1% DTI)
β₃ = -0.00002 (per $ income)
β₄ = -0.15 (per year)

Example prediction:
Credit Score: 720
DTI: 25%
Income: $60,000
Employment: 3 years

Log-odds = 5.2 - 0.008×720 + 0.12×25 - 0.00002×60000 - 0.15×3
Log-odds = 5.2 - 5.76 + 3.0 - 1.2 - 0.45
Log-odds = 0.79

P(Default) = 1 / (1 + e^(-0.79)) = 0.687 → Wait, this is wrong
P(Default) = e^0.79 / (1 + e^0.79) = 2.203 / 3.203 = 0.688

Actually, higher log-odds = higher probability. Let me recalculate with proper context.

Correct interpretation (these are log-odds of GOOD performance):
P(Good) = 0.688 → P(Default) = 0.312 (31.2%)

Better model formulation for default:
Log-odds(Default) = negative coefficients for good factors
```

### 5.2 Logistic Regression

#### 5.2.1 Binary Classification - Fraud Detection

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Features for fraud detection
# X: [transaction_amount, time_since_last, merchant_risk_score, 
#     is_international, velocity_24h]

# Training data (simplified)
X = np.array([
    [45, 2, 0.2, 0, 3],      # Legitimate
    [1200, 0.1, 0.8, 1, 12], # Fraud
    [85, 5, 0.3, 0, 2],      # Legitimate
    # ... more samples
])
y = np.array([0, 1, 0, ...])  # 0=legitimate, 1=fraud

model = LogisticRegression()
model.fit(X, y)

# Coefficients interpretation (odds ratios)
# Amount: coef = 0.003 → OR = e^0.003 = 1.003
#   Each $1 increase increases fraud odds by 0.3%
# Time since last: coef = -2.5 → OR = e^(-2.5) = 0.082
#   Short time (< 1 min) significantly increases fraud risk

# Prediction
new_transaction = np.array([[500, 0.05, 0.9, 1, 15]])
fraud_prob = model.predict_proba(new_transaction)[0][1]
# Output: 0.87 (87% fraud probability)

# Payment Network Decision Rule
if fraud_prob > 0.7:
    action = "Decline + Alert"
elif fraud_prob > 0.3:
    action = "SMS Verification"
else:
    action = "Approve"
```

### 5.3 Regularization

#### 5.3.1 Ridge vs Lasso for Credit Scoring

```
Problem: 100+ features, multicollinearity, overfitting risk

Ridge Regression (L2):
- Penalty: λ × Σβ²
- Shrinks coefficients toward 0 but rarely to 0
- Good for correlated features (e.g., income and credit limit)
- Payment Network use: When all features are potentially relevant

Lasso Regression (L1):
- Penalty: λ × Σ|β|
- Can drive coefficients to exactly 0 (feature selection)
- Good for high-dimensional data
- Payment Network use: Initial feature selection from 500+ variables

Elastic Net:
- Combines L1 and L2: α×L1 + (1-α)×L2
- Best of both worlds
- Payment Network standard for credit risk models

Cross-validation for λ selection:
- 5-fold CV on training data
- Select λ that minimizes validation error
- Payment Network typically uses λ between 0.001 and 0.1
```

### 5.4 Model Diagnostics

#### 5.4.1 Residual Analysis

```
Checking regression assumptions:

1. Linearity: Residuals vs Fitted plot
   - Should show random scatter around 0
   - Pattern indicates non-linearity
   - Payment Network fix: Log-transform transaction amounts

2. Homoscedasticity: Scale-Location plot
   - Equal variance across fitted values
   - Fan shape = heteroscedasticity
   - Payment Network fix: Weighted least squares

3. Normality: Q-Q plot of residuals
   - Points should follow diagonal line
   - Heavy tails common in financial data
   - Payment Network fix: Robust standard errors

4. Independence: Durbin-Watson test
   - DW ≈ 2: No autocorrelation
   - DW < 2: Positive autocorrelation (common in time series)
   - Payment Network fix: Time series models (ARIMA, GARCH)
```

---

## 6. Time Series Analysis for Financial Data

### 6.1 Components of Time Series

**Payment Network Daily Transaction Volume Decomposition:**

```
Y(t) = T(t) + S(t) + C(t) + ε(t)

Where:
T(t) = Trend (long-term growth/decline)
S(t) = Seasonality (weekly, monthly patterns)
C(t) = Cyclical (economic cycles)
ε(t) = Irregular (random noise)

Example - Payment Network Transaction Data:

Date         Volume    Trend    Seasonal    Irregular
2024-01-01   1.2M      1.15M    +0.15M      -0.10M  (New Year spike)
2024-01-02   1.0M      1.16M    -0.10M      -0.06M  (Post-holiday drop)
...

Observed patterns:
- Trend: 3% annual growth
- Seasonal: +20% Nov-Dec (holidays), -15% Jan
- Weekly: +30% Friday-Saturday, -20% Sunday-Monday
```

### 6.2 Moving Averages

#### 6.2.1 Simple Moving Average - Trend Detection

```python
import pandas as pd

# Daily transaction amounts
daily_spend = pd.Series([...])  # 365 days

# 7-day SMA (weekly smoothing)
sma_7 = daily_spend.rolling(window=7).mean()

# 30-day SMA (monthly trend)
sma_30 = daily_spend.rolling(window=30).mean()

# Trading signal for fraud monitoring
# If daily > 2× SMA: Potential anomaly
anomaly_threshold = sma_30 * 2
alerts = daily_spend > anomaly_threshold
```

#### 6.2.2 Exponential Moving Average

```
EMA gives more weight to recent observations

Formula: EMA_t = α × Price_t + (1-α) × EMA_{t-1}
Where α = 2/(N+1)

Payment Network Application - Fraud Detection:
- EMA of transaction amount per customer
- If current > 3× EMA: Flag for review
- Adapts to customer's evolving spending pattern

Example:
Customer A 30-day EMA: $150
Today's transaction: $600
Ratio: 4× → Decline and call customer
```

### 6.3 ARIMA Models

#### 6.3.1 ARIMA for Transaction Forecasting

```
ARIMA(p,d,q) model selection:

1. Differentiation (d):
   - d=0: Stationary data
   - d=1: First difference (growth rate)
   - d=2: Second difference (acceleration)
   
   ADF Test for stationarity:
   H₀: Unit root (non-stationary)
   Payment Network daily transactions: d=1 typically sufficient

2. AR component (p):
   - PACF plot identifies p
   - Significant lags in PACF
   - Payment Network: p=1 or p=2 (yesterday affects today)

3. MA component (q):
   - ACF plot identifies q
   - Significant lags in ACF
   - Payment Network: q=1 (shocks persist one day)

Final Model: ARIMA(1,1,1)
- 1 autoregressive term
- 1 difference for stationarity
- 1 moving average term

Forecast accuracy: MAPE ≈ 5-8% for 7-day forecast
```

### 6.4 Seasonal Decomposition

#### 6.4.1 STL Decomposition for Payment Network Data

```python
from statsmodels.tsa.seasonal import STL

# Monthly transaction volume (3 years)
monthly_volume = pd.Series([...], index=pd.date_range('2022-01', periods=36, freq='M'))

# STL decomposition
stl = STL(monthly_volume, seasonal=13)  # 13-period seasonal
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Payment Network Insights:
# Seasonal component shows:
# - Peak: November (holiday shopping)
# - Trough: January (post-holiday)
# - Secondary peak: July (travel season)

# Residual analysis:
# Large residuals = anomalies (investigate)
```

### 6.5 Volatility Models (GARCH)

#### 6.5.1 Modeling Transaction Volatility

```
GARCH(1,1) for variance modeling:

σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}

Where:
ω = long-term average variance
α = weight of previous shock
β = weight of previous variance

Payment Network Application - Risk Management:

Daily transaction amount variance:
ω = 10000
α = 0.1
β = 0.85

Interpretation:
- 85% of volatility persists from day to day
- 10% from yesterday's shock
- Long-term avg variance: $10,000

Value at Risk calculation:
VaR_95% = Mean - 1.645×σ_t
Dynamic VaR adjusts daily based on recent volatility
```

---

## 7. Fraud Detection Statistics

### 7.1 Anomaly Detection Methods

#### 7.1.1 Z-Score Method

```
Flag transactions with |Z| > 3:

Z = (X - μ) / σ

Customer profile:
μ (mean transaction) = $150
σ (std dev) = $45

Transaction: $500
Z = (500 - 150) / 45 = 7.78

Decision: Z > 3 → Decline transaction

Limitations:
- Assumes normal distribution (financial data is skewed)
- Sensitive to outliers in historical data
- Payment Network improvement: Use median and MAD instead
```

#### 7.1.2 Modified Z-Score (Median-Based)

```
More robust to outliers:

M_i = 0.6745 × (X_i - median(X)) / MAD

Where MAD = median(|X_i - median(X)|)

Same customer:
Median = $140
MAD = $30

Transaction: $500
M = 0.6745 × (500 - 140) / 30 = 8.1

Threshold: |M| > 3.5 suggests outlier
Decision: Decline

Advantage: Robust to customer's own outlier history
```

#### 7.1.3 Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Features for each transaction
features = [
    'amount',
    'amount_vs_user_avg',
    'amount_vs_user_max',
    'time_since_last_txn',
    'merchant_category',
    'merchant_risk_score',
    'is_international',
    'velocity_1h',
    'velocity_24h',
    'unique_merchants_24h'
]

# Train isolation forest
iso_forest = IsolationForest(
    contamination=0.01,  # Expected 1% fraud rate
    n_estimators=100,
    random_state=42
)

iso_forest.fit(X_train)

# Predict
anomaly_scores = iso_forest.decision_function(X_test)
predictions = iso_forest.predict(X_test)  # -1 = anomaly, 1 = normal

# Payment Network threshold tuning:
# Score < -0.3: High confidence fraud
# Score -0.3 to 0: Medium risk
# Score > 0: Legitimate
```

### 7.2 Imbalanced Classification Metrics

#### 7.2.1 Confusion Matrix for Fraud Detection

```
                    Predicted
                    Fraud    Legit    Total
Actual   Fraud      850      150      1000 (1%)
         Legit      500      98,500   99,000 (99%)
         Total      1350     98,650   100,000

Metrics:
Accuracy = (850 + 98,500) / 100,000 = 99.35%

Precision = 850 / 1350 = 62.96%
  (Of flagged transactions, 63% are actually fraud)

Recall = 850 / 1000 = 85%
  (Caught 85% of all frauds)

Specificity = 98,500 / 99,000 = 99.5%
  (Correctly approved 99.5% of legitimate)

False Positive Rate = 500 / 99,000 = 0.51%
  (0.51% of legitimate declined)

F1-Score = 2 × (0.63 × 0.85) / (0.63 + 0.85) = 0.72

Payment Network Business Impact:
- Each false positive = angry customer + customer service cost
- Each false negative = fraud loss ($120 avg)
- Optimize threshold for cost minimization
```

#### 7.2.2 Precision-Recall Trade-off

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Find optimal threshold
# Payment Network typically optimizes for F-beta with β=0.5 (precision weighted)
beta = 0.5
f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)
optimal_idx = np.argmax(f_scores)
optimal_threshold = thresholds[optimal_idx]

# At optimal threshold:
print(f"Threshold: {optimal_threshold:.3f}")
print(f"Precision: {precisions[optimal_idx]:.3f}")
print(f"Recall: {recalls[optimal_idx]:.3f}")
print(f"F{beta}-Score: {f_scores[optimal_idx]:.3f}")
```

#### 7.2.3 ROC Curve and AUC

```
ROC Curve: Plot TPR vs FPR at different thresholds

AUC Interpretation:
- 0.5: Random classifier (no discrimination)
- 0.7: Acceptable
- 0.8: Good
- 0.9: Excellent
- 1.0: Perfect

Payment Network Fraud Model Performance:
Training AUC: 0.95
Validation AUC: 0.92
Test AUC: 0.91

No significant overfitting (drop of 0.04)
Model generalizes well

Precision-Recall AUC (better for imbalanced data):
PR-AUC = 0.78
Random baseline = 0.01 (fraud rate)
Lift: 78× better than random
```

### 7.3 Benford's Law for Fraud Detection

```
Benford's Law: First digit distribution in natural data:
Digit 1: 30.1%
Digit 2: 17.6%
Digit 3: 12.5%
...
Digit 9: 4.6%

Payment Network Application:
Legitimate transaction amounts follow Benford's Law
Fraudulent/manipulated data often doesn't

Chi-square test:
χ² = Σ((O-E)²/E)

If χ² > critical value: Potential fraud/manipulation

Example:
Merchant transactions first digits:
1: 25% (Expected 30.1%)
2: 18% (Expected 17.6%)
...
9: 8% (Expected 4.6%)

χ² = 45.2, df=8, p < 0.001
Conclusion: Significant deviation from Benford's Law
Action: Investigate merchant for potential fraud
```

---

## 8. Credit Risk Statistics

### 8.1 Probability of Default (PD) Models

#### 8.1.1 Logistic Regression for PD

```
PD Model for 12-month default prediction:

log(PD/(1-PD)) = β₀ + β₁×CreditScore + β₂×DTI + β₃×Inquiries + ...

Model coefficients:
Intercept: 10.5
Credit Score (/100): -0.45
DTI Ratio: 0.08
Recent Inquiries: 0.15
Delinquencies (12mo): 0.85
Employment (years): -0.20

Example calculation:
Applicant:
- Credit Score: 680
- DTI: 32%
- Inquiries: 2
- Delinquencies: 1
- Employment: 3 years

Log-odds = 10.5 - 0.45×6.8 + 0.08×32 + 0.15×2 + 0.85×1 - 0.20×3
Log-odds = 10.5 - 3.06 + 2.56 + 0.30 + 0.85 - 0.60
Log-odds = 10.55

PD = 1 / (1 + e^(-10.55)) ≈ 0.999... Error in calculation

Wait, this gives very high PD. Let me recalculate with realistic coefficients:

More realistic model:
Intercept: -8.5
Credit Score: -0.015 per point

Log-odds = -8.5 - 0.015×680 + 0.08×32 + 0.15×2 + 0.85×1 - 0.20×3
Log-odds = -8.5 - 10.2 + 2.56 + 0.30 + 0.85 - 0.60
Log-odds = -15.59

PD = 1 / (1 + e^(15.59)) ≈ 0.0000017 (very low)

Let me try a borderline case:
Credit Score: 620, DTI: 45%, 4 inquiries, 2 delinquencies

Log-odds = -8.5 - 9.3 + 3.6 + 0.6 + 1.7 - 0.6 = -12.5
Still too low. Better realistic intercept: -4

Log-odds = -4 - 9.3 + 3.6 + 0.6 + 1.7 - 0.6 = -8.0
PD = 1/(1+e^8) = 0.0003

Let me use properly calibrated example:
```

**Realistic PD Model:**
```
Intercept: -2.5
Credit Score: -0.004 per point
DTI: 0.05 per percent
Inquiries: 0.10 each
Delinquencies: 0.40 each
Employment years: -0.08 per year

Applicant A (Low Risk):
- Score: 750, DTI: 15%, Inquiries: 0, Delinq: 0, Emp: 5yr
Log-odds = -2.5 - 3.0 + 0.75 + 0 + 0 - 0.40 = -5.15
PD = 0.58% (Prime rate)

Applicant B (High Risk):
- Score: 620, DTI: 42%, Inquiries: 4, Delinq: 2, Emp: 1yr
Log-odds = -2.5 - 2.48 + 2.1 + 0.4 + 0.8 - 0.08 = -1.76
PD = 14.6% (Subprime)
```

### 8.2 Loss Given Default (LGD)

#### 8.2.1 Estimating Recovery Rates

```
LGD = 1 - Recovery Rate

Factors affecting LGD:
- Collateral value
- Seniority of debt
- Economic conditions
- Industry type

Payment Network Credit Cards (unsecured):
Historical LGD: 80-90%
(Only 10-20% recovered on defaulted balances)

LGD Model (Beta regression):
E[Recovery] = 1 / (1 + e^(-Xβ))

Coefficients:
Credit limit: -0.0002 (higher limit = lower recovery %)
Account age: 0.02 (older accounts = higher recovery)
Prior defaults: -0.15 (repeat defaulters = lower recovery)
Economic index: 0.10 (better economy = higher recovery)
```

### 8.3 Exposure at Default (EAD)

```
EAD = Current Balance + Expected Drawdown

For credit cards:
Expected Drawdown = (Credit Limit - Balance) × CCF

CCF (Credit Conversion Factor) model:
CCF = β₀ + β₁×Utilization + β₂×CreditLimit + β₃×Tenure

Example:
Credit Limit: $10,000
Current Balance: $2,000
Utilization: 20%
Tenure: 2 years

CCF = 0.10 + 0.30×0.20 - 0.00001×10000 + 0.02×2
CCF = 0.10 + 0.06 - 0.10 + 0.04 = 0.10 (10%)

Expected Drawdown = ($10,000 - $2,000) × 0.10 = $800
EAD = $2,000 + $800 = $2,800
```

### 8.4 Expected Loss (EL) Calculation

```
EL = PD × LGD × EAD

Example for a portfolio segment:

Segment: Gold Card, Credit Score 650-700
- Average PD: 3.5%
- Average LGD: 85%
- Average EAD: $5,200

EL = 0.035 × 0.85 × $5,200 = $154.70 per account

For 10,000 accounts in segment:
Total Expected Loss = $1,547,000

Pricing Implication:
Required revenue per account to cover EL: $154.70
Plus cost of funds, operating costs, profit margin
→ Minimum annual fee + interest required
```

### 8.5 Survival Analysis

#### 8.5.1 Kaplan-Meier Estimator

```python
from lifelines import KaplanMeierFitter

# Time to default data (months)
# Event: 1 = defaulted, 0 = censored (still active or closed good)
durations = [12, 18, 24, 6, 36, 15, ...]
events = [1, 0, 1, 1, 0, 1, ...]

kmf = KaplanMeierFitter()
kmf.fit(durations, events)

# Survival probability at 24 months
survival_24m = kmf.survival_function_at_times(24)
print(f"Probability of not defaulting within 24 months: {survival_24m:.3f}")
# Output: 0.925 (92.5%)

# Plot survival curve
kmf.plot_survival_function()

# Payment Network Insights:
# - Steepest decline in first 12 months (new account risk)
# - Plateau after 36 months (seasoned accounts)
```

#### 8.5.2 Cox Proportional Hazards

```python
from lifelines import CoxPHFitter

# Features affecting time to default
features = pd.DataFrame({
    'credit_score': [680, 720, 650, ...],
    'dti': [25, 18, 35, ...],
    'income': [60000, 85000, 45000, ...],
    'tenure': [2, 5, 1, ...]
})

# Fit Cox model
cph = CoxPHFitter()
cph.fit(features, durations, events)

# Hazard ratios
cph.print_summary()

# Output:
# credit_score: HR = 0.98 (per point)
#   20-point increase → 33% lower hazard (0.98^20 = 0.67)
# dti: HR = 1.05 (per %)
#   10% increase → 63% higher hazard (1.05^10 = 1.63)
```

---

## 9. Customer Retention & Churn Analytics

### 9.1 Retention Metrics for Credit Card Portfolio

#### 9.1.1 Key Retention KPIs

| Metric | Formula | Payment Network Business Use | Industry Benchmark |
|--------|---------|-------------------|-------------------|
| **Retention Rate** | (Customers at End - New) / Customers at Start | Overall portfolio health | 85-92% annual |
| **Churn Rate** | 1 - Retention Rate | Loss monitoring | 8-15% annual |
| **Monthly Active Rate** | Active Users / Total Users | Engagement tracking | 60-75% |
| **Revenue Retention** | Revenue from Cohort Year 2 / Year 1 | Expansion vs. Contraction | 100-110% (net) |
| **Net Revenue Retention** | (Recurring Revenue + Expansion - Contraction) / Base | Growth indicator | >100% for healthy |

**Payment Network Platinum Card Retention Analysis:**
```
Cohort: Premium cardholders acquired Jan 2023

Year 1 (2023): 10,000 new members
Year 2 (2024): 9,200 active members

Retention Rate = 9,200 / 10,000 = 92%
Churn Rate = 8%

Revenue Analysis:
Year 1 Average Revenue per Member: $850
Year 2 Average Revenue per Member: $920 (upsell + usage increase)

Revenue Retention = (9,200 × $920) / (10,000 × $850)
                = $8,464,000 / $8,500,000
                = 99.6%

Net Revenue Retention > 100% despite 8% churn
→ Remaining customers spend more, offsetting churn
```

#### 9.1.2 Cohort Analysis

**Monthly Cohort Retention Matrix:**
```
Cohort     Month 0  Month 1  Month 3  Month 6  Month 12  Month 24
2023-01    100%     95%      92%      89%      85%       78%
2023-02    100%     94%      91%      88%      83%       -
2023-03    100%     96%      93%      90%      86%       -
2023-04    100%     93%      89%      85%      -         -
...

Insights:
1. Drop-off steepest in Month 1 (4-7%)
2. Stabilization after Month 6
3. 2023-04 cohort underperforming → investigate onboarding issue
```

**Python Implementation:**
```python
import pandas as pd
import numpy as np

# Create cohort analysis
def cohort_analysis(transactions, customer_id, order_date, period='M'):
    """
    Calculate cohort retention matrix for credit card customers
    """
    # Convert to datetime
    transactions[order_date] = pd.to_datetime(transactions[order_date])
    
    # Get first transaction date (acquisition)
    transactions['cohort_month'] = transactions.groupby(customer_id)[order_date] \
        .transform('min').dt.to_period(period)
    
    # Get period of each transaction
    transactions['period_month'] = transactions[order_date].dt.to_period(period)
    
    # Calculate period number (months since acquisition)
    transactions['period_number'] = (transactions['period_month'] -
                                     transactions['cohort_month']).apply(attrgetter('n'))
    
    # Create cohort table
    cohort_data = transactions.groupby(['cohort_month', 'period_number'])[customer_id] \
        .nunique().reset_index()
    
    cohort_table = cohort_data.pivot(index='cohort_month',
                                     columns='period_number',
                                     values=customer_id)
    
    # Calculate retention rates
    cohort_sizes = cohort_table.iloc[:, 0]
    retention_matrix = cohort_table.divide(cohort_sizes, axis=0)
    
    return retention_matrix

# Payment Network Usage
retention_matrix = cohort_analysis(
    transactions=amex_transactions,
    customer_id='card_member_id',
    order_date='transaction_date'
)

# Identify at-risk cohorts
latest_retention = retention_matrix.iloc[:, -1]
underperforming = latest_retention[latest_retention < 0.80]
print(f"Cohorts with <80% retention: {underperforming.index.tolist()}")
```

### 9.2 Churn Prediction Models

#### 9.2.1 Statistical Churn Indicators

**Early Warning Signals:**
```
Behavioral Red Flags (30-60 days before churn):

1. Transaction Frequency Decline:
   - Current 30-day transactions < 50% of previous 90-day average
   - Statistical test: Z-score < -2

2. Spend Amount Reduction:
   - Current month spend < 30% of 3-month rolling average
   - t-test: p-value < 0.05

3. Engagement Drop:
   - No app login for 45+ days
   - Customer service calls > 3 (frustration indicator)

4. Payment Pattern Change:
   - Minimum payment only (vs. full payment historically)
   - Late payment (was never late before)

5. Reward Redemption Drop:
   - No redemptions in 90 days (losing interest)
```

**Churn Risk Score Formula:**
```
Risk Score = w₁×(1 - Activity_Z) + w₂×(1 - Engagement_Z) + w₃×Payment_Change

Where:
Activity_Z = (Current_Month_Txns - μ_6mo) / σ_6mo
Engagement_Z = (App_Sessions - μ_historical) / σ_historical
Payment_Change = 1 if only minimum payment, 0 otherwise

Weights (from logistic regression):
w₁ = 0.40 (transaction activity)
w₂ = 0.30 (digital engagement)
w₃ = 0.30 (payment behavior)

Risk Tiers:
0.0-0.3: Low Risk (green)
0.3-0.6: Medium Risk (yellow) → Marketing nurture
0.6-0.8: High Risk (orange) → Retention team outreach
0.8-1.0: Critical Risk (red) → Immediate intervention
```

#### 9.2.2 Survival Analysis for Churn

**Kaplan-Meier Survival Curve:**
```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Customer tenure data
# Event = 1 if churned, 0 if censored (still active)

# By card tier
platinum_tenures = [24, 18, 36, 12, 48, 6, 15, ...]
platinum_events = [0, 1, 0, 1, 0, 1, 0, ...]  # 1=churned

gold_tenures = [12, 8, 24, 6, 18, 3, 9, ...]
gold_events = [1, 1, 0, 1, 0, 1, 1, ...]

# Fit KM curves
kmf_platinum = KaplanMeierFitter()
kmf_platinum.fit(platinum_tenures, platinum_events, label='Platinum')

kmf_gold = KaplanMeierFitter()
kmf_gold.fit(gold_tenures, gold_events, label='Gold')

# Plot
kmf_platinum.plot_survival_function()
kmf_gold.plot_survival_function()

# Statistical test
results = logrank_test(
    platinum_tenures, gold_tenures,
    platinum_events, gold_events
)
print(f"p-value: {results.p_value:.4f}")
# p < 0.05 → Significant difference in retention between tiers

# Median survival time
print(f"Platinum median tenure: {kmf_platinum.median_survival_time_} months")
print(f"Gold median tenure: {kmf_gold.median_survival_time_} months")
```

**Cox Proportional Hazards for Churn:**
```python
from lifelines import CoxPHFitter

# Features affecting churn
features = pd.DataFrame({
    'annual_spend': [24000, 12000, 36000, 8000, ...],
    'transactors': [1, 0, 1, 0, ...],  # 1=pays in full
    'reward_redemptions_12mo': [4, 1, 8, 0, ...],
    'customer_service_calls': [0, 3, 1, 5, ...],
    'months_since_acquisition': [12, 6, 24, 3, ...]
})

durations = [24, 6, 36, 3, ...]  # months until churn or censoring
events = [0, 1, 0, 1, ...]  # 1=churned

cph = CoxPHFitter()
cph.fit(features, durations, events)

# Interpret hazard ratios
cph.print_summary()

# Example Output:
# Variable                   HR    p-value
# annual_spend ($10K)       0.85    0.001  (15% lower churn per $10K spend)
# transactors               0.45    0.0001 (55% lower churn if pay in full)
# reward_redemptions        0.90    0.01   (10% lower per redemption)
# customer_service_calls    1.35    0.001  (35% higher per call)
```

### 9.3 Customer Lifetime Value (CLV) Deep Dive

#### 9.3.1 Probabilistic CLV Models

**Buy Till You Die Models (BTYD):**
```python
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Transaction data: frequency, recency, T (age)
# frequency: number of repeat purchases
# recency: time between first and last purchase
# T: time between first purchase and end of observation

summary = pd.DataFrame({
    'frequency': [5, 2, 8, 1, 0, ...],  # repeat transactions
    'recency': [12, 4, 18, 2, 0, ...],   # weeks
    'T': [24, 12, 36, 8, 6, ...],       # customer age in weeks
    'monetary_value': [250, 150, 400, 100, 0, ...]  # avg transaction
})

# Fit BG/NBD model (predict transactions)
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Predict purchases in next 12 months
summary['predicted_purchases_12m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    52,  # weeks
    summary['frequency'],
    summary['recency'],
    summary['T']
)

# Fit Gamma-Gamma model (predict spend)
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(summary['frequency'], summary['monetary_value'])

# Predict average order value
summary['predicted_avg_order'] = ggf.conditional_expected_average_profit(
    summary['frequency'],
    summary['monetary_value']
)

# Calculate CLV
summary['clv_12m'] = (
    summary['predicted_purchases_12m'] *
    summary['predicted_avg_order'] *
    0.15  # profit margin
)

# Payment Network Application:
# Top 20% CLV customers → VIP treatment
# Bottom 20% CLV customers → Nurture or don't acquire
```

#### 9.3.2 CLV Segmentation

**Value-Based Segmentation:**
```
CLV Calculation Components:

Revenue Streams:
1. Interchange Revenue = Annual Spend × 2.5%
2. Interest Revenue = Revolving Balance × APR
3. Fee Revenue = Annual Fee + Late Fees + Other

Costs:
1. Cost of Funds = Average Balance × 3%
2. Rewards Cost = Points Earned × $0.01
3. Operational Cost = $50/year
4. Acquisition Cost = $300 (amortized over 5 years)

CLV Formula:
CLV = Σ(t=1 to T) [Revenue_t - Cost_t] × Retention_Probability_t / (1 + r)^t

Payment Network Example - Platinum Member:
Year 1:
- Spend: $24,000 → Interchange: $600
- APR Revenue: $200
- Annual Fee: $695
- Costs: $240 (rewards) + $50 (ops) + $100 (acquisition)
- Year 1 Profit: $1,105

Year 2+ (assuming 90% retention):
- Reduced acquisition cost
- Higher interchange (increased usage)
- Retention probability compounds

5-Year CLV (r = 10%): $4,850

Segmentation:
CLV > $5,000: Platinum Elite (dedicated concierge)
CLV $2,000-5,000: Gold Segment (premium offers)
CLV $500-2,000: Green Segment (standard)
CLV < $500: At-risk (retention campaigns)
```

### 9.4 A/B Testing for Retention

#### 9.4.1 Retention Campaign Testing

**Test Design:**
```
Objective: Reduce churn with proactive retention offers

Hypothesis: Customers receiving "exclusive bonus points" offer
           have 15% lower churn than control

Treatment: 5,000 bonus points (worth $50) for $1,000 spend
Control: No offer

Sample Size Calculation:
Baseline churn (control): 8%
Target churn (treatment): 6.8% (15% relative reduction)
Power: 80%
Significance: 95%

Required per group: ~7,500 customers
Total: 15,000 customers
Duration: 6 months observation

Randomization:
- Stratified by CLV quartile
- Stratified by tenure (new vs. established)
- Block randomization within strata

Success Metrics:
Primary: 6-month retention rate
Secondary: Incremental spend, offer redemption rate
```

**Statistical Analysis:**
```python
from scipy import stats

# Results after 6 months
control_retained = 6900
control_total = 7500
treatment_retained = 7050
treatment_total = 7500

# Proportions
p_control = control_retained / control_total  # 0.92
p_treatment = treatment_retained / treatment_total  # 0.94

# Two-proportion z-test
p_pooled = (control_retained + treatment_retained) / (control_total + treatment_total)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
z = (p_treatment - p_control) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

# Result
print(f"Control retention: {p_control:.3f}")
print(f"Treatment retention: {p_treatment:.3f}")
print(f"Lift: {(p_treatment - p_control) / p_control:.1%}")
print(f"p-value: {p_value:.4f}")

# If p < 0.05:
# Calculate ROI
offer_cost = 7500 * 50  # $375,000
incremental_customers = treatment_retained - control_retained  # 150
clv_per_customer = 4850  # from earlier
incremental_clv = incremental_customers * clv_per_customer  # $727,500
roi = (incremental_clv - offer_cost) / offer_cost  # 94% ROI
```

### 9.5 Engagement Score Models

#### 9.5.1 RFM Analysis (Recency, Frequency, Monetary)

```python
import pandas as pd
from datetime import datetime, timedelta

# Calculate RFM scores
def calculate_rfm(transactions, analysis_date=None):
    if analysis_date is None:
        analysis_date = transactions['date'].max()
    
    rfm = transactions.groupby('customer_id').agg({
        'date': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Create quintile scores (1-5, 5 being best)
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
    
    # Combined RFM score
    rfm['rfm_score'] = rfm['r_score'].astype(str) + \
                       rfm['f_score'].astype(str) + \
                       rfm['m_score'].astype(str)
    
    return rfm

# Payment Network Segment Definitions
segments = {
    '555': 'Champions',      # Best customers
    '554': 'Champions',
    '545': 'Champions',
    '544': 'Champions',
    '455': 'Loyal Customers', # High frequency & monetary
    '445': 'Loyal Customers',
    '354': 'Potential Loyalists', # Recent, growing
    '355': 'Potential Loyalists',
    '345': 'Potential Loyalists',
    '155': 'New Customers',   # Recent but low frequency
    '154': 'New Customers',
    '144': 'New Customers',
    '553': 'At Risk',         # Haven't purchased recently
    '552': 'At Risk',
    '543': 'At Risk',
    '551': 'Cannot Lose Them', # Were high value
    '541': 'Cannot Lose Them',
    '511': 'Lost',            # Haven't purchased in long time
    '411': 'Lost',
    '111': 'Lost'
}

# Apply segments
rfm['segment'] = rfm['rfm_score'].map(segments)
rfm['segment'] = rfm['segment'].fillna('Others')

# Action matrix
actions = {
    'Champions': 'Reward them. Early adopter for new products.',
    'Loyal Customers': 'Upsell higher value products.',
    'Potential Loyalists': 'Offer membership/loyalty programs.',
    'New Customers': 'Provide onboarding support.',
    'At Risk': 'Send personalized reactivation campaigns.',
    'Cannot Lose Them': 'Win them back via renewals/helpful products.',
    'Lost': 'Revive interest with reach-out campaigns.'
}
```

#### 9.5.2 Engagement Scoring Algorithm

```
Multi-Factor Engagement Score (0-100):

Components:
1. Transaction Engagement (30%)
   - Frequency score: min(txns/month ÷ 10, 1) × 100
   - Consistency score: std dev of monthly txns inverse
   
2. Digital Engagement (25%)
   - App logins: min(logins/week ÷ 3, 1) × 100
   - Feature usage: % of available features used
   
3. Financial Engagement (25%)
   - Utilization rate: balance ÷ limit
   - Payment behavior: 100 if full payer, 50 if revolver
   
4. Rewards Engagement (20%)
   - Redemption rate: redemptions ÷ points earned
   - Program participation: enrollment in bonus categories

Formula:
Engagement_Score = 0.30×Transaction + 0.25×Digital +
                   0.25×Financial + 0.20×Rewards

Payment Network Tier Thresholds:
90-100: Highly Engaged (product advocates)
70-89: Engaged (target for upsell)
50-69: Moderate (nurture with content)
30-49: Low (risk of churn)
0-29: Inactive (immediate intervention)
```

### 9.6 Win-Back Campaign Statistics

#### 9.6.1 Reactivation Probability Modeling

```python
from sklearn.ensemble import GradientBoostingClassifier

# Features for win-back prediction
features = pd.DataFrame({
    'days_since_last_txn': [90, 180, 45, 270, ...],
    'lifetime_spend': [5000, 1200, 8000, 600, ...],
    'tenure_months': [24, 12, 36, 6, ...],
    'previous_complaints': [0, 2, 0, 1, ...],
    'card_tier': [3, 1, 3, 1, ...],  # 3=Platinum, 1=Green
    'reward_balance': [50000, 5000, 75000, 1000, ...],
    'acquisition_channel': ['referral', 'paid', 'organic', 'paid', ...]
})

# Target: 1 if reactivated within 60 days of campaign
target = [1, 0, 1, 0, ...]

# Train model
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)
model.fit(features, target)

# Predict reactivation probability
reactivation_prob = model.predict_proba(features)[:, 1]

# Optimal targeting
# Only target if: Expected Value > Campaign Cost
# Expected Value = P(reactivation) × (CLV - Reactivation_Cost)

# Payment Network Win-Back ROI Calculation:
campaign_cost = 25  # per customer (email + offer)
clv = 2000  # estimated for reactivated customer

expected_value = reactivation_prob * (clv - campaign_cost)
roi = expected_value - campaign_cost

# Target customers with positive ROI
target_customers = features[expected_value > 0]
```

#### 9.6.2 Offer Optimization

```
Test Matrix for Win-Back Offers:

Offer Type                | Redemption Rate | Reactivation Rate | Cost | ROI
--------------------------|-----------------|-------------------|------|----
10% statement credit      | 8%              | 12%               | $50  | 2.5x
Bonus points (10K)        | 15%             | 18%               | $100 | 2.1x
Annual fee waiver         | 22%             | 25%               | $95  | 3.2x
No offer (control)        | -               | 3%                | $0   | -

Statistical Significance:
Chi-square test on reactivation rates:
χ² = 45.2, df = 3, p < 0.001

Conclusion: All offers significantly better than control
Best: Annual fee waiver (highest reactivation, good ROI)
```

---

## 10. Bayesian Statistics in Finance

### 9.1 Bayesian Inference for Fraud Scoring

#### 9.1.1 Updating Fraud Probability

```
Prior belief: P(Fraud) = 0.01 (1% base rate)

Evidence: Transaction is international
P(International|Fraud) = 0.30
P(International|Legit) = 0.05

Posterior calculation:
P(Fraud|International) = P(International|Fraud) × P(Fraud) / P(International)

P(International) = P(Int|Fraud)P(Fraud) + P(Int|Legit)P(Legit)
P(International) = 0.30×0.01 + 0.05×0.99 = 0.003 + 0.0495 = 0.0525

P(Fraud|International) = 0.30 × 0.01 / 0.0525 = 0.057 (5.7%)

Updated fraud probability: 5.7× increase from 1% to 5.7%

Additional evidence: Unusual amount (>3σ)
P(Unusual|Fraud) = 0.80
P(Unusual|Legit) = 0.01

New posterior using previous posterior as prior:
P(Fraud|Int ∧ Unusual) = 0.80 × 0.057 / (0.80×0.057 + 0.01×0.943)
P(Fraud|Int ∧ Unusual) = 0.0456 / (0.0456 + 0.0094)
P(Fraud|Int ∧ Unusual) = 0.829 (82.9%)

Decision: Decline transaction (probability > 70%)
```

### 9.2 Bayesian A/B Testing

```python
import numpy as np
from scipy import stats

# Variant A: 400 conversions / 10,000 visitors
# Variant B: 520 conversions / 10,000 visitors

# Bayesian approach with Beta priors
# Prior: Beta(1, 1) = Uniform

# Posteriors
alpha_A, beta_A = 1 + 400, 1 + 9600  # Beta(401, 9601)
alpha_B, beta_B = 1 + 520, 1 + 9480  # Beta(521, 9481)

# Sample from posteriors
samples_A = np.random.beta(alpha_A, beta_A, 100000)
samples_B = np.random.beta(alpha_B, beta_B, 100000)

# Probability that B > A
prob_B_better = np.mean(samples_B > samples_A)
print(f"P(Conversion_B > Conversion_A) = {prob_B_better:.3f}")
# Output: 0.997 (99.7% probability)

# Expected lift
lift = (samples_B - samples_A) / samples_A
expected_lift = np.mean(lift)
ci_lower, ci_upper = np.percentile(lift, [2.5, 97.5])
print(f"Expected lift: {expected_lift:.1%} (95% CI: {ci_lower:.1%} to {ci_upper:.1%})")

# Payment Network Decision: 99.7% > 95% threshold → Roll out B
```

### 9.3 Credible Intervals vs Confidence Intervals

```
Frequentist 95% CI: "If we repeated this experiment many times, 
                    95% of intervals would contain true parameter"

Bayesian 95% Credible Interval: "There is 95% probability that 
                                 true parameter is in this interval"

Payment Network Example - Default Rate:

Frequentist:
- Observed: 45 defaults / 1000 accounts = 4.5%
- 95% CI: (3.3%, 5.9%)

Bayesian (with Beta(2, 98) prior):
- Posterior: Beta(47, 1053)
- 95% Credible Interval: (3.4%, 6.0%)

Bayesian advantage for Payment Network:
- Incorporate prior knowledge (industry default rates)
- Smaller sample sizes needed
- Direct probability interpretation
```

---

## 10. Statistical Metrics for Model Evaluation

### 10.1 Classification Metrics Summary

| Metric | Formula | Payment Network Use Case | Good Value |
|--------|---------|---------------|------------|
| **Accuracy** | (TP+TN)/Total | Overall correctness | >95% (imbalanced) |
| **Precision** | TP/(TP+FP) | Minimize false declines | >60% |
| **Recall** | TP/(TP+FN) | Catch fraud | >80% |
| **Specificity** | TN/(TN+FP) | Customer experience | >99% |
| **F1-Score** | 2PR/(P+R) | Balanced performance | >0.70 |
| **AUC-ROC** | Area under ROC | Discrimination power | >0.90 |
| **AUC-PR** | Area under PR | Imbalanced data | >0.50 |
| **LogLoss** | -Σ(y·log(p)) | Probabilistic accuracy | <0.30 |
| **Brier Score** | MSE of probabilities | Calibration | <0.10 |
| **Kolmogorov-Smirnov** | max|CDF₁-CDF₀| | Score separation | >0.40 |

### 10.2 Kolmogorov-Smirnov (KS) Statistic

```
KS measures separation between distributions:

KS = max|CDF(fraud) - CDF(legitimate)|

Interpretation:
- KS < 0.2: Poor separation
- 0.2 ≤ KS < 0.4: Fair
- 0.4 ≤ KS < 0.6: Good
- KS ≥ 0.6: Excellent

Payment Network Fraud Model:
KS = 0.52 at score cutoff 650

Meaning: At score 650, 75% of frauds score below
         while only 23% of legitimate score below
         Difference: 52 percentage points

Regulatory requirement: KS > 0.30 for approval
```

### 10.3 Gini Coefficient

```
Gini = 2 × AUC - 1

Interpretation:
- Gini = 0: No discrimination
- Gini = 1: Perfect discrimination
- Gini < 0.4: Poor
- 0.4 ≤ Gini < 0.6: Acceptable
- 0.6 ≤ Gini < 0.8: Good
- Gini ≥ 0.8: Excellent

Payment Network Credit Risk Models:
Application scoring: Gini = 0.75
Behavioral scoring: Gini = 0.82
Fraud detection: Gini = 0.85

Trend monitoring:
- Gini drop > 5% in 3 months → Model degradation
- Investigate: Data drift, concept drift, population change
```

### 10.4 Population Stability Index (PSI)

```
PSI measures distribution shift between samples:

PSI = Σ(%Actual - %Expected) × ln(%Actual/%Expected)

Interpretation:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.25: Moderate change (monitor)
- PSI ≥ 0.25: Significant change (investigate)

Payment Network Application:
Compare current month score distribution vs. development sample

Score Band  Expected  Actual  %Exp  %Act  Diff  ln(%Act/%Exp)  Contribution
300-400     1000      1200    2%    2.4%  0.4%  0.182          0.0007
400-500     5000      5500    10%   11%   1%    0.095          0.0010
...
Total PSI = 0.18

Action: Moderate change - increase monitoring frequency
```

### 10.5 Characteristic Stability Index (CSI)

```
Similar to PSI but for individual features:

CSI tracks feature distribution shifts over time

Payment Network Example - Credit Score Feature:

Score Range  Dev %  Current %  Diff  Weight
300-500      5%     8%        3%    0.0012
500-600      15%    18%       3%    0.0005
600-700      35%    32%       -3%   0.0003
700-800      35%    30%       -5%   0.0008
800+         10%    12%       2%    0.0004

CSI = 0.0032

Interpretation: Minimal drift in credit score distribution
No action required

If CSI > 0.10 for key feature:
- Investigate data quality
- Check for policy changes
- Consider feature engineering
```

### 10.6 Calibration Metrics

#### 10.6.1 Brier Score

```
Brier Score = (1/N) × Σ(p_i - y_i)²

Where:
p_i = predicted probability
y_i = actual outcome (0 or 1)

Interpretation:
- 0: Perfect calibration
- 0.25: Random guessing (for binary)
- Lower is better

Payment Network PD Model Calibration:

Score Bucket  Predicted PD  Actual Default Rate  Difference
0-100         0.5%          0.4%                 -0.1%
100-200       1.2%          1.5%                 +0.3%
200-300       3.0%          2.8%                 -0.2%
300-400       7.0%          7.5%                 +0.5%
400-500       15.0%         14.2%                -0.8%
500+          30.0%         31.0%                +1.0%

Overall Brier Score: 0.042
Well-calibrated model (typical range 0.03-0.08)
```

#### 10.6.2 Hosmer-Lemeshow Test

```
H₀: Model is well-calibrated
H₁: Model is poorly calibrated

Test statistic:
HL = Σ((O_i - E_i)² / (E_i×(1-E_i/n_i)))

Where:
O_i = observed defaults in bin i
E_i = expected defaults in bin i
n_i = number of observations in bin i

Follows χ² distribution with g-2 degrees of freedom
(g = number of bins, typically 10)

Payment Network Model:
HL statistic: 7.2
df = 8
p-value: 0.51

Decision: Fail to reject H₀
Conclusion: Model is well-calibrated
```

---

## 11. Interview Questions & Solutions

### 11.1 Descriptive Statistics Questions

#### Q1: How would you handle outliers in Payment Network transaction data?

**Answer:**
```
Three approaches for outlier treatment:

1. Detection Methods:
   - Z-score: |Z| > 3 (assumes normality)
   - IQR: Values outside [Q1-1.5×IQR, Q3+1.5×IQR]
   - Modified Z-score: More robust, uses MAD
   - Isolation Forest: Multivariate outliers

2. Treatment Strategies:
   a) Capping (Winsorization): Set max at 99th percentile
      - Preserves sample size
      - Reduces impact of extreme values
   
   b) Transformation: Log transformation
      - Compresses right tail
      - Makes distribution more normal
   
   c) Separate modeling: Build separate models for segments
      - Regular spenders (95%)
      - High-value customers (5%)

3. Payment Network-Specific Consideration:
   DON'T automatically remove outliers!
   - Large transactions might be legitimate (luxury purchases)
   - Flag for review rather than delete
   - Use domain knowledge: Is $50K transaction reasonable for Premium cardholder?
```

#### Q2: Explain the difference between mean, median, and mode in credit card analytics.

**Answer:**
```
Mean: Average value
- Use: Overall portfolio metrics, revenue projections
- Issue: Skewed by luxury purchases
- Example: Mean transaction $185 (pulled up by $10K purchases)

Median: Middle value
- Use: "Typical" customer behavior, targeting
- Advantage: Robust to outliers
- Example: Median transaction $78 (better represents typical spend)

Mode: Most frequent value
- Use: Identifying common purchase amounts
- Payment Network use: Detecting structured transactions (money laundering)
- Example: Mode = $25 (many small purchases)

Payment Network Application:
- Marketing: Use MEDIAN for segment definitions
- Fraud: Flag deviations from customer's MODE
- Revenue: Use MEAN with trimming for forecasts
```

### 11.2 Probability Questions

#### Q3: Payment Network processes 1M transactions/day with 0.1% fraud rate. What's the probability of 1100+ frauds in a day?

**Answer:**
```
Given:
n = 1,000,000
p = 0.001
λ = np = 1000 (use Poisson approximation)

Using Normal approximation (valid for large n):
μ = np = 1000
σ = √(np(1-p)) = √(1000×0.999) ≈ 31.6

P(X ≥ 1100) = P(Z ≥ (1099.5 - 1000)/31.6)
P(X ≥ 1100) = P(Z ≥ 3.15)
P(X ≥ 1100) = 1 - 0.9992 = 0.0008 (0.08%)

Interpretation:
- Very unlikely under normal conditions
- If observed: Indicates fraud attack or system issue
- Payment Network action: Trigger alert, investigate immediately

Exact Poisson:
P(X ≥ 1100) = 1 - Σ(k=0 to 1099) [e^(-1000) × 1000^k / k!]
Using software: P ≈ 0.0008
```

#### Q4: A customer has 95% legitimate transaction history. A new transaction scores 0.8 on fraud model (high risk). What's the probability it's actually fraud?

**Answer:**
```
Bayes' Theorem application:

Prior: P(Fraud) = 0.05, P(Legit) = 0.95

Likelihood (from model validation):
P(Score=0.8|Fraud) = 0.70 (70% of frauds score 0.8)
P(Score=0.8|Legit) = 0.02 (2% of legitimate score 0.8)

Posterior:
P(Fraud|Score=0.8) = P(Score|Fraud) × P(Fraud) / P(Score=0.8)

P(Score=0.8) = 0.70×0.05 + 0.02×0.95 = 0.035 + 0.019 = 0.054

P(Fraud|Score=0.8) = 0.70 × 0.05 / 0.054 = 0.648 (64.8%)

Interpretation:
- Despite high model score (0.8), only 65% fraud probability
- Base rate (5% fraud) significantly impacts posterior
- Payment Network decision: Request additional verification (not auto-decline)
```

### 11.3 Hypothesis Testing Questions

#### Q5: How would you test if a new fraud detection model is better than the current one?

**Answer:**
```
Paired comparison approach:

1. Data Collection:
   - Random sample of 10,000 transactions
   - Score each with both models
   - Record actual outcome (fraud/legitimate)

2. Metric Selection:
   Primary: AUC-ROC (overall discrimination)
   Secondary: Precision at fixed recall (e.g., 80%)

3. Hypothesis Test:
   H₀: AUC_new = AUC_current
   H₁: AUC_new > AUC_current

4. Test Method:
   DeLong's test for comparing two AUCs
   
   Test statistic follows normal distribution
   
   Example results:
   AUC_new = 0.945
   AUC_current = 0.928
   DeLong z-score = 4.2
   p-value = 0.000013

5. Decision:
   p < 0.05 → Reject H₀
   New model significantly better

6. Business Validation:
   - Shadow mode for 1 week
   - Compare business metrics (false positive rate)
   - Rollout if business metrics improve
```

#### Q6: Payment Network wants to test if weekend spending is higher than weekday. How would you design this test?

**Answer:**
```
Paired t-test design:

1. Data Collection:
   - Sample: 100 customers with both weekday and weekend transactions
   - Metric: Average daily spend

2. Hypotheses:
   H₀: μ_weekend = μ_weekday
   H₁: μ_weekend > μ_weekday (one-tailed)

3. Test Procedure:
   Calculate difference for each customer: D = Weekend - Weekday
   
   Test statistic:
   t = D̄ / (s_D/√n)
   
   df = n - 1 = 99

4. Example Calculation:
   D̄ = $45 (weekend higher)
   s_D = $120
   n = 100
   
   t = 45 / (120/10) = 3.75
   
   Critical value (α=0.05, one-tailed): 1.66
   p-value < 0.001

5. Conclusion:
   Reject H₀
   Weekend spending significantly higher by $45/day on average

6. Payment Network Application:
   - Target weekend promotions
   - Adjust fraud thresholds for weekend patterns
   - Staff customer service for higher volume

Note: Check assumptions
- Differences approximately normal (n=100, CLT applies)
- Independent customers
- No significant outliers in differences
```

### 11.4 Regression Questions

#### Q7: Explain multicollinearity and how it affects credit risk models.

**Answer:**
```
Multicollinearity: High correlation among predictors

Problem in Credit Risk:
- Income and Credit Limit: r = 0.85
- Credit Score and Interest Rate: r = -0.80

Effects:
1. Unstable coefficient estimates
   - Small data changes → large coefficient swings
   - Difficult interpretation

2. Inflated standard errors
   - t-statistics reduced
   - May fail to detect significant predictors

3. Model overfitting
   - Poor generalization

Detection:
- VIF (Variance Inflation Factor): VIF > 10 indicates problem
- Correlation matrix: |r| > 0.8 concerning
- Condition number: > 30 indicates multicollinearity

Solutions:
1. Remove one variable:
   - Drop Credit Limit, keep Income (more fundamental)

2. Combine variables:
   - Create ratio: Credit Limit / Income

3. Regularization:
   - Ridge regression handles multicollinearity well
   - Payment Network standard for credit models

4. PCA:
   - Create uncorrelated components
   - Trade-off: Interpretability

Payment Network Practice:
- Check VIF during model development
- Prioritize business-interpretable variables
- Use domain knowledge for variable selection
```

#### Q8: How would you interpret logistic regression coefficients in a fraud model?

**Answer:**
```
Logistic regression: log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ...

Interpretation methods:

1. Log-Odds Scale:
   Coefficient β represents change in log-odds per unit change in X
   
   Example: β_amount = 0.001
   - Each $1 increase increases log-odds of fraud by 0.001
   - $100 increase increases log-odds by 0.1

2. Odds Ratio (more intuitive):
   OR = e^β
   
   Example: OR_amount = e^0.001 = 1.001
   - Each $1 increase multiplies fraud odds by 1.001
   - Each $100 increase: OR = e^0.1 = 1.105 (10.5% increase)
   
   Example: β_international = 1.2
   OR = e^1.2 = 3.32
   International transactions have 3.32× higher odds of fraud

3. Probability Scale (marginal effects):
   Effect depends on baseline probability
   
   At p=0.01 (1% fraud probability):
   $100 increase → p increases to ~1.1%
   
   At p=0.5 (50% fraud probability):
   $100 increase → p increases to ~52.5%

Payment Network Interpretation:
- Report odds ratios to business
- "International transactions are 3.3× more likely to be fraud"
- Use for policy decisions and customer communication
```

### 11.5 Time Series Questions

#### Q9: How would you forecast Payment Network transaction volume for next quarter?

**Answer:**
```
Forecasting approach:

1. Data Exploration:
   - Daily transaction volume (3 years)
   - Check for trends, seasonality, outliers
   - Decompose: Trend + Seasonal + Residual

2. Model Selection:
   
   Option A: ARIMA
   - Good for short-term forecasts
   - Captures autocorrelation
   - ARIMA(1,1,1)(1,1,1)₇ for weekly seasonality
   
   Option B: Prophet (Facebook)
   - Handles multiple seasonalities
   - Robust to missing data/outliers
   - Easy to add holidays (Black Friday, etc.)
   
   Option C: Machine Learning
   - XGBoost with lag features
   - External regressors (marketing spend, economic indicators)

3. Model Building (Prophet example):
   ```python
   from fbprophet import Prophet
   
   model = Prophet(
       yearly_seasonality=True,
       weekly_seasonality=True,
       holidays=amex_holiday_calendar
   )
   model.add_regressor('marketing_spend')
   model.fit(df)
   
   future = model.make_future_dataframe(periods=90)
   forecast = model.predict(future)
   ```

4. Evaluation:
   - Train: 2 years, Test: 6 months
   - Metrics: MAPE, RMSE
   - Target: MAPE < 5%

5. Uncertainty:
   - Report prediction intervals
   - Plan for 10% above forecast (capacity)

6. Payment Network Specific:
   - Include marketing campaign calendar
   - Adjust for new product launches
   - Monitor for macroeconomic shifts
```

#### Q10: How would you detect anomalies in real-time transaction streams?

**Answer:**
```
Real-time anomaly detection system:

1. Feature Engineering (rolling windows):
   - Transaction amount vs. 30-day median
   - Transactions per hour (velocity)
   - Unique merchants in 24h
   - Geographic distance from last transaction

2. Detection Methods:

   a) Statistical (Z-score):
      - Real-time mean/std calculation (exponential weighting)
      - Flag if |Z| > 3
      - Fast, interpretable

   b) Isolation Forest:
      - Pre-trained model
      - Score each transaction
      - Threshold: Score < -0.5

   c) LSTM Autoencoder:
      - Reconstruction error > threshold
      - Captures sequence patterns

3. Implementation Architecture:
   ```
   Transaction Stream → Kafka → Feature Store → 
   Model Scoring → Rule Engine → Decision
                         ↓
                    Alert System
   ```

4. Threshold Calibration:
   - Historical validation
   - Target: < 0.5% false positive rate
   - Review queue capacity: 100 cases/hour

5. Feedback Loop:
   - Confirm fraud/legitimate
   - Update model weekly
   - Track detection rate

6. Payment Network Requirements:
   - Latency: < 50ms scoring
   - Availability: 99.99%
   - Explainable decisions for compliance
```

### 11.6 Risk Model Questions

#### Q11: Explain the difference between PD, LGD, and EAD models.

**Answer:**
```
Three components of credit risk:

1. PD (Probability of Default):
   - Likelihood borrower defaults in next 12 months
   - Output: Probability (0-1)
   - Model: Logistic regression, ML classifiers
   - Features: Credit score, DTI, payment history
   
   Payment Network Example:
   Customer with Score 720, DTI 20%: PD = 1.2%

2. LGD (Loss Given Default):
   - Percentage of exposure lost if default occurs
   - Output: Percentage (0-100%)
   - Model: Beta regression, OLS on recovery
   - Features: Collateral, seniority, economic conditions
   
   Payment Network Example:
   Credit card (unsecured): LGD = 85%
   (Only 15% recovered through collections)

3. EAD (Exposure at Default):
   - Outstanding balance at time of default
   - Output: Dollar amount
   - Model: Credit conversion factor on undrawn amount
   - Features: Current balance, credit limit, utilization trend
   
   Payment Network Example:
   Current: $2,000, Limit: $10,000
   EAD = $2,000 + ($8,000 × 0.10 CCF) = $2,800

4. Expected Loss (EL):
   EL = PD × LGD × EAD
   
   Example:
   PD = 3%, LGD = 85%, EAD = $2,800
   EL = 0.03 × 0.85 × $2,800 = $71.40

5. Use Cases:
   - Pricing: Interest rate must cover EL + costs + profit
   - Provisioning: Reserve capital for expected losses
   - Portfolio management: Identify high-risk segments
```

#### Q12: How would you validate a credit risk model for regulatory compliance?

**Answer:**
```
SR 11-7 / CCAR Model Validation Framework:

1. Conceptual Soundness:
   - Theory justification for model approach
   - Peer comparison
   - Expert judgment review

2. Data Quality:
   - Source system documentation
   - Data lineage
   - Missing value treatment
   - Outlier analysis

3. Statistical Performance:
   
   a) Discrimination:
      - Gini coefficient > 0.40
      - KS statistic > 0.30
      - AUC-ROC > 0.70
   
   b) Calibration:
      - Brier score < 0.10
      - Hosmer-Lemeshow test (p > 0.05)
      - Bin-wise actual vs. predicted comparison
   
   c) Stability:
      - PSI < 0.25
      - CSI monitoring for key variables

4. Out-of-Time Validation:
   - Test on period not used in training
   - Minimum 12 months post-development
   - Different economic conditions

5. Sensitivity Analysis:
   - Stress test scenarios
   - What-if: Unemployment +5%
   - What-if: Housing prices -20%

6. Benchmarking:
   - Compare to vendor models
   - Compare to simple alternative (scorecard)
   - Champion-Challenger framework

7. Documentation:
   - Model Development Document
   - Validation Report
   - Ongoing monitoring plan

8. Governance:
   - Independent validation team
   - Model Risk Committee approval
   - Annual recertification

Payment Network Specific:
- CCAR stress testing requirements
- FRB and OCC examinations
- Model inventory management
```

### 12.7 Customer Retention Questions

#### Q13: How would you calculate and interpret customer retention rate for a credit card portfolio?

**Answer:**
```
Retention Rate Formula:
Retention Rate = (Customers at End - New Customers) / Customers at Start

Payment Network Example - Annual Retention:
Q1 2024:
- Customers at Start (Jan 1): 100,000
- New Acquisitions: 15,000
- Customers at End (Dec 31): 108,000

Retention Rate = (108,000 - 15,000) / 100,000
               = 93,000 / 100,000
               = 93%

Interpretation:
- 93% of existing customers retained
- 7% churned (cancelled or inactive)
- Industry benchmark: 85-92% for credit cards
- Payment Network Platinum typically higher: 94-96%

Cohort-Specific Retention:
Different calculation for more insight:
- Month 1 Retention: 95% (5% early churn)
- Month 6 Retention: 90%
- Month 12 Retention: 86%
- Month 24 Retention: 82%

Revenue Retention (more important metric):
Revenue Retention = Revenue Year 2 / Revenue Year 1

Example:
Cohort Year 1 Revenue: $8,500,000
Same Cohort Year 2 Revenue: $8,800,000

Revenue Retention = 103.5%

Interpretation:
- Despite losing 14% of customers
- Remaining customers spent 20% more
- Net revenue GROWTH from cohort
```

#### Q14: Design a churn prediction model for credit card customers.

**Answer:**
```
Churn Prediction Model Design:

1. Problem Definition:
   - Define churn: No transaction in 90 days OR account closure
   - Prediction window: 30-60 days ahead
   - Balance precision/recall for intervention cost

2. Feature Engineering:

   Behavioral Features:
   - Transaction frequency (current vs. historical)
   - Average transaction amount trend
   - Days since last transaction
   - Unique merchant count (diversity indicator)
   
   Engagement Features:
   - App login frequency
   - Customer service interactions
   - Reward redemption rate
   - Payment behavior changes
   
   Financial Features:
   - Utilization rate trend
   - Payment amount vs. minimum
   - Balance growth/decline
   - Cash advance usage
   
   Temporal Features:
   - Seasonality indicators
   - Time since acquisition
   - Lifecycle stage

3. Model Selection:

   Option A: Logistic Regression (baseline)
   - Pros: Interpretable, fast
   - Cons: May miss non-linear patterns
   
   Option B: Gradient Boosting (XGBoost/LightGBM)
   - Pros: Handles non-linearity, feature interactions
   - Cons: Less interpretable
   
   Option C: Survival Model (Cox PH)
   - Pros: Time-to-event prediction, handles censoring
   - Cons: Requires more data

4. Model Performance:

   Key Metrics:
   - AUC-ROC: > 0.80
   - Precision@10%: > 40% (of top 10% risk, 40% actually churn)
   - Recall@20%: > 60% (catch 60% of churners in top 20% risk)
   
   Calibration:
   - Predicted probability should match actual churn rate
   - Hosmer-Lemeshow test p > 0.05

5. Intervention Strategy:

   Risk Tiers:
   - Low Risk (0-30%): No action
   - Medium Risk (30-60%): Automated offers
   - High Risk (60-80%): Personalized outreach
   - Critical Risk (80-100%): Retention team call

6. Business Impact:

   ROI Calculation:
   - Cost of intervention: $50 per customer
   - CLV of saved customer: $2,000
   - Break-even: Save 1 in 40 customers
   
   Target: 30% lift in retention for contacted customers
```

#### Q15: How would you calculate Customer Lifetime Value (CLV) for a credit card customer?

**Answer:**
```
CLV Components for Credit Cards:

1. Revenue Streams:
   - Interchange: 2.5% × Annual Spend
   - Interest: APR × Average Revolving Balance
   - Fees: Annual Fee + Late/Overlimit Fees
   - Other: Foreign exchange, cash advance

2. Cost Components:
   - Cost of Funds: 3% × Average Balance
   - Rewards: Points Earned × $0.01
   - Operations: $50/year servicing cost
   - Acquisition: $300 (amortized over 5 years)
   - Expected Loss: PD × LGD × EAD

3. CLV Formula:

   Historical CLV:
   CLV = Σ(Revenue_t - Cost_t) for t=1 to T

   Predictive CLV:
   CLV = Σ[(Revenue_t - Cost_t) × Retention_Probability_t] / (1 + r)^t
   
   Where:
   - Retention_Probability_t: Probability customer still active
   - r: Discount rate (10% typical)
   - T: Planning horizon (5 years)

4. Payment Network Example Calculation:

   Customer Profile:
   - Card: Gold ($250 annual fee)
   - Annual Spend: $24,000
   - APR: 22.99%
   - Revolving Balance: $3,000 average
   - Tenure: 3 years

   Year 1 Calculation:
   Revenue:
   - Interchange: $24,000 × 2.5% = $600
   - Interest: $3,000 × 22.99% = $690
   - Annual Fee: $250
   - Total Revenue: $1,540
   
   Costs:
   - Cost of Funds: $3,000 × 3% = $90
   - Rewards: ($24,000 × 1 pt/$) × $0.01 = $240
   - Operations: $50
   - Acquisition: $60 ($300/5 years)
   - Expected Loss: $30 (1% × 80% × $3,750)
   - Total Cost: $470
   
   Year 1 Profit: $1,070
   
   Year 2-5:
   - Similar calculation
   - Retention probability: 92% each year
   - No acquisition cost
   
   5-Year CLV:
   Year 1: $1,070 × 1.00 = $1,070
   Year 2: $1,070 × 0.92 / 1.10 = $895
   Year 3: $1,070 × 0.85 / 1.21 = $751
   Year 4: $1,070 × 0.78 / 1.33 = $627
   Year 5: $1,070 × 0.72 / 1.46 = $528
   
   Total CLV: $3,871

5. CLV Applications:
   
   - Acquisition Budget: Spend up to CLV × 30% to acquire
   - Segmentation: Different service tiers by CLV
   - Retention Priority: Focus on high CLV at-risk customers
   - Product Development: Features for high CLV segments
```

#### Q16: Explain how you would design an A/B test to improve customer retention.

**Answer:**
```
Retention A/B Test Design:

1. Objective & Hypothesis:
   
   Objective: Reduce 6-month churn rate with proactive engagement
   
   Hypothesis: Customers receiving "exclusive" bonus points offer
   will have 15% lower churn than control group

2. Success Metrics:
   
   Primary: 6-month retention rate
   Secondary:
   - Offer redemption rate
   - Incremental spend
   - Engagement score improvement
   
   Guardrail Metrics:
   - Customer satisfaction (don't annoy customers)
   - Support call volume

3. Sample Size Calculation:
   
   Parameters:
   - Baseline churn: 8%
   - Target churn: 6.8% (15% relative reduction)
   - Power: 80%
   - Significance: α = 0.05
   
   Calculation:
   Effect size: h = 2×(arcsin(√0.068) - arcsin(√0.08)) = 0.043
   n = 2 × (1.96 + 0.84)² / 0.043² ≈ 7,500 per group
   
   Total: 15,000 customers

4. Randomization:
   
   Stratified Randomization:
   - Stratum 1: CLV Quartile (4 levels)
   - Stratum 2: Tenure (New <1yr, Established 1-3yr, Veteran >3yr)
   - Stratum 3: Card Tier (Green, Gold, Platinum)
   
   Total strata: 4 × 3 × 3 = 36
   Ensures balanced groups across important dimensions

5. Treatment Design:
   
   Control: No intervention
   Treatment:
   - Email: "Exclusive offer just for you"
   - 5,000 bonus points for $1,000 spend in 30 days
   - Points worth ~$50

6. Statistical Analysis:
   
   Two-proportion z-test:
   H₀: p_control = p_treatment
   H₁: p_control > p_treatment (one-tailed)
   
   Example Results:
   Control: 6,900/7,500 = 92.0% retention
   Treatment: 7,050/7,500 = 94.0% retention
   
   Z = (0.94 - 0.92) / SE = 2.45
   p-value = 0.007
   
   Conclusion: Reject H₀, treatment significantly better

7. Business Impact:
   
   Incremental retained: 150 customers
   CLV per customer: $3,871
   Incremental value: $580,650
   
   Campaign cost: 7,500 × $50 = $375,000
   
   ROI: ($580,650 - $375,000) / $375,000 = 55%
   
   Recommendation: Roll out to all eligible customers

8. Post-Experiment:
   
   - Monitor long-term effects (12 months)
   - Check for novelty effects
   - Segment analysis (which groups respond best)
   - Iterate on offer design
```

#### Q17: How would you use cohort analysis to understand customer retention?

**Answer:**
```
Cohort Analysis Framework:

1. Cohort Definition:
   
   Time-Based Cohorts:
   - Acquisition month (most common)
   - First transaction date
   - Marketing campaign period
   
   Behavior-Based Cohorts:
   - Acquisition channel (organic, paid, referral)
   - First product used
   - Initial spend tier

2. Retention Matrix Construction:

   Example - Monthly Acquisition Cohorts:
   
   Cohort     Month 0  Month 1  Month 3  Month 6  Month 12
   2024-01    100%     95%      92%      89%      85%
   2024-02    100%     94%      90%      87%      -
   2024-03    100%     96%      93%      -        -
   2024-04    100%     92%      -        -        -

3. Key Insights:

   Diagonal Analysis (Same Age):
   - Month 1 retention declining: 95% → 94% → 96% → 92%
   - 2024-04 cohort underperforming
   - Investigation: What changed in April? (new onboarding?)

   Horizontal Analysis (Cohort Maturity):
   - 2024-01: Steady decline, stabilizes at 85%
   - Normal pattern for credit cards
   - Most churn happens in months 1-3

   Vertical Analysis (Specific Time):
   - Month 6 across cohorts: 89%, 87%
   - Consistent performance

4. Cohort Metrics:

   Cohort Lifetime Value:
   CLV_cohort = Σ(Retention_Month_n × Monthly_Revenue_n)
   
   Cohort Payback Period:
   Months to recover acquisition cost
   
   Cohort Quality Score:
   Composite of retention, revenue, engagement

5. Payment Network Application:

   Cohort Analysis Insights:
   
   a) Channel Performance:
      - Referral cohorts: 95% Month-12 retention
      - Paid social cohorts: 82% Month-12 retention
      - Action: Shift budget to referral programs
   
   b) Product Impact:
      - Platinum cardholders: 96% retention
      - Green cardholders: 88% retention
      - Action: Upsell strategies for Green cohorts
   
   c) Seasonal Effects:
      - Holiday acquisition cohorts: Lower retention
      - Gift recipients, not decision makers
      - Action: Different onboarding for gift cohorts

6. Statistical Significance:

   Testing Cohort Differences:
   Chi-square test on retention rates
   Log-rank test on survival curves
   
   Example:
   Referral vs. Paid cohort at Month 6:
   χ² = 18.5, df = 1, p < 0.001
   Conclusion: Significant difference in retention
```

---

## Appendix A: Payment Network-Specific Statistical Formulas

### A.1 Credit Limit Assignment

```
Initial Limit = max(Base, min(Cap, Income × Multiplier × RiskFactor))

Where:
Base = $1,000 (minimum)
Cap = $50,000 (maximum for new customers)
Multiplier = 0.15 (15% of annual income)
RiskFactor = 1.0 - 0.002 × (850 - CreditScore)

Example:
Income: $80,000, Score: 750
RiskFactor = 1.0 - 0.002 × 100 = 0.8
Limit = max(1000, min(50000, 80000 × 0.15 × 0.8))
Limit = max(1000, min(50000, 9600)) = $9,600
```

### A.2 Minimum Payment Calculation

```
Minimum Payment = max(Floor, min(Cap, Balance × Rate + Fees + Interest))

Where:
Floor = $35 (or full balance if <$35)
Cap = Balance (can't exceed balance)
Rate = 1% (of statement balance)
Fees = Late fees, overlimit fees
Interest = APR/12 × Balance

Example:
Balance: $2,500
APR: 24.99%
No fees

Interest = 0.2499/12 × 2500 = $52.06
Min Payment = max(35, min(2500, 2500×0.01 + 52.06))
Min Payment = max(35, 77.06) = $77.06
```

### A.3 Reward Points Valuation

```
Point Value = Redemption Amount / Points Required

Payment Network Benchmarks:
- Statement credit: $0.006/point
- Travel (Amex Travel): $0.01/point
- Transfer to airlines: $0.015-0.02/point
- Gift cards: $0.008/point

Optimal Strategy:
Transfer to airline partners for premium cabin
Value: 1.5-2 cents per point

Cost to Payment Network:
Interchange fee: 2.5% of transaction
Reward rate: 1 point/$ (1% cost)
Net revenue: 1.5%
```

### A.4 Customer Lifetime Value (CLV)

```
CLV = Σ(t=1 to T) [E[Revenue_t] - E[Cost_t]] / (1 + r)^t

Revenue components:
- Interchange: 2.5% × Spend_t
- Interest: APR × revolvers
- Fees: Annual + Late + Other

Cost components:
- Rewards: Point value × Earned
- Cost of funds: 2% × Balance
- Operations: $50/year
- Expected loss: PD × LGD × EAD

Simplified Payment Network Formula:
CLV = (Annual Spend × 0.025 + Interest Revenue - Reward Cost - OpEx) × 
      Tenure × Retention Rate - Acquisition Cost

Example:
Annual Spend: $24,000
Interest Revenue: $200
Reward Cost: $240 (1%)
OpEx: $50
Tenure: 5 years
Retention: 90%/year
Acquisition: $300

Annual Profit = 600 + 200 - 240 - 50 = $510
5-Year CLV = 510 × 4.1 - 300 = $1,791
```

---

## Appendix B: Quick Reference Tables

### B.1 Critical Values

| Test | 90% | 95% | 99% |
|------|-----|-----|-----|
| Z (two-tailed) | ±1.645 | ±1.96 | ±2.576 |
| Z (one-tailed) | 1.282 | 1.645 | 2.326 |
| t (df=30, two) | ±1.697 | ±2.042 | ±2.750 |
| t (df=100, two) | ±1.660 | ±1.984 | ±2.626 |
| χ² (df=1) | 2.706 | 3.841 | 6.635 |
| χ² (df=5) | 9.236 | 11.070 | 15.086 |
| F (df1=5, df2=100, α=0.05) | - | 2.30 | - |

### B.2 Distribution Formulas

| Distribution | PMF/PDF | Mean | Variance |
|--------------|---------|------|----------|
| Binomial | C(n,k)p^k(1-p)^(n-k) | np | np(1-p) |
| Poisson | e^(-λ)λ^k/k! | λ | λ |
| Normal | (1/σ√2π)e^(-(x-μ)²/2σ²) | μ | σ² |
| Exponential | λe^(-λx) | 1/λ | 1/λ² |
| Log-Normal | - | e^(μ+σ²/2) | (e^σ²-1)e^(2μ+σ²) |
| Beta | x^(α-1)(1-x)^(β-1)/B(α,β) | α/(α+β) | αβ/((α+β)²(α+β+1)) |

### B.3 Model Performance Thresholds

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Accuracy | <85% | 85-92% | 92-97% | >97% |
| Precision | <40% | 40-60% | 60-80% | >80% |
| Recall | <60% | 60-75% | 75-90% | >90% |
| F1-Score | <0.50 | 0.50-0.70 | 0.70-0.85 | >0.85 |
| AUC-ROC | <0.70 | 0.70-0.80 | 0.80-0.90 | >0.90 |
| AUC-PR | <0.30 | 0.30-0.50 | 0.50-0.70 | >0.70 |
| Gini | <0.40 | 0.40-0.60 | 0.60-0.80 | >0.80 |
| KS | <0.20 | 0.20-0.40 | 0.40-0.60 | >0.60 |
| PSI | >0.25 | 0.10-0.25 | <0.10 | <0.05 |

---

## Summary

This guide covers essential statistics concepts for Data Science and Data Analyst interviews in the Finance domain, with specific focus on Payment Network/credit card industry applications:

**Key Areas Mastered:**
1. Descriptive statistics for transaction data analysis
2. Probability distributions for risk modeling
3. Hypothesis testing for A/B tests and model validation
4. Regression analysis for credit scoring
5. Time series for forecasting and anomaly detection
6. Specialized fraud and credit risk statistics
7. Bayesian methods for updating beliefs
8. Model evaluation metrics for financial services

**Payment Network Domain Knowledge:**
- Credit card portfolio metrics (PD, LGD, EAD, CLV)
- Fraud detection statistical methods
- Regulatory compliance (SR 11-7, CCAR)
- Real-world transaction and risk modeling scenarios

**Interview Preparation Tips:**
- Practice with real financial datasets when possible
- Understand business impact of statistical decisions
- Be ready to explain trade-offs (precision vs. recall in fraud)
- Know regulatory requirements for financial models
- Demonstrate ability to communicate technical concepts to business stakeholders
