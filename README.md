# Customer Intelligence & Support Copilot — Teaching Kit

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## How students should learn the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

## Part B — Segments and lapse
RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes. The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV and a saved joblib model. The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

## Part C — Ticket triage and privacy
Empty and overlong tickets are skipped before model calls. Email, phone and order ID patterns are redacted before prompting. Retries handle simulated transient errors. JSON category values are allow-listed; malformed outputs are accepted only when exactly one allowed category is detectable using `re.findall`. Human review receives guardrail/invalid/budget cases, fraud goes to security, premium customers go to the premium desk, and the rest go to the support queue.

## Part D — Policy retrieval and refund tools
Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences. Refund rules are implemented in Python tools backed by SQLite: completed orders delivered in over 45 minutes qualify for a full refund; orders over INR 1000 escalate instead of auto-issuing. This is a teaching simulation, not a payment integration.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.

## My Part B Result 

- Best k = 4
- Silhouette Score = 0.525
- Cohort = 1156 customers
- Lapsed = 443 customers (38.3%)
- Baseline Accuracy = 0.616 (61.6%)
- Output files: lapse_model.joblib, retention_list.csv, PartB_output.txt
# Instamart Refund Agent

An LLM agent that handles late-delivery refunds safely using tools, policy guardrails, and human approval.

## Part A - Data
SQLite DB `data/instamart.db` with 3 tables:
- `customers(customer_id, name, ...)`
- `products(product_id, name, price, ...)`
- `orders(order_id, customer_id, product_id, quantity, order_ts, status, discount_pct, order_amount, delivery_minutes)`

## Part B - Policy
Policy in `data/policy.md`:
- Delivery > 45 mins = eligible for refund
- Refund = full order_amount
- Amount < INR 1000 = auto-issue
- Amount >= INR 1000 = escalate to human support lead, do NOT auto-issue

## Part C - Tools
In `src/tools.py`:
- `get_order(order_id)` - fetch order from SQLite
- `calculate_refund(order_amount, delivery_minutes)` - applies policy rule
- `issue_refund(order_id, refund_inr)` - issues refund (guardrailed)
- `escalate_to_human(order_id, reason)` - human approval path

## Part D - Agent (Verified Run)
`src/refund_agent.py` uses ReAct loop with LLM + tools, logs tool calls, llm_calls, tokens_used.

**Verified Output:**
Input: "My order ORD-100048 was delivered late, please refund"
- Order: ORD-100048, Rs 128.25, 65.0 mins
- calculate_refund -> Rs 128.25
- issue_refund -> ORD-100048, Rs 128.25
- llm_calls: 5, tokens_used: 872
- Result: APPROVED, no escalation needed
if __name__ == "__main__":
    # 1. Main run - few-shot
    result, summary = run_pipeline()
    print("MAIN:", summary)
    print(result["status"].value_counts().to_dict())
    print(result["route"].value_counts().to_dict())

    # 2. Budget demo - small budget forces budget_exhausted
    result2, summary2 = run_pipeline(token_budget=200)
    print("BUDGET DEMO:", summary2)

    # 3. Zero-shot demo - ZERO prompt
    result3, summary3 = run_pipeline(prompt=ZERO)
    print("ZERO-SHOT:", summary3)
    # This overwrites triage_budget_demo.csv too, that's expected - run main again last if you want clean file
    result, summary = run_pipeline()
    print("MAIN AGAIN:", summary)