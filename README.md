# Amazon ML Challenge 2026 — Business Entity Resolution

Comprehensive ML pipeline and integration repository for **Member 3 (Integration, Inference, Validation, and Final Submission Packaging)**, unifying Member 1's blocking algorithms and Member 2's feature engineering & machine learning model.

---

## 1. Integrated System Architecture

```text
[Raw Business Record TSV Files]
  │
  ▼
Stage 1: Candidate Generation / Blocking (Member 1)
  ├── Module: src/preprocessing/blocking.py
  ├── Normalization: src/preprocessing/normalize_names.py & normalize_address.py
  ├── Inverted Indexing: Character / Name / Address blocking keys
  └── Output: output/candidate_pairs.tsv (Audit & Candidate Candidate File)
  │
  ▼
Stage 2: Feature Engineering & Matching Model Scoring (Member 2)
  ├── Module: src/features.py (31 Pairwise String Distances & Char TF-IDF Cosine Similarities)
  ├── Model: src/model.py (EntityResolutionModel: LogisticRegression / RandomForest)
  ├── Training & Evaluation: src/train.py & src/evaluate.py (Macro-F0.5 threshold sweep)
  └── Adapter: src/adapters.py (Member1CandidateGeneratorAdapter & Member2MatchingModelAdapter)
  │
  ▼
Stage 3: Submission Output & Official Validation (Member 3)
  ├── Formatting & Singletons: src/submission.py (output/matching_results.tsv)
  ├── Evaluation Metric: src/validation.py (Local Macro-F0.5 Calculator)
  ├── Pipeline Harness: src/pipeline.py (End-to-End Orchestrator)
  └── Official Rules Validator: student_resource/utils/validate_submission.py
```

---

## 2. Workspace Setup

### Activate Virtual Environment

```powershell
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install / verify pinned dependencies
pip install -r requirements.txt
```

---

## 3. Running Synthetic Integration Tests

Execute the complete 3-member synthetic unit and integration test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_full_synthetic_integration.py
.\.venv\Scripts\python.exe -m unittest tests/test_adapter_synthetic.py
.\.venv\Scripts\python.exe -m unittest tests/test_synthetic_pipeline.py
```

---

## 4. Real Data Execution Commands

### Step 1: Generate Candidate Pairs on Training Data (Member 1)

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.blocking
```

### Step 2: Train Model & Sweep Thresholds (Member 2)

```powershell
.\.venv\Scripts\python.exe -m src.train `
  --candidate_pairs dataset/train/candidate_pairs.tsv `
  --s1_data dataset/train/train_source1.tsv `
  --s2_data dataset/train/train_source2.tsv `
  --s3_data dataset/train/train_source3.tsv `
  --gt_data dataset/train/train_ground_truth.tsv `
  --out_dir models/
```

### Step 3: End-to-End Test Inference & Submission Generation (Member 3)

```powershell
.\.venv\Scripts\python.exe -m src.pipeline `
  --mode inference `
  --test-dir dataset/test `
  --output-dir output `
  --model-dir models `
  --check-ids
```

### Step 4: Run Official Submission Validator

```powershell
.\.venv\Scripts\python.exe student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test `
  --check-ids
```

---

## 5. Final Package Structure

Per `problem_statement.pdf` [Page 5], the final submission archive `<team_name>_submission.zip` contains:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv      # Leaderboard evaluation file
│   └── candidate_pairs.tsv       # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                  # Source code (adapters, preprocessing, features, model, pipeline, submission, validation)
│       ├── README.md             # End-to-end run instructions
│       └── requirements.txt      # Pinned environment requirements
└── Documentation_template.md     # Methodology write-up
```
