# Amazon ML Challenge 2026: Member 3 Integration Architecture & Design Report

This document details the technical integration architecture for **Member 3 (Integration, Inference, Validation, Output Generation, and Packaging)** for the Amazon ML Challenge 2026.

Every specification below is explicitly tagged as either **[PDF Confirmed]** (mandatory competition requirement from `problem_statement.pdf`) or **[Proposed Design / Suggestion]** (our clean software engineering recommendation).

---

### 1. Expected Inputs and Outputs of Member 1's Module (Candidate Generation / Blocking)

* **Inputs**:
  * **[PDF Confirmed]** Tab-separated business records from Source 1, Source 2, and Source 3: `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv` (`entity_id`, `business_name`, `business_address`, `country`) [PDF Page 1].
  * **[PDF Confirmed]** Ground truth labels for training split: `train_ground_truth.tsv` (used to evaluate candidate recall ceiling) [PDF Page 2, 4].
* **Outputs**:
  * **[PDF Confirmed]** Output file `output/candidate_pairs.tsv` with tab-separated columns: `source1_entity_id` \t `candidate_entity_ids` (comma-separated list of candidate S2/S3 IDs) [PDF Page 3, 4].
  * **[Proposed Design / Suggestion]** Python function/method return value: A dictionary or mapping `Dict[str, List[str]]` mapping each `source1_entity_id` to its list of candidate `entity_id` strings (or a 2-column DataFrame `source1_entity_id`, `candidate_entity_id`).

---

### 2. Expected Inputs and Outputs of Member 2's Module (Feature Engineering & Matching Model)

* **Inputs**:
  * **[PDF Confirmed]** Source business records (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) [PDF Page 1].
  * **[PDF Confirmed]** The candidate pairs produced by Member 1's blocking stage [PDF Page 3, 4].
  * **[PDF Confirmed]** Trained model artifact (must be MIT/Apache 2.0 licensed, $\le$ 8 Billion parameters) [PDF Page 5].
* **Outputs**:
  * **[PDF Confirmed]** Output file `output/matching_results.tsv` with tab-separated columns: `source1_entity_id` \t `matched_entity_ids` (comma-separated list of matched S2/S3 IDs, or empty string for singletons) [PDF Page 3].
  * **[Proposed Design / Suggestion]** Python function/method return value: Probability scores or binary match predictions for candidate pairs.

---

### 3. Interface Between Modules & My Inference Pipeline

* **Pipeline Flow**:
  * **[PDF Confirmed]** Sequential flow: Raw Data $\rightarrow$ Candidate Generation (Blocking) $\rightarrow$ `candidate_pairs.tsv` $\rightarrow$ Matching Model Scoring $\rightarrow$ `matching_results.tsv` [PDF Page 4, 5].
  * **[PDF Confirmed]** **Strict Subset Requirement**: Every ID listed in `matching_results.tsv` MUST be present in `candidate_pairs.tsv` for that `source1_entity_id` [PDF Page 4].
* **Proposed Python Interface (Suggestion)**:
  ```python
  # 1. Member 1 Interface
  candidate_dict = candidate_generator.generate_candidates(
      s1_df, s2_df, s3_df
  )

  # 2. Member 3 Pipeline saves intermediate output
  save_candidate_pairs("output/candidate_pairs.tsv", candidate_dict)

  # 3. Member 2 Interface
  match_dict = matching_model.predict_matches(
      candidate_dict, s1_df, s2_df, s3_df
  )

  # 4. Member 3 Pipeline formats, validates & saves final submission
  save_matching_results("output/matching_results.tsv", match_dict)
```

---

### 4. How to Handle Source 1 Entities with No Matches (Singletons)

* **[PDF Confirmed]** Singletons must have an entry row in both `matching_results.tsv` and `candidate_pairs.tsv` [PDF Page 3, 4].
* **[PDF Confirmed]** The `matched_entity_ids` (or `candidate_entity_ids`) field must be left as an **empty string** (e.g., `S1-00003\t`) [PDF Page 3].
* **[PDF Confirmed]** Singletons are included in the macro-$F_{0.5}$ score: predicting an empty string for a singleton earns **1.0**, while predicting any false match yields **0.0** [PDF Page 6].
* **[Proposed Design / Suggestion]** Post-processing fallback: Our pipeline will perform a full outer join/alignment against the complete list of entity IDs in `test_source1.tsv`. Any missing `source1_entity_id` will automatically default to an empty match list (`""`).

---

### 5. Preserving Source Entity IDs, Row Ordering, and Format

* **[PDF Confirmed]** **File Format**: Tab-separated (`sep="\t"`), no quotes around lists, no spaces in comma-separated IDs (e.g., `S1-00001\tS2-00047,S3-00812`) [PDF Page 1, 3].
* **[PDF Confirmed]** **ID Validation**: `matched_entity_ids` and `candidate_entity_ids` must ONLY contain IDs starting with `S2-` or `S3-` that exist in the test set. No self-matches to `S1-` are permitted [PDF Page 3, 5].
* **[PDF Confirmed]** **Exact Row Count**: Exactly **1,732,544 rows** (matching the number of records in `test_source1.tsv`) [PDF Page 3, 5].
* **[Proposed Design / Suggestion]** **Row Ordering**: Maintain the exact original row index order of `test_source1.tsv` during prediction assembly to guarantee 100% ID alignment and zero missing/duplicate rows.

---

### 6. Automated Validation Suite for Output Files

Our integration validation script (`src/submission.py` / `src/validation.py`) will automatically enforce 8 checks before any submission:

1. **[PDF Confirmed] File Existence Check**: Verify `output/matching_results.tsv` and `output/candidate_pairs.tsv` exist [PDF Page 3, 5].
2. **[PDF Confirmed] Tab Separator Check**: Verify files parse correctly with `sep="\t"` [PDF Page 1, 4].
3. **[PDF Confirmed] Exact Header Check**:
   * `matching_results.tsv`: `source1_entity_id` \t `matched_entity_ids` [PDF Page 3]
   * `candidate_pairs.tsv`: `source1_entity_id` \t `candidate_entity_ids` [PDF Page 4]
4. **[PDF Confirmed] Exact Row Count Check**: Verify total rows == **1,732,544** [PDF Page 3, 5].
5. **[PDF Confirmed] Unique Primary Keys Check**: Zero duplicate `source1_entity_id` rows [PDF Page 3, 5].
6. **[PDF Confirmed] Unique List IDs Check**: Zero duplicate IDs inside any single comma-separated prediction string [PDF Page 3, 5].
7. **[PDF Confirmed] Valid Entity Prefix Check**: All predicted IDs start with `S2-` or `S3-` and exist in `test_source2.tsv` / `test_source3.tsv` [PDF Page 3, 5].
8. **[PDF Confirmed] Candidate Subset Rule Check**: Every ID in `matching_results.tsv` must exist in `candidate_pairs.tsv` for that entity [PDF Page 4].

---

### 7. Checklist of Deliverables to Request From Teammates

#### From Member 1 (Data Loading, Preprocessing & Candidate Generation):
- [ ] Runnable Python module (`src/candidate_generation.py`).
- [ ] Blocking hyperparameter configuration (TF-IDF threshold, n-gram size, top-K candidate limits).
- [ ] Written summary for `Documentation_template.md` detailing candidate generation strategy [PDF Page 5, 7].

#### From Member 2 (Feature Engineering & Matching Model):
- [ ] Runnable Python module (`src/feature_engineering.py` & `src/matching_model.py`).
- [ ] Saved model weights/artifacts placed under `models/` (`.pt` or `.joblib`).
- [ ] Calibrated decision score threshold tuned for Macro-$F_{0.5}$ optimization.
- [ ] Written summary for `Documentation_template.md` detailing model architecture and feature engineering [PDF Page 5, 7].

#### Requirements for Final Packaging (Both Members):
- [ ] Exact list of external Python libraries and pinned versions for `requirements.txt` [PDF Page 5].
