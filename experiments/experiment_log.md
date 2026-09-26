# Experiment Log — Business Entity Resolution

## Experiment 01: Initial Dataset Inspection & Verification
- **Date**: 2026-09-25
- **Member**: Member 1
- **Focus**: Dataset identification, structural integrity, schema validation, distributions, ground truth analysis.
- **Output Report**: [experiments/01_data_inspection.txt](file:///C:/Users/Rashika/Documents/amazon-ml-challenge-2026/experiments/01_data_inspection.txt)

### Summary of Dataset Files
- **train_source1.tsv**: 2,206,821 rows, 4 columns (deduplicated reference source)
- **train_source2.tsv**: 5,034,616 rows, 4 columns (3.36% missing address)
- **train_source3.tsv**: 5,285,603 rows, 4 columns (3.33% missing address)
- **train_ground_truth.tsv**: 2,206,821 rows, 2 columns (labels for train_source1)
- **test_source1.tsv**: 1,732,544 rows, 4 columns (deduplicated reference source to match)
- **test_source2.tsv**: 4,887,273 rows, 4 columns (2.65% missing address)
- **test_source3.tsv**: 5,082,316 rows, 4 columns (2.68% missing address)
- **Total records**: 26,435,994 rows across 7 files (~2.4 GB uncompressed)

### Key Ground Truth Findings
- Total S1 Entities: 2,206,821
- Zero Matches (Singletons): 123,247 (5.58%)
- Exactly One Match: 119,157 (5.40%)
- Multiple Matches: 1,964,417 (89.02%)
- Match count range: 0 to 11 (Mean: 3.46, Median: 3.0)
- Total matched pairs: 7,638,365 (3,693,619 to S2, 3,944,746 to S3)
- Distractor rates in S2 and S3: ~26.6% in S2 and ~25.4% in S3 have no match in S1.

### Key Distribution & Preprocessing Findings
- **Zero entity ID duplicates** across all individual files.
- **100% ID alignment**: Every entity in `train_source1.tsv` has a corresponding row in `train_ground_truth.tsv`. All matched IDs in GT exist in S2/S3.
- **Country shift**: Training contains `US` (~60%) and `India` (~40%). Test introduces `France` (~14.4%–15.0%), with India (~47%) and US (~38%).
- **Multilingual content**: India records include Hindi, Kannada, Telugu, etc., and French records include French text and accented characters. Addressing and name tokenization must handle non-ASCII / Unicode scripts.
- **Missing values**: `business_address` has ~2.6% to ~3.4% missing values in Source 2 and Source 3 across train and test. S1 has 0% missing address.
