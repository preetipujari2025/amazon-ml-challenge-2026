"""Integration Test for Member 3 Framework using Artificial Synthetic Data.

This test uses zero real competition dataset files and touches zero ground-truth test labels.
It creates a tiny temporary test environment to verify:
1. Candidate pairs output formatting.
2. Matching results output formatting.
3. Singleton handling (empty string after tab).
4. Full validation pass using student_resource/utils/validate_submission.py (with --check-ids).
5. Macro-F0.5 local metric calculation logic.
"""

import os
import shutil
import tempfile
import unittest

from src.submission import generate_submission_outputs, run_official_validator
from src.validation import evaluate_macro_f05
from src.pipeline import run_pipeline


class SyntheticCandidateGenerator:
    """Mock candidate generator for artificial test data."""

    def generate_candidates(self, s1_data, s2_data, s3_data):
        return {
            "S1-0001": ["S2-0010", "S3-0040"],
            "S1-0002": ["S2-0020"],
            "S1-0003": ["S3-0030"],
            "S1-0004": ["S3-0040"],
            "S1-0005": []  # Singleton entity
        }


class SyntheticMatchingModel:
    """Mock matching model for artificial test data."""

    def predict_matches(self, candidate_pairs, s1_data, s2_data, s3_data):
        return {
            "S1-0001": ["S2-0010"],          # Correct match
            "S1-0002": ["S2-0020"],          # Correct match
            "S1-0003": ["S3-0030"],          # Correct match
            "S1-0004": [],                   # Singleton prediction
            "S1-0005": []                    # True singleton prediction
        }


class TestSyntheticPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = os.path.join(self.temp_dir, "dataset", "test")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Create artificial test_source1.tsv (5 entities)
        s1_file = os.path.join(self.test_dir, "test_source1.tsv")
        with open(s1_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-0001\tAlpha Corp\t100 Main St\tUS\n")
            f.write("S1-0002\tBeta Inc\t200 Oak Ave\tUS\n")
            f.write("S1-0003\tGamma LLC\t300 Pine Rd\tIndia\n")
            f.write("S1-0004\tDelta Co\t400 Maple Dr\tUS\n")
            f.write("S1-0005\tEpsilon Enterprise\t500 Elm St\tFrance\n")

        # 2. Create artificial test_source2.tsv
        s2_file = os.path.join(self.test_dir, "test_source2.tsv")
        with open(s2_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-0010\tAlpha Corporation\t100 Main Street\tUS\n")
            f.write("S2-0020\tBeta Incorporated\t200 Oak Avenue\tUS\n")

        # 3. Create artificial test_source3.tsv
        s3_file = os.path.join(self.test_dir, "test_source3.tsv")
        with open(s3_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-0030\tGamma Limited\t300 Pine Road\tIndia\n")
            f.write("S3-0040\tDelta Company\t400 Maple Drive\tUS\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_pipeline_output_generation_and_validation(self):
        """Test full pipeline execution and official validator on synthetic data."""
        gen = SyntheticCandidateGenerator()
        model = SyntheticMatchingModel()

        # Run pipeline
        output_files = run_pipeline(
            test_dir=self.test_dir,
            output_dir=self.output_dir,
            candidate_generator=gen,
            matching_model=model,
            validate=False
        )

        self.assertTrue(os.path.exists(output_files["matching"]))
        self.assertTrue(os.path.exists(output_files["candidate"]))

        # Verify exact line counts (Header + 5 rows = 6 lines) using rstrip('\r\n')
        with open(output_files["matching"], "r", encoding="utf-8") as f:
            matching_lines = [line.rstrip("\r\n") for line in f if line.rstrip("\r\n")]
        self.assertEqual(len(matching_lines), 6)
        self.assertEqual(matching_lines[0], "source1_entity_id\tmatched_entity_ids")
        # Check singleton formatting on line 6 (S1-0005\t)
        self.assertEqual(matching_lines[5], "S1-0005\t")

        # Run official validator with --check-ids
        exit_code = run_official_validator(
            matching_path=output_files["matching"],
            test_dir=self.test_dir,
            candidate_path=output_files["candidate"],
            check_ids=True
        )
        self.assertEqual(exit_code, 0, "Synthetic output failed official submission validator!")

    def test_macro_f05_evaluation_metric(self):
        """Test local Macro-F0.5 calculation logic."""
        ground_truth = {
            "S1-0001": ["S2-0010"],
            "S1-0002": ["S2-0020"],
            "S1-0003": ["S3-0030"],
            "S1-0004": ["S3-0040"],  # Model predicted [] -> score 0.0
            "S1-0005": []            # Model predicted [] -> singleton score 1.0
        }

        predictions = SyntheticMatchingModel().predict_matches(None, None, None, None)

        macro_f05, breakdown = evaluate_macro_f05(predictions, ground_truth)

        # S1-0001: 1.0, S1-0002: 1.0, S1-0003: 1.0, S1-0004: 0.0, S1-0005: 1.0
        # Expected average = (1.0 + 1.0 + 1.0 + 0.0 + 1.0) / 5 = 4.0 / 5 = 0.8
        self.assertAlmostEqual(macro_f05, 0.8)
        self.assertEqual(breakdown["total_entities"], 5)
        self.assertEqual(breakdown["singleton_count"], 1)
        self.assertEqual(breakdown["singleton_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
