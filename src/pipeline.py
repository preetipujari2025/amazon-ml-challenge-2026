"""End-to-End Integration Pipeline Harness for Member 3.

Connects:
1. Member 1 Module: Candidate Generation / Blocking
2. Member 3 Sub-pipeline: Saves output/candidate_pairs.tsv
3. Member 2 Module: Matching Model Inference
4. Member 3 Sub-pipeline: Formats output/matching_results.tsv and validates rules
"""

import os
from typing import Dict, List, Optional, Any
from src.interfaces import CandidateGeneratorProtocol, MatchingModelProtocol
from src.submission import generate_submission_outputs, run_official_validator


class PlaceholderCandidateGenerator:
    """Integration Placeholder for Member 1's Candidate Generator.

    Member 1 will replace this placeholder with their actual CandidateGenerator class.
    """

    def generate_candidates(
        self,
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Mock implementation: returns a basic candidate set for integration testing."""
        print("[MEMBER 1 PLACEHOLDER] Running candidate generation...")
        # In actual code, Member 1 will implement TF-IDF / Blocking logic here
        return {}


class PlaceholderMatchingModel:
    """Integration Placeholder for Member 2's Matching Model.

    Member 2 will replace this placeholder with their actual MatchingModel class.
    """

    def predict_matches(
        self,
        candidate_pairs: Dict[str, List[str]],
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Mock implementation: returns match predictions for candidates."""
        print("[MEMBER 2 PLACEHOLDER] Running matching model scoring...")
        # In actual code, Member 2 will run feature engineering & model scoring here
        return {}


def run_pipeline(
    test_dir: str,
    output_dir: str = "output",
    candidate_generator: Optional[CandidateGeneratorProtocol] = None,
    matching_model: Optional[MatchingModelProtocol] = None,
    validate: bool = True,
    check_ids: bool = False
) -> Dict[str, str]:
    """Execute end-to-end integration pipeline.

    Args:
        test_dir: Directory containing test_source1.tsv.
        output_dir: Directory to save generated submission TSV files.
        candidate_generator: Instance conforming to CandidateGeneratorProtocol (Member 1).
        matching_model: Instance conforming to MatchingModelProtocol (Member 2).
        validate: Whether to run student_resource/utils/validate_submission.py after generation.
        check_ids: Whether to check ID existence against test_source2/3.tsv during validation.

    Returns:
        Dict of paths to generated output files.
    """
    if candidate_generator is None:
        candidate_generator = PlaceholderCandidateGenerator()
    if matching_model is None:
        matching_model = PlaceholderMatchingModel()

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("=== Pipeline Step 1: Candidate Generation (Member 1) ===")
    candidates = candidate_generator.generate_candidates(s1_path, s2_path, s3_path)

    print("=== Pipeline Step 2: Matching Model Scoring (Member 2) ===")
    matches = matching_model.predict_matches(candidates, s1_path, s2_path, s3_path)

    print("=== Pipeline Step 3: Formatted Submission Generation (Member 3) ===")
    output_files = generate_submission_outputs(
        output_dir=output_dir,
        s1_filepath=s1_path,
        matching_dict=matches,
        candidate_dict=candidates
    )

    if validate:
        print("=== Pipeline Step 4: Official Rules Validation (Member 3) ===")
        exit_code = run_official_validator(
            matching_path=output_files["matching"],
            test_dir=test_dir,
            candidate_path=output_files.get("candidate"),
            check_ids=check_ids
        )
        if exit_code != 0:
            print("WARNING: Output validation failed! Check logs above for details.")
        else:
            print("SUCCESS: Output files passed validation!")

    return output_files
