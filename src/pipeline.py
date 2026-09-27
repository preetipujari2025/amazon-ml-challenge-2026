"""End-to-End Integration Pipeline Harness for Member 3.

Connects:
1. Member 1 Module: Candidate Generation / Blocking (src.preprocessing.blocking)
2. Member 3 Sub-pipeline: Saves output/candidate_pairs.tsv
3. Member 2 Module / Adapter: Matching Model Inference (Member2MatchingModelAdapter)
4. Member 3 Sub-pipeline: Formats output/matching_results.tsv and validates rules

Supports command-line execution for inference and training setup.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Any
from src.interfaces import CandidateGeneratorProtocol, MatchingModelProtocol
from src.submission import generate_submission_outputs, run_official_validator


class PlaceholderCandidateGenerator:
    """Fallback Placeholder for Member 1's Candidate Generator."""

    def generate_candidates(
        self,
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Mock implementation: returns empty candidate set."""
        print("[MEMBER 1 PLACEHOLDER] Running fallback candidate generation...")
        return {}


class PlaceholderMatchingModel:
    """Fallback Placeholder for Member 2's Matching Model."""

    def predict_matches(
        self,
        candidate_pairs: Dict[str, List[str]],
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Mock implementation: returns empty match predictions."""
        print("[MEMBER 2 PLACEHOLDER] Running fallback matching model scoring...")
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
        try:
            from src.adapters import Member1CandidateGeneratorAdapter
            candidate_generator = Member1CandidateGeneratorAdapter()
        except ImportError:
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


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — End-to-End Pipeline Harness (Member 3)"
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "train"],
        default="inference",
        help="Pipeline mode: 'inference' to generate outputs, 'train' to fit models."
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path to folder containing test_source1/2/3.tsv"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to folder for saving generated submission TSV files"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Path to folder containing trained model artifacts"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for matching probabilities (overrides threshold.json)"
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Enable ID existence check in validator against test_source2/3.tsv"
    )

    args = parser.parse_args()

    if args.mode == "inference":
        # Candidate Generator (Member 1)
        try:
            from src.adapters import Member1CandidateGeneratorAdapter
            candidate_gen = Member1CandidateGeneratorAdapter()
            print("[INFO] Using Member 1 Candidate Generator Adapter.")
        except ImportError:
            print("[INFO] Member 1 Candidate Generator Adapter not available. Using Placeholder.")
            candidate_gen = PlaceholderCandidateGenerator()

        # Matching Model Adapter (Member 2)
        fg_path = os.path.join(args.model_dir, "feature_generator.joblib")
        model_path = os.path.join(args.model_dir, "logistic_regression.joblib")
        threshold_path = os.path.join(args.model_dir, "threshold.json")

        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, "random_forest.joblib")

        if os.path.exists(fg_path) and os.path.exists(model_path):
            from src.adapters import Member2MatchingModelAdapter
            th = args.threshold if args.threshold is not None else 0.5
            matching_model = Member2MatchingModelAdapter(
                fg_path=fg_path,
                model_path=model_path,
                threshold_path=threshold_path if os.path.exists(threshold_path) else None,
                threshold=th
            )
            print(f"[INFO] Loaded Member 2 model ({os.path.basename(model_path)}) and feature generator.")
        else:
            print("[INFO] Model artifacts not found under models/. Using PlaceholderMatchingModel.")
            matching_model = PlaceholderMatchingModel()

        run_pipeline(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            candidate_generator=candidate_gen,
            matching_model=matching_model,
            validate=True,
            check_ids=args.check_ids
        )

    elif args.mode == "train":
        print("To train Member 2's models on generated candidates, run:")
        print("  python -m src.train --candidate_pairs output/candidate_pairs.tsv ...")


if __name__ == "__main__":
    main()
