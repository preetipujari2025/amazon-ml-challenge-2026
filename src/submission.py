"""Submission Formatter and Pre-submission Validation Suite for Member 3.

Handles generating:
- output/matching_results.tsv (final entity matches)
- output/candidate_pairs.tsv (blocking candidate set)

Strictly adheres to official TSV formatting rules:
1. Tab-separated values (sep="\t").
2. Exact header names:
   - matching_results.tsv: ['source1_entity_id', 'matched_entity_ids']
   - candidate_pairs.tsv: ['source1_entity_id', 'candidate_entity_ids']
3. Preserves exact row count and ordering of test_source1.tsv.
4. Singletons mapped to empty string (no quotes, no extra spaces).
5. Invokes student_resource/utils/validate_submission.py for local verification.
"""

import csv
import os
import sys
import subprocess
from typing import Dict, List, Optional, Sequence

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_s1_entity_ids(s1_filepath: str) -> List[str]:
    """Read the exact ordered list of source1 entity IDs from test_source1.tsv.

    Args:
        s1_filepath: Path to test_source1.tsv file.

    Returns:
        List of source1_entity_id strings in original file order.
    """
    s1_ids = []
    with open(s1_filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=DELIM)
        header = next(reader, None)  # Skip header
        for row in reader:
            if row:
                s1_ids.append(row[0].strip())
    return s1_ids


def write_id_list_tsv(
    output_filepath: str,
    header: List[str],
    id_mapping: Dict[str, Sequence[str]],
    s1_id_order: List[str]
) -> str:
    """Write an ID list TSV file (matching_results.tsv or candidate_pairs.tsv).

    Args:
        output_filepath: Absolute or relative output file path.
        header: 2-element header list [s1_col, target_col].
        id_mapping: Dict mapping source1_id -> sequence of target S2/S3 IDs.
        s1_id_order: Ordered list of all source1_entity_id values required.

    Returns:
        The written output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

    with open(output_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=DELIM, lineterminator="\n")
        writer.writerow(header)

        for s1_id in s1_id_order:
            target_ids = id_mapping.get(s1_id, [])
            if target_ids:
                # Comma-separated list with no spaces, no quoting
                formatted_ids = ",".join(target_ids)
            else:
                # Empty string for singletons / no matches
                formatted_ids = ""
            writer.writerow([s1_id, formatted_ids])

    return output_filepath


def generate_submission_outputs(
    output_dir: str,
    s1_filepath: str,
    matching_dict: Dict[str, Sequence[str]],
    candidate_dict: Optional[Dict[str, Sequence[str]]] = None
) -> Dict[str, str]:
    """Generate both output TSV files (matching_results.tsv & candidate_pairs.tsv).

    Args:
        output_dir: Directory where outputs should be saved (e.g., 'output').
        s1_filepath: Path to test_source1.tsv.
        matching_dict: Dict mapping s1_id -> predicted matched IDs.
        candidate_dict: Optional dict mapping s1_id -> candidate IDs.

    Returns:
        Dict with paths to written output files.
    """
    s1_order = read_s1_entity_ids(s1_filepath)

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    write_id_list_tsv(matching_path, MATCHING_HEADER, matching_dict, s1_order)
    results = {"matching": matching_path}

    if candidate_dict is not None:
        candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
        write_id_list_tsv(candidate_path, CANDIDATE_HEADER, candidate_dict, s1_order)
        results["candidate"] = candidate_path

    return results


def run_official_validator(
    matching_path: str,
    test_dir: str,
    candidate_path: Optional[str] = None,
    check_ids: bool = False,
    validator_script: Optional[str] = None
) -> int:
    """Run the official submission validator script via subprocess.

    Args:
        matching_path: Path to matching_results.tsv.
        test_dir: Directory containing test_source1.tsv (and source2/3 if check_ids=True).
        candidate_path: Path to candidate_pairs.tsv.
        check_ids: If True, enables ID existence verification against test_source2/3.tsv.
        validator_script: Path to validate_submission.py.

    Returns:
        Exit code (0 = PASS, 1 = FAIL).
    """
    if validator_script is None:
        validator_script = os.path.join("student_resource", "utils", "validate_submission.py")

    if not os.path.exists(validator_script):
        print(f"WARNING: Validator script not found at {validator_script}")
        return -1

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--test-dir", test_dir
    ]

    if candidate_path and os.path.exists(candidate_path):
        cmd.extend(["--candidate", candidate_path])

    if check_ids:
        cmd.append("--check-ids")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode
