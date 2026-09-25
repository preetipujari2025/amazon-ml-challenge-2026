"""Module Interfaces for Amazon ML Challenge 2026.

This module defines standard Python Protocol interfaces for:
- Member 1: Candidate Generation / Blocking module
- Member 2: Feature Engineering & Matching Model module

Using Python standard library typing.Protocol for zero-dependency contract enforcement.
"""

from typing import Dict, List, Any, Protocol


class CandidateGeneratorProtocol(Protocol):
    """Interface for Member 1's Candidate Generation (Blocking) module."""

    def generate_candidates(
        self,
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Generate candidate S2/S3 entity IDs for each Source 1 entity ID.

        Args:
            s1_data: Source 1 entity records (filepath, list of dicts, or DataFrame).
            s2_data: Source 2 entity records.
            s3_data: Source 3 entity records.

        Returns:
            Dict mapping source1_entity_id -> list of candidate entity_ids (S2-xxx, S3-yyy).
            For singletons (entities with no candidates), the list should be empty [].
        """
        ...


class MatchingModelProtocol(Protocol):
    """Interface for Member 2's Matching Model module."""

    def predict_matches(
        self,
        candidate_pairs: Dict[str, List[str]],
        s1_data: Any,
        s2_data: Any,
        s3_data: Any
    ) -> Dict[str, List[str]]:
        """Score candidate pairs and predict final entity matches.

        Args:
            candidate_pairs: Dict mapping source1_entity_id -> list of candidate entity_ids.
            s1_data: Source 1 entity records.
            s2_data: Source 2 entity records.
            s3_data: Source 3 entity records.

        Returns:
            Dict mapping source1_entity_id -> list of matched entity_ids (S2-xxx, S3-yyy).
            For singletons or entities with no matches, the list should be empty [].
        """
        ...
