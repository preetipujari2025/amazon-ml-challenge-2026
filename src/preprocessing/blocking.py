"""
Candidate blocking for business entity resolution.

Generates candidate pairs between Source 1 and Sources 2/3
using normalized business names and addresses.

Memory strategy
---------------
Source 2 and Source 3 are streamed row-by-row into the inverted
blocking index.  We never hold a full copy of either file in RAM.
Source 1 is also streamed when we look up candidates.

The inverted index maps a blocking key to the set of candidate entity
IDs that share that key.  Its size is bounded by the number of distinct
keys (which is far smaller than the raw row count) plus the number of
candidate IDs per key.
"""

from __future__ import annotations

import csv
import re
import sys
from itertools import chain
from pathlib import Path
from typing import Generator, Iterable

from src.preprocessing.normalize_address import normalize_address
from src.preprocessing.normalize_names import normalize_name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_tsv(path: Path) -> Generator[dict[str, str], None, None]:
    """Yield rows from a TSV file one at a time (streaming, low memory)."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _name_key(name: str) -> str:
    """Return a blocking key derived from a normalized business name."""
    return normalize_name(name or "")


def _address_key(address: str) -> str:
    """Return a blocking key derived from a normalized business address."""
    normalized = normalize_address(address or "")
    # Strip all non-alphanumeric characters for a stable, compact key.
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def _row_keys(row: dict[str, str]) -> set[str]:
    """Return the set of blocking keys for a single row."""
    keys: set[str] = set()
    name_k = _name_key(row.get("business_name", ""))
    addr_k = _address_key(row.get("business_address", ""))
    if name_k:
        keys.add(f"name:{name_k}")
    if addr_k:
        keys.add(f"address:{addr_k}")
    return keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_blocking_index(
    rows: Iterable[dict[str, str]],
) -> dict[str, set[str]]:
    """
    Build an inverted blocking index from *rows*.

    Maps each blocking key to the set of entity IDs that share it.
    ``rows`` may be a generator -- it is consumed exactly once and never
    stored wholesale in memory.
    """
    index: dict[str, set[str]] = {}
    for row in rows:
        entity_id = row.get("entity_id", "")
        if not entity_id:
            continue
        for key in _row_keys(row):
            index.setdefault(key, set()).add(entity_id)
    return index


def generate_candidate_pairs(
    source1_rows: Iterable[dict[str, str]],
    source2_rows: Iterable[dict[str, str]],
    source3_rows: Iterable[dict[str, str]],
) -> list[tuple[str, str]]:
    """
    Generate candidate Source1-to-(Source2/Source3) entity pairs.

    A pair is generated when the records share at least one blocking key
    (normalized business name OR normalized address).

    Memory-conscious design
    -----------------------
    * Source 2 and Source 3 are streamed together via ``itertools.chain``
      into ``build_blocking_index``.  Neither file is ever materialised
      as a complete list.
    * Source 1 is also streamed; only its per-row keys are kept in memory
      at one time.
    * Duplicate pairs are de-duplicated via a ``set`` of tuples, which is
      far smaller than the raw row data.
    """
    # --- Phase 1: build inverted index from Source2 + Source3 (streaming) ---
    print(
        "[blocking] Building inverted index from Source 2 + Source 3 ...",
        file=sys.stderr,
        flush=True,
    )

    index = build_blocking_index(chain(source2_rows, source3_rows))

    print(
        f"[blocking] Index built: {len(index):,} unique blocking keys.",
        file=sys.stderr,
        flush=True,
    )

    # --- Phase 2: scan Source1 and look up candidates ---
    print(
        "[blocking] Scanning Source 1 to generate candidate pairs ...",
        file=sys.stderr,
        flush=True,
    )

    candidates: set[tuple[str, str]] = set()
    source1_count = 0

    for row in source1_rows:
        source1_id = row.get("entity_id", "")
        if not source1_id:
            continue
        source1_count += 1

        for key in _row_keys(row):
            for candidate_id in index.get(key, set()):
                candidates.add((source1_id, candidate_id))

        if source1_count % 200_000 == 0:
            print(
                f"[blocking]   Processed {source1_count:,} Source-1 rows, "
                f"{len(candidates):,} candidate pairs so far ...",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"[blocking] Done. {source1_count:,} Source-1 rows scanned, "
        f"{len(candidates):,} candidate pairs total.",
        file=sys.stderr,
        flush=True,
    )

    return sorted(candidates)


def write_candidate_pairs(
    candidates: Iterable[tuple[str, str]],
    output_path: Path,
) -> None:
    """Write candidate pairs to a TSV file with header source1_id / candidate_id."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_id", "candidate_id"])
        for source1_id, candidate_id in candidates:
            writer.writerow([source1_id, candidate_id])


def run_blocking(
    dataset_root: Path,
    output_path: Path,
) -> int:
    """
    Run candidate blocking on the full training data and write results.

    Parameters
    ----------
    dataset_root:
        Root directory that contains the ``train/`` subdirectory.
    output_path:
        Destination TSV file for candidate pairs.

    Returns
    -------
    int
        Number of candidate pairs written.
    """
    train_dir = dataset_root / "train"

    source1 = train_dir / "train_source1.tsv"
    source2 = train_dir / "train_source2.tsv"
    source3 = train_dir / "train_source3.tsv"

    for p in (source1, source2, source3):
        if not p.exists():
            raise FileNotFoundError(f"Training file not found: {p}")

    candidates = generate_candidate_pairs(
        _read_tsv(source1),
        _read_tsv(source2),
        _read_tsv(source3),
    )

    write_candidate_pairs(candidates, output_path)

    return len(candidates)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    dataset_root = (
        project_root
        / "6ab10eb3b23ba_student_resource"
        / "student_resource"
        / "dataset"
    )

    output_path = project_root / "experiments" / "candidate_pairs.tsv"

    print(f"[blocking] Dataset root : {dataset_root}", file=sys.stderr)
    print(f"[blocking] Output path  : {output_path}", file=sys.stderr)

    count = run_blocking(dataset_root, output_path)

    print(f"Generated {count:,} candidate pairs.")
    print(f"Output: {output_path}")