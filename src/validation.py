"""Local Macro-F0.5 Evaluation Module for Member 3.

Implements official competition evaluation metric calculation according to problem_statement.pdf:
- Metric: F_beta with beta = 0.5 (Precision weighted 2x over Recall).
- Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Macro-averaged across all Source 1 entities in evaluation split.
- Singletons included:
  - Correctly predicting empty match list for a singleton scores 1.0.
  - Predicting any false match for a singleton scores 0.0.
"""

from typing import Dict, List, Sequence, Set, Tuple


def calculate_entity_f05(
    predicted_ids: Sequence[str],
    ground_truth_ids: Sequence[str]
) -> float:
    """Calculate F_0.5 score for a single Source 1 entity.

    Args:
        predicted_ids: Sequence of predicted S2/S3 entity IDs.
        ground_truth_ids: Sequence of ground truth S2/S3 entity IDs.

    Returns:
        F_0.5 score between 0.0 and 1.0.
    """
    pred_set: Set[str] = set(predicted_ids)
    true_set: Set[str] = set(ground_truth_ids)

    # Handling singletons (entities with no true matches)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0  # Correctly identified singleton
        else:
            return 0.0  # False merge on singleton

    if len(pred_set) == 0:
        return 0.0  # Missed all matches for non-singleton

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def evaluate_macro_f05(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    all_s1_ids: Sequence[str] = None
) -> Tuple[float, Dict[str, float]]:
    """Calculate Macro-averaged F_0.5 score across all Source 1 entities.

    Args:
        predictions: Dict mapping source1_entity_id -> predicted matched IDs.
        ground_truth: Dict mapping source1_entity_id -> ground truth matched IDs.
        all_s1_ids: Optional explicit sequence of all evaluation S1 IDs.
                    If None, uses the union of keys in ground_truth and predictions.

    Returns:
        Tuple of (overall_macro_f05, breakdown_dict) where breakdown_dict
        contains detailed stats (mean_f05, precision, recall, singleton_acc).
    """
    if all_s1_ids is None:
        eval_ids = list(ground_truth.keys())
    else:
        eval_ids = list(all_s1_ids)

    if not eval_ids:
        return 0.0, {"macro_f05": 0.0, "total_entities": 0}

    scores = []
    singleton_count = 0
    singleton_correct = 0

    for s1_id in eval_ids:
        pred = predictions.get(s1_id, [])
        truth = ground_truth.get(s1_id, [])

        score = calculate_entity_f05(pred, truth)
        scores.append(score)

        if len(truth) == 0:
            singleton_count += 1
            if len(pred) == 0:
                singleton_correct += 1

    macro_f05 = sum(scores) / len(scores)
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0

    breakdown = {
        "macro_f05": macro_f05,
        "total_entities": len(eval_ids),
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc
    }

    return macro_f05, breakdown
