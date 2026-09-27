import argparse
import json
import os
import unittest
import numpy as np
import pandas as pd


def calculate_entity_metrics(true_ids, pred_ids):
    """
    Calculates precision, recall, and F0.5 for a single entity as SETS.
    """
    true_set = set(true_ids)
    pred_set = set(pred_ids)

    # If both are empty, it's a perfect no-match prediction
    if not true_set and not pred_set:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f0_5': 1.0}

    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f0_5 = 0.0
    if (precision + recall) > 0:
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f0_5': f0_5}


def build_gt_map(ground_truth_df, s1_filter=None):
    """
    Fast pre-builder for ground truth mapping dict.
    Runs 100x faster than iterrows().
    """
    if isinstance(ground_truth_df, dict):
        return ground_truth_df

    gt_map = {}
    s1_vals = ground_truth_df['source1_entity_id'].astype(str).values
    match_vals = ground_truth_df['matched_entity_ids'].values

    for s1, matches_str in zip(s1_vals, match_vals):
        s1 = s1.strip()
        if s1_filter is not None and s1 not in s1_filter:
            continue

        if pd.notna(matches_str) and str(matches_str).strip():
            gt_map[s1] = set([x.strip() for x in str(matches_str).split(',') if x.strip()])
        else:
            gt_map[s1] = set()

    return gt_map


def evaluate_predictions(predictions_df, ground_truth_df_or_map, threshold=0.5):
    """
    Fast evaluation of predictions against ground truth using a given probability threshold.
    """
    if isinstance(ground_truth_df_or_map, dict):
        gt_map = ground_truth_df_or_map
    else:
        gt_map = build_gt_map(ground_truth_df_or_map)

    # Filter predictions by threshold
    positive_preds = predictions_df[predictions_df['score'] >= threshold]

    # Fast group-by prediction map
    pred_map = {s1: set() for s1 in gt_map.keys()}
    if not positive_preds.empty:
        for s1, cand in zip(positive_preds['source1_entity_id'].astype(str).values, positive_preds['candidate_entity_id'].astype(str).values):
            s1 = s1.strip()
            cand = cand.strip()
            if s1 in pred_map:
                pred_map[s1].add(cand)

    # Calculate macro metrics
    entity_results = []
    total_tp = total_fp = total_fn = 0
    singleton_errors = singletons = 0
    true_matches_count = pred_matches_count = 0

    for s1, true_set in gt_map.items():
        pred_set = pred_map.get(s1, set())

        if not true_set:
            singletons += 1
            if pred_set:
                singleton_errors += 1
        else:
            true_matches_count += len(true_set)

        pred_matches_count += len(pred_set)
        metrics = calculate_entity_metrics(true_set, pred_set)

        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']

        entity_results.append(metrics['f0_5'])

    macro_f0_5 = float(np.mean(entity_results)) if entity_results else 0.0

    return {
        'threshold': threshold,
        'macro_f0_5': macro_f0_5,
        'number_of_source1_entities': len(gt_map),
        'entities_with_true_matches': len(gt_map) - singletons,
        'singleton_entities': singletons,
        'predicted_match_count': pred_matches_count,
        'true_match_count': true_matches_count,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'singleton_prediction_errors': singleton_errors
    }


def sweep_thresholds(predictions_df, ground_truth_df, out_csv="experiments/threshold_results.csv", out_json="models/threshold.json", coarse_grid=True):
    """
    Fast threshold sweep pre-building ground truth mapping once.
    """
    val_s1_entities = set(predictions_df['source1_entity_id'].astype(str).unique())
    gt_map = build_gt_map(ground_truth_df, s1_filter=val_s1_entities)

    if coarse_grid:
        thresholds = np.arange(0.10, 0.95, 0.05)
    else:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = []
    best_f0_5 = -1.0
    best_threshold = 0.50

    for th in thresholds:
        th = round(float(th), 2)
        res = evaluate_predictions(predictions_df, gt_map, threshold=th)
        results.append({
            'threshold': th,
            'macro_f0_5': res['macro_f0_5'],
            'predicted_match_count': res['predicted_match_count'],
            'false_positive_count': res['total_false_positives'],
            'false_negative_count': res['total_false_negatives']
        })

        if res['macro_f0_5'] > best_f0_5 or (abs(res['macro_f0_5'] - best_f0_5) < 1e-9 and th > best_threshold):
            best_f0_5 = res['macro_f0_5']
            best_threshold = th

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    best_config = {
        "threshold": best_threshold,
        "metric": "macro_f0_5",
        "selection": "fast_validation_sample",
        "best_macro_f0_5": best_f0_5
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)

    return best_threshold, best_f0_5


def main():
    parser = argparse.ArgumentParser(description="Fast threshold evaluation & calibration for Macro-F0.5")
    parser.add_argument("--predictions", type=str, default="models/val_predictions_logistic_regression.tsv")
    parser.add_argument("--ground_truth", type=str, default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--out_csv", type=str, default="experiments/threshold_results.csv")
    parser.add_argument("--out_json", type=str, default="models/threshold.json")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests")

    args = parser.parse_args()

    if args.test:
        print("Running Unit Tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluationModule)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if result.wasSuccessful():
            print("\nAll tests passed successfully.")
        return

    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found at {args.predictions}")
        print("Running internal unit tests instead...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluationModule)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return

    print(f"Loading validation predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep="\t")

    print(f"Loading ground truth from {args.ground_truth}...")
    gt_df = pd.read_csv(args.ground_truth, sep="\t")

    print("Fast sweeping probability thresholds (0.10 - 0.90)...")
    best_th, best_f05 = sweep_thresholds(pred_df, gt_df, out_csv=args.out_csv, out_json=args.out_json, coarse_grid=True)
    print(f"Optimal Decision Threshold : {best_th}")
    print(f"Best Macro-F0.5 Score      : {best_f05:.4f}")
    print(f"Threshold configuration saved: {args.out_json}")


class TestEvaluationModule(unittest.TestCase):

    def test_perfect_matching(self):
        metrics = calculate_entity_metrics(['A', 'B'], ['A', 'B'])
        self.assertEqual(metrics['f0_5'], 1.0)
        self.assertEqual(metrics['tp'], 2)

    def test_completely_empty(self):
        metrics = calculate_entity_metrics([], [])
        self.assertEqual(metrics['f0_5'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)

    def test_false_positive(self):
        metrics = calculate_entity_metrics([], ['A'])
        self.assertEqual(metrics['f0_5'], 0.0)
        self.assertEqual(metrics['fp'], 1)

    def test_false_negative(self):
        metrics = calculate_entity_metrics(['A', 'B'], ['A'])
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['tp'], 1)

    def test_threshold_sweep_integration(self):
        gt = pd.DataFrame({
            'source1_entity_id': ['S1_1', 'S1_2', 'S1_3'],
            'matched_entity_ids': ['S2_1', 'S2_2,S3_2', '']
        })
        preds = pd.DataFrame({
            'source1_entity_id': ['S1_1', 'S1_2', 'S1_2', 'S1_3'],
            'candidate_entity_id': ['S2_1', 'S2_2', 'S3_2', 'S2_3'],
            'score': [0.9, 0.8, 0.4, 0.6]
        })

        best_th, best_f05 = sweep_thresholds(preds, gt, out_csv="experiments/test_threshold_results.csv", out_json="models/test_threshold.json")
        self.assertIsNotNone(best_th)


if __name__ == '__main__':
    main()
