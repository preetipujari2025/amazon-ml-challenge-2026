import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
import warnings
import unittest

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

def evaluate_predictions(predictions_df, ground_truth_df, threshold=0.5):
    """
    Evaluates predictions against ground truth using a given probability threshold.
    predictions_df: must have ['source1_entity_id', 'candidate_entity_id', 'score']
    ground_truth_df: must have ['source1_entity_id', 'matched_entity_ids']
    """
    if not all(col in predictions_df.columns for col in ['source1_entity_id', 'candidate_entity_id', 'score']):
        raise ValueError("predictions_df missing required columns.")
    if not all(col in ground_truth_df.columns for col in ['source1_entity_id', 'matched_entity_ids']):
        raise ValueError("ground_truth_df missing required columns.")
        
    # Validations
    if predictions_df['score'].isna().any():
        raise ValueError("Score column contains missing values.")
    if not pd.api.types.is_numeric_dtype(predictions_df['score']):
        raise ValueError("Scores must be numeric.")
    if (predictions_df['score'] < 0).any() or (predictions_df['score'] > 1).any():
        raise ValueError("Scores must be probabilities between 0 and 1.")
    if predictions_df['source1_entity_id'].isna().any() or predictions_df['candidate_entity_id'].isna().any():
        raise ValueError("Entity IDs cannot be missing.")
        
    # Ground truth mapping
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1 = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids'])
        if matches_str and matches_str.lower() != 'nan' and matches_str.strip():
            gt_map[s1] = set([x.strip() for x in matches_str.split(',') if x.strip()])
        else:
            gt_map[s1] = set()

    # Create predictions mapping
    pred_map = {s1: set() for s1 in gt_map.keys()} # Initialize all GT entities
    
    # Filter predictions by threshold
    positive_preds = predictions_df[predictions_df['score'] >= threshold]
    
    for _, row in positive_preds.iterrows():
        s1 = row['source1_entity_id']
        cand = row['candidate_entity_id']
        if s1 not in pred_map:
            pred_map[s1] = set()
        pred_map[s1].add(cand)
        
    # Calculate macro metrics
    entity_results = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_errors = 0
    singletons = 0
    true_matches_count = 0
    pred_matches_count = 0
    
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
        
        entity_results.append({
            'source1_entity_id': s1,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f0_5': metrics['f0_5']
        })
        
    results_df = pd.DataFrame(entity_results)
    
    return {
        'threshold': threshold,
        'macro_precision': results_df['precision'].mean(),
        'macro_recall': results_df['recall'].mean(),
        'macro_f0_5': results_df['f0_5'].mean(),
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

def sweep_thresholds(predictions_df, ground_truth_df, out_csv="experiments/threshold_results.csv", out_json="models/threshold.json"):
    """
    Sweeps threshold from 0.05 to 0.95 with step 0.01.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    
    best_f0_5 = -1.0
    best_threshold = -1.0
    
    for th in thresholds:
        th = round(th, 2)
        res = evaluate_predictions(predictions_df, ground_truth_df, threshold=th)
        results.append({
            'threshold': th,
            'macro_precision': res['macro_precision'],
            'macro_recall': res['macro_recall'],
            'macro_f0_5': res['macro_f0_5'],
            'predicted_match_count': res['predicted_match_count'],
            'false_positive_count': res['total_false_positives'],
            'false_negative_count': res['total_false_negatives']
        })
        
        # Tie breaker: choose higher threshold
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
        "selection": "validation",
        "best_macro_f0_5": best_f0_5
    }
    with open(out_json, "w") as f:
        json.dump(best_config, f, indent=4)
        
    return best_threshold, best_f0_5


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
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 0.5)
        
    def test_multiple_correct_matches(self):
        metrics = calculate_entity_metrics(['A', 'B', 'C'], ['A', 'B', 'C'])
        self.assertEqual(metrics['f0_5'], 1.0)
        
    def test_mixed(self):
        metrics = calculate_entity_metrics(['A', 'B'], ['B', 'C'])
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertAlmostEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 0.5)
        self.assertAlmostEqual(metrics['f0_5'], 0.5)
        
    def test_duplicate_prediction_ids(self):
        metrics = calculate_entity_metrics(['A'], ['A', 'A'])
        self.assertEqual(metrics['f0_5'], 1.0) # SET logic removes duplicates
        
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
        
        # At threshold 0.5:
        # S1_1: pred={S2_1} -> f0.5 = 1.0
        # S1_2: pred={S2_2} (S3_2 is 0.4 < 0.5) -> precision=1.0, recall=0.5 -> f0.5=0.8333
        # S1_3: pred={S2_3} -> fp=1, precision=0.0 -> f0.5 = 0.0
        # Macro F0.5 = (1.0 + 0.8333 + 0.0) / 3 = 0.6111
        
        res_05 = evaluate_predictions(preds, gt, threshold=0.5)
        self.assertAlmostEqual(res_05['macro_f0_5'], (1.0 + 5/6 + 0.0)/3)
        self.assertEqual(res_05['singleton_prediction_errors'], 1)
        
        # At threshold 0.7:
        # S1_1: 1.0
        # S1_2: 0.8333
        # S1_3: 1.0 (since score 0.6 < 0.7, pred={})
        # Macro F0.5 = (1.0 + 0.8333 + 1.0) / 3 = 0.9444
        res_07 = evaluate_predictions(preds, gt, threshold=0.7)
        self.assertAlmostEqual(res_07['macro_f0_5'], (1.0 + 5/6 + 1.0)/3)
        
        # Test threshold sweep
        best_th, best_f05 = sweep_thresholds(preds, gt, out_csv="experiments/test_threshold_results.csv", out_json="models/test_threshold.json")
        self.assertEqual(best_th, 0.8) # Should pick higher threshold yielding best score

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Predictions")
    parser.add_argument("--predictions", type=str, default="models/val_predictions_logistic_regression.tsv",
                        help="Path to validation predictions TSV.")
    parser.add_argument("--gt_data", type=str, default="dataset/train/train_ground_truth.tsv",
                        help="Path to ground truth TSV.")
    parser.add_argument("--out_csv", type=str, default="experiments/threshold_results.csv",
                        help="Path to threshold sweep output CSV.")
    parser.add_argument("--out_json", type=str, default="models/threshold.json",
                        help="Path to best threshold output JSON.")
    parser.add_argument("--test", action="store_true", help="Run unit tests instead of evaluation.")
    args = parser.parse_args()

    if args.test:
        print("Running Unit Tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluationModule)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
    else:
        if not os.path.exists(args.predictions):
            print(f"Predictions file not found: {args.predictions}")
            print("Run python src/train.py first to generate predictions.")
            exit(1)
            
        print(f"Loading predictions from {args.predictions}...")
        pred_df = pd.read_csv(args.predictions, sep='\t')
        print(f"Loading ground truth from {args.gt_data}...")
        gt_df = pd.read_csv(args.gt_data, sep='\t', dtype=str)
        
        # Filter ground truth to entities present in predictions to evaluate fairly
        pred_s1 = set(pred_df['source1_entity_id'])
        gt_sub = gt_df[gt_df['source1_entity_id'].isin(pred_s1)].reset_index(drop=True)
        print(f"Evaluating {len(pred_df):,} predictions for {len(gt_sub):,} validation S1 entities...")
        
        best_th, best_f05 = sweep_thresholds(pred_df, gt_sub, out_csv=args.out_csv, out_json=args.out_json)
        print(f"\nEvaluation Results:")
        print(f"  Best Threshold: {best_th:.2f}")
        print(f"  Best Macro F0.5: {best_f05:.4f}")
        print(f"  Results saved to: {args.out_csv} and {args.out_json}")

